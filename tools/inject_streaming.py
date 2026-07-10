#!/usr/bin/env python3
"""Inject streaming plane-building into the installed vLLM-Moet, post-patch.

The stock patch host-stages ALL NVFP4 experts (~380 GiB for GLM-5.2) to CPU
during load, THEN builds 2-bit planes per layer in process_weights_after_loading.
Peak host RAM = the full staged set, which OOMs a RunPod container cgroup
(377 GiB on 2 cards, 565 GiB on 4). This injects a per-layer streaming build:
as each layer's expert weights finish loading (detected GAP layers later, since
GLM shards load layer-ordered), build that layer's planes and free its NVFP4
immediately. Peak drops to ~GAP layers + the accumulating 2-bit base.

Opt-in via VLLM_MOE_W2_STREAM=1 (default off => byte-identical to stock, so the
proven DS4 single-card path is unaffected). Fail-safe: process_weights_after_
loading still builds any layer not streamed, so a mistimed trigger cannot
corrupt output (an un-built layer just builds late as today). Idempotent-safe:
each layer builds exactly once (guarded on layer._moe_w2_key), keys stay 0..N-1
contiguous, and each layer stores its own key.
"""
import os, re, vllm

root = os.path.dirname(vllm.__file__)
cubit = os.path.join(root, "model_executor/layers/quantization/utils/moe_w2_cubit.py")
mopt = os.path.join(root, "model_executor/layers/quantization/modelopt.py")

# ---- 1. append streaming helpers to moe_w2_cubit.py ----
HELPERS = '''

# --- VLLM_MOE_W2 streaming build (bound host RAM during load) ------------
_STREAM_LAYERS = {}
_STREAM_MAX_SEEN = [-1]
def _stream_gap():
    try: return int(os.getenv("VLLM_MOE_W2_STREAM_GAP", "3"))
    except Exception: return 3
def register_stream_layer(layer_idx, layer):
    _STREAM_LAYERS[layer_idx] = layer
def _stream_build_one(layer):
    if getattr(layer, "_moe_w2_key", None) is not None:
        return
    try:
        from vllm.model_executor.layers.fused_moe.config import (
            FUSED_MOE_UNQUANTIZED_CONFIG)
    except Exception:
        FUSED_MOE_UNQUANTIZED_CONFIG = None
    key = len(_LAYERS)
    build_layer_planes_nvfp4(layer, key)
    layer._moe_w2_key = key
    qm = getattr(layer, "quant_method", None)
    if qm is not None:
        qm._moe_w2_active = True
        if FUSED_MOE_UNQUANTIZED_CONFIG is not None:
            qm.moe_quant_config = FUSED_MOE_UNQUANTIZED_CONFIG
    try:
        logger.info("moe_w2: streamed-built layer key=%d (host-RAM bound)", key)
    except Exception:
        pass
def stream_touch(layer_idx):
    if layer_idx > _STREAM_MAX_SEEN[0]:
        _STREAM_MAX_SEEN[0] = layer_idx
        upto = _STREAM_MAX_SEEN[0] - _stream_gap()
        for idx in sorted(list(_STREAM_LAYERS)):
            if idx <= upto:
                _stream_build_one(_STREAM_LAYERS.pop(idx))
'''

src = open(cubit).read()
assert "def build_layer_planes_nvfp4(" in src, "build_layer_planes_nvfp4 missing in moe_w2_cubit"
assert "register_stream_layer" not in src, "already injected"
open(cubit, "w").write(src + HELPERS)
print("[inject] moe_w2_cubit.py: streaming helpers appended")

# ---- 2. modelopt.py create_weights: wrap expert weight_loaders (opt-in) ----
mopt_src = open(mopt).read()
STAGE_ANCHOR = ('''            for pname in ("w13_weight", "w13_weight_scale",
                          "w2_weight", "w2_weight_scale"):
                p_ = getattr(layer, pname)
                p_.data = p_.data.cpu()''')
assert STAGE_ANCHOR in mopt_src, "create_weights .cpu() staging anchor not found"
STREAM_SETUP = STAGE_ANCHOR + '''
            import os as _os_strm
            if _os_strm.getenv("VLLM_MOE_W2_STREAM", "0") == "1":
                import re as _re_s
                _m_s = _re_s.search(r"\\.layers\\.(\\d+)\\.",
                                    getattr(layer, "layer_name", "") or "")
                if _m_s is not None:
                    _idx_s = int(_m_s.group(1))
                    moe_w2_cubit.register_stream_layer(_idx_s, layer)
                    for _pn_s in ("w13_weight", "w13_weight_scale",
                                  "w2_weight", "w2_weight_scale"):
                        _p_s = getattr(layer, _pn_s)
                        _real_s = getattr(_p_s, "weight_loader", None)
                        if _real_s is not None:
                            def _mk_s(_real, _idx):
                                def _wl_s(*a, **k):
                                    r = _real(*a, **k)
                                    try:
                                        moe_w2_cubit.stream_touch(_idx)
                                    except Exception:
                                        pass
                                    return r
                                return _wl_s
                            _p_s.weight_loader = _mk_s(_real_s, _idx_s)'''
mopt_src = mopt_src.replace(STAGE_ANCHOR, STREAM_SETUP, 1)

# ---- 3. modelopt.py process_weights: guard against double-build (fail-safe) ----
PW_ANCHOR = ('''            key = len(moe_w2_cubit._LAYERS)
            moe_w2_cubit.build_layer_planes_nvfp4(layer, key)
            layer._moe_w2_key = key''')
assert PW_ANCHOR in mopt_src, "process_weights build anchor not found"
PW_GUARDED = ('''            if getattr(layer, "_moe_w2_key", None) is None:
                key = len(moe_w2_cubit._LAYERS)
                moe_w2_cubit.build_layer_planes_nvfp4(layer, key)
                layer._moe_w2_key = key''')
mopt_src = mopt_src.replace(PW_ANCHOR, PW_GUARDED, 1)
open(mopt, "w").write(mopt_src)
print("[inject] modelopt.py: streaming loader-wrap + process_weights guard applied")

# ---- syntax check ----
import py_compile
py_compile.compile(cubit, doraise=True)
py_compile.compile(mopt, doraise=True)
print("[inject] syntax OK; streaming injection complete")
