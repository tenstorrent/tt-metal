import os

# LTX_FAST=1 is a single-switch alias for the served fast-mode env bundle (bf8 DiT-linear quant +
# step-cut denoise schedule + traced + host weight cache), mirroring ltx_server build_worker_env's
# fast branch so a bare `LTX_FAST=1 pytest ...` reproduces it. Runs at collection (module top level,
# not a fixture) because pipeline_ltx_distilled reads LTX_S1/S2_SIGMAS and module.py reads
# TT_DIT_HOST_WEIGHT_CACHE at import — a fixture would land after that. setdefault so an explicit
# override still wins. Values must stay in sync with build_worker_env; the tree switch it also does
# (TT_METAL_HOME -> ltx-rt) can't be set from here, so run this from an ltx-rt checkout.
if os.environ.get("LTX_FAST", "0") in ("1", "true", "True"):
    os.environ.setdefault("LTX_QUANT", "all_bf8_lofi")
    os.environ.setdefault("LTX_S1_SIGMAS", "1.0,0.9875,0.975,0.909375,0.725,0.421875,0.0")
    os.environ.setdefault("LTX_S2_SIGMAS", "0.909375,0.0")
    os.environ.setdefault("LTX_TRACED", "1")
    os.environ.setdefault("TT_DIT_HOST_WEIGHT_CACHE", "1")
