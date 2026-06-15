# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""deep-plan_13 §10 measurement: per-projection unified-op (MatmulDecodeLinear)
DEVICE FW timing for the 12 pi05 projections across the 3 stages, at FULL-SEQ M
(one block-forward = one full-S pass; the wrapper M-splits into 32-row tiles).

Each (stage, proj) runs in its OWN tracy subprocess (ONLY_STAGE + ONLY_PROJ) so
the tracy host post-processor does not segfault on the VLM N=16384 volume. One
projection => one signpost region MMD_<stage>_<proj>. Warm-up OUTSIDE the
signpost; N_ITERS full-seq forwards INSIDE. Region MatmulDecodeDeviceOperation
DEVICE FW sum / N_ITERS = mmd_device_fw_us (the per-block-forward unified number).

This is the unified-op leg of the §10.6 table; the native best-explicit leg is
profile_native_explicit_sweep_stages.py (report_natexp). Tracy-only device time.

Run (one proj at a time, serial; see run_unified_mmsweep.sh):
  ONLY_STAGE=VLM ONLY_PROJ=gate python -m tracy -p -r -v --op-support-count 100000 \
    -m 'pytest profile_unified_mmsweep_stages.py::test_profile -x -s'
"""
from __future__ import annotations

import os

import torch
import ttnn

# Self-contained bootstrap: make the vendored _lib package importable.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _lib.matmul_decode_linear import MatmulDecodeLinear

try:
    from tracy import signpost
except Exception:
    def signpost(*a, **k):
        pass

TT_METAL_COMMIT = "e4500c1fae97c103b16fc24fc7010b852992a9e6"
SEED = 1234
BF8 = ttnn.bfloat8_b
BF16 = ttnn.bfloat16

N_ITERS = int(os.environ.get("N_ITERS", "5"))
N_REPEAT = int(os.environ.get("N_REPEAT", "1"))  # >1 => emit MMD_<...>_r<n> regions for epsilon
ONLY_STAGE = os.environ.get("ONLY_STAGE", "")
ONLY_PROJ = os.environ.get("ONLY_PROJ", "")

# (stage, S, [(proj, K, N, weight_dtype, pad_n, pad_k)])  -- exactly the §10 scope.
STAGES = {
    "SigLIP": (256, [
        ("qkv", 1152, 4608, BF16, None, None),
        ("o", 1536, 1152, BF16, None, None),
        ("fc1", 1152, 4304, BF8, 4320, None),
        ("fc2", 4304, 1152, BF8, None, 4320),
    ]),
    "VLM": (288, [
        ("qkv", 2048, 2560, BF8, None, None),
        ("o", 2048, 2048, BF8, None, None),
        ("gate", 2048, 16384, BF8, None, None),
        ("up", 2048, 16384, BF8, None, None),
        ("down", 16384, 2048, BF8, None, None),
    ]),
    "DENOISE": (64, [
        ("gate", 1024, 4096, BF8, None, None),
        ("up", 1024, 4096, BF8, None, None),
        ("down", 4096, 1024, BF8, None, None),
    ]),
}


def _build(dev, K, N, weight_dtype, pad_n, pad_k, role):
    torch.manual_seed(SEED)
    w = torch.randn(K, N) * 0.02
    return MatmulDecodeLinear(
        dev, w, bias=None, out_dtype=BF16, weight_dtype=weight_dtype,
        role=role, pad_n=pad_n, pad_k=pad_k)


def _forward(dev, mmd, S, K):
    """One full-seq block-forward: feed [1, S, K]; wrapper M-splits into 32-row tiles."""
    x = ttnn.from_torch(torch.randn(1, S, K) * 0.5, dtype=BF16,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    y = mmd(x)
    ttnn.deallocate(x)
    ttnn.deallocate(y)


def _run_proj(dev, stage, S, proj, K, N, weight_dtype, pad_n, pad_k):
    mmd = _build(dev, K, N, weight_dtype, pad_n, pad_k, f"{stage.lower()}_{proj}")
    desc = mmd.describe()
    print(f"  [{stage}.{proj}] plan: mode={desc['mode']} n_chunks={desc['n_chunks']} "
          f"k_split_G={desc['k_split_G']} npc={mmd.npc} stream_k={mmd.stream_k} "
          f"a_cores={desc['a_cores']} K={desc['K']} N={desc['N']}", flush=True)
    # warm-up OUTSIDE signpost (compile + weight stage already done in __init__)
    _forward(dev, mmd, S, mmd.K_orig)
    ttnn.synchronize_device(dev)
    for rep in range(N_REPEAT):
        tag = f"MMD_{stage}_{proj}" + (f"_r{rep}" if N_REPEAT > 1 else "")
        signpost(header=tag)
        for _ in range(N_ITERS):
            _forward(dev, mmd, S, mmd.K_orig)
        ttnn.synchronize_device(dev)


def test_profile(dev):
    for stage, (S, projs) in STAGES.items():
        if ONLY_STAGE and stage != ONLY_STAGE:
            continue
        print(f"\n==== UNIFIED mmsweep {stage} S={S} ====", flush=True)
        for proj, K, N, wd, pn, pk in projs:
            if ONLY_PROJ and proj != ONLY_PROJ:
                continue
            _run_proj(dev, stage, S, proj, K, N, wd, pn, pk)
