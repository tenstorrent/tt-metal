# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3 knob sweep (run with run_safe_pytest.sh --profile; read the CSV in order).
Forces (R, Q, block_c, COMBINE_FANIN) on the flagged collapse config, one dispatch per variant.
A measurement tool, not a correctness gate (correctness: test_mhc_post.py / golden)."""
import os

import pytest
import torch
import ttnn

from ttnn.operations.mhc_post import mhc_post
from ttnn.operations.mhc_post import mhc_post_program_descriptor as desc

N = 4


def _variants():
    spec = os.environ.get("MHC_SWEEP", "")
    out = []
    for item in filter(None, spec.split(";")):
        v = [int(x) for x in item.split(",")] + [0, 0, 0][: 7 - len(item.split(","))]
        out.append(tuple(v[:7]))
    return out


def test_knob_sweep(device, monkeypatch):
    T, C = int(os.environ.get("MHC_T", 160)), int(os.environ.get("MHC_C", 5120))
    mode = os.environ.get("MHC_MODE", "collapse")
    torch.manual_seed(0)
    bf = lambda t: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    f32 = lambda t: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mixes = f32(torch.rand(1, 1, T, N * (N + 2)))
    kw = dict(post_col=N, comb_col=2 * N)
    if mode in ("stats", "collapse"):
        kw["stats_weight"] = f32(torch.randn(1, 1, N * C, 32) * 0.01)
    if mode == "collapse":
        ng = N * (N + 2)
        kw.update(collapse_pre=mixes, hc_scale=[0.3, 0.2, 0.4], hc_base=[0.1 * (k % 5) for k in range(ng)])
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
    )
    io = f32 if os.environ.get("MHC_DT") == "fp32" else bf
    F, X = io(torch.randn(1, 1, T, C)), io(torch.randn(1, 1, T, N * C))
    post, comb = f32(torch.rand(1, 1, T, N)), f32(torch.rand(1, 1, T, N * N))
    ref = None
    orig_split, orig_bc, orig_fan = desc.choose_split, desc.choose_block_c, desc.combine_fanin
    for R, Q, bc, fan, io, od, cd in [(0,) * 7] + _variants():
        monkeypatch.setattr(desc, "choose_split", (lambda *a, **k: (R, Q)) if R else orig_split)
        monkeypatch.setattr(desc, "choose_block_c", (lambda *a, **k: bc) if bc else orig_bc)
        monkeypatch.setattr(desc, "combine_fanin", (lambda Q, f=fan: f) if fan else orig_fan)
        monkeypatch.setattr(desc, "IO_DEPTH", io if io else 2)
        monkeypatch.setattr(desc, "OUT_DEPTH", od if od else 2)
        monkeypatch.setattr(desc, "COEF_DEPTH", cd if cd else 2)
        if os.environ.get("MHC_SEP"):  # separate post / comb (plain only)
            out = mhc_post(F, X, post, comb, compute_kernel_config=cfg)
        else:
            out = mhc_post(F, X, mixes, mixes, compute_kernel_config=cfg, **kw)
        ttnn.synchronize_device(device)
        res = [ttnn.to_torch(t) for t in (out if isinstance(out, tuple) else (out,))]
        if ref is None:
            ref = res
        else:
            for a, b in zip(ref, res):
                assert torch.allclose(a.float(), b.float(), rtol=1e-3, atol=1e-3), (R, Q, bc, fan)
        print("VARIANT", R, Q, bc, fan)
