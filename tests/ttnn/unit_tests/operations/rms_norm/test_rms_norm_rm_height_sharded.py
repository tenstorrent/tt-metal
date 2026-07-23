# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Refinement 5a regression test — RM input + HEIGHT_SHARDED (tilize-on-resident-shard).
# DO NOT DELETE — documents the RM-input HEIGHT-sharded coverage.
#
# R5 landed HEIGHT_SHARDED for TILE input (zero-copy resident tile-shard). R5a completes
# the corner for ROW_MAJOR input, whose resident row-shard is full-W RM sticks (not tiles).
# Each core still keeps FULL-W rows, so the RMS reduce stays LOCAL per core (no cross-core
# combine, phase=0, only the standard W%32 mask). The op:
#   * backs cb_shard_in / cb_shard_out ZERO-COPY on the resident RM shards (no DRAM/remote
#     read — the shard is consumed locally),
#   * reader loopback-repacks the resident sticks into tile-padded cb_x_sticks (local NoC
#     loopback via my_x/my_y),
#   * compute tilizes cb_x_sticks -> allocated cb_x_in (whole tile-row resident), runs the
#     R3-resident indexed two-pass, then untilizes cb_out -> cb_out_sticks per block,
#   * NEW writer loopback-writes the valid columns of cb_out_sticks into the RM output shard.
# gamma is streamed per block (RM sticks, or no gamma; RM input + TILE gamma is INVALID).
#
# Cases exercise: tile-aligned + W/H/both non-aligned, last-core-short (per_h < shard),
# wide W (L1 pressure), fp32, 2D/3D/4D ranks, RM gamma + no gamma (incl. mixed precision).

import pytest
import torch

import ttnn

from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
RM = ttnn.ROW_MAJOR_LAYOUT


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _ref(x, g=None, eps=1e-6):
    x = x.float()
    o = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if g is not None:
        o = o * g.float().reshape(-1)
    return o


def _run(device, shape, gamma_layout, dt, gdt):
    torch.manual_seed(0)
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    ti = torch.randn(shape).to(tdt)
    # RM input HEIGHT shard: sh = per-core rows (granule 1), sw = W padded to 8/4.
    cfg = auto_shard_config(list(shape), HEIGHT, layout=RM, dtype=dt, device=device)
    xt = ttnn.from_torch(ti, dtype=dt, layout=RM, device=device, memory_config=cfg)
    tg = gt = None
    if gamma_layout is not None:
        W = shape[-1]
        gtdt = torch.float32 if gdt == ttnn.float32 else torch.bfloat16
        tg = torch.randn(W).to(gtdt)
        gt = ttnn.from_torch(tg.reshape(1, 1, 1, W), dtype=gdt, layout=gamma_layout, device=device)
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = dt == ttnn.float32
    cc.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cc, memory_config=xt.memory_config())
    assert out.memory_config().memory_layout == HEIGHT
    assert out.layout == RM
    got = ttnn.to_torch(out)
    exp = _ref(ti, tg)
    p = _pcc(got, exp)
    md = (got.float() - exp.float()).abs().max().item()
    assert p > 0.99, f"{shape} gl={gamma_layout} {dt}/{gdt}: PCC={p} maxdiff={md}"


bf16, f32 = ttnn.bfloat16, ttnn.float32

# (shape, gamma_layout, input_dtype, gamma_dtype)
_CASES = [
    # --- RM gamma ---
    ((1, 1, 256, 512), RM, bf16, bf16),  # loose-case shape (per_h short on last core)
    ((1, 1, 256, 512), RM, bf16, f32),  # mixed: bf16 act + f32 RM gamma
    ((1, 1, 32, 50), RM, bf16, bf16),  # W non-aligned
    ((1, 1, 50, 128), RM, bf16, bf16),  # H non-aligned
    ((1, 1, 17, 50), RM, bf16, f32),  # both non-aligned, mixed
    ((2, 1, 100, 47), RM, bf16, bf16),  # both non-aligned, 4D
    ((4, 8, 32, 256), RM, bf16, bf16),  # 4D, last-core-short (per_h=10, total=1024)
    ((1, 32, 4096), RM, bf16, bf16),  # 3D, wide W
    ((1024, 1024), RM, bf16, bf16),  # 2D, large
    ((128, 512), RM, bf16, f32),  # 2D mixed
    ((1, 1, 32, 64), RM, f32, f32),  # fp32
    ((1, 1, 32, 8192), RM, f32, f32),  # fp32 wide W (streaming path: resident cb_x_in would OOM)
    ((128, 8192), RM, f32, bf16),  # fp32 wide, 2D, mixed gamma
    # --- no gamma ---
    ((1, 1, 256, 512), None, bf16, None),
    ((1, 1, 32, 50), None, bf16, None),  # W non-aligned, no gamma
    ((1, 1, 50, 128), None, bf16, None),  # H non-aligned, no gamma
    ((1, 1, 32, 8192), None, bf16, None),  # wide W, few rows (L1 pressure)
    ((128, 8192), None, f32, None),  # fp32 wide, no gamma
    ((128, 512), None, bf16, None),  # 2D no gamma
    ((1, 1, 32, 64), None, f32, None),  # fp32 no gamma
]


@pytest.mark.parametrize("shape,gl,dt,gdt", _CASES)
def test_rms_norm_rm_height_sharded(device, shape, gl, dt, gdt):
    _run(device, shape, gl, dt, gdt)
