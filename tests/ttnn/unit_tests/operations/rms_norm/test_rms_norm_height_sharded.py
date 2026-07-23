# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Refinement 5 regression test — HEIGHT_SHARDED (local per-core reduction).
# DO NOT DELETE — documents the HEIGHT-sharded coverage.
#
# HEIGHT sharding splits rows across cores; each core keeps FULL-W rows, so the RMS
# reduce stays LOCAL per core (the row-parallel scheme with the row-shard resident in
# each core's L1). The op backs cb_x_in / cb_out ZERO-COPY on the sharded buffers
# (cb_descriptor_from_sharded_tensor — no NoC read/write) and reuses the interleaved
# R3 resident indexed two-pass compute; gamma is streamed per block so cb_gamma stays
# small (fits any W). TILE input only (RM input HEIGHT is deferred to R5a / EXCLUDED).
# These cases exercise: tile-aligned + W/H/both non-aligned, per_h>1 (R>grid), wide W,
# fp32, bf8b, 2D/3D/4D ranks, and TILE / RM / no gamma (incl. mixed precision).

import pytest
import torch

import ttnn

from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
TILE = ttnn.TILE_LAYOUT
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
    cfg = auto_shard_config(list(shape), HEIGHT, layout=TILE, dtype=dt, device=device)
    xt = ttnn.from_torch(ti, dtype=dt, layout=TILE, device=device, memory_config=cfg)
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
    got = ttnn.to_torch(out)
    exp = _ref(ti, tg)
    p = _pcc(got, exp)
    md = (got.float() - exp.float()).abs().max().item()
    assert p > 0.99, f"{shape} gl={gamma_layout} {dt}/{gdt}: PCC={p} maxdiff={md}"


bf16, f32, bf8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b

# (shape, gamma_layout, input_dtype, gamma_dtype)
_CASES = [
    # --- TILE gamma ---
    ((1, 1, 256, 512), TILE, bf16, bf16),  # loose-case shape, 8 cores x 1 tile-row
    ((1, 1, 8192, 256), TILE, bf16, bf16),  # per_h>1 (R=256 > grid)
    ((4, 8, 32, 256), TILE, bf16, bf16),  # multi-batch, R=32
    ((1, 1, 32, 50), TILE, bf16, bf16),  # W non-aligned
    ((1, 1, 50, 128), TILE, bf16, bf16),  # H non-aligned
    ((1, 1, 17, 50), TILE, bf16, bf16),  # both non-aligned
    ((1, 1, 32, 8192), TILE, bf16, bf16),  # wide W, 1 core (L1 pressure)
    ((4, 128, 512), TILE, bf16, bf16),  # 3D
    ((128, 512), TILE, bf16, bf16),  # 2D
    ((1, 1, 256, 512), TILE, f32, f32),  # fp32
    ((1, 1, 256, 512), TILE, bf8, bf8),  # bf8b tile-aligned
    ((1, 1, 256, 512), TILE, bf16, f32),  # mixed: bf16 act + f32 TILE gamma
    # --- RM gamma (TILE input) ---
    ((1, 1, 256, 512), RM, bf16, bf16),
    ((1, 1, 256, 512), RM, bf16, f32),  # mixed: bf16 act + f32 RM gamma
    ((1, 1, 32, 50), RM, bf16, bf16),  # W non-aligned
    ((1, 1, 17, 50), RM, bf16, f32),  # both non-aligned, mixed
    ((1, 1, 32, 4096), RM, bf16, bf16),  # wide
    ((128, 512), RM, bf16, f32),  # 2D mixed
    # --- no gamma ---
    ((1, 1, 256, 512), None, bf16, None),
    ((2, 1, 128, 100), None, bf16, None),  # W non-aligned, no gamma
    ((1, 1, 64, 8192), None, bf16, None),  # wide, no gamma
    ((1024, 1024), None, bf16, None),  # 2D no gamma
]


@pytest.mark.parametrize("shape,gl,dt,gdt", _CASES)
def test_rms_norm_height_sharded(device, shape, gl, dt, gdt):
    _run(device, shape, gl, dt, gdt)
