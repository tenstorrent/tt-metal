# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness harness for rms_norm Refinement 6c (two-stage hierarchical gather).

The R6c lever replaces the flat K-1 -> master gather with a 2-stage reduce over a 2D
reduction rectangle (stage 1: each grid row's NX cores -> row-leader; stage 2: the NY
row-leaders -> root). It engages ONLY on the pure tiled WIDTH-sharded cross-core path
for a clean 2D rectangle with fan-in saving (NX-1)*(NY-1) >= 13 and a single round
(HT_LOCAL=1). This file pins the correctness edges the perf/ablation harnesses don't:

  * 2D WIDTH groups (8x4, 7x4) that ENGAGE two-stage — aligned AND non-aligned W (the
    partial-scaler masking must stay correct through both gather stages), gamma / no-gamma.
  * A 1-D WIDTH group (n x 1) and a small-saving 2D group (7x2, saving 6 < 13) that FALL
    BACK to the byte-identical flat gather — verifying the gate does not break those.

Soft PCC gate 0.9995 (the perf cases' pcc_threshold). --dev + non-dev both catch a race
in the new SEM_GATHER2 stage-2 handshake.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm

_ML = ttnn.TensorMemoryLayout

# (rows, W, sh, sw, gx, gy, gamma, note) — WIDTH-sharded 2D + fallback geometries.
# A WIDTH shard fully tiles the (padded) width: W_padded == K*sw, K = gx*gy, sw a multiple
# of 32 (per_w_t = sw//32 tiles/core). Aligned W = K*sw; a non-aligned last tile uses
# W = K*sw - (32 - p) (the globally-last core, slice K-1, is the partial holder). Two-stage
# engages iff gx>1 and gy>1 and (gx-1)*(gy-1) >= 13.
CASES = [
    # --- 2D groups that ENGAGE two-stage ---
    (32, 32 * 32 * 5, 32, 32 * 5, 8, 4, True, "8x4 aligned per_w_t=5 (engages, saving 21)"),  # W=5120
    (32, 28 * 32, 32, 32, 7, 4, True, "7x4 aligned per_w_t=1 (engages, saving 18)"),  # W=896
    (32, 31 * 32 + 12, 32, 32, 8, 4, True, "8x4 NON-aligned W=1004 (partial holder thru 2 stages)"),
    (32, 28 * 64 - 20, 32, 64, 7, 4, True, "7x4 NON-aligned W=1772 per_w_t=2 (partial holder)"),
    (32, 32 * 32 * 5, 32, 32 * 5, 8, 4, False, "8x4 aligned NO-gamma (engages)"),  # W=5120
    # --- geometries that FALL BACK to flat (gate) — must stay correct ---
    (32, 8 * 32, 32, 32, 8, 1, True, "8x1 1-D (flat fallback)"),  # W=256
    (32, 14 * 96 - 20, 32, 96, 7, 2, True, "7x2 NON-aligned W=1324 saving 6<13 (flat fallback)"),
]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else (torch.dot(a, b).item()) / denom


def _cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


@pytest.mark.parametrize(
    "rows,W,sh,sw,gx,gy,has_gamma,note",
    CASES,
    ids=[f"{gx}x{gy}_{r}x{w}_{'g' if hg else 'ng'}" for (r, w, _sh, _sw, gx, gy, hg, _n) in CASES],
)
def test_two_stage(rows, W, sh, sw, gx, gy, has_gamma, note, device):
    torch.manual_seed(0)
    shape = (1, 1, rows, W)
    ti = torch.randn(shape, dtype=torch.bfloat16)
    tg = torch.randn(W, dtype=torch.bfloat16) if has_gamma else None
    exp = ti.float() * torch.rsqrt(ti.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    if has_gamma:
        exp = exp * tg.float().reshape(-1)

    cfg = shard_config(
        [sh, sw], (gx, gy), _ML.WIDTH_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    xt = ttnn.from_torch(ti, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    gt = None
    if has_gamma:
        gt = ttnn.from_torch(
            tg.reshape(1, 1, 1, W),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=_cfg(), memory_config=cfg)
    p = _pcc(ttnn.to_torch(out), exp)
    print(f"\nR6C {note}: PCC={p:.6f}")
    assert p >= 0.9995, f"{note}: PCC {p} < 0.9995"
