# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 6e correctness — two-phase (tile-index) reduce-mcast fold distribution.

R6e distributes the group MASTER's serial K-partial fold across up to min(C, K)
"folder" cores by tile-index (each folder gathers+folds a disjoint set of the C
batched tile-rows, scatters its finalized 1/RMS to the root, root assembles + mcasts
back). Engages ONLY on the pure tiled cross-core BLOCK path with C>1 multi-round
batching (Ht_local % C == 0). C=1 / WIDTH / RM / logical / two-stage stay byte-identical.

These are CORRECTNESS cases (perf is measured separately in test_rms_norm_perf_r6.py).
Small multi-round BLOCK geometries that hit the two-phase branch:
  * (1,1,2048,256) BLOCK (512,64) grid (4,4): Ht_local=16, C=8 -> 2 rounds, K=4,
    num_folders=4, owned_count=2 per folder (exercises multi-owned-row folds).
  * (1,1,8192,1024) BLOCK (1024,128) grid (8,8): Ht_local=32, C=8 -> 4 rounds, K=8,
    num_folders=8, owned_count=1 (the perf target's topology).
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm

_ML = ttnn.TensorMemoryLayout


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else (torch.dot(a, b).item()) / denom


def _reference(x, gamma):
    e = x.to(torch.float32)
    e = e * torch.rsqrt(e.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    if gamma is not None:
        e = e * gamma.to(torch.float32).reshape(-1)
    return e


def _cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


# (rows, W, shard_h, shard_w, grid_x, grid_y, has_gamma)
CASES = [
    (2048, 256, 512, 64, 4, 4, True),  # Ht_local=16, C=8, 2 rounds, K=4, folders=4, owned=2
    (2048, 256, 512, 64, 4, 4, False),  # no-gamma variant
    (1024, 512, 512, 128, 4, 4, True),  # Ht_local=16, K=4, per_w_t=4
    (8192, 1024, 1024, 128, 8, 8, True),  # perf target topology (Ht_local=32, folders=8, owned=1)
]


@pytest.mark.parametrize(
    "rows,W,sh,sw,gx,gy,has_gamma",
    CASES,
    ids=[f"blk_{r}x{w}_{gx}x{gy}_{'g' if hg else 'ng'}" for (r, w, _sh, _sw, gx, gy, hg) in CASES],
)
def test_two_phase_block(rows, W, sh, sw, gx, gy, has_gamma, device):
    torch.manual_seed(0)
    shape = (1, 1, rows, W)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(W, dtype=torch.bfloat16) if has_gamma else None
    expected = _reference(torch_input, torch_gamma)

    in_cfg = shard_config(
        [sh, sw], (gx, gy), _ML.BLOCK_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_cfg
    )
    ttnn_gamma = None
    if has_gamma:
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    out = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=1e-6, compute_kernel_config=_cfg(), memory_config=in_cfg)
    result = ttnn.to_torch(out)
    pcc = _pcc(result, expected)
    print(f"\nR6E ({rows},{W}) grid {gx}x{gy} gamma={has_gamma} PCC={pcc:.6f}")
    assert pcc >= 0.995, f"PCC {pcc} < 0.995 for shape {shape}"
