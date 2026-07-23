# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
# R6 ablation harness (NOT the golden suite) — isolate the cross-core collective cost.
# Gap-free 7-wide WIDTH grids (logical x=0..6 -> virtual x=1..7, contiguous) so the mcast
# broadcast engages. Fixed per_w_t=1 (shard_w=32), varying K via grid y -> isolates the
# collective's K-dependence at constant per-core work. Plus a HT_LOCAL sweep for BLOCK.
from __future__ import annotations
import pytest, torch, ttnn
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm

N_TRIALS = 8
_ML = ttnn.TensorMemoryLayout

# (rows, W, shard_h, shard_w, gx, gy) — WIDTH, gap-free 7-wide, per_w_t=1 (shard_w=32)
ABLATE = [
    (32, 32 * 7, 32, 32, 7, 1),  # K=7,  HT=1, per_w_t=1
    (32, 32 * 14, 32, 32, 7, 2),  # K=14, HT=1, per_w_t=1
    (32, 32 * 21, 32, 32, 7, 3),  # K=21, HT=1, per_w_t=1
    (32, 32 * 28, 32, 32, 7, 4),  # K=28, HT=1, per_w_t=1
    # HT_LOCAL sweep (BLOCK, gap-free 7-wide row groups, per_w_t=1): rows -> HT_LOCAL
    (32 * 4, 32 * 7, 32 * 4, 32, 7, 4),  # HT=4  per group, K=7, 4 groups
    (32 * 16, 32 * 7, 32 * 16, 32, 7, 4),  # HT=16 per group, K=7
    (32 * 32, 32 * 7, 32 * 32, 32, 7, 4),  # HT=32 per group, K=7
]


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return 1.0 if d == 0 else torch.dot(a, b).item() / d


@pytest.mark.parametrize(
    "rows,W,sh,sw,gx,gy", ABLATE, ids=[f"K{gx*gy}_HT{r//32}_{r}x{w}" for (r, w, _sh, _sw, gx, gy) in ABLATE]
)
def test_ablate(rows, W, sh, sw, gx, gy, device):
    torch.manual_seed(0)
    shape = (1, 1, rows, W)
    ti = torch.randn(shape, dtype=torch.bfloat16)
    tg = torch.randn(W, dtype=torch.bfloat16)
    exp = ti.float() * torch.rsqrt(ti.float().pow(2).mean(-1, keepdim=True) + 1e-6) * tg.float().reshape(-1)
    ml = _ML.BLOCK_SHARDED if gy > 1 and sh > 32 else _ML.WIDTH_SHARDED
    cfg = shard_config([sh, sw], (gx, gy), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    xt = ttnn.from_torch(ti, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    gt = ttnn.from_torch(
        tg.reshape(1, 1, 1, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi2
    cc.fp32_dest_acc_en = False
    cc.math_approx_mode = False
    out = None
    for _ in range(N_TRIALS):
        out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cc, memory_config=cfg)
    p = _pcc(ttnn.to_torch(out), exp)
    print(f"\nABLATE K={gx*gy} HT={rows//32} W={W} PCC={p:.5f}")
    assert p >= 0.999, p
