# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 6 perf probe: sharded perf cases. Run under `scripts/run_safe_pytest.sh --profile`."""
import pytest
import torch
import ttnn

import os
from ttnn.operations.rms_norm import rms_norm, rms_norm_program_descriptor as pd

if "RMS_SGCAP" in os.environ:
    pd.SHARD_G_CAP = int(os.environ["RMS_SGCAP"])

_ML = ttnn.TensorMemoryLayout
CASES = [
    (32, 1024, _ML.WIDTH_SHARDED, [32, 128], (8, 1)),
    (32, 2304, _ML.WIDTH_SHARDED, [32, 256], (9, 1)),
    (32, 5120, _ML.WIDTH_SHARDED, [32, 160], (8, 4)),
    (32, 7168, _ML.WIDTH_SHARDED, [32, 256], (7, 4)),
    (640, 6144, _ML.BLOCK_SHARDED, [64, 704], (9, 10)),
    (640, 7168, _ML.BLOCK_SHARDED, [64, 1024], (7, 10)),
    (7168, 1024, _ML.BLOCK_SHARDED, [736, 128], (8, 10)),
    (7168, 2048, _ML.BLOCK_SHARDED, [736, 192], (11, 10)),
    (8192, 1024, _ML.BLOCK_SHARDED, [1024, 128], (8, 8)),
]


@pytest.mark.parametrize("rows,W,ml,shard,grid", CASES, ids=[f"{c[0]}x{c[1]}" for c in CASES])
def test_perf(device, rows, W, ml, shard, grid):
    x = torch.randn(1, 1, rows, W).bfloat16()
    g = torch.randn(1, 1, 1, W).bfloat16()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))})
    orient = ttnn.ShardOrientation.ROW_MAJOR
    mc = ttnn.MemoryConfig(ml, ttnn.BufferType.L1, ttnn.ShardSpec(cores, shard, orient))
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    out = rms_norm(tx, gamma=None if os.environ.get("RMS_NOGAMMA") else tg, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    ref = (
        x.float()
        * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        * (1 if os.environ.get("RMS_NOGAMMA") else g.float())
    )
    r = ttnn.to_torch(out).float()
    assert torch.corrcoef(torch.stack([r.flatten(), ref.flatten()]))[0, 1] > 0.9995
