# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-cell composite bench for the ttnn.topk -> topk_large_indices routing
boundary questions (review #53464): pow2-4096 arm and tiny-width large-k
floor. One cell per invocation; run under `python -m tracy -r -v` and sum
DEVICE KERNEL DURATION over the report CSV.

Usage: _topk_route_cells_bench.py <H> <W> <k> <routed|stock>
`stock` forces the stock engine by passing sub_core_grids (the routing
predicate declines whenever sub_core_grids is provided).
"""
import sys
import torch
import ttnn
from loguru import logger

H, W, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
ARM = sys.argv[4]
WARMUP, ITERS = 3, 30

device = ttnn.open_device(device_id=0)
device.enable_program_cache()

torch.manual_seed(2005)
torch_input = torch.randn(1, 1, H, W, dtype=torch.bfloat16) * 0.9
tt_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

kwargs = dict(dim=-1, largest=True, sorted=True)
if ARM == "stock":
    grid = device.compute_with_storage_grid_size()
    kwargs["sub_core_grids"] = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
    )

for _ in range(WARMUP + ITERS):
    values, indices = ttnn.topk(tt_input, K, **kwargs)
    ttnn.synchronize_device(device)

logger.info(f"BENCH_DONE H={H} W={W} k={K} arm={ARM} iters={WARMUP + ITERS}")
ttnn.close_device(device)
