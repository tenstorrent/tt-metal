# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Tracy bench for one stable-sort cell (comparator-vs-fused A/B legs).

Underscore-prefixed so plain pytest collection never picks it up; run as
    SORT_BENCH_W=8192 SORT_BENCH_STABLE=1 \
        python -m tracy -r -v tests/ttnn/unit_tests/operations/data_movement/_sort_stable_bench.py
and read DEVICE KERNEL DURATION for the Sort ops from the ops_perf_results CSV.
The input is tie-heavy (5 distinct levels) so the stability machinery is exercised.
"""
import os
import torch
import ttnn

W = int(os.environ.get("SORT_BENCH_W", "8192"))
STABLE = os.environ.get("SORT_BENCH_STABLE", "1") == "1"
WARMUP, ITERS = 2, 5

dev = ttnn.open_device(device_id=0)
dev.enable_program_cache()
g = torch.Generator().manual_seed(0)
levels = torch.tensor([-1.5, -0.5, -0.5, 0.5, 1.5], dtype=torch.bfloat16)
inp = levels[torch.randint(0, len(levels), (32, W), generator=g)]
x = ttnn.from_torch(inp, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

for _ in range(WARMUP):
    for descending in (True, False):
        v, i = ttnn.sort(x, dim=-1, descending=descending, stable=STABLE)
ttnn.synchronize_device(dev)

for _ in range(ITERS):
    for descending in (True, False):
        v, i = ttnn.sort(x, dim=-1, descending=descending, stable=STABLE)
ttnn.synchronize_device(dev)
ttnn.close_device(dev)
