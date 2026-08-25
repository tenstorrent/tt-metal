# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Tracy bench for one sort cell with configurable H (mergesort campaign).

Underscore-prefixed so plain pytest collection never picks it up; run as
    SORT_BENCH_H=32 SORT_BENCH_W=2048 SORT_BENCH_STABLE=1 \
        python -m tracy -r -v tests/ttnn/unit_tests/operations/data_movement/_mergesort_bench.py
The input is tie-heavy (5 distinct levels) so the stability machinery is exercised.
"""
import os
import torch
import ttnn

H = int(os.environ.get("SORT_BENCH_H", "32"))
W = int(os.environ.get("SORT_BENCH_W", "2048"))
STABLE = os.environ.get("SORT_BENCH_STABLE", "1") == "1"
WARMUP, ITERS = 2, 5

dev = ttnn.open_device(device_id=0)
dev.enable_program_cache()
g = torch.Generator().manual_seed(0)
levels = torch.tensor([-1.5, -0.5, -0.5, 0.5, 1.5], dtype=torch.bfloat16)
inp = levels[torch.randint(0, len(levels), (H, W), generator=g)]
LAYOUT = ttnn.ROW_MAJOR_LAYOUT if os.environ.get("SORT_BENCH_LAYOUT", "TILE") == "RM" else ttnn.TILE_LAYOUT
x = ttnn.from_torch(inp, ttnn.bfloat16, layout=LAYOUT, device=dev)

PREALLOC_U32 = os.environ.get("SORT_BENCH_PREALLOC_U32", "0") == "1"
outs = None
if PREALLOC_U32:
    vals = ttnn.zeros_like(x)
    idxs = ttnn.zeros((H, W), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=dev)

for _ in range(WARMUP):
    for descending in (True, False):
        if PREALLOC_U32:
            ttnn.sort(x, dim=-1, descending=descending, stable=STABLE, out=(vals, idxs))
        else:
            v, i = ttnn.sort(x, dim=-1, descending=descending, stable=STABLE)
ttnn.synchronize_device(dev)

for _ in range(ITERS):
    for descending in (True, False):
        if PREALLOC_U32:
            ttnn.sort(x, dim=-1, descending=descending, stable=STABLE, out=(vals, idxs))
        else:
            v, i = ttnn.sort(x, dim=-1, descending=descending, stable=STABLE)
ttnn.synchronize_device(dev)
ttnn.close_device(dev)
