# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Tracy bench for one stable-topk cell (comparator-vs-fast A/B legs).

Underscore-prefixed so plain pytest collection never picks it up; run as
    TOPK_BENCH_W=65536 TOPK_BENCH_K=64 [TOPK_BENCH_PREALLOC_U32=1] \
        python -m tracy -r -v tests/ttnn/unit_tests/operations/reduce/_topk_stable_cell_bench.py
and read DEVICE KERNEL DURATION for the TopK ops from the ops_perf_results CSV.
The input is tie-heavy (8 distinct levels) so the stability machinery is exercised.
"""
import os
import torch
import ttnn

W = int(os.environ.get("TOPK_BENCH_W", "65536"))
K = int(os.environ.get("TOPK_BENCH_K", "32"))
PREALLOC_U32 = os.environ.get("TOPK_BENCH_PREALLOC_U32", "0") == "1"
STABLE = os.environ.get("TOPK_BENCH_STABLE", "1") == "1"
WARMUP, ITERS = 2, 5

dev = ttnn.open_device(device_id=0)
dev.enable_program_cache()
torch.manual_seed(0)
inp = torch.randint(-4, 4, (1, 1, 32, W)).to(torch.bfloat16)
x = ttnn.from_torch(inp, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

out = None
if PREALLOC_U32:
    vt = ttnn.from_torch(
        torch.zeros((1, 1, 32, K), dtype=torch.bfloat16), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
    )
    it = ttnn.from_torch(
        torch.zeros((1, 1, 32, K), dtype=torch.int32), ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=dev
    )
    out = (vt, it)


def run(largest):
    if out is not None:
        return ttnn.topk(x, K, dim=-1, largest=largest, sorted=True, stable=STABLE, output_tensor=out)
    return ttnn.topk(x, K, dim=-1, largest=largest, sorted=True, stable=STABLE)


for _ in range(WARMUP):
    for largest in (True, False):
        run(largest)
ttnn.synchronize_device(dev)

for _ in range(ITERS):
    for largest in (True, False):
        run(largest)
ttnn.synchronize_device(dev)
ttnn.close_device(dev)
