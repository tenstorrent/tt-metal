# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Tracy bench for wide stable topk (W=65536, k=32, both directions).

Underscore-prefixed so plain pytest collection never picks it up; run it as
    python -m tracy -r -v tests/ttnn/unit_tests/operations/reduce/_topk_wide_stable_bench.py
and read DEVICE KERNEL DURATION for the TopK ops from the ops_perf_results CSV.
The input is tie-heavy (8 distinct levels) so the stability machinery is exercised,
not just the value network.
"""
import torch
import ttnn

W, K = 65536, 32
WARMUP, ITERS = 2, 5

dev = ttnn.open_device(device_id=0)
dev.enable_program_cache()
torch.manual_seed(0)
inp = torch.randint(-4, 4, (1, 1, 32, W)).to(torch.bfloat16)
x = ttnn.from_torch(inp, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

for _ in range(WARMUP):
    for largest in (True, False):
        v, i = ttnn.topk(x, K, dim=-1, largest=largest, sorted=True, stable=True)
ttnn.synchronize_device(dev)

for _ in range(ITERS):
    for largest in (True, False):
        v, i = ttnn.topk(x, K, dim=-1, largest=largest, sorted=True, stable=True)
ttnn.synchronize_device(dev)
ttnn.close_device(dev)
