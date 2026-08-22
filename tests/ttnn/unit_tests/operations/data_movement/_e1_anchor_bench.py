# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E1 anchor bench: topk_large_indices leaf+merge-level marginal costs (Tracy DKD).

    E1_ROWS=1 E1_W=4096 python -m tracy -r -v .../_e1_anchor_bench.py
"""
import os
import torch
import ttnn

ROWS = int(os.environ.get("E1_ROWS", "1"))
W = int(os.environ.get("E1_W", "4096"))
K = int(os.environ.get("E1_K", "2048"))
WARMUP, ITERS = 2, 10

dev = ttnn.open_device(device_id=0)
dev.enable_program_cache()
g = torch.Generator().manual_seed(0)
inp = torch.randn((ROWS, W), generator=g).bfloat16()
x = ttnn.from_torch(inp, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

for _ in range(WARMUP):
    i = ttnn.experimental.topk_large_indices(x, k=K)
ttnn.synchronize_device(dev)
for _ in range(ITERS):
    i = ttnn.experimental.topk_large_indices(x, k=K)
ttnn.synchronize_device(dev)
ttnn.close_device(dev)
