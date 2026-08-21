# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-cell bench for topk_large_indices width scaling (Pavle's prefill sweep).
Usage: _topk_li_width_sweep.py <rows> <W> <k> [iters]
Run under `python -m tracy -r -v`; parse ops_perf_results CSV afterward.
"""
import sys
import torch
import ttnn
from loguru import logger

ROWS, W, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
# Optional: search only the first VALID columns of a wider preallocated
# buffer (the DSA indexer's growing-prefill calling shape).
VALID = int(sys.argv[5]) if len(sys.argv) > 5 else None
WARMUP = 3

device = ttnn.open_device(device_id=0)
device.enable_program_cache()

torch.manual_seed(2005)
# Chunked generation to keep host memory sane at 1M width.
torch_input = torch.randn(ROWS, W, dtype=torch.bfloat16) * 0.9
tt_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.Layout.ROW_MAJOR, device=device)

for _ in range(WARMUP + ITERS):
    indices = ttnn.experimental.topk_large_indices(tt_input, k=K, valid_length=VALID)
    ttnn.synchronize_device(device)

logger.info(f"BENCH_DONE rows={ROWS} W={W} k={K} iters={WARMUP + ITERS}")
ttnn.close_device(device)
