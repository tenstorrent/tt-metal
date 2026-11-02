# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 36 scoping: cost of transposing a [5120, 1024] bf16 tile tensor to [1024, 5120] (the w2 forward's
output in the transposed orientation) with ttnn.transpose, and of the [1536, 1024] / [5120, 1024] qkv / out cases."""

import time

import numpy as np
import torch
import ttnn

device = ttnn.open_device(device_id=0)
for rows, cols in ((5120, 1024), (1536, 1024)):
    t = ttnn.from_torch(
        torch.from_numpy(np.random.default_rng(0).standard_normal((1, 1, rows, cols)).astype(np.float32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    fn = lambda: ttnn.transpose(t, -2, -1)
    out = fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(20):
        out = fn()
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) * 1e6 / 20
    print(
        f"transpose [{rows}, {cols}] -> [{cols}, {rows}] bf16: {us:.0f} us ({rows * cols * 2 / us / 1e3:.0f} GB/s read)",
        flush=True,
    )
ttnn.close_device(device)
