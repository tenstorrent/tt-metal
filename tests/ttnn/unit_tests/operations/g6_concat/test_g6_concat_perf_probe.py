# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf probe (not an acceptance test): runs the Qwen 79 row_gather shape once for profiling/ablation."""

import torch
import ttnn
from ttnn.operations.g6_concat import g6_concat


def test_qwen79_row_gather_16in(device):
    torch.manual_seed(0)
    xs = [torch.randn(1, 1, 1, 62080).to(torch.bfloat16) for _ in range(16)]
    tt = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for x in xs]
    out = g6_concat(tt, 2)
    ttnn.synchronize_device(device)
