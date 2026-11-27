# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-kernel-time bench for g6_expand (run with `scripts/run_safe_pytest.sh --profile`).

One warm-up call (compile) then one measured call per production width; read the
DEVICE KERNEL DURATION column of the ops_perf_results CSV (last row per width).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_expand import g6_expand


@pytest.mark.parametrize("width", [512, 256, 196, 185, 207])
def test_g6_expand_perf_bench(device, width):
    torch.manual_seed(0)
    x = torch.randn((1, 1, 1, width), dtype=torch.float32).to(torch.bfloat16)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(3):
        out = g6_expand(tt_x, [1, 1, width, width])
    ttnn.synchronize_device(device)
    ttnn.deallocate(out)
