# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1996800}], indirect=True)
def test_with_ops(device):
    torch.manual_seed(0)
    m = 1024
    k = 1024
    n = 1024
    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)

    a = ttnn.to_device(a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    b = ttnn.to_device(b, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

    ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(100):
        ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))
    ttnn.end_trace_capture(device, tid, cq_id=0)

    for i in range(5):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(device, tid)
