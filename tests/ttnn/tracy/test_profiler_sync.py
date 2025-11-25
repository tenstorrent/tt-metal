# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_with_watcher, is_watcher_enabled


def test_with_ops(device):
    torch.manual_seed(0)

    m = 1024
    k = 1024
    n = 1024
    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)
    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = a @ b
    output = a @ b

    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)

    a = ttnn.to_device(a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    b = ttnn.to_device(b, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

    output = ttnn.matmul(a, b, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=8, x=8))

    output = ttnn.matmul(a, b, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=8, x=8))


@skip_with_watcher("Test is not passing with watcher enabled due to overflow of the code size")
def test_mesh_device(
    mesh_device,
):
    if is_watcher_enabled():
        pytest.skip("Test is not passing with watcher enabled due to overflow of the code size")

    logger.debug("Testing All Devices")
