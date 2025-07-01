# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "shape,shard_shape",
    [
        # Batch sharding
        # Batch dims are equal for shape and shard_shape, all other dims of shard shape are 1 (or tile size for the lower two dims)
        ((10, 4, 32 * 17, 32 * 17), (10, 1, 32, 32)),
        ((5, 7, 32 * 11, 32 * 11), (5, 1, 32, 32)),
        ((5, 1, 32 * 11, 32 * 11), (5, 1, 32, 32)),
        ((5, 1, 32, 32 * 11), (5, 1, 32, 32)),
        ((5, 1, 32, 32), (5, 1, 32, 32)),
        ((1, 5, 32 * 11, 32 * 11), (1, 1, 32, 32)),
        ((1, 1, 32, 32), (1, 1, 32, 32)),
    ],
)
def test_reduce_on_batch(shape, shard_shape, device):
    torch.manual_seed(0)

    torch_input_tensor = torch_random(shape, -100, 100, dtype=torch.float32)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=0, keepdim=True)

    grid_size = device.compute_with_storage_grid_size()
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))
    grid = ttnn.CoreRangeSet([core_range])

    memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape(shard_shape), grid))
    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )

    output_tensor = ttnn.sum(input_tensor, dim=0, keepdim=True)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)
