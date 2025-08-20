# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
        # Batch dims are equal for shape and shard_shape, all other dims of shard shape are 1 (or tile size for the lower two dims, or other combinations)
        ((10, 4, 32 * 17, 32 * 17), (10, 1, 32, 32)),
        ((5, 7, 32 * 11, 32 * 11), (5, 1, 32, 32)),
        ((5, 1, 32 * 11, 32 * 11), (5, 1, 32, 32)),
        ((5, 1, 32, 32 * 11), (5, 1, 32, 32)),
        ((5, 1, 32, 32), (5, 1, 32, 32)),
        ((1, 5, 32 * 11, 32 * 11), (1, 1, 32, 32)),
        ((9, 5, 32 * 11, 32 * 11), (9, 1, 32, 32)),
        ((1, 1, 32, 32), (1, 1, 32, 32)),
        ((10, 1, 320, 320), (1, 32, 32)),
        ((10, 4, 32 * 16, 32 * 16), (10, 1, 32, 32)),
        ((10, 4, 32 * 16, 32 * 16), (10, 1, 64, 64)),
        ((10, 4, 32 * 16, 32 * 16), (10, 2, 32, 64)),
        ((10, 65, 32, 32), (10, 1, 32, 32)),
        ((10, 65, 64, 64), (10, 1, 32, 32)),
        ((10, 4, 32 * 16, 32 * 16), (5, 2, 32, 64)),  # half batch sharding
        ((10, 5, 32 * 11, 32 * 11), (10, 2, 64, 64)),  # tensor dimensions not evenly divided by shard dimensions
    ],
)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("interleaved", [False, True])
def test_reduce_on_batch(shape, shard_shape, dim, interleaved, device):
    torch.manual_seed(0)

    torch_input_tensor = torch_random(shape, -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=True)

    grid_size = device.compute_with_storage_grid_size()
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))
    grid = ttnn.CoreRangeSet([core_range])

    memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape(shard_shape), grid))
    output_shard_shape = list(shard_shape)
    output_shard_shape[0] = 1
    memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape(shard_shape), grid))
    output_memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape(output_shard_shape), grid))
    if interleaved:
        memory_config = None
        output_memory_config = None
    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=True, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.995)
