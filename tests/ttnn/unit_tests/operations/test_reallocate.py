# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


from models.utility_functions import is_wormhole_b0, is_blackhole


@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("num_allocs", [1, 2, 3, 4])
def test_reallocate_interleaved(device, mem_config, num_allocs):
    width = 1024
    height = 128
    depth = 2
    batch = 2

    if num_allocs == 2 and mem_config == ttnn.DRAM_MEMORY_CONFIG:
        pytest.xfail("#7732: dram tensor corruption after move")

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 7),
            ),
        }
    )

    torch_tensors = []
    tensors = []
    for i in range(num_allocs):
        it = torch.rand((batch, depth, height, width), dtype=torch.bfloat16)
        torch_tensors.append(it)
        device_tensor = ttnn.to_device(
            ttnn.from_torch(it, dtype=ttnn.bfloat16), device=device, memory_config=mem_config
        )
        tensors.append(device_tensor)

    for i in range(num_allocs - 1):
        tensors[i].deallocate()

    assert ttnn.is_tensor_storage_on_device(tensors[-1])
    assert tensors[-1].is_allocated()

    initial_address = tensors[-1].buffer_address()
    tensors[-1] = ttnn.reallocate(tensors[-1])
    new_address = tensors[-1].buffer_address()

    if mem_config == ttnn.DRAM_MEMORY_CONFIG:
        assert new_address <= initial_address
    else:
        assert new_address >= initial_address

    assert_with_pcc(torch_tensors[-1], ttnn.to_torch(tensors[-1]), 0.9999)


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.HEIGHT])
@pytest.mark.parametrize(
    "input_shape, core_grid",
    (
        ([1, 1, 32, 32], ttnn.CoreGrid(x=1, y=1)),
        ([1, 1, 256, 256], ttnn.CoreGrid(x=2, y=2)),
        ([1, 1, 4, 34], ttnn.CoreGrid(x=1, y=1)),  # Checks unaligned RM shard
        ([2, 2, 128, 1024], ttnn.CoreGrid(x=4, y=4)),
    ),
)
def test_reallocate_sharded(device, input_shape, core_grid, strategy, layout):
    if (input_shape[-1] % ttnn.TILE_SIZE != 0 or input_shape[-2] % ttnn.TILE_SIZE != 0) and layout == ttnn.TILE_LAYOUT:
        pytest.skip("Shards must be aligned with tile layout")

    input_memory_config = ttnn.create_sharded_memory_config(
        input_shape, core_grid, strategy, ttnn.ShardOrientation.ROW_MAJOR
    )

    torch_input_tensor = torch.rand(input_shape).to(dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=layout)

    dummy_tensor = torch.rand([1, 1, 512, 512])
    dummy_tensor = ttnn.from_torch(dummy_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    dummy_tensor = ttnn.to_device(dummy_tensor, device, ttnn.L1_MEMORY_CONFIG)

    input_tensor = ttnn.to_device(input_tensor, device, input_memory_config)

    ttnn.deallocate(dummy_tensor)  # make L1 space for reallocation
    output_tensor = ttnn.reallocate(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_input_tensor, output_tensor, 1.0)
