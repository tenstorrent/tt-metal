# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


from models.utility_functions import is_wormhole_b0, is_blackhole


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="#7733: fix for sharding on whb0")
@pytest.mark.parametrize(
    "mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG]
)
@pytest.mark.parametrize("num_allocs", [1, 2, 3, 4])
def test_ttnn_reallocate(device, mem_config, num_allocs):
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

    # If sharded, creat actual memory config
    if mem_config == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG:
        shard_spec = ttnn.ShardSpec(
            shard_grid, [batch * height * depth // 8, width], ttnn.ShardOrientation.ROW_MAJOR, False
        )
        mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
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
