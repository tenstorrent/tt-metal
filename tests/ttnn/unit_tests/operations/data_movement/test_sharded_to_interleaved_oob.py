# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_equal


def test_s2i_dram_height_sharded(device):
    torch_weight_tensor = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    torch_input_tensor = torch.rand([1, 1, 320, 32], dtype=torch.bfloat16)

    core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 7)),
        }
    )
    input_shard_shape = (32, 32)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )
    input_tensor = ttnn.from_torch(
        torch_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=input_memory_config
    )

    # Allocate a dummy tensor so we can put weight after in DRAM
    output_tensor = ttnn.from_torch(
        torch_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )  # Alternatively: ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
    weight_tensor = ttnn.from_torch(
        torch_weight_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )  # This will be allocated after output tensor in DRAM

    # Deallocated output to create slot and then reshard from L1 again
    ttnn.deallocate(output_tensor)
    output_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    assert_equal(torch_input_tensor, ttnn.to_torch(output_tensor))
    assert_equal(torch_weight_tensor, ttnn.to_torch(weight_tensor))


# Shard wider than the logical row, as pool leaves it: the interleaved page must be the logical row,
# not the shard width. Height sharding is the one strategy that never clamps its write, so the guard
# catches an over-wide stick. The block grid has 3 padding-only columns that must be skipped.
@pytest.mark.parametrize(
    "memory_layout, shard_shape, grid",
    [
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (32, 96), (1, 2)),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (32, 32), (6, 2)),
    ],
    ids=["height_sharded_wider_than_row", "block_sharded_spare_columns"],
)
def test_s2i_rm_shard_wider_than_row(device, memory_layout, shard_shape, grid):
    shape = [1, 1, 64, 90]
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)

    core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1)),
        }
    )
    input_shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, input_shard_spec)
    input_tensor = ttnn.from_torch(
        torch_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=input_memory_config
    )
    assert input_tensor.padded_shape[-1] > input_tensor.shape[-1], "test setup must pad the shard width"
    assert_equal(torch_input_tensor, ttnn.to_torch(input_tensor))

    # Hole the exact size of the unpadded output, guard right behind it.
    output_tensor = ttnn.from_torch(
        torch_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    torch_guard_tensor = torch.full(shape, -1.0, dtype=torch.bfloat16)
    guard_tensor = ttnn.from_torch(
        torch_guard_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.deallocate(output_tensor)

    output_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    assert list(output_tensor.padded_shape) == shape, "an interleaved row-major page is one logical row"
    assert_equal(torch_input_tensor, ttnn.to_torch(output_tensor))
    assert_equal(torch_guard_tensor, ttnn.to_torch(guard_tensor))
