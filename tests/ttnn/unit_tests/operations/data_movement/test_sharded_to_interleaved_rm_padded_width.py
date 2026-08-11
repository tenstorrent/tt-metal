# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

# Shard width wider than the logical width, as pool produces when it pads channels.
# (memory_layout, shape, shard_shape, grid)
SHARDINGS = [
    (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [1, 1, 64, 40], (32, 64), (1, 2)),
    (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [1, 1, 64, 90], (32, 96), (1, 2)),
    (ttnn.TensorMemoryLayout.WIDTH_SHARDED, [1, 1, 32, 40], (32, 64), (1, 1)),
    (ttnn.TensorMemoryLayout.WIDTH_SHARDED, [1, 1, 32, 100], (32, 64), (2, 1)),
    (ttnn.TensorMemoryLayout.BLOCK_SHARDED, [1, 1, 64, 100], (32, 64), (2, 2)),
]
SHARDING_IDS = ["hs_40in64", "hs_90in96", "ws_40in64", "ws_100in128", "bs_100in128"]


def make_sharded(device, memory_layout, shape, shard_shape, grid):
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))
    memory_config = ttnn.MemoryConfig(
        memory_layout,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet({core_range}), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )
    torch_input = torch.rand(shape).bfloat16()
    sharded = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    assert sharded.padded_shape[-1] > sharded.shape[-1], "test setup must produce a padded shard width"
    return torch_input, sharded


@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
@pytest.mark.parametrize("memory_layout, shape, shard_shape, grid", SHARDINGS, ids=SHARDING_IDS)
def test_s2i_rm_output_is_unpadded(device, memory_config, memory_layout, shape, shard_shape, grid):
    torch_input, sharded = make_sharded(device, memory_layout, shape, shard_shape, grid)

    output = ttnn.sharded_to_interleaved(sharded, memory_config)

    assert list(output.padded_shape) == shape
    assert output.volume() == output.logical_volume()
    assert torch.equal(ttnn.to_torch(output), torch_input)


@pytest.mark.parametrize("memory_layout, shape, shard_shape, grid", SHARDINGS, ids=SHARDING_IDS)
def test_s2i_rm_padded_width_downstream_pad(device, memory_layout, shape, shard_shape, grid):
    torch_input, sharded = make_sharded(device, memory_layout, shape, shard_shape, grid)

    output = ttnn.sharded_to_interleaved(sharded, ttnn.DRAM_MEMORY_CONFIG)
    padded = ttnn.pad(output, [(0, 0), (0, 0), (0, 0), (0, 8)], 0.0)

    assert torch.equal(ttnn.to_torch(padded), torch.nn.functional.pad(torch_input, (0, 8)))


@pytest.mark.parametrize("memory_layout, shape, shard_shape, grid", SHARDINGS, ids=SHARDING_IDS)
def test_s2i_rm_padded_width_no_neighbour_clobber(device, memory_layout, shape, shard_shape, grid):
    torch_input, sharded = make_sharded(device, memory_layout, shape, shard_shape, grid)

    # Free a hole and put a guard right after it, so an over-wide stick write lands in the guard.
    hole = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    torch_guard = torch.full(shape, -1.0).bfloat16()
    guard = ttnn.from_torch(
        torch_guard,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(hole)

    output = ttnn.sharded_to_interleaved(sharded, ttnn.DRAM_MEMORY_CONFIG)

    assert torch.equal(ttnn.to_torch(output), torch_input)
    assert torch.equal(ttnn.to_torch(guard), torch_guard)


# Shard grid with more columns than the data needs: the trailing columns hold only padding.
@pytest.mark.parametrize("grid_x", [3, 4, 6], ids=["exact", "one_spare_col", "three_spare_cols"])
def test_s2i_rm_block_sharded_over_provisioned_grid(device, grid_x):
    shape = [1, 1, 64, 90]
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, 1))
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet({core_range}), (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    torch_input = torch.rand(shape).bfloat16()
    sharded = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    assert torch.equal(ttnn.to_torch(sharded), torch_input), "test setup: sharded tensor must round-trip"

    output = ttnn.sharded_to_interleaved(sharded, ttnn.DRAM_MEMORY_CONFIG)

    assert list(output.padded_shape) == shape
    assert torch.equal(ttnn.to_torch(output), torch_input)
