# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


def l1_mem_config(grid_x, grid_y, shard_h, shard_w, layout):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])
    shard_spec = ttnn.ShardSpec(grid, (shard_h, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, shard_spec)


WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

# A width shard fills one core of the grid, so an x-by-y grid holds x*y shards. The partial last shard
# is only detected when the core count multiplies both extents: these 2D-grid cases with a padded
# capacity (4*32 > 112, 8*16 > 112) took the no-padding factory and wrote one row of the last shard
# from the wrong offset.
WIDTH_SHARDED_CASES = [
    ((2, 2, 32, 32), (2, 4, 32, 16)),
    ((2, 2, 32, 32), (4, 2, 32, 16)),
]

WIDTH_SHARDED_CONTROLS = [
    ((4, 1, 32, 32), (8, 1, 32, 16)),
    ((2, 4, 32, 16), (2, 2, 32, 32)),
    ((2, 2, 32, 32), (8, 1, 32, 16)),
]


@pytest.mark.parametrize("src, dst", WIDTH_SHARDED_CASES + WIDTH_SHARDED_CONTROLS)
def test_reshard_width_sharded_2d_grid_partial_last_shard(device, src, dst):
    torch.manual_seed(0)
    shape = (1, 1, 32, 112)
    x_torch = torch.randn(shape).bfloat16()
    src_config = l1_mem_config(*src, WS)
    dst_config = l1_mem_config(*dst, WS)

    x = ttnn.from_torch(
        x_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=src_config
    )
    y = ttnn.reshard(x, dst_config)

    passing, reason = comp_equal(x_torch, ttnn.to_torch(y))
    assert passing, reason


def test_reshard_height_sharded_2d_grid_control(device):
    torch.manual_seed(0)
    shape = (1, 1, 112, 64)
    x_torch = torch.randn(shape).bfloat16()
    src_config = l1_mem_config(2, 2, 32, 64, HS)
    dst_config = l1_mem_config(2, 4, 16, 64, HS)

    x = ttnn.from_torch(
        x_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=src_config
    )
    y = ttnn.reshard(x, dst_config)

    passing, reason = comp_equal(x_torch, ttnn.to_torch(y))
    assert passing, reason
