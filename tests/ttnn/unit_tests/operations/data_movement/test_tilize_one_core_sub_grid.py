# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""tilize must not run its single-core reader on WIDTH or BLOCK sharded inputs.

The single-core factory reads a whole row from one input page, which is only valid when a row lives
inside one shard. WIDTH_SHARDED and BLOCK_SHARDED inputs split a row across shards, so the read
crosses shard boundaries and returns other shards' bytes. A `sub_core_grids` of one core (or
use_low_perf) used to select that factory and the call returned wrong values with no error.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

WIDTH_SHARDED = ttnn.TensorMemoryLayout.WIDTH_SHARDED
HEIGHT_SHARDED = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _core_grid(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _sharded_config(memory_layout, shard_shape, grid=None):
    grid = grid if grid is not None else _core_grid(0, 0, 3, 0)
    return ttnn.MemoryConfig(
        memory_layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )


def _row_major_input(device, shape, memory_config):
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    return torch_input, tt_input


def test_tilize_one_core_sub_grid_width_sharded(device):
    # The four shards hold 64 columns each, so one row spans all four shards and the single-core
    # factory's full-row read crossed shard boundaries.
    memory_config = _sharded_config(WIDTH_SHARDED, [32, 64])
    torch_input, tt_input = _row_major_input(device, (1, 1, 32, 256), memory_config)
    output = ttnn.tilize(
        tt_input,
        memory_config=memory_config,
        use_multicore=True,
        sub_core_grids=_core_grid(0, 0, 0, 0),
    )
    assert_equal(ttnn.to_torch(output).float(), torch_input.float())


def test_tilize_one_core_sub_grid_height_sharded(device):
    # Control that passes with and without the fix: a whole row lies inside one height shard.
    memory_config = _sharded_config(HEIGHT_SHARDED, [32, 64])
    torch_input, tt_input = _row_major_input(device, (1, 1, 128, 64), memory_config)
    output = ttnn.tilize(
        tt_input,
        memory_config=memory_config,
        use_multicore=True,
        sub_core_grids=_core_grid(0, 0, 0, 0),
    )
    assert_equal(ttnn.to_torch(output).float(), torch_input.float())


def test_tilize_two_core_sub_grid_width_sharded(device):
    # Control that passes with and without the fix: two cores already avoid the single-core factory.
    memory_config = _sharded_config(WIDTH_SHARDED, [32, 64])
    torch_input, tt_input = _row_major_input(device, (1, 1, 32, 256), memory_config)
    output = ttnn.tilize(
        tt_input,
        memory_config=memory_config,
        use_multicore=True,
        sub_core_grids=_core_grid(0, 0, 1, 0),
    )
    assert_equal(ttnn.to_torch(output).float(), torch_input.float())


def test_tilize_one_core_sub_grid_interleaved(device):
    # Control that passes with and without the fix: interleaved input rows are contiguous.
    torch_input, tt_input = _row_major_input(device, (1, 1, 32, 256), ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.tilize(
        tt_input,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        use_multicore=True,
        sub_core_grids=_core_grid(0, 0, 0, 0),
    )
    assert_equal(ttnn.to_torch(output).float(), torch_input.float())
