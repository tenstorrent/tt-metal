# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for ttnn.experimental.deepseek.width_to_height_shard.

The op turns a TILE-layout WIDTH_SHARDED input into a ROW_MAJOR HEIGHT_SHARDED output whose
shard is the *whole* input tensor: each input core untilizes its column slice and writes it into
the hub core's output shard, and once every slice has landed the hub multicasts the assembled
tensor to every core in ``output_core_range_set``.

So for a [1, 1, 1, 4096] input width-sharded over 64 cores (64 columns per core), every output
core ends up holding a full 1x4096 row-major copy of the logical input.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

TILE = 32


def _core_range_set(grid_x: int, grid_y: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])


def _width_sharded_config(core_range_set: ttnn.CoreRangeSet, height: int, width: int) -> ttnn.MemoryConfig:
    num_cores = core_range_set.num_cores()
    assert width % num_cores == 0, f"width {width} must divide evenly over {num_cores} cores"
    shard_spec = ttnn.ShardSpec(core_range_set, (height, width // num_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


@pytest.mark.parametrize("shape", [(1, 1, 1, 4096)], ids=lambda s: "x".join(str(d) for d in s))
@pytest.mark.parametrize("input_grid", [(4, 4), (8, 8)], ids=lambda g: f"in{g[0]}x{g[1]}")
# The op multicasts over the bounding box of the output set, so cover both a full-grid broadcast and
# a smaller one whose bounding box is a strict subset of the input cores.
@pytest.mark.parametrize("output_grid", [(6, 6), (4, 4)], ids=lambda g: f"out{g[0]}x{g[1]}")
def test_width_to_height_shard(device, shape, input_grid, output_grid):
    torch.manual_seed(0)

    width = shape[-1]
    logical_height = shape[-2]
    padded_height = max(TILE, logical_height)
    input_core_range_set = _core_range_set(*input_grid)
    output_core_range_set = _core_range_set(*output_grid)
    num_output_cores = output_core_range_set.num_cores()

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_width_sharded_config(input_core_range_set, padded_height, width),
    )

    tt_output = ttnn.experimental.deepseek.width_to_height_shard(tt_input, output_core_range_set)
    print(tt_output.memory_config().shard_spec.shape, tt_output.shape)
    assert tt_output.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tt_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert tuple(tt_output.memory_config().shard_spec.shape) == (logical_height, width)

    torch_output = ttnn.to_torch(tt_output)
    per_core = torch_output.reshape(num_output_cores, logical_height, width)
    expected = torch_input.reshape(logical_height, width)

    for core_id in range(num_output_cores):
        assert_equal(expected, per_core[core_id])
