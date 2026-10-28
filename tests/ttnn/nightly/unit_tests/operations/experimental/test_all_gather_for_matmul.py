# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for ttnn.experimental.deepseek.all_gather_for_matmul.

WIDTH_SHARDED input: each core untilizes its column slice if the data is TILE, then the
slices are gathered on a hub and the hub multicasts the assembled tensor to
``output_core_range_set``. ROW_MAJOR width shards skip untilize.

HEIGHT_SHARDED single-core input: the full tensor already lives on one core, so that core
untilizes if needed and multicasts (no gather). TILE and ROW_MAJOR are accepted.

Both paths replicate all M rows, where M is the product of every logical dim but the last,
so the output shard is the whole dense (M, K) tensor and the output's logical shape is
``[1, ..., 1, num_output_cores * M, K]``.
"""

import math

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

TILE = 32


def _core_range_set(grid_x: int, grid_y: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])


def _single_core_range_set(x: int, y: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))])


def _width_sharded_config(core_range_set: ttnn.CoreRangeSet, height: int, width: int) -> ttnn.MemoryConfig:
    num_cores = core_range_set.num_cores()
    assert width % num_cores == 0, f"width {width} must divide evenly over {num_cores} cores"
    shard_spec = ttnn.ShardSpec(core_range_set, (height, width // num_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def _height_sharded_config(core_range_set: ttnn.CoreRangeSet, height: int, width: int) -> ttnn.MemoryConfig:
    shard_spec = ttnn.ShardSpec(core_range_set, (height, width), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _input_shard_height(shape, layout) -> int:
    """Flattened padded height: TILE pads dim -2 of every leading-dim slice up to a whole tile."""
    rows_per_slice = shape[-2]
    if layout == ttnn.TILE_LAYOUT:
        rows_per_slice = math.ceil(rows_per_slice / TILE) * TILE
    return math.prod(shape[:-2]) * rows_per_slice


def _assert_replicated(tt_output, torch_input, num_output_cores):
    width = torch_input.shape[-1]
    rows = math.prod(torch_input.shape[:-1])
    assert tt_output.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tt_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert tuple(tt_output.memory_config().shard_spec.shape) == (rows, width)
    expected_shape = (1,) * (torch_input.dim() - 2) + (num_output_cores * rows, width)
    assert tuple(tt_output.shape) == expected_shape

    torch_output = ttnn.to_torch(tt_output)
    per_core = torch_output.reshape(num_output_cores, rows, width)
    expected = torch_input.reshape(rows, width)
    for core_id in range(num_output_cores):
        assert_equal(expected, per_core[core_id])


def _run_width_sharded(device, shape, layout, input_grid, output_grid):
    torch.manual_seed(0)

    width = shape[-1]
    input_core_range_set = _core_range_set(*input_grid)
    num_input_cores = input_core_range_set.num_cores()
    if width % num_input_cores != 0:
        pytest.skip(f"width {width} does not split evenly over {num_input_cores} cores")
    if layout == ttnn.TILE_LAYOUT and (width // num_input_cores) % TILE != 0:
        pytest.skip(f"width {width} does not split into tile-wide shards over {num_input_cores} cores")
    output_core_range_set = _core_range_set(*output_grid)

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=_width_sharded_config(input_core_range_set, _input_shard_height(shape, layout), width),
    )

    tt_output = ttnn.experimental.deepseek.all_gather_for_matmul(tt_input, output_core_range_set)
    _assert_replicated(tt_output, torch_input, output_core_range_set.num_cores())


def _run_height_sharded(device, shape, layout, input_core, output_grid):
    torch.manual_seed(0)

    grid = device.compute_with_storage_grid_size()
    if input_core[0] >= grid.x or input_core[1] >= grid.y:
        pytest.skip(f"input core {input_core} exceeds device grid {grid.x}x{grid.y}")

    width = shape[-1]
    input_core_range_set = _single_core_range_set(*input_core)
    output_core_range_set = _core_range_set(*output_grid)

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=_height_sharded_config(input_core_range_set, _input_shard_height(shape, layout), width),
    )

    tt_output = ttnn.experimental.deepseek.all_gather_for_matmul(tt_input, output_core_range_set)
    _assert_replicated(tt_output, torch_input, output_core_range_set.num_cores())


def _shape_id(shape) -> str:
    return "x".join(str(d) for d in shape)


# A TILE width shard must be a whole tile wide, so TILE cases need width / num_cores to be a
# multiple of 32. ROW_MAJOR only needs the width to divide evenly over the input cores.
@pytest.mark.parametrize("shape", [(1, 1, 1, 4096), (1, 1, 16, 2048), (1, 1, 16, 512)], ids=_shape_id)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("input_grid", [(4, 4), (8, 8)], ids=lambda g: f"in{g[0]}x{g[1]}")
# The op multicasts over the bounding box of the output set, so cover both a full-grid broadcast and
# a smaller one whose bounding box is a strict subset of the input cores.
@pytest.mark.parametrize("output_grid", [(6, 6), (4, 4)], ids=lambda g: f"out{g[0]}x{g[1]}")
def test_all_gather_for_matmul(device, shape, layout, input_grid, output_grid):
    _run_width_sharded(device, shape, layout, input_grid, output_grid)


@pytest.mark.parametrize("shape", [(1, 1, 1, 4096), (1, 1, 16, 512), (16, 512)], ids=_shape_id)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("input_core", [(0, 0), (5, 5)], ids=lambda c: f"src{c[0]}x{c[1]}")
@pytest.mark.parametrize("output_grid", [(6, 6), (4, 4)], ids=lambda g: f"out{g[0]}x{g[1]}")
def test_all_gather_for_matmul_height_sharded(device, shape, layout, input_core, output_grid):
    _run_height_sharded(device, shape, layout, input_core, output_grid)


# M > 1 beyond a single tile-row: M spanning several tile-rows (aligned and not), and M spread over
# leading dims, where every slice carries its own tile padding in TILE layout. Every output core
# must hold the dense M x K tensor.
MULTI_ROW_WIDTH_SHAPES = [
    (1, 1, 64, 2048),  # two full tile-rows
    (1, 1, 40, 2048),  # M not tile-aligned across tile-rows
    (1, 4, 1, 2048),  # one row per user, users on a leading dim
    (1, 2, 16, 2048),  # two padded slices
    (2, 3, 5, 2048),  # several leading dims
    (3, 7, 2048),  # rank 3
]


@pytest.mark.parametrize("shape", MULTI_ROW_WIDTH_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("input_grid", [(4, 4), (8, 8)], ids=lambda g: f"in{g[0]}x{g[1]}")
@pytest.mark.parametrize("output_grid", [(6, 6), (4, 4)], ids=lambda g: f"out{g[0]}x{g[1]}")
def test_all_gather_for_matmul_multi_row(device, shape, layout, input_grid, output_grid):
    _run_width_sharded(device, shape, layout, input_grid, output_grid)


MULTI_ROW_HEIGHT_SHAPES = [
    (1, 1, 64, 512),
    (1, 1, 40, 512),
    (1, 4, 1, 512),
    (1, 2, 16, 512),
    (2, 3, 5, 256),
    (3, 7, 256),
]


@pytest.mark.parametrize("shape", MULTI_ROW_HEIGHT_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("input_core", [(0, 0), (5, 5)], ids=lambda c: f"src{c[0]}x{c[1]}")
@pytest.mark.parametrize("output_grid", [(6, 6), (4, 4)], ids=lambda g: f"out{g[0]}x{g[1]}")
def test_all_gather_for_matmul_height_sharded_multi_row(device, shape, layout, input_core, output_grid):
    _run_height_sharded(device, shape, layout, input_core, output_grid)
