# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)

from enum import Enum

from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull


def run_reshard_test(
    device,
    input_shape,
    input_layout,
    input_shard_grid,
    input_shard_shape,
    input_shard_orientation,
    input_sharding_scheme,
    output_shard_grid,
    output_shard_shape,
    output_shard_orientation,
    output_sharding_scheme,
    tt_dtype,
):
    input_shard_grid_set = set()
    for _input_shard_grid in input_shard_grid:
        compute_grid_start = ttnn.CoreCoord(_input_shard_grid[0][0], _input_shard_grid[0][1])
        compute_grid_end = ttnn.CoreCoord(_input_shard_grid[1][0], _input_shard_grid[1][1])
        input_shard_grid_set.add(ttnn.CoreRange(compute_grid_start, compute_grid_end))

    input_shard_grid = ttnn.CoreRangeSet(input_shard_grid_set)

    output_shard_grid_set = set()
    for _output_shard_grid in output_shard_grid:
        compute_grid_start = ttnn.CoreCoord(_output_shard_grid[0][0], _output_shard_grid[0][1])
        compute_grid_end = ttnn.CoreCoord(_output_shard_grid[1][0], _output_shard_grid[1][1])
        output_shard_grid_set.add(ttnn.CoreRange(compute_grid_start, compute_grid_end))

    output_shard_grid = ttnn.CoreRangeSet(output_shard_grid_set)

    output_shard_spec = ttnn.ShardSpec(output_shard_grid, output_shard_shape, output_shard_orientation, False)
    output_mem_config = ttnn.MemoryConfig(output_sharding_scheme, ttnn.BufferType.L1, output_shard_spec)
    if input_layout == ttnn.ROW_MAJOR_LAYOUT and tt_dtype == ttnn.bfloat8_b:
        pytest.skip("Illegal layout/dtype config")

    dram_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    torch_tensor = torch.randn(input_shape).bfloat16()
    tt_tensor_sharded = ttnn.Tensor(torch_tensor, tt_dtype).to(input_layout)
    tt_tensor_sharded = tt_tensor_sharded.to(device, dram_memory_config)
    tt_tensor_sharded = ttnn.interleaved_to_sharded(
        tt_tensor_sharded,
        input_shard_grid,
        input_shard_shape,
        input_sharding_scheme,
        input_shard_orientation,
        output_dtype=tt_dtype,
    )

    tt_tensor_reshard = ttnn.reshard(tt_tensor_sharded, output_mem_config)

    tt_tensor_interleaved = ttnn.sharded_to_interleaved(
        tt_tensor_reshard,
        dram_memory_config,
    )

    tt_tensor_interleaved = tt_tensor_interleaved.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    torch_tensor_after_round_trip = tt_tensor_interleaved.to_torch()

    return torch_tensor, torch_tensor_after_round_trip


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "input_shape, input_layout, input_shard_grid,  input_shard_shape, input_shard_orientation, input_sharding_scheme, output_shard_grid, output_shard_shape, output_shard_orientation, output_sharding_scheme",
    [
        (
            [1, 1, 64, 64],
            ttnn.TILE_LAYOUT,
            [[(0, 0), (0, 1)]],
            (64, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [[(0, 0), (0, 1)]],
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            [1, 1, 128, 64],
            ttnn.TILE_LAYOUT,
            [[(0, 0), (0, 1)]],
            (64, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (1, 3)]],
            (32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        (
            [1, 1, 32, 128],
            ttnn.TILE_LAYOUT,
            [[(0, 0), (0, 3)]],
            (32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (0, 1)]],
            (32, 64),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        (
            [1, 1, 32, 2304],
            ttnn.TILE_LAYOUT,
            [[(0, 0), (0, 7)]],
            (32, 288),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (0, 1)]],
            (32, 1152),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        (
            [1, 1, 32, 16],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (0, 0)]],
            (32, 16),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (0, 1)]],
            (16, 16),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        (
            [1, 1, 32, 8192],
            ttnn.TILE_LAYOUT,
            [[(0, 0), (7, 7)]],
            (32, 128),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [[(0, 0), (0, 7)]],
            (32, 1024),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("tt_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_reshard(
    device,
    input_shape,
    input_layout,
    input_shard_grid,
    input_shard_shape,
    input_shard_orientation,
    input_sharding_scheme,
    output_shard_grid,
    output_shard_shape,
    output_shard_orientation,
    output_sharding_scheme,
    tt_dtype,
):
    torch_tensor, torch_tensor_after_round_trip = run_reshard_test(
        device,
        input_shape,
        input_layout,
        input_shard_grid,
        input_shard_shape,
        input_shard_orientation,
        input_sharding_scheme,
        output_shard_grid,
        output_shard_shape,
        output_shard_orientation,
        output_sharding_scheme,
        tt_dtype,
    )

    assert torch_tensor.shape == torch_tensor_after_round_trip.shape
    if tt_dtype != ttnn.bfloat8_b:
        passing, output = comp_equal(torch_tensor, torch_tensor_after_round_trip)
    else:
        passing, output = comp_pcc(torch_tensor, torch_tensor_after_round_trip)
    assert passing, output


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shape, input_layout, input_shard_grid,  input_shard_shape, input_shard_orientation, input_sharding_scheme, output_shard_grid, output_shard_shape, output_shard_orientation, output_sharding_scheme",
    [
        (
            [1, 1, 62720, 256],
            ttnn.TILE_LAYOUT,
            [[(0, 0), (7, 6)]],
            (1120, 256),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [[(0, 0), (7, 5)], [(0, 6), (0, 6)]],
            (1280, 256),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("tt_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_reshard_rn50(
    device,
    input_shape,
    input_layout,
    input_shard_grid,
    input_shard_shape,
    input_shard_orientation,
    input_sharding_scheme,
    output_shard_grid,
    output_shard_shape,
    output_shard_orientation,
    output_sharding_scheme,
    tt_dtype,
):
    torch_tensor, torch_tensor_after_round_trip = run_reshard_test(
        device,
        input_shape,
        input_layout,
        input_shard_grid,
        input_shard_shape,
        input_shard_orientation,
        input_sharding_scheme,
        output_shard_grid,
        output_shard_shape,
        output_shard_orientation,
        output_sharding_scheme,
        tt_dtype,
    )

    assert torch_tensor.shape == torch_tensor_after_round_trip.shape
    if tt_dtype != ttnn.bfloat8_b:
        passing, output = comp_equal(torch_tensor, torch_tensor_after_round_trip)
    else:
        passing, output = comp_pcc(torch_tensor, torch_tensor_after_round_trip)
    assert passing, output


@pytest.mark.parametrize(
    "input_shape, input_layout, input_shard_grid,  input_shard_shape, input_shard_orientation, input_sharding_scheme, output_shard_grid, output_shard_shape, output_shard_orientation, output_sharding_scheme",
    [
        (
            [1, 1, 32, 6272],
            ttnn.TILE_LAYOUT,
            [[(0, 0), (6, 6)]],
            (32, 128),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [[(0, 0), (0, 6)]],
            (32, 1024),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("tt_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_reshard_with_program_cache(
    device,
    use_program_cache,
    input_shape,
    input_layout,
    input_shard_grid,
    input_shard_shape,
    input_shard_orientation,
    input_sharding_scheme,
    output_shard_grid,
    output_shard_shape,
    output_shard_orientation,
    output_sharding_scheme,
    tt_dtype,
):
    torch_tensor, torch_tensor_after_round_trip = run_reshard_test(
        device,
        input_shape,
        input_layout,
        input_shard_grid,
        input_shard_shape,
        input_shard_orientation,
        input_sharding_scheme,
        output_shard_grid,
        output_shard_shape,
        output_shard_orientation,
        output_sharding_scheme,
        tt_dtype,
    )

    assert torch_tensor.shape == torch_tensor_after_round_trip.shape
    if tt_dtype != ttnn.bfloat8_b:
        passing, output = comp_equal(torch_tensor, torch_tensor_after_round_trip)
    else:
        passing, output = comp_pcc(torch_tensor, torch_tensor_after_round_trip)
    assert passing, output

    torch_tensor1, torch_tensor_after_round_trip1 = run_reshard_test(
        device,
        input_shape,
        input_layout,
        input_shard_grid,
        input_shard_shape,
        input_shard_orientation,
        input_sharding_scheme,
        output_shard_grid,
        output_shard_shape,
        output_shard_orientation,
        output_sharding_scheme,
        tt_dtype,
    )

    assert torch_tensor1.shape == torch_tensor_after_round_trip1.shape
    if tt_dtype != ttnn.bfloat8_b:
        passing, output = comp_equal(torch_tensor1, torch_tensor_after_round_trip1)
    else:
        passing, output = comp_pcc(torch_tensor1, torch_tensor_after_round_trip1)
    assert passing, output

    assert device.num_program_cache_entries() == 3
