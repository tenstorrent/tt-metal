# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)

from models.utility_functions import skip_for_grayskull, skip_for_blackhole


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
    grid_size = device.compute_with_storage_grid_size()
    input_shard_grid_set = set()
    for _input_shard_grid in input_shard_grid:
        compute_grid_start = ttnn.CoreCoord(_input_shard_grid[0][0], _input_shard_grid[0][1])
        compute_grid_end = ttnn.CoreCoord(_input_shard_grid[1][0], _input_shard_grid[1][1])
        if compute_grid_end.x > grid_size.x - 1 or compute_grid_end.y > grid_size.y - 1:
            pytest.skip("Shard Grid exceeds device grid size")
        input_shard_grid_set.add(ttnn.CoreRange(compute_grid_start, compute_grid_end))

    input_shard_grid = ttnn.CoreRangeSet(input_shard_grid_set)

    output_shard_grid_set = set()
    for _output_shard_grid in output_shard_grid:
        compute_grid_start = ttnn.CoreCoord(_output_shard_grid[0][0], _output_shard_grid[0][1])
        compute_grid_end = ttnn.CoreCoord(_output_shard_grid[1][0], _output_shard_grid[1][1])
        if compute_grid_end.x > grid_size.x - 1 or compute_grid_end.y > grid_size.y - 1:
            pytest.skip("Shard Grid exceeds device grid size")
        output_shard_grid_set.add(ttnn.CoreRange(compute_grid_start, compute_grid_end))

    output_shard_grid = ttnn.CoreRangeSet(output_shard_grid_set)

    output_shard_spec = ttnn.ShardSpec(output_shard_grid, output_shard_shape, output_shard_orientation)
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


@skip_for_blackhole("Hitting assertion in reshard op on BH, see #12349")
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
        (
            [1, 1, 1320, 32],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (7, 5)], [(0, 6), (6, 6)]],
            (24, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [[(0, 0), (7, 4)], [(0, 5), (1, 5)]],
            (32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            [1, 1, 4, 256],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (3, 0)]],
            (4, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [[(0, 0), (0, 1)]],
            (4, 128),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            [1, 1, 4, 256],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (1, 0)]],
            (4, 128),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [[(0, 0), (0, 7)]],
            (4, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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


@skip_for_blackhole("GH Issue #15234")
@pytest.mark.parametrize(
    "input_shape, input_layout, input_shard_grid, input_shard_shape, input_shard_orientation, input_sharding_scheme, input_buffer_type, output_shard_grid, output_shard_shape, output_shard_orientation, output_sharding_scheme, output_buffer_type",
    [
        (
            [1, 1, 768, 64],
            ttnn.TILE_LAYOUT,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            (96, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))}),
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
        ),
        (
            [1, 1, 768, 64],
            ttnn.TILE_LAYOUT,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))}),
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            (96, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
        ),
        (
            [1, 1, 768, 64],
            ttnn.TILE_LAYOUT,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            (96, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            (192, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
        ),
        (
            [1, 1, 768, 64],
            ttnn.TILE_LAYOUT,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            (192, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            (96, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
        ),
        (
            [1, 1, 768, 64],
            ttnn.TILE_LAYOUT,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            (96, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            (192, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
        ),
        (
            [1, 1, 768, 64],
            ttnn.TILE_LAYOUT,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            (192, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            (96, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
        ),
        (
            [1, 1, 1, 96],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            (1, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            (1, 48),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
        ),
        (
            [1, 1, 32, 512],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            (32, 128),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            (32, 256),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
        ),
        (
            [1, 1, 2, 256],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            (2, 128),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            (2, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
        ),
        (
            [1, 1, 16, 256],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            (16, 128),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            (16, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
        ),
        (
            [1, 1, 1, 96],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                }
            ),
            (1, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                }
            ),
            (1, 96),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
        ),
        (
            [1, 1, 4, 84480],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3)),
                }
            ),
            (4, 2656),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                }
            ),
            (4, 21120),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
        ),
    ],
)
def test_dram_reshard(
    device,
    input_shape,
    input_layout,
    input_shard_grid,
    input_shard_shape,
    input_shard_orientation,
    input_sharding_scheme,
    input_buffer_type,
    output_shard_grid,
    output_shard_shape,
    output_shard_orientation,
    output_sharding_scheme,
    output_buffer_type,
):
    input_shard_spec = ttnn.ShardSpec(input_shard_grid, input_shard_shape, input_shard_orientation)
    input_mem_config = ttnn.MemoryConfig(input_sharding_scheme, input_buffer_type, input_shard_spec)
    output_shard_spec = ttnn.ShardSpec(output_shard_grid, output_shard_shape, output_shard_orientation)
    output_mem_config = ttnn.MemoryConfig(output_sharding_scheme, output_buffer_type, output_shard_spec)

    input = torch.randn(input_shape).bfloat16()
    input_tensor = ttnn.Tensor(input, ttnn.bfloat16, device=device, layout=input_layout, mem_config=input_mem_config)

    output_tensor = ttnn.reshard(input_tensor, output_mem_config)

    output = ttnn.to_torch(output_tensor)
    passing, output_log = comp_pcc(input, output, 1.0)
    assert passing, output_log


@skip_for_blackhole("GH Issue #15234")
@pytest.mark.parametrize(
    "input_shape, input_layout, input_shard_grid, input_shard_shape, input_shard_orientation, input_sharding_scheme, input_buffer_type, output_shard_grid, output_shard_shape, output_shard_orientation, output_sharding_scheme, output_buffer_type",
    [
        (  # tests reshard_multi_core_same_width
            [1, 1, 768, 64],
            ttnn.TILE_LAYOUT,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            (96, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))}),
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
        ),
        (  # test reshard_multi_core_same_height
            [1, 1, 16, 256],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            (16, 128),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            (16, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
        ),
    ],
)
def test_dram_reshard_with_program_cache(
    use_program_cache,
    device,
    input_shape,
    input_layout,
    input_shard_grid,
    input_shard_shape,
    input_shard_orientation,
    input_sharding_scheme,
    input_buffer_type,
    output_shard_grid,
    output_shard_shape,
    output_shard_orientation,
    output_sharding_scheme,
    output_buffer_type,
):
    dtype = ttnn.bfloat8_b
    for _ in range(4):
        dummy_tensor = (
            ttnn.Tensor(torch.rand([1, 1, 128, 512]), dtype).to(ttnn.TILE_LAYOUT).to(device, ttnn.L1_MEMORY_CONFIG)
        )
        test_dram_reshard(
            device,
            input_shape,
            input_layout,
            input_shard_grid,
            input_shard_shape,
            input_shard_orientation,
            input_sharding_scheme,
            input_buffer_type,
            output_shard_grid,
            output_shard_shape,
            output_shard_orientation,
            output_sharding_scheme,
            output_buffer_type,
        )
        dummy_tensor = (
            ttnn.Tensor(torch.rand([2, 2, 128, 64]), dtype).to(ttnn.TILE_LAYOUT).to(device, ttnn.L1_MEMORY_CONFIG)
        )

    assert device.num_program_cache_entries() == 1
