# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)

from models.common.utility_functions import skip_for_blackhole


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


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
    # torch_tensor = torch.randn(input_shape).bfloat16()
    torch_tensor = random_torch_tensor(tt_dtype, input_shape)
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
@pytest.mark.parametrize("tt_dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.int32])
def test_reshard_with_program_cache(
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


@pytest.mark.parametrize(
    "input_shape, input_layout, input_shard_grid,  input_shard_shape, input_shard_orientation, input_sharding_scheme, output_shard_grid, output_shard_shape, output_shard_orientation, output_sharding_scheme",
    [
        # block sharded with different page sizes
        (
            [1, 1, 64, 320],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (7, 1)]],
            (32, 40),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (4, 1)]],
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        (
            [1, 1, 192, 160],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (4, 2)]],
            (64, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (2, 2)]],
            (64, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        # width sharded with different page sizes
        (
            [1, 1, 32, 160],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (4, 0)]],
            (32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [[(0, 0), (2, 0)]],
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            [1, 1, 5, 192],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (2, 0)]],
            (5, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [[(0, 0), (5, 0)]],
            (5, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("tt_dtype", [ttnn.bfloat16])
def test_reshard_diff_width(
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
            [1, 1, 16384, 320],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (7, 7)]],
            (2048, 40),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (4, 7)]],
            (2048, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        # Test case: input [1, 1, 4096, 320] sharded (512, 40) on 8x8 grid, reshard to (512, 64) on 5x8 grid
        (
            [1, 1, 4096, 320],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (7, 7)]],
            (512, 40),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (4, 7)]],
            (512, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        # Test case: input [1, 1, 4096, 640] sharded (512, 80) on 8x8 grid, reshard to (512, 96) on 7x8 grid
        (
            [1, 1, 4096, 640],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (7, 7)]],
            (512, 80),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (6, 7)]],
            (512, 96),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        # Test case: input [1, 1, 1024, 640] sharded (128, 80) on 8x8 grid, reshard to (128, 96) on 7x8 grid
        (
            [1, 1, 1024, 640],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (7, 7)]],
            (128, 80),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (6, 7)]],
            (128, 96),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        # Test case: input [1, 1, 4096, 1920] sharded (512, 240) on 8x8 grid, reshard to (512, 288) on 7x8 grid
        (
            [1, 1, 4096, 1920],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (7, 7)]],
            (512, 240),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (6, 7)]],
            (512, 288),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        # Test case: input [1, 1, 4096, 1280] sharded (512, 160) on 8x8 grid, reshard to (512, 192) on 7x8 grid
        (
            [1, 1, 4096, 1280],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (7, 7)]],
            (512, 160),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (6, 7)]],
            (512, 192),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        # Test case: input [1, 1, 4096, 960] sharded (512, 120) on 8x8 grid, reshard to (512, 192) on 5x8 grid
        (
            [1, 1, 4096, 960],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (7, 7)]],
            (512, 120),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (4, 7)]],
            (512, 192),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        # input and output with different page size and have padding
        (
            [1, 1, 4096, 1248],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (7, 7)]],
            (512, 160),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            [[(0, 0), (6, 7)]],
            (512, 192),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        # width sharded with different page sizes
        (
            [1, 1, 1024, 1248],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (7, 0)]],
            (1024, 160),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [[(0, 0), (6, 0)]],
            (1024, 192),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            [1, 1, 32, 160],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (2, 0)]],
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [[(0, 0), (4, 0)]],
            (32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("tt_dtype", [ttnn.bfloat16])
@pytest.mark.timeout(120)
def test_sd_reshard(
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
    passing, output = comp_equal(torch_tensor, torch_tensor_after_round_trip)
    assert passing, output


@pytest.mark.parametrize(
    "input_shape, core_grid_y, core_grid_x",
    [
        # SentenceBERT embeddings reshard configuration
        # Shape: [batch_size, 1, sequence_length, hidden_size]
        # Original error: NOC target address overflow during interleaved to block sharded reshard
        ([8, 1, 384, 768], 8, 6),
        # DP mode with 2 devices (16 batch total)
        ([16, 1, 384, 768], 8, 6),
    ],
)
@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_reshard_interleaved_to_block_sharded(
    device,
    input_shape,
    core_grid_y,
    core_grid_x,
    in_dtype,
    out_dtype,
):
    """
    Test reshard from interleaved L1 to BLOCK sharded memory config.

    This test is based on a NOC error found in SentenceBERT embeddings:
    - Input: interleaved L1 tensor with shape [batch, 1, seq_len, hidden_size]
    - Output: BLOCK sharded on core_grid with dtype conversion
    - The error was: "unicast write 3203369245 bytes... (NOC target address overflow)"
      with garbage values indicating memory corruption in reshard kernel parameters
    """
    grid_size = device.compute_with_storage_grid_size()
    if core_grid_x > grid_size.x or core_grid_y > grid_size.y:
        pytest.skip(f"Core grid ({core_grid_x}, {core_grid_y}) exceeds device grid size ({grid_size.x}, {grid_size.y})")

    # Create input tensor in interleaved L1 memory
    torch_tensor = torch.randn(input_shape).bfloat16()
    tt_tensor = ttnn.from_torch(torch_tensor, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_tensor = ttnn.to_memory_config(tt_tensor, ttnn.L1_MEMORY_CONFIG)

    # Create BLOCK sharded memory config (same as SentenceBERT embeddings)
    output_mem_config = ttnn.create_sharded_memory_config(
        tt_tensor.shape,
        core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Perform reshard with dtype conversion
    tt_tensor_resharded = ttnn.to_memory_config(tt_tensor, output_mem_config, dtype=out_dtype)

    # Verify result
    tt_tensor_interleaved = ttnn.to_memory_config(tt_tensor_resharded, ttnn.L1_MEMORY_CONFIG)
    torch_result = ttnn.to_torch(tt_tensor_interleaved)

    if out_dtype == ttnn.bfloat8_b:
        # bfloat8_b has lower precision
        passing, output = comp_pcc(torch_tensor, torch_result, 0.99)
    else:
        passing, output = comp_equal(torch_tensor, torch_result)

    assert passing, output


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [
        # SentenceBERT embeddings shape after unsqueeze: [batch, 1, seq_len, hidden_size]
        (8, 384, 768),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_reshard_sentencebert_embeddings(
    device,
    batch_size,
    seq_len,
    hidden_size,
    layout,
):
    """
    Test reshard operation matching SentenceBERT embeddings pattern.

    This reproduces the exact operation from ttnn_sentencebert_embeddings.py:
    - Input: interleaved L1 tensor from embedding addition
    - Output: BLOCK sharded on 8x6 grid with bfloat8_b dtype
    - The NOC error kernel was: writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp
    - Error: "unicast write 3203369245 bytes to L1[addr=0x6448f51d]" (NOC target address overflow)
    """
    grid_size = device.compute_with_storage_grid_size()
    if grid_size.x < 6 or grid_size.y < 8:
        pytest.skip(f"Device grid size ({grid_size.x}, {grid_size.y}) too small for 8x6 grid")

    # bfloat8_b only works with TILE_LAYOUT
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        out_dtype = ttnn.bfloat16
    else:
        out_dtype = ttnn.bfloat8_b

    # Shape after unsqueeze in SentenceBERT: [batch, 1, seq_len, hidden_size]
    input_shape = [batch_size, 1, seq_len, hidden_size]

    # Simulate the embeddings computation (word + token_type + position embeddings)
    torch_tensor = torch.randn(input_shape).bfloat16()

    # Create tensor in interleaved L1 (result of embedding additions)
    tt_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=layout, device=device)
    tt_tensor = ttnn.to_memory_config(tt_tensor, ttnn.L1_MEMORY_CONFIG)

    # Create BLOCK sharded memory config (exact match to SentenceBERT embeddings)
    output_mem_config = ttnn.create_sharded_memory_config(
        tt_tensor.shape,
        core_grid=ttnn.CoreGrid(y=8, x=6),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Perform reshard with dtype conversion (this is where NOC error occurred)
    tt_tensor_resharded = ttnn.to_memory_config(tt_tensor, output_mem_config, dtype=out_dtype)

    # Verify result
    tt_tensor_interleaved = ttnn.to_memory_config(tt_tensor_resharded, ttnn.L1_MEMORY_CONFIG)
    torch_result = ttnn.to_torch(tt_tensor_interleaved)

    if out_dtype == ttnn.bfloat8_b:
        passing, output = comp_pcc(torch_tensor, torch_result, 0.99)
    else:
        passing, output = comp_equal(torch_tensor, torch_result)
    assert passing, output


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576}],
    indirect=True,
)
def test_reshard_sentencebert_embeddings_full(device):
    """
    Minimal repro for NOC error from SentenceBERT.

    Bug: sharded_to_interleaved corrupts memory, causing subsequent
    interleaved-to-sharded reshard to use garbage values for NOC transfer.

    Sequence that triggers bug:
    1. HEIGHT_SHARDED L1 tensor
    2. sharded_to_interleaved (corrupts memory)
    3. to_memory_config to BLOCK sharded <- NOC error here

    Error: "unicast write 3203347578 bytes... (NOC target address overflow)"
    Kernel: writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp
    """
    grid_size = device.compute_with_storage_grid_size()
    if grid_size.x < 6 or grid_size.y < 8:
        pytest.skip(f"Device grid size ({grid_size.x}, {grid_size.y}) too small for 8x6 grid")

    batch_size = 8
    seq_len = 384
    hidden_size = 768

    # Create L1 HEIGHT sharded memory config (8x8 grid, shard shape (1, seq_len))
    l1_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    l1_shard_spec = ttnn.ShardSpec(l1_grid, (1, seq_len), ttnn.ShardOrientation.ROW_MAJOR)
    l1_sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec)

    # Step 1: Create HEIGHT_SHARDED L1 tensor
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.int32)
    tt_input = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_input = ttnn.to_memory_config(tt_input, l1_sharded_config)

    # Step 2: sharded_to_interleaved - THIS CORRUPTS MEMORY
    tt_input_interleaved = ttnn.sharded_to_interleaved(tt_input, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_input_interleaved)  # Don't need this, just corrupts state

    # Step 3: Create new tensor and reshard to BLOCK sharded
    embeddings = torch.randn(batch_size, seq_len, hidden_size).bfloat16()
    tt_embeddings = ttnn.from_torch(embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, ttnn.L1_MEMORY_CONFIG)

    # Step 4: Reshard to BLOCK sharded - NOC ERROR OCCURS HERE
    output_mem_config = ttnn.create_sharded_memory_config(
        tt_embeddings.shape,
        core_grid=ttnn.CoreGrid(y=8, x=6),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, output_mem_config, dtype=ttnn.bfloat8_b)

    ttnn.synchronize_device(device)
    assert tt_embeddings.shape == ttnn.Shape([batch_size, seq_len, hidden_size])


# ============================================================================
# Variants to narrow down the corruption source
# ============================================================================


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576}],
    indirect=True,
)
def test_reshard_variant1_skip_sharded_to_interleaved(device):
    """
    Variant 1: Skip sharded_to_interleaved, keep deallocate.
    Expected: PASS (if sharded_to_interleaved is the culprit)
    """
    grid_size = device.compute_with_storage_grid_size()
    if grid_size.x < 6 or grid_size.y < 8:
        pytest.skip(f"Device grid size ({grid_size.x}, {grid_size.y}) too small for 8x6 grid")

    batch_size = 8
    seq_len = 384
    hidden_size = 768

    l1_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    l1_shard_spec = ttnn.ShardSpec(l1_grid, (1, seq_len), ttnn.ShardOrientation.ROW_MAJOR)
    l1_sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec)

    # Create HEIGHT_SHARDED tensor
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.int32)
    tt_input = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_input = ttnn.to_memory_config(tt_input, l1_sharded_config)

    # SKIP sharded_to_interleaved - just deallocate
    ttnn.deallocate(tt_input)

    # Create new tensor and reshard to BLOCK sharded
    embeddings = torch.randn(batch_size, seq_len, hidden_size).bfloat16()
    tt_embeddings = ttnn.from_torch(embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, ttnn.L1_MEMORY_CONFIG)

    output_mem_config = ttnn.create_sharded_memory_config(
        tt_embeddings.shape,
        core_grid=ttnn.CoreGrid(y=8, x=6),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, output_mem_config, dtype=ttnn.bfloat8_b)

    ttnn.synchronize_device(device)
    assert tt_embeddings.shape == ttnn.Shape([batch_size, seq_len, hidden_size])


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576}],
    indirect=True,
)
def test_reshard_variant2_skip_deallocates(device):
    """
    Variant 2: Keep sharded_to_interleaved, skip deallocates.
    Expected: FAIL if sharded_to_interleaved alone corrupts, PASS if deallocation is needed
    """
    grid_size = device.compute_with_storage_grid_size()
    if grid_size.x < 6 or grid_size.y < 8:
        pytest.skip(f"Device grid size ({grid_size.x}, {grid_size.y}) too small for 8x6 grid")

    batch_size = 8
    seq_len = 384
    hidden_size = 768

    l1_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    l1_shard_spec = ttnn.ShardSpec(l1_grid, (1, seq_len), ttnn.ShardOrientation.ROW_MAJOR)
    l1_sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec)

    # Create HEIGHT_SHARDED tensor
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.int32)
    tt_input = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_input = ttnn.to_memory_config(tt_input, l1_sharded_config)

    # Do sharded_to_interleaved but SKIP deallocates
    tt_input_interleaved = ttnn.sharded_to_interleaved(tt_input, ttnn.L1_MEMORY_CONFIG)
    # NO deallocate - tensors stay allocated

    # Create new tensor and reshard to BLOCK sharded
    embeddings = torch.randn(batch_size, seq_len, hidden_size).bfloat16()
    tt_embeddings = ttnn.from_torch(embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, ttnn.L1_MEMORY_CONFIG)

    output_mem_config = ttnn.create_sharded_memory_config(
        tt_embeddings.shape,
        core_grid=ttnn.CoreGrid(y=8, x=6),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, output_mem_config, dtype=ttnn.bfloat8_b)

    ttnn.synchronize_device(device)
    assert tt_embeddings.shape == ttnn.Shape([batch_size, seq_len, hidden_size])

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_input_interleaved)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576}],
    indirect=True,
)
def test_reshard_variant3_skip_height_sharded(device):
    """
    Variant 3: Start with interleaved tensor instead of HEIGHT_SHARDED.
    Expected: PASS (if HEIGHT_SHARDED is required to trigger bug)
    """
    grid_size = device.compute_with_storage_grid_size()
    if grid_size.x < 6 or grid_size.y < 8:
        pytest.skip(f"Device grid size ({grid_size.x}, {grid_size.y}) too small for 8x6 grid")

    batch_size = 8
    seq_len = 384
    hidden_size = 768

    # Create INTERLEAVED tensor (not HEIGHT_SHARDED)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.int32)
    tt_input = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # Keep as interleaved, don't convert to HEIGHT_SHARDED

    ttnn.deallocate(tt_input)

    # Create new tensor and reshard to BLOCK sharded
    embeddings = torch.randn(batch_size, seq_len, hidden_size).bfloat16()
    tt_embeddings = ttnn.from_torch(embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, ttnn.L1_MEMORY_CONFIG)

    output_mem_config = ttnn.create_sharded_memory_config(
        tt_embeddings.shape,
        core_grid=ttnn.CoreGrid(y=8, x=6),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, output_mem_config, dtype=ttnn.bfloat8_b)

    ttnn.synchronize_device(device)
    assert tt_embeddings.shape == ttnn.Shape([batch_size, seq_len, hidden_size])


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576}],
    indirect=True,
)
def test_reshard_variant4_only_deallocate_interleaved(device):
    """
    Variant 4: Only deallocate the interleaved tensor (not the sharded one).
    Tests if deallocating the sharded tensor matters.
    """
    grid_size = device.compute_with_storage_grid_size()
    if grid_size.x < 6 or grid_size.y < 8:
        pytest.skip(f"Device grid size ({grid_size.x}, {grid_size.y}) too small for 8x6 grid")

    batch_size = 8
    seq_len = 384
    hidden_size = 768

    l1_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    l1_shard_spec = ttnn.ShardSpec(l1_grid, (1, seq_len), ttnn.ShardOrientation.ROW_MAJOR)
    l1_sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec)

    # Create HEIGHT_SHARDED tensor
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.int32)
    tt_input = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_input = ttnn.to_memory_config(tt_input, l1_sharded_config)

    # Do sharded_to_interleaved
    tt_input_interleaved = ttnn.sharded_to_interleaved(tt_input, ttnn.L1_MEMORY_CONFIG)
    # Only deallocate interleaved, keep sharded
    ttnn.deallocate(tt_input_interleaved)

    # Create new tensor and reshard to BLOCK sharded
    embeddings = torch.randn(batch_size, seq_len, hidden_size).bfloat16()
    tt_embeddings = ttnn.from_torch(embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, ttnn.L1_MEMORY_CONFIG)

    output_mem_config = ttnn.create_sharded_memory_config(
        tt_embeddings.shape,
        core_grid=ttnn.CoreGrid(y=8, x=6),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, output_mem_config, dtype=ttnn.bfloat8_b)

    ttnn.synchronize_device(device)
    assert tt_embeddings.shape == ttnn.Shape([batch_size, seq_len, hidden_size])

    # Cleanup
    ttnn.deallocate(tt_input)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576}],
    indirect=True,
)
def test_reshard_variant5_only_deallocate_sharded(device):
    """
    Variant 5: Only deallocate the sharded tensor (not the interleaved one).
    Tests if deallocating the interleaved tensor matters.
    """
    grid_size = device.compute_with_storage_grid_size()
    if grid_size.x < 6 or grid_size.y < 8:
        pytest.skip(f"Device grid size ({grid_size.x}, {grid_size.y}) too small for 8x6 grid")

    batch_size = 8
    seq_len = 384
    hidden_size = 768

    l1_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    l1_shard_spec = ttnn.ShardSpec(l1_grid, (1, seq_len), ttnn.ShardOrientation.ROW_MAJOR)
    l1_sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec)

    # Create HEIGHT_SHARDED tensor
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.int32)
    tt_input = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_input = ttnn.to_memory_config(tt_input, l1_sharded_config)

    # Do sharded_to_interleaved
    tt_input_interleaved = ttnn.sharded_to_interleaved(tt_input, ttnn.L1_MEMORY_CONFIG)
    # Only deallocate sharded, keep interleaved
    ttnn.deallocate(tt_input)

    # Create new tensor and reshard to BLOCK sharded
    embeddings = torch.randn(batch_size, seq_len, hidden_size).bfloat16()
    tt_embeddings = ttnn.from_torch(embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, ttnn.L1_MEMORY_CONFIG)

    output_mem_config = ttnn.create_sharded_memory_config(
        tt_embeddings.shape,
        core_grid=ttnn.CoreGrid(y=8, x=6),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, output_mem_config, dtype=ttnn.bfloat8_b)

    ttnn.synchronize_device(device)
    assert tt_embeddings.shape == ttnn.Shape([batch_size, seq_len, hidden_size])

    # Cleanup
    ttnn.deallocate(tt_input_interleaved)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576}],
    indirect=True,
)
def test_reshard_variant6_reverse_dealloc_order(device):
    """
    Variant 6: Reverse deallocation order (interleaved first, then sharded).
    Tests if deallocation order matters.
    """
    grid_size = device.compute_with_storage_grid_size()
    if grid_size.x < 6 or grid_size.y < 8:
        pytest.skip(f"Device grid size ({grid_size.x}, {grid_size.y}) too small for 8x6 grid")

    batch_size = 8
    seq_len = 384
    hidden_size = 768

    l1_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    l1_shard_spec = ttnn.ShardSpec(l1_grid, (1, seq_len), ttnn.ShardOrientation.ROW_MAJOR)
    l1_sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec)

    # Create HEIGHT_SHARDED tensor
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.int32)
    tt_input = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_input = ttnn.to_memory_config(tt_input, l1_sharded_config)

    # Do sharded_to_interleaved
    tt_input_interleaved = ttnn.sharded_to_interleaved(tt_input, ttnn.L1_MEMORY_CONFIG)
    # Reverse order: deallocate interleaved first, then sharded
    ttnn.deallocate(tt_input_interleaved)
    ttnn.deallocate(tt_input)

    # Create new tensor and reshard to BLOCK sharded
    embeddings = torch.randn(batch_size, seq_len, hidden_size).bfloat16()
    tt_embeddings = ttnn.from_torch(embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, ttnn.L1_MEMORY_CONFIG)

    output_mem_config = ttnn.create_sharded_memory_config(
        tt_embeddings.shape,
        core_grid=ttnn.CoreGrid(y=8, x=6),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_embeddings = ttnn.to_memory_config(tt_embeddings, output_mem_config, dtype=ttnn.bfloat8_b)

    ttnn.synchronize_device(device)
    assert tt_embeddings.shape == ttnn.Shape([batch_size, seq_len, hidden_size])
