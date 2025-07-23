# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)


def run_nd_reshard_test(
    device,
    input_shape,
    layout,
    input_shard_shape,
    input_shard_orientation,
    output_shard_shape,
    output_shard_orientation,
    input_grid=None,
    output_grid=None,
    dtype=ttnn.bfloat16,
    input_buffer_type=ttnn.BufferType.L1,
    output_buffer_type=ttnn.BufferType.L1,
):
    def create_full_grid(device, buffer_type):
        if buffer_type == ttnn.BufferType.DRAM:
            grid_size = device.dram_grid_size()
        else:
            grid_size = device.compute_with_storage_grid_size()
        grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
        )
        return grid

    # Default grid covers entire device if not specified
    if input_grid is None:
        input_grid = create_full_grid(device, input_buffer_type)

    # Default output grid is the same as input grid if not specified
    if output_grid is None:
        if input_buffer_type == output_buffer_type:
            output_grid = input_grid
        else:
            output_grid = create_full_grid(device, output_buffer_type)

    input_memory_config = ttnn.MemoryConfig(
        input_buffer_type, ttnn.NdShardSpec(ttnn.Shape(input_shard_shape), input_grid, input_shard_orientation)
    )
    output_memory_config = ttnn.MemoryConfig(
        output_buffer_type, ttnn.NdShardSpec(ttnn.Shape(output_shard_shape), output_grid, output_shard_orientation)
    )

    # Create appropriate torch tensor based on dtype
    if dtype in [ttnn.bfloat16, ttnn.bfloat8_b]:
        torch_tensor = torch.randn(input_shape).bfloat16()
    elif dtype in [ttnn.float32]:
        torch_tensor = torch.randn(input_shape).float()
    elif dtype in [ttnn.uint8]:
        torch_tensor = torch.randint(0, 255, input_shape, dtype=torch.uint8)
    elif dtype in [ttnn.uint16]:
        torch_tensor = torch.randint(-32768, 32767, input_shape, dtype=torch.int16)
    elif dtype in [ttnn.uint32]:
        torch_tensor = torch.randint(-32768, 32767, input_shape, dtype=torch.int32)
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

    if layout == ttnn.ROW_MAJOR_LAYOUT and dtype == ttnn.bfloat8_b:
        pytest.skip("bfloat8_b is not supported for ROW_MAJOR_LAYOUT")

    tt_tensor_sharded = ttnn.from_torch(
        torch_tensor, dtype=dtype, device=device, layout=layout, memory_config=input_memory_config
    )
    tt_tensor_output_sharded = ttnn.from_torch(
        torch_tensor, dtype=dtype, device=device, layout=layout, memory_config=output_memory_config
    )

    tt_tensor_resharded = ttnn.reshard(tt_tensor_sharded, output_memory_config)

    round_trip_tensor = ttnn.to_torch(tt_tensor_resharded)
    expected_resharded_tensor = ttnn.to_torch(tt_tensor_output_sharded)
    return torch_tensor, round_trip_tensor, expected_resharded_tensor


# TODO: Make implementation for cases when tile size changes
@pytest.mark.parametrize(
    "input_shape, layout, input_shard_shape, input_shard_orientation, output_shard_shape, output_shard_orientation, input_grid",
    [
        # Single core reshard
        (
            [1, 1, 64, 64],
            ttnn.TILE_LAYOUT,
            (64, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        ),
        (
            [1, 3, 64, 64],
            ttnn.TILE_LAYOUT,
            (3, 32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            (3, 64, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        ),
        # ROW_MAJOR_LAYOUT test cases
        # Case below won't work since page size changes
        # ([1, 1, 64, 32], ttnn.ROW_MAJOR_LAYOUT, (32, 32), ttnn.ShardOrientation.ROW_MAJOR, (32, 64), ttnn.ShardOrientation.COL_MAJOR, None),
        (
            [1, 1, 64, 32],
            ttnn.ROW_MAJOR_LAYOUT,
            (32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (64, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            None,
        ),
        (
            [1, 8, 128, 64],
            ttnn.ROW_MAJOR_LAYOUT,
            (4, 64, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            (2, 128, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            None,
        ),
        # TILE_LAYOUT with COL_MAJOR orientation
        (
            [1, 1, 32, 64],
            ttnn.TILE_LAYOUT,
            (32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            (32, 64),
            ttnn.ShardOrientation.COL_MAJOR,
            None,
        ),
        (
            [1, 1, 32, 128],
            ttnn.TILE_LAYOUT,
            (32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            None,
        ),
        # 2D tensor shapes
        (
            [128, 256],
            ttnn.TILE_LAYOUT,
            (64, 128),
            ttnn.ShardOrientation.ROW_MAJOR,
            (32, 256),
            ttnn.ShardOrientation.ROW_MAJOR,
            None,
        ),
        # 3D tensor shapes
        # Padded shards
        (
            [1, 96, 128],
            ttnn.TILE_LAYOUT,
            (1, 32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            (1, 64, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            None,
        ),
        (
            [1, 96, 128],
            ttnn.TILE_LAYOUT,
            (1, 32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            (1, 64, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            None,
        ),
        (
            [1, 128, 96],
            ttnn.TILE_LAYOUT,
            (1, 32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            (1, 96, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            None,
        ),
        (
            [1, 128, 96],
            ttnn.TILE_LAYOUT,
            (1, 32, 64),
            ttnn.ShardOrientation.COL_MAJOR,
            (1, 96, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            None,
        ),
        # Mixed dimensions with 4D inputs
        (
            [2, 3, 32, 32],
            ttnn.TILE_LAYOUT,
            (2, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (3, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            None,
        ),
        (
            [4, 2, 32, 32],
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (2, 32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            None,
        ),
        # Grid size > Num required banks
        (
            [1, 1, 64, 64],
            ttnn.TILE_LAYOUT,
            (32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))]),
        ),
        # Output is the same as input
        (
            [1, 1, 128, 128],
            ttnn.TILE_LAYOUT,
            (32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))]),
        ),
        # 2D Tensor with 2D CoreRangeSet (2x2 grid)
        (
            [128, 128],
            ttnn.TILE_LAYOUT,
            (64, 64),
            ttnn.ShardOrientation.COL_MAJOR,
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))]),
        ),
        (
            [96, 160],
            ttnn.ROW_MAJOR_LAYOUT,
            (32, 64),
            ttnn.ShardOrientation.COL_MAJOR,
            (64, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))]),
        ),
        # 3D Tensor with 2D CoreRangeSet
        (
            [32, 64, 64],
            ttnn.TILE_LAYOUT,
            (32, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (32, 32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))]),
        ),
        (
            [32, 64, 96],
            ttnn.ROW_MAJOR_LAYOUT,
            (32, 32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            (32, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(3, 3))]),
        ),
        # 4D Tensor with 2D CoreRangeSet
        (
            [2, 16, 64, 64],
            ttnn.TILE_LAYOUT,
            (1, 16, 64, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            (2, 1, 64, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))]),
        ),
        (
            [4, 3, 128, 128],
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 3, 32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            (4, 3, 1, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 2))]),
        ),
        # Width sharded
        (
            [64, 256],
            ttnn.TILE_LAYOUT,
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            (64, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))]),
        ),
        (
            [32, 32, 64],
            ttnn.TILE_LAYOUT,
            (32, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (32, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))]),
        ),
        (
            [3, 32, 64, 64],
            ttnn.TILE_LAYOUT,
            (3, 32, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (3, 32, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 3))]),
        ),
        # High sharded
        (
            [128, 64],
            ttnn.TILE_LAYOUT,
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]),
        ),
        (
            [32, 96, 32],
            ttnn.TILE_LAYOUT,
            (32, 32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            (32, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]),
        ),
        (
            [2, 3, 128, 64],
            ttnn.TILE_LAYOUT,
            (2, 3, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (2, 3, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(3, 2))]),
        ),
        # Multiple disjoint core ranges in the CoreRangeSet
        (
            [256, 128],
            ttnn.TILE_LAYOUT,
            (64, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            (32, 64),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(1, 2)),
                ]
            ),
        ),
        (
            [1, 64, 64, 32],
            ttnn.TILE_LAYOUT,
            (1, 32, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (1, 32, 32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 1)),
                ]
            ),
        ),
        # Shard dimensions larger than input dimensions (with padding)
        (
            [32, 96],
            ttnn.TILE_LAYOUT,
            (64, 128),
            ttnn.ShardOrientation.ROW_MAJOR,
            (32, 128),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        ),
        (
            [2, 32, 32],  # 3D tensor with some dimensions smaller than shard
            ttnn.ROW_MAJOR_LAYOUT,
            (4, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (2, 64, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))]),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint32])
def test_nd_reshard(
    device,
    input_shape,
    layout,
    input_shard_shape,
    input_shard_orientation,
    output_shard_shape,
    output_shard_orientation,
    input_grid,
    dtype,
):
    torch.manual_seed(1)
    torch_tensor, torch_tensor_after_round_trip, expected_resharded_tensor = run_nd_reshard_test(
        device,
        input_shape,
        layout,
        input_shard_shape,
        input_shard_orientation,
        output_shard_shape,
        output_shard_orientation,
        input_grid,
        None,
        dtype,
    )
    if dtype != ttnn.bfloat8_b:
        passing, output = comp_equal(torch_tensor, torch_tensor_after_round_trip)
        passing2, output2 = comp_equal(torch_tensor_after_round_trip, expected_resharded_tensor)
        passing3, output3 = comp_equal(torch_tensor, expected_resharded_tensor)
    else:
        passing, output = comp_pcc(torch_tensor, torch_tensor_after_round_trip)
        passing2, output2 = comp_pcc(torch_tensor_after_round_trip, expected_resharded_tensor)
        passing3, output3 = comp_pcc(torch_tensor, expected_resharded_tensor)
    assert passing3, output3
    assert passing2, output2
    assert passing, output


# Bigger cases for benchmarking
@pytest.mark.parametrize(
    "input_shape, layout, input_shard_shape, input_shard_orientation, output_shard_shape, output_shard_orientation",
    [
        (
            [1, 1, 2048, 4096],
            ttnn.TILE_LAYOUT,
            (64, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
        (
            [1, 3, 1024, 1024],
            ttnn.TILE_LAYOUT,
            (1, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (3, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
        (
            [3, 4, 1024, 1024],
            ttnn.TILE_LAYOUT,
            (1, 1, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (3, 4, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
        (
            [2, 3, 4, 512, 512],
            ttnn.TILE_LAYOUT,
            (2, 1, 1, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (1, 3, 4, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    ],
)
def test_benchmark_nd_reshard(
    device,
    input_shape,
    layout,
    input_shard_shape,
    input_shard_orientation,
    output_shard_shape,
    output_shard_orientation,
):
    for _ in range(5):
        run_nd_reshard_test(
            device,
            input_shape,
            layout,
            input_shard_shape,
            input_shard_orientation,
            output_shard_shape,
            output_shard_orientation,
        )


@pytest.mark.parametrize(
    "input_shape, layout, input_shard_shape, input_shard_orientation, output_shard_shape, output_shard_orientation, input_grid, output_grid",
    [
        (
            [128, 64],
            ttnn.TILE_LAYOUT,
            (32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 1)),
                ]
            ),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 1)),
                ]
            ),
        ),
        (
            [4, 64, 96],
            ttnn.TILE_LAYOUT,
            (1, 32, 64),
            ttnn.ShardOrientation.COL_MAJOR,
            (2, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))]),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 4), ttnn.CoreCoord(5, 5))]),
        ),
        (
            [3, 4, 64, 32],
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 2, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            (3, 2, 32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 2)),
                ]
            ),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 1)),
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint32])
def test_nd_reshard_different_input_output_grid(
    device,
    input_shape,
    layout,
    input_shard_shape,
    input_shard_orientation,
    output_shard_shape,
    output_shard_orientation,
    input_grid,
    output_grid,
    dtype,
):
    torch.manual_seed(1)
    torch_tensor, torch_tensor_after_round_trip, expected_resharded_tensor = run_nd_reshard_test(
        device,
        input_shape,
        layout,
        input_shard_shape,
        input_shard_orientation,
        output_shard_shape,
        output_shard_orientation,
        input_grid,
        None,
        dtype,
    )
    if dtype != ttnn.bfloat8_b:
        passing, output = comp_equal(torch_tensor, torch_tensor_after_round_trip)
        passing2, output2 = comp_equal(torch_tensor_after_round_trip, expected_resharded_tensor)
        passing3, output3 = comp_equal(torch_tensor, expected_resharded_tensor)
    else:
        passing, output = comp_pcc(torch_tensor, torch_tensor_after_round_trip)
        passing2, output2 = comp_pcc(torch_tensor_after_round_trip, expected_resharded_tensor)
        passing3, output3 = comp_pcc(torch_tensor, expected_resharded_tensor)
    assert passing3, output3
    assert passing2, output2
    assert passing, output


# TODO: Fix DRAM cases
@pytest.mark.skip(reason="DRAM resharding is not supported yet")
@pytest.mark.parametrize(
    "input_shape, layout, input_shard_shape, input_shard_orientation, output_shard_shape, output_shard_orientation, input_grid, output_grid, dtype, input_buffer_type, output_buffer_type",
    [
        # DRAM -> L1
        (
            [128, 128],
            ttnn.TILE_LAYOUT,
            (64, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))]),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(5, 5))]),
            ttnn.bfloat16,
            ttnn.BufferType.DRAM,
            ttnn.BufferType.L1,
        ),
        (
            [3, 32, 64],
            ttnn.TILE_LAYOUT,
            (1, 32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            (3, 32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))]),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 0))]),
            ttnn.uint32,
            ttnn.BufferType.DRAM,
            ttnn.BufferType.L1,
        ),
        # L1 -> DRAM transfers
        (
            [64, 128],
            ttnn.TILE_LAYOUT,
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            (64, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 2))]),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))]),
            ttnn.bfloat16,
            ttnn.BufferType.L1,
            ttnn.BufferType.DRAM,
        ),
        (
            [2, 3, 64, 32],
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 3, 32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            (2, 1, 64, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))]),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 1))]),
            ttnn.bfloat16,
            ttnn.BufferType.L1,
            ttnn.BufferType.DRAM,
        ),
        # DRAM to DRAM transfers with different grids
        (
            [4, 256, 256],
            ttnn.TILE_LAYOUT,
            (1, 128, 128),
            ttnn.ShardOrientation.ROW_MAJOR,
            (4, 64, 256),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))]),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(5, 5))]),
            ttnn.bfloat16,
            ttnn.BufferType.DRAM,
            ttnn.BufferType.DRAM,
        ),
        (
            [18, 128, 64],
            ttnn.TILE_LAYOUT,
            (1, 64, 96),
            ttnn.ShardOrientation.ROW_MAJOR,
            (1, 32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(4, 6), ttnn.CoreCoord(6, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 3), ttnn.CoreCoord(3, 3)),
                ]
            ),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 5))]),
            ttnn.uint32,
            ttnn.BufferType.DRAM,
            ttnn.BufferType.L1,
        ),
        (
            [2, 3, 2, 3, 4],
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 1, 2, 2, 4),
            ttnn.ShardOrientation.COL_MAJOR,
            (3, 3, 1, 1, 4),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 4))]),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))]),
            ttnn.bfloat16,
            ttnn.BufferType.L1,
            ttnn.BufferType.DRAM,
        ),
    ],
)
def test_DRAM_nd_reshard(
    device,
    input_shape,
    layout,
    input_shard_shape,
    input_shard_orientation,
    output_shard_shape,
    output_shard_orientation,
    dtype,
    input_grid,
    output_grid,
    input_buffer_type,
    output_buffer_type,
):
    torch.manual_seed(1)
    torch_tensor, torch_tensor_after_round_trip, expected_resharded_tensor = run_nd_reshard_test(
        device,
        input_shape,
        layout,
        input_shard_shape,
        input_shard_orientation,
        output_shard_shape,
        output_shard_orientation,
        input_grid,
        output_grid,
        dtype,
        input_buffer_type,
        output_buffer_type,
    )
    if dtype != ttnn.bfloat8_b:
        passing, output = comp_equal(torch_tensor, torch_tensor_after_round_trip)
        passing2, output2 = comp_equal(torch_tensor_after_round_trip, expected_resharded_tensor)
        passing3, output3 = comp_equal(torch_tensor, expected_resharded_tensor)
    else:
        passing, output = comp_pcc(torch_tensor, torch_tensor_after_round_trip)
        passing2, output2 = comp_pcc(torch_tensor_after_round_trip, expected_resharded_tensor)
        passing3, output3 = comp_pcc(torch_tensor, expected_resharded_tensor)
    assert passing3, output3
    assert passing2, output2
    assert passing, output
