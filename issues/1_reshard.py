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
        # Test case for 3-channel unaligned data (HEIGHT_SHARDED)
        (
            [1, 1, 64, 3],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (1, 0)]],
            (64, 3),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [[(0, 0), (2, 0)]],
            (32, 3),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            [1, 1, 64, 16],
            ttnn.ROW_MAJOR_LAYOUT,
            [[(0, 0), (1, 0)]],
            (64, 16),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [[(0, 0), (2, 0)]],
            (32, 16),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
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
