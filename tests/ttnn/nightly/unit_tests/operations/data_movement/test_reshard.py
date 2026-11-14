# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5, 6, 7, 9, 10, 17, 33, 65])
@pytest.mark.parametrize("input_buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
def test_reshard_unaligned_height_sharded_scratch_race(device, channels, input_buffer_type):
    """Regression for #51217: 32x4 -> 64x2 (L1 output) lands the local split on a remote-shard boundary; fails pre-fix, passes with the disjoint-halves scratch split."""
    grid_size = device.compute_with_storage_grid_size()
    if grid_size.x < 4:
        pytest.skip("Test requires at least 4 cores in the x dimension")

    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})

    input_shard_spec = ttnn.ShardSpec(input_shard_grid, (32, channels), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_buffer_type, input_shard_spec)
    output_shard_spec = ttnn.ShardSpec(output_shard_grid, (64, channels), ttnn.ShardOrientation.ROW_MAJOR)
    # L1 output is what selects the reader path (use_scratch = unaligned && local_is_output).
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_tensor = torch.rand([1, 1, 128, channels]).bfloat16().float()
    input_tensor = ttnn.Tensor(
        torch_tensor, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
    )

    output_tensor = ttnn.reshard(input_tensor, output_mem_config)
    torch_tensor_after_round_trip = ttnn.to_torch(output_tensor)

    assert torch_tensor.shape == torch_tensor_after_round_trip.shape
    passing, output = comp_equal(torch_tensor, torch_tensor_after_round_trip)
    assert passing, output


def test_reshard_width_sharded_multirange_corerangeset_grid(device):
    """Regression for the multi-CoreRange CoreRangeSet grid truncation at reshard_device_operation.cpp:73-74.

    A width-sharded tensor whose shard grid is a CoreRangeSet containing more than one CoreRange
    (for example, a ring of per-core 1x1 ranges, or 22 shards that wrap onto a second row of an
    8-wide compute grid) was previously resharded using only the first range's grid_size as the
    full shard count. Cores beyond ranges()[0] received wrong per-shard offsets and the final
    output shard was silently corrupted. This test constructs that layout with fp32 arange data
    so every element is uniquely identifiable, and asserts bitwise identity against the input.
    L1 only — DRAM banks are 1D (every shard core must have y == 0), so a wrapped multi-range
    shard grid is not representable on DRAM; the production reach lives in L1 anyway.
    """
    grid_size = device.compute_with_storage_grid_size()
    if grid_size.x < 8 or grid_size.y < 6:
        pytest.skip("Test requires a compute grid at least 8x6 to wrap 22 cores onto a second row")

    rows = 64
    width = 1032
    w_in = 48  # 1032 / 48 = 21.5 -> 22 input cores (wraps 8x2 + 6x1)
    w_out = 24  # 1032 / 24 = 43 output cores (wraps 8x5 + 3x1)
    assert (width + w_in - 1) // w_in == 22
    assert (width + w_out - 1) // w_out == 43

    def make_wrapped_grid(n_cores, grid_x):
        full_rows = n_cores // grid_x
        remainder = n_cores - full_rows * grid_x
        ranges = []
        if full_rows > 0:
            ranges.append(
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1))
            )
        if remainder > 0:
            ranges.append(
                ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(remainder - 1, full_rows))
            )
        return ttnn.CoreRangeSet(ranges)

    input_shard_grid = make_wrapped_grid(22, 8)
    output_shard_grid = make_wrapped_grid(43, 8)
    assert len(input_shard_grid.ranges()) == 2, "input grid must be multi-range to exercise the bug"
    assert len(output_shard_grid.ranges()) == 2, "output grid must be multi-range to exercise the bug"

    input_shard_spec = ttnn.ShardSpec(input_shard_grid, (rows, w_in), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )
    output_shard_spec = ttnn.ShardSpec(output_shard_grid, (rows, w_out), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )

    torch_tensor = torch.arange(rows * width, dtype=torch.float32).reshape(1, 1, rows, width)

    input_tensor = ttnn.from_torch(
        torch_tensor, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor = ttnn.interleaved_to_sharded(input_tensor, input_mem_config)
    output_tensor = ttnn.reshard(input_tensor, output_mem_config)
    output_torch = ttnn.to_torch(output_tensor)

    passing, output = comp_equal(torch_tensor, output_torch)
    assert passing, output
