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
