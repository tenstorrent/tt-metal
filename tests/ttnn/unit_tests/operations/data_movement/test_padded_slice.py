# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal


# Last-dim slice with output padding.
@pytest.mark.parametrize(
    "begins, ends, shard_shape, core_grid",
    (
        (
            [0, 0, 0, 16],
            [1, 1, 64, 64],
            (8, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        ),
        (
            [0, 0, 0, 0],
            [1, 1, 64, 40],
            (8, 48),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        ),
    ),
)
def test_padded_slice_rm_last_dim_pads_sliced_row(device, begins, ends, shard_shape, core_grid):
    requested_num_cores = core_grid.num_cores()
    device_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    if device_num_cores < requested_num_cores:
        pytest.skip(f"Not enough cores to run test case (need {requested_num_cores} but have {device_num_cores})")

    torch.manual_seed(0)
    torch_input = torch.randn([1, 1, 64, 64], dtype=torch.bfloat16)
    sliced = torch_input[..., begins[3] : ends[3]]
    pad = shard_shape[1] - sliced.shape[-1]
    expected = torch.nn.functional.pad(sliced, (0, pad)) if pad > 0 else sliced

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    actual = ttnn.experimental.padded_slice(tt_input, begins, ends, [1, 1, 1, 1], memory_config=output_mem_config)
    actual = ttnn.to_torch(actual)

    passed, message = assert_equal(expected, actual)
    assert passed, message
