# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
import math

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
    skip_for_blackhole,
)


def arange_row_major(shape, *, dtype=torch.bfloat16, device=None):
    numel = math.prod(shape)  # or: functools.reduce(operator.mul, shape, 1)
    return torch.arange(1, numel + 1, dtype=dtype, device=device).reshape(shape)


@pytest.mark.parametrize("C", [8, 16])
@pytest.mark.parametrize(
    "HW, core_grid, padded_sharded_dim",
    (
        (
            32,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                }
            ),
            32,
        ),
        (
            128,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                }
            ),
            64,
        ),
        (
            168960,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                }
            ),
            2688,
        ),  # UNet Shallow
    ),
)
def test_convert_to_hwc(device, C, HW, core_grid, padded_sharded_dim):
    device_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    requested_num_cores = core_grid.num_cores()
    if device_num_cores < requested_num_cores:
        pytest.skip(f"Not enough cores to run test case (need {requested_num_cores} but have {device_num_cores})")

    input_tensor = torch.randn([1, 1, C, HW], dtype=torch.bfloat16)
    expected = input_tensor.transpose(2, 3)

    input_shard_shape = (C, padded_sharded_dim)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    output_shard_shape = (padded_sharded_dim, C)
    output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    input_tensor = ttnn.Tensor(
        input_tensor, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
    )

    actual = ttnn.experimental.convert_to_hwc(input_tensor, memory_config=output_mem_config, dtype=ttnn.bfloat16)
    actual = ttnn.to_torch(actual)

    assert_with_pcc(expected, actual, 0.99999)

    # return actual
