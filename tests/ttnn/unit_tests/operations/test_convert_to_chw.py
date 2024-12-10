# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
    skip_for_blackhole,
)


@skip_for_grayskull()
@skip_for_blackhole()
@pytest.mark.parametrize("C", [1, 2, 4])
@pytest.mark.parametrize(
    "HW, core_grid",
    (
        (32, ttnn.CoreGrid(x=1, y=1)),
        (64, ttnn.CoreGrid(x=1, y=1)),
        (64, ttnn.CoreGrid(x=2, y=1)),
        (1056 * 160, ttnn.CoreGrid(x=8, y=6)),
    ),
)
def test_convert_to_chw(device, C, HW, core_grid):
    input_tensor = torch.randn([1, 1, HW, C], dtype=torch.bfloat16)
    expected = input_tensor.transpose(2, 3)

    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_memory_config = ttnn.create_sharded_memory_config(
        [1, 1, HW, 32], core_grid, ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR
    )
    input_tensor = ttnn.to_device(input_tensor, device, input_memory_config)

    output_memory_config = ttnn.create_sharded_memory_config(
        [1, 1, 32, HW], core_grid, ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR
    )
    actual = ttnn.experimental.convert_to_chw(input_tensor, memory_config=output_memory_config)
    actual = ttnn.to_torch(actual)

    assert_with_pcc(expected, actual, 0.9999999)


@skip_for_grayskull()
@skip_for_blackhole()
def test_convert_to_chw_with_program_cache(device, use_program_cache):
    C, HW = 8, 128
    core_grid = ttnn.CoreGrid(x=2, y=1)

    for _ in range(2):
        test_convert_to_chw(device, C, HW, core_grid)
        test_convert_to_chw(device, C, HW, core_grid)
        dummy_shape = [1, 1, 128, 128]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = (
            ttnn.Tensor(py_dummy_tensor, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, ttnn.L1_MEMORY_CONFIG)
        )

    assert device.num_program_cache_entries() == 1
