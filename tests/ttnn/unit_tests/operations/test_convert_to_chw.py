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

    assert_with_pcc(expected, ttnn.to_torch(actual), 1.0)

    return actual


@skip_for_grayskull()
@skip_for_blackhole()
@pytest.mark.parametrize("C", [1, 2, 4])
@pytest.mark.parametrize(
    "HW, core_grid, padded_sharded_dim",
    (
        (96, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}), 64),
        (84480, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}), 2656),
        (168960, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}), 2656),
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
def test_convert_to_chw_padded(device, C, HW, core_grid, padded_sharded_dim):
    if device.core_grid.num_cores < core_grid.num_cores():
        pytest.skip(
            "Not enough cores to run test case (need {core_grid.num_cores()} but have {device.core_grid.num_cores}"
        )
    input_tensor = torch.randn([1, 1, HW, C], dtype=torch.bfloat16)
    expected = input_tensor.transpose(2, 3)

    input_shard_shape = (padded_sharded_dim, 32)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    output_shard_shape = (C, padded_sharded_dim)
    output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device, memory_config=input_mem_config)

    actual = ttnn.experimental.convert_to_chw(input_tensor, memory_config=output_mem_config)

    assert_with_pcc(expected, ttnn.to_torch(actual), 1.0)

    return actual


@skip_for_grayskull()
@skip_for_blackhole()
def test_convert_to_chw_with_program_cache(device, use_program_cache):
    C, HW, core_grid = 2, 256, ttnn.CoreGrid(x=2, y=1)

    C_padded, HW_padded, padded_sharded_dim = 4, 96, 64
    core_grid_padded = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})

    a, b, c = None, None, None
    for _ in range(8):
        a = test_convert_to_chw_padded(device, C_padded, HW_padded, core_grid_padded, padded_sharded_dim)
        b = test_convert_to_chw(device, C, HW, core_grid)
        c = test_convert_to_chw_padded(device, C_padded, HW_padded, core_grid_padded, padded_sharded_dim)
        dummy_shape = [1, 1, 256, 128]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = (
            ttnn.Tensor(py_dummy_tensor, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, ttnn.L1_MEMORY_CONFIG)
        )

    assert device.num_program_cache_entries() == 2
