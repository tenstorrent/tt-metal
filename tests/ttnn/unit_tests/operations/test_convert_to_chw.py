# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
    skip_for_blackhole,
)


@skip_for_grayskull()
@skip_for_blackhole()
@pytest.mark.parametrize("input_data_type", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("C", [1, 2, 4])
@pytest.mark.parametrize(
    "HW, core_grid",
    (
        (32, ttnn.CoreGrid(x=1, y=1)),
        (64, ttnn.CoreGrid(x=1, y=1)),
        (64, ttnn.CoreGrid(x=2, y=1)),
        (1056 * 160, ttnn.CoreGrid(x=8, y=6)),
        (1024 * 128, ttnn.CoreGrid(x=8, y=8)),
        (2048 * 128, ttnn.CoreGrid(x=8, y=8)),
        (4096 * 128, ttnn.CoreGrid(x=8, y=8)),
    ),
)
def test_convert_to_chw(device, C, HW, core_grid, input_data_type):
    torch.manual_seed(0)
    device_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    requested_num_cores = core_grid.x * core_grid.y
    if device_num_cores < requested_num_cores:
        pytest.skip(f"Not enough cores to run test case (need {requested_num_cores} but have {device_num_cores})")
    input_tensor = torch.randn([1, 1, HW, C], dtype=torch.bfloat16)
    expected = input_tensor.transpose(2, 3)

    input_tensor = ttnn.from_torch(input_tensor, dtype=input_data_type, layout=ttnn.TILE_LAYOUT)
    input_memory_config = ttnn.create_sharded_memory_config(
        [1, 1, HW, 32], core_grid, ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR
    )
    input_tensor = ttnn.to_device(input_tensor, device, input_memory_config)

    output_memory_config = ttnn.create_sharded_memory_config(
        [1, 1, C, HW], core_grid, ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR
    )
    actual = ttnn.experimental.convert_to_chw(input_tensor, memory_config=output_memory_config, dtype=ttnn.bfloat16)

    if input_data_type == ttnn.bfloat8_b:
        expected_pcc = 0.9999  # bfloat8_b can't be exatcly compared to torch bfloat16
        assert_with_pcc(expected, ttnn.to_torch(actual), expected_pcc)
    else:
        assert_equal(expected, ttnn.to_torch(actual))

    return actual


@skip_for_grayskull()
@skip_for_blackhole()
@pytest.mark.parametrize("input_data_type", [ttnn.bfloat16, ttnn.bfloat8_b])
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
def test_convert_to_chw_padded(device, C, HW, core_grid, padded_sharded_dim, input_data_type):
    device_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    requested_num_cores = core_grid.num_cores()
    if device_num_cores < requested_num_cores:
        pytest.skip(f"Not enough cores to run test case (need {requested_num_cores} but have {device_num_cores})")
    input_tensor = torch.randn([1, 1, HW, C], dtype=torch.bfloat16)
    expected = input_tensor.transpose(2, 3)

    input_shard_shape = (padded_sharded_dim, 32)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    output_shard_shape = (C, padded_sharded_dim)
    output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    input_tensor = ttnn.from_torch(input_tensor, dtype=input_data_type, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device, memory_config=input_mem_config)

    actual = ttnn.experimental.convert_to_chw(input_tensor, memory_config=output_mem_config, dtype=ttnn.bfloat16)

    if input_data_type == ttnn.bfloat8_b:
        expected_pcc = 0.9999  # bfloat8_b can't be exatcly compared to torch bfloat16
        assert_with_pcc(expected, ttnn.to_torch(actual), expected_pcc)
    else:
        assert_equal(expected, ttnn.to_torch(actual))

    return actual


@skip_for_grayskull()
@skip_for_blackhole()
def test_convert_to_chw_with_program_cache(device):
    C, HW, core_grid = 2, 256, ttnn.CoreGrid(x=2, y=1)

    C_padded, HW_padded, padded_sharded_dim = 4, 96, 64
    core_grid_padded = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})

    a, b, c = None, None, None
    for iter in range(8):
        if iter < 4:
            in_dtype = ttnn.bfloat16
        else:
            in_dtype = ttnn.bfloat8_b
        a = test_convert_to_chw_padded(device, C_padded, HW_padded, core_grid_padded, padded_sharded_dim, in_dtype)
        b = test_convert_to_chw(device, C, HW, core_grid, in_dtype)
        c = test_convert_to_chw_padded(device, C_padded, HW_padded, core_grid_padded, padded_sharded_dim, in_dtype)
        dummy_shape = [1, 1, 256, 128]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = (
            ttnn.Tensor(py_dummy_tensor, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, ttnn.L1_MEMORY_CONFIG)
        )

    assert device.num_program_cache_entries() == 4
