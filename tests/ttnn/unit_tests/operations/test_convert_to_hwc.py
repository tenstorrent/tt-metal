# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_equal


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

    passed, message = assert_equal(expected, actual)
    assert passed, message


@pytest.mark.parametrize("C", [8, 16])
@pytest.mark.parametrize(
    "HW, input_core_grid, output_core_grid, input_padded_sharded_dim, output_padded_sharded_dim",
    (
        (
            32,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                }
            ),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                }
            ),
            32,
            32,
        ),
        (
            84480,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0)),
                }
            ),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3)),
                }
            ),
            14080,
            2656,
        ),
        (
            168960,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0)),
                }
            ),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                }
            ),
            14080,
            2688,
        ),  # UNet Shallow
    ),
)
def test_convert_to_hwc_dram(
    device, C, HW, input_core_grid, output_core_grid, input_padded_sharded_dim, output_padded_sharded_dim
):
    worker_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    requested_num_cores = output_core_grid.num_cores()
    if worker_num_cores < requested_num_cores:
        pytest.skip(f"Not enough cores to run test case (need {requested_num_cores} but have {worker_num_cores})")

    dram_num_cores = device.dram_grid_size().x * device.dram_grid_size().y
    requested_num_dram_cores = input_core_grid.num_cores()
    if dram_num_cores < requested_num_dram_cores:
        pytest.skip(
            f"Not enough DRAM cores to run test case (need {requested_num_dram_cores} but have {dram_num_cores})"
        )

    input_tensor = torch.randn([1, 1, C, HW], dtype=torch.bfloat16)
    expected = input_tensor.transpose(2, 3)

    input_shard_shape = (C, input_padded_sharded_dim)
    input_shard_spec = ttnn.ShardSpec(input_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, input_shard_spec)

    output_shard_shape = (output_padded_sharded_dim, C)
    output_shard_spec = ttnn.ShardSpec(output_core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    input_tensor = ttnn.Tensor(
        input_tensor, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
    )

    actual = ttnn.experimental.convert_to_hwc(input_tensor, memory_config=output_mem_config, dtype=ttnn.bfloat16)
    actual = ttnn.to_torch(actual)

    passed, message = assert_equal(expected, actual)
    assert passed, message
