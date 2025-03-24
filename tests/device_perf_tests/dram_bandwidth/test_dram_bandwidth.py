# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch
import ttnn

from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import is_blackhole


@pytest.mark.parametrize(
    "shape, core_grid",
    [
        ([1, 1, 10 * 1024 * 1024 // 32 // 64, 32 * 64], (8, 8)),
        ([1, 1, 20 * 1024 * 1024 // 32 // 64, 32 * 64], (8, 8)),
    ],
    ids=["40MB_DRAM_INTERLEAVED", "80MB_DRAM_INTERLEAVED"],
)
def test_dram_interleaved(
    device,
    shape,
    core_grid,
):
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y == 64:
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    # helper function to create L1 width sharded memory config
    def create_width_sharded_memory_config(output_core_grid, input_shape):
        if isinstance(output_core_grid, tuple):
            output_core_range_set = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(output_core_grid[0] - 1, output_core_grid[1] - 1)
                    ),
                ]
            )
        else:
            output_core_range_set = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in output_core_grid
                ]
            )
        padded_out_w = math.ceil(input_shape[3] / output_core_range_set.num_cores() / 32) * 32
        output_memory_config = ttnn.create_sharded_memory_config(
            shape=(
                input_shape[0] * input_shape[1] * input_shape[2],
                padded_out_w,
            ),
            core_grid=output_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        return output_memory_config

    torch_input_tensor = torch.randint(low=0, high=10, size=shape, dtype=torch.int32)
    ttnn_input_tensor_dram_interleaved = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_width_sharded_memory_config = create_width_sharded_memory_config(core_grid, shape)

    ttnn_input_tensor_width_sharded = ttnn.to_memory_config(
        ttnn_input_tensor_dram_interleaved, memory_config=output_width_sharded_memory_config
    )

    print(ttnn_input_tensor_dram_interleaved.memory_config())
    print(ttnn_input_tensor_width_sharded.memory_config())


@pytest.mark.parametrize(
    "shape",
    [
        [12, 1, 128, 1024],
        [12, 1, 256, 1024],
    ],
    ids=["6MB_DRAM_SHARDED", "12MB_DRAM_SHARDED"],
)
def test_dram_sharded(
    device,
    shape,
):
    grid = device.compute_with_storage_grid_size()
    assert grid.x * grid.y == 64, "Only valid on 64 cores grid"

    # helper function to create dram sharded memory config with height sharded tensor
    # where dram cores is equal to batch size
    def create_dram_sharded_batch_height_sharded_memory_config(device, input_shape):
        dram_grid_size = device.dram_grid_size()

        assert (
            input_shape[0] % (dram_grid_size.x * dram_grid_size.y) == 0
        ), "Input shape must be divisible by dram grid size"

        dram_shard_spec = ttnn.ShardSpec(
            grid=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            shard_shape=[
                input_shape[1] * input_shape[2],
                input_shape[3],
            ],
            shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_dram_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return sharded_dram_mem_config

    # helper function to create L1 sharded memory config with height sharded tensor
    # where only optimal read/write worker cores are used
    def create_batch_height_sharded_memory_config(device, input_shape):
        dram_grid_size = device.dram_grid_size()

        assert (
            input_shape[0] % (dram_grid_size.x * dram_grid_size.y) == 0
        ), "Input shape must be divisible by dram grid size"

        output_core_grid = device.get_optimal_dram_bank_to_logical_worker_assignment()
        output_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(core_coord, core_coord) for core_coord in output_core_grid]
        )

        output_memory_config = ttnn.create_sharded_memory_config(
            shape=(
                input_shape[1] * input_shape[2],
                input_shape[3],
            ),
            core_grid=output_core_range_set,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        return output_memory_config

    torch_input_tensor = torch.randint(low=0, high=10, size=shape, dtype=torch.int32)

    dram_sharded_memory_config = create_dram_sharded_batch_height_sharded_memory_config(device, shape)

    ttnn_input_tensor_dram_sharded = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_sharded_memory_config,
    )

    output_memory_config = create_batch_height_sharded_memory_config(device, shape)

    ttnn_input_tensor_l1_sharded = ttnn.to_memory_config(
        ttnn_input_tensor_dram_sharded, memory_config=output_memory_config
    )


@pytest.mark.skipif(is_blackhole(), reason="Blackhole")
@pytest.mark.parametrize(
    "test, transfer_size_MB, expected_speed_GBps",
    [
        ["test_dram_interleaved[40MB_DRAM_INTERLEAVED]", 40, 200],
        ["test_dram_interleaved[80MB_DRAM_INTERLEAVED]", 80, 200],
        ["test_dram_sharded[6MB_DRAM_SHARDED]", 6, 230],
        ["test_dram_sharded[12MB_DRAM_SHARDED]", 12, 230],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal(test, transfer_size_MB, expected_speed_GBps):
    subdir = f"dram_bandwidth"
    batch_size = 1
    num_iterations = 3
    margin = 0.03
    command = f"pytest tests/device_perf_tests/dram_bandwidth/test_dram_bandwidth.py::{test}"
    cols = ["DEVICE KERNEL"]

    expected_perf_ns = 10**9 * transfer_size_MB / (expected_speed_GBps * 1024)

    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    expected_perf_cols = {inference_time_key: expected_perf_ns}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)

    print(post_processed_results)
    check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
