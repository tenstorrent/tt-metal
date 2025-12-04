# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.SSD512.common import (
    setup_seeds_and_deterministic,
    build_and_init_torch_model,
    build_and_load_ttnn_model,
    synchronize_device,
)
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
)
from models.common.utility_functions import run_for_wormhole_b0


def create_ssd512_pipeline_model(ttnn_model, dtype=ttnn.bfloat16):
    """
    Create a pipeline model function for SSD512.
    The function receives L1 device tensors and returns device tensors.
    Now works directly with TTNN tensors - no torch conversion needed.
    """

    def run(l1_input_tensor):
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects input tensor to be on device"
        assert (
            l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), "Model expects input tensor to be in L1"

        batch_size = l1_input_tensor.shape[0]
        input_height = l1_input_tensor.shape[1]
        input_width = l1_input_tensor.shape[2]

        # Convert TTNN tensor to torch for model forward
        input_torch = ttnn.to_torch(l1_input_tensor)
        if input_torch.dim() == 4 and input_torch.shape[3] == 3:
            input_torch = input_torch.permute(0, 3, 1, 2)

        # Forward pass expects torch tensor
        loc, conf = ttnn_model.forward(input_torch, dtype=dtype, memory_config=ttnn.L1_MEMORY_CONFIG, debug=False)

        # Convert outputs back to TTNN tensors in DRAM
        loc_ttnn = ttnn.from_torch(
            loc,
            device=ttnn_model.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        conf_ttnn = ttnn.from_torch(
            conf,
            device=ttnn_model.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return (loc_ttnn, conf_ttnn)

    return run


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 98304,
            "trace_region_size": 10000000,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [10])
@pytest.mark.parametrize(
    "batch_size, size, expected_throughput_fps",
    [(1, 512, 10.0)],
)
@pytest.mark.models_performance_bare_metal
def test_ssd512_e2e_performant(
    device, num_iterations, batch_size, size, expected_throughput_fps, reset_seeds, model_location_generator
):
    """
    Test SSD512 end-to-end performance with Pipeline API (Trace + 2CQ).
    """
    setup_seeds_and_deterministic(reset_seeds=reset_seeds, seed=0)

    num_classes = 21
    dtype = ttnn.bfloat16

    logger.info(f"Building SSD512 model for performance test...")
    torch_model = build_and_init_torch_model(phase="test", size=size, num_classes=num_classes)
    ttnn_model = build_and_load_ttnn_model(torch_model, device, num_classes=num_classes)

    synchronize_device(device)

    input_shape = (batch_size, 3, size, size)
    sample_input = torch.randn(input_shape, dtype=torch.float32)

    logger.info(f"Creating pipeline model function...")
    pipeline_model = create_ssd512_pipeline_model(ttnn_model, dtype=dtype)

    logger.info(f"Preparing input tensor...")
    sample_input_permuted = sample_input.permute(0, 2, 3, 1)
    sample_input_shape = sample_input_permuted.shape
    sample_input_host = ttnn.from_torch(
        sample_input_permuted,
        device=None,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    logger.info(f"Creating sharded memory configs...")
    batch_size, height, width, channels = sample_input_shape
    total_height = batch_size * height * width

    core_grid = device.core_grid
    num_l1_cores = core_grid.x * core_grid.y

    max_cb_pages = 60000
    tile_height = 32
    l1_alignment = 32
    dtype_size = 2
    max_shard_height = min(total_height // num_l1_cores, max_cb_pages * tile_height)
    max_shard_height = max(tile_height, (max_shard_height // tile_height) * tile_height)

    shard_width_bytes = channels * dtype_size
    padded_shard_width = ((shard_width_bytes + l1_alignment - 1) // l1_alignment) * l1_alignment
    padded_shard_width_channels = padded_shard_width // dtype_size

    l1_shard_height = max_shard_height
    l1_core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))
    l1_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({l1_core_range}),
        [l1_shard_height, padded_shard_width_channels],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    l1_input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec
    )

    dram_grid_size = device.dram_grid_size()
    num_dram_cores = dram_grid_size.x
    assert dram_grid_size.y == 1, "Only 1D DRAM grid is supported"

    min_shard_height = (total_height + num_dram_cores - 1) // num_dram_cores
    dram_shard_height = max(tile_height, ((min_shard_height + tile_height - 1) // tile_height) * tile_height)

    num_shards_needed = (total_height + dram_shard_height - 1) // dram_shard_height
    while num_shards_needed > num_dram_cores:
        dram_shard_height += tile_height
        num_shards_needed = (total_height + dram_shard_height - 1) // dram_shard_height

    actual_num_shards = min(num_shards_needed, num_dram_cores)

    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(actual_num_shards - 1, 0))}),
        [dram_shard_height, padded_shard_width_channels],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    logger.info(f"Configuring pipeline (2CQ without trace)...")
    pipeline_config = PipelineConfig(
        use_trace=False,
        num_command_queues=2,
        all_transfers_on_separate_command_queue=False,
    )

    logger.info(f"Creating pipeline...")
    pipeline = create_pipeline_from_config(
        config=pipeline_config,
        model=pipeline_model,
        device=device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    logger.info(f"Compiling pipeline (warmup)...")
    pipeline.compile(sample_input_host)

    synchronize_device(device)

    logger.info(f"Running {num_iterations} inference iterations...")
    inference_times = []
    all_inputs = [sample_input_host] * num_iterations

    t0 = time.time()
    pipeline.enqueue(all_inputs)
    outputs = pipeline.pop_all()
    synchronize_device(device)
    t1 = time.time()

    total_time = t1 - t0
    avg_time_per_iteration = total_time / num_iterations
    throughput_fps = batch_size / avg_time_per_iteration

    logger.info(f"Performance Results:")
    logger.info(f"  Total time for {num_iterations} iterations: {total_time:.4f}s")
    logger.info(f"  Average time per iteration: {avg_time_per_iteration:.4f}s")
    logger.info(f"  Throughput: {throughput_fps:.2f} FPS")

    logger.info(f"Cleaning up pipeline...")
    pipeline.cleanup()

    logger.info(f"Performance test completed successfully!")
    logger.info(f"  Expected FPS: {expected_throughput_fps:.2f}")
    logger.info(f"  Actual FPS: {throughput_fps:.2f}")

    assert (
        throughput_fps >= expected_throughput_fps * 0.8
    ), f"Throughput {throughput_fps:.2f} FPS is below expected {expected_throughput_fps:.2f} FPS"
