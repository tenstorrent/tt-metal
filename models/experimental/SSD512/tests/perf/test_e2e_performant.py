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
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
)
from models.common.utility_functions import run_for_wormhole_b0


def create_ssd512_pipeline_model(ttnn_model, dtype=ttnn.bfloat16):
    """
    Create a pipeline model function for SSD512.
    The function receives L1 device tensors and returns device tensors.
    """

    def run(l1_input_tensor):
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects input tensor to be on device"
        assert (
            l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), "Model expects input tensor to be in L1"

        input_for_model = ttnn.to_memory_config(l1_input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        if input_for_model.layout != ttnn.TILE_LAYOUT:
            input_for_model = ttnn.to_layout(input_for_model, ttnn.TILE_LAYOUT)

        loc, conf = ttnn_model.forward(input_for_model, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG, debug=False)

        if loc.layout != ttnn.ROW_MAJOR_LAYOUT:
            loc = ttnn.to_layout(loc, ttnn.ROW_MAJOR_LAYOUT)
        if conf.layout != ttnn.ROW_MAJOR_LAYOUT:
            conf = ttnn.to_layout(conf, ttnn.ROW_MAJOR_LAYOUT)

        loc = ttnn.to_memory_config(loc, ttnn.DRAM_MEMORY_CONFIG)
        conf = ttnn.to_memory_config(conf, ttnn.DRAM_MEMORY_CONFIG)

        return (loc, conf)

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
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize(
    "batch_size, size, expected_compile_time, expected_throughput_fps",
    [(1, 512, 25.4, 39.3)],
)
@pytest.mark.models_performance_bare_metal
def test_ssd512_e2e_performant(
    device,
    num_iterations,
    batch_size,
    size,
    expected_compile_time,
    expected_throughput_fps,
    reset_seeds,
    model_location_generator,
):
    """
    Test SSD512 end-to-end performance with Pipeline API (Trace + 2CQ).
    """
    setup_seeds_and_deterministic(reset_seeds=reset_seeds, seed=0)

    num_classes = 21
    dtype = ttnn.bfloat16

    logger.info("Building SSD512 model...")
    torch_model = build_and_init_torch_model(phase="test", size=size, num_classes=num_classes)
    ttnn_model = build_and_load_ttnn_model(torch_model, device, num_classes=num_classes)

    synchronize_device(device)

    input_shape = (batch_size, 3, size, size)
    sample_input = torch.randn(input_shape, dtype=torch.float32)

    logger.info("Creating pipeline model...")
    pipeline_model = create_ssd512_pipeline_model(ttnn_model, dtype=dtype)

    logger.info("Preparing input tensor...")
    sample_input_permuted = sample_input.permute(0, 2, 3, 1)
    sample_input_shape = sample_input_permuted.shape
    ttnn_input_tensor = ttnn.from_torch(
        sample_input_permuted,
        device=None,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    logger.info("Creating memory configs...")
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

    # Create DRAM input memory config
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

    logger.info(f"Configuring pipeline (2CQ with trace and overlapped input)...")
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False),
        model=pipeline_model,
        device=device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    input_tensors = [ttnn_input_tensor] * num_iterations

    logger.info("Compiling pipeline (warmup)...")
    start = time.time()
    pipeline.compile(ttnn_input_tensor)
    end = time.time()

    compile_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    logger.info(f"Running {num_iterations} inference iterations...")
    start = time.time()
    _ = pipeline.enqueue(input_tensors).pop_all()  # Execute pipeline, outputs not needed for perf test
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={num_iterations * batch_size / (end-start) : .2f} fps")

    total_num_samples = batch_size
    prep_perf_report(
        model_name="ssd512-trace-2cq",
        batch_size=total_num_samples,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=total_num_samples / expected_throughput_fps,
        comments=f"batch_{batch_size}-size_{size}",
    )

    logger.info("Performance test completed!")
