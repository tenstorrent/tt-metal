# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("hidden_size", [896])
@pytest.mark.parametrize("output_size", [2112])
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 550912,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_wq_kv_a_linear_trace_mode(
    device, batch_size, hidden_size, output_size, warmup_iters, num_iters
):
    """
    Test the fused wq_kv_a linear operation with trace mode for performance measurement.

    This test captures a trace of the linear operation and executes it multiple times
    to measure performance. Uses signposts for Tracy profiling integration.

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Width sharded memory configuration
    """
    torch.manual_seed(0)

    input_shape = [1, 1, batch_size, hidden_size]
    weight_shape = [hidden_size, output_size]

    # Create random tensors for golden reference
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_weight_tensor = torch.randn(weight_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor @ torch_weight_tensor

    # Create ttnn tensors
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Setup WIDTH_SHARDED memory config
    shard_height = batch_size
    shard_width = hidden_size // 28  # 896 / 28 = 32

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(3, 6),
            )
        }
    )

    shard_shape = [shard_height, shard_width]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    width_sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    # Convert input to width sharded
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, width_sharded_mem_config)

    # Compile run
    logger.info("Compiling linear operation")
    tt_output_tensor = ttnn.linear(
        tt_input_tensor,
        tt_weight_tensor,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.linear(
            tt_input_tensor,
            tt_weight_tensor,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.linear(
            tt_input_tensor,
            tt_weight_tensor,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler = BenchmarkProfiler()
    profiler.start("warmup")
    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)
    profiler.end("warmup")
    ttnn.synchronize_device(device)

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("main")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    profiler.end("main")
    signpost("stop")
    ttnn.synchronize_device(device)

    # Calculate performance metrics
    warmup_time_ms = profiler.get_duration("warmup") * 1000
    main_time_ms = profiler.get_duration("main") * 1000
    avg_time_per_iter_us = (main_time_ms / num_iters) * 1000

    logger.info(f"Warmup time: {warmup_time_ms:.2f} ms ({warmup_iters} iterations)")
    logger.info(f"Main trace time: {main_time_ms:.2f} ms ({num_iters} iterations)")
    logger.info(f"Average time per iteration: {avg_time_per_iter_us:.2f} µs")

    # Verify correctness
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    torch_output_from_tt = ttnn.to_torch(tt_output_tensor)

    assert torch_output_from_tt.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, torch_output_from_tt, 0.99)

    logger.info("✓ Trace mode test passed with correct output")
