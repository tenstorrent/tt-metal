# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "op_name, input_shape, output_shape",
    [
        (
            "wo_tilize",
            [1, 1, 32, 16384],
            [1, 1, 32, 16384],
        ),
    ],
    ids=["wo_tilize"],
)
@pytest.mark.parametrize("memory_config_type", ["interleaved", "width_sharded"])
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 1671168,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_tilize_trace_mode(
    device,
    batch_size,
    op_name,
    input_shape,
    output_shape,
    memory_config_type,
    warmup_iters,
    num_iters,
):
    """
    Test the tilize operation from mla1d.py with trace mode.

    This operation converts from ROW_MAJOR layout to TILE_LAYOUT:
    - wo_tilize (line 1903): [1, 1, 32, 16384] ROW_MAJOR → [1, 1, 32, 16384] TILE_LAYOUT
      Context: After all_gather in decode path, converts v_out before wo matmul
      Input: L1 ROW_MAJOR (from all_gather)
      Output: L1 TILE_LAYOUT

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Input: L1 memory, ROW_MAJOR
    - Output: L1 memory, TILE_LAYOUT
    - Memory config: INTERLEAVED or WIDTH_SHARDED (parameterized)
      - INTERLEAVED: no sharding
      - WIDTH_SHARDED: shard_spec grid [0:0-7:1], shape [16, 1024], ROW_MAJOR
    """
    torch.manual_seed(0)

    # Create random tensor for input
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    # Golden output - same shape, just layout conversion
    torch_output_tensor = torch_input_tensor.clone()

    # Verify expected output shape
    assert (
        list(torch_output_tensor.shape) == output_shape
    ), f"Output shape mismatch: {list(torch_output_tensor.shape)} != {output_shape}"

    # Configure memory config based on type
    if memory_config_type == "width_sharded":
        # WIDTH_SHARDED: grid=[{x:0,y:0}-{x:7,y:1}], shape=[16, 1024], ROW_MAJOR
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
            [32, 1024],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
            [32, 1024],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )
    else:
        # INTERLEAVED
        input_memory_config = ttnn.L1_MEMORY_CONFIG
        output_memory_config = ttnn.L1_MEMORY_CONFIG

    # Create ttnn tensor with L1 memory config in ROW_MAJOR layout
    # This simulates the output from all_gather which is in ROW_MAJOR
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=input_memory_config,
    )

    # Compile run
    logger.info(f"Compiling tilize operation: {op_name}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Input layout: ROW_MAJOR")
    logger.info(f"  Memory config: {memory_config_type}")
    logger.info(f"  Output shape: {output_shape}")
    logger.info(f"  Output layout: TILE_LAYOUT")

    tt_output_tensor = ttnn.tilize(tt_input_tensor, memory_config=output_memory_config)
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.tilize(tt_input_tensor, memory_config=output_memory_config)
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.tilize(tt_input_tensor, memory_config=output_memory_config)
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

    # Verify correctness
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    torch_output_from_tt = ttnn.to_torch(tt_output_tensor)

    assert (
        torch_output_from_tt.shape == torch_output_tensor.shape
    ), f"Shape mismatch: {torch_output_from_tt.shape} != {torch_output_tensor.shape}"

    assert_equal(torch_output_tensor, torch_output_from_tt)

    logger.info(f"✓ Trace mode {op_name} test passed with correct output")
