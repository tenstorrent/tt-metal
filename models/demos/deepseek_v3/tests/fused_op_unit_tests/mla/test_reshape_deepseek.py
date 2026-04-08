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
    "op_name, input_shape, output_shape, memory_config_in, memory_config_out",
    [
        (
            "q_reshape_decode",
            [1, 1, 32, 3072],  # After wq_b linear
            [32, 1, 16, 192],  # Reshape to [bsz, 1, num_heads_local, qk_head_dim]
            ttnn.L1_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ),
        (
            "v_out_reshape_decode",
            [1, 32, 128, 128],  # After all_gather: [1, bsz, num_heads, v_head_dim]
            [1, 1, 32, 16384],  # Reshape to [1, 1, bsz, num_heads * v_head_dim]
            ttnn.L1_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ),
    ],
    ids=["q_reshape_decode", "v_out_reshape_decode"],
)
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 1335296,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_reshape_trace_mode(
    device,
    batch_size,
    op_name,
    input_shape,
    output_shape,
    memory_config_in,
    memory_config_out,
    warmup_iters,
    num_iters,
):
    """
    Test the reshape operations from mla1d.py with trace mode.

    This operation reshapes tensors:
    1. q_reshape_decode (line 1227): [1, 1, 32, 3072] → [1, 32, 16, 192]
    2. v_out_reshape_decode (line 1327): [1, 32, 128, 128] → [1, 1, 32, 16384]

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - L1 interleaved memory layout
    """
    torch.manual_seed(0)

    # Create random tensor for input
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    # Golden output - apply reshape
    torch_output_tensor = torch_input_tensor.reshape(output_shape)

    # Verify expected output shape
    assert (
        list(torch_output_tensor.shape) == output_shape
    ), f"Output shape mismatch: {list(torch_output_tensor.shape)} != {output_shape}"

    # Create ttnn tensor with specified memory config
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config_in,
    )

    # Compile run
    logger.info(f"Compiling reshape operation: {op_name}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Output shape: {output_shape}")

    tt_output_tensor = ttnn.reshape(tt_input_tensor, output_shape)
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.reshape(tt_input_tensor, output_shape)
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.reshape(tt_input_tensor, output_shape)
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
