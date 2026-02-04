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
    "op_name, input_shape, padding, pad_value, output_shape",
    [
        (
            "kvpe_pad",
            [1, 1, 32, 576],
            [(0, 0), (0, 31), (0, 0), (0, 0)],  # Pad dim=1 from 1 to 32 (TILE_SIZE)
            0,
            [1, 32, 32, 576],
        ),
    ],
    ids=["kvpe_pad"],
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
def test_deepseek_v3_mla_pad_trace_mode(
    device,
    batch_size,
    op_name,
    input_shape,
    padding,
    pad_value,
    output_shape,
    warmup_iters,
    num_iters,
):
    """
    Test the pad operation from mla1d.py with trace mode.

    This operation pads tensors:
    1. kvpe_pad (line 1175): [1, 1, 32, 576] → [1, 32, 32, 576], padding dim=1 to TILE_SIZE

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - L1 interleaved memory layout
    """
    torch.manual_seed(0)

    # Create random tensor for input
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    # Golden output - apply padding
    # Convert padding format from [(d0_before, d0_after), (d1_before, d1_after), ...]
    # to torch.nn.functional.pad format (reverse order, flattened)
    # torch.pad uses (last_dim_before, last_dim_after, second_last_before, second_last_after, ...)
    torch_padding = []
    for pad_pair in reversed(padding):
        torch_padding.extend(pad_pair)

    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=pad_value)

    # Verify expected output shape
    assert (
        list(torch_output_tensor.shape) == output_shape
    ), f"Output shape mismatch: {list(torch_output_tensor.shape)} != {output_shape}"

    # Create ttnn tensor with L1 interleaved memory config
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Compile run
    logger.info(f"Compiling pad operation: {op_name}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Padding: {padding}")
    logger.info(f"  Pad value: {pad_value}")
    logger.info(f"  Output shape: {output_shape}")

    tt_output_tensor = ttnn.pad(tt_input_tensor, padding, pad_value)
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.pad(tt_input_tensor, padding, pad_value)
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.pad(tt_input_tensor, padding, pad_value)
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
