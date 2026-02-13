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
    "op_name, concat_spec, concat_dim, output_shape, memory_config",
    [
        (
            "kvpe_concat",
            ([1, 1, 32, 512], [1, 1, 32, 64]),
            -1,
            [1, 1, 32, 576],
            ttnn.L1_MEMORY_CONFIG,
        ),
        (
            "q_concat",
            ([1, 32, 16, 512], [1, 32, 16, 64]),
            -1,
            [1, 32, 16, 576],
            ttnn.L1_MEMORY_CONFIG,
        ),
    ],
    ids=[
        "kvpe_concat",
        "q_concat",
    ],
)
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 595968,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_concat_trace_mode(
    device,
    batch_size,
    op_name,
    concat_spec,
    concat_dim,
    output_shape,
    memory_config,
    warmup_iters,
    num_iters,
):
    """
    Test the concat operations from mla1d.py with trace mode.

    These operations concatenate tensor dimensions:
    1. kvpe_concat (line 1172): [1, 1, 32, 512] | [1, 1, 32, 64] → [1, 32, 1, 576], dims=(-1)
    2. q_concat (line 1177): [1, 32, 16, 512] | [32, 1, 16, 64] → [1, 32, 16, 576], dims=(-1)

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - L1 interleaved memory layout
    """
    torch.manual_seed(0)

    # Create random tensor for input
    torch_input_tensor_1 = torch.randn(concat_spec[0], dtype=torch.bfloat16)
    torch_input_tensor_2 = torch.randn(concat_spec[1], dtype=torch.bfloat16)

    # Golden output - apply concat
    torch_output_tensor = torch.cat([torch_input_tensor_1, torch_input_tensor_2], dim=concat_dim)

    # Verify expected output shape
    assert (
        list(torch_output_tensor.shape) == output_shape
    ), f"Output shape mismatch: {list(torch_output_tensor.shape)} != {output_shape}"

    # Create ttnn tensor with L1 interleaved memory config
    tt_input_tensor_1 = ttnn.from_torch(
        torch_input_tensor_1,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )
    tt_input_tensor_2 = ttnn.from_torch(
        torch_input_tensor_2,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )

    # Compile run
    logger.info(f"Compiling concat operation: {op_name}")
    logger.info(f"  Input shapes: {concat_spec[0]} and {concat_spec[1]}")
    logger.info(f"  Concat dim: {concat_dim}")
    logger.info(f"  Output shape: {output_shape}")
    logger.info(f"  Memory config: {memory_config}")

    tt_output_tensor = ttnn.concat([tt_input_tensor_1, tt_input_tensor_2], concat_dim, memory_config=memory_config)
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.concat([tt_input_tensor_1, tt_input_tensor_2], concat_dim, memory_config=memory_config)
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.concat([tt_input_tensor_1, tt_input_tensor_2], concat_dim, memory_config=memory_config)
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
