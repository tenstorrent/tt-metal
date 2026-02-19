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
    "op_name, input_shape, permute_dims, output_shape",
    [
        (
            "kv_rope_permute_pre",
            [1, 1, 32, 64],
            (0, 2, 1, 3),
            [1, 32, 1, 64],
        ),
        (
            "kv_rope_permute_post",
            [1, 32, 1, 64],
            (0, 2, 1, 3),
            [1, 1, 32, 64],
        ),
        (
            "kvpe_permute",
            [1, 1, 32, 576],
            (0, 2, 1, 3),
            [1, 32, 32, 576],  # After pad [1,32,32,576] then permute (dims 1&2 both 32)
        ),
        (
            "q_nope_permute_pre_linear",
            [32, 1, 16, 128],
            (1, 2, 0, 3),
            [1, 16, 32, 128],
        ),
        (
            "q_nope_permute_post_linear",
            [1, 16, 32, 512],
            (0, 2, 1, 3),
            [1, 32, 16, 512],
        ),
        (
            "q_rope_permute",
            [32, 1, 16, 64],
            (1, 0, 2, 3),
            [1, 32, 16, 64],
        ),
        (
            "attn_out_permute_pre_linear",
            [1, 4, 128, 512],
            (0, 2, 1, 3),
            [1, 128, 4, 512],
        ),
        (
            "v_out_permute",
            [1, 128, 4, 128],
            (0, 2, 1, 3),
            [1, 4, 128, 128],
        ),
    ],
    ids=[
        "kv_rope_permute_pre",
        "kv_rope_permute_post",
        "kvpe_permute",
        "q_nope_permute_pre_linear",
        "q_nope_permute_post_linear",
        "q_rope_permute",
        "attn_out_permute_pre_linear",
        "v_out_permute",
    ],
)
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
def test_deepseek_v3_mla_permute_trace_mode(
    device,
    batch_size,
    op_name,
    input_shape,
    permute_dims,
    output_shape,
    warmup_iters,
    num_iters,
):
    """
    Test the permute operations from mla1d.py with trace mode.

    These operations transpose tensor dimensions:
    1. kv_rope_permute_pre (line 1153): [1, 1, 32, 64] → [1, 32, 1, 64], dims=(0, 2, 1, 3)
    2. kv_rope_permute_post (line 1169): [1, 32, 1, 64] → [1, 1, 32, 64], dims=(0, 2, 1, 3)
    3. kvpe_permute (lines 1191-1194): [1, 1, 32, 576] → pad → [1, 32, 32, 576] → permute(0,2,1,3) → [1, 32, 32, 576]
       Includes pad operation before permute (model perf: pad 45µs + permute 24µs)
    4. q_nope_permute_pre_linear (line 1238): [1, 32, 16, 192] → [32, 16, 1, 192], dims=(1, 2, 0, 3)
    5. q_nope_permute_post_linear (line 1242): [1, 16, 32, 512] → [1, 32, 16, 512], dims=(0, 2, 1, 3)
    6. q_rope_permute (line 1247): [1, 32, 16, 64] → [32, 1, 16, 64], dims=(1, 0, 2, 3)
    7. attn_out_permute_pre_linear (line 1304): [1, 4, 128, 512] → [1, 128, 4, 512], dims=(0, 2, 1, 3)
    8. v_out_permute (line 1310): [1, 128, 4, 128] → [1, 4, 128, 128], dims=(0, 2, 1, 3)

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - L1 interleaved memory layout
    """
    torch.manual_seed(0)

    # Create random tensor for input
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    # Golden output - apply pad (if kvpe_permute) then permute
    if op_name == "kvpe_permute":
        # Pad dim 1 from 1 to 32, matching model line 1191
        torch_output_tensor = torch.nn.functional.pad(
            torch_input_tensor, (0, 0, 0, 0, 0, 31, 0, 0), mode="constant", value=0
        )
        torch_output_tensor = torch_output_tensor.permute(permute_dims)
    else:
        torch_output_tensor = torch_input_tensor.permute(permute_dims)

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
    logger.info(f"Compiling permute operation: {op_name}")
    logger.info(f"  Input shape: {input_shape}")
    if op_name == "kvpe_permute":
        logger.info(f"  Pad operation: dim 1 from 1 to 32")
    logger.info(f"  Permute dims: {permute_dims}")
    logger.info(f"  Output shape: {output_shape}")

    if op_name == "kvpe_permute":
        # Pad dim 1 from 1 to 32, matching model line 1191
        tt_padded_tensor = ttnn.pad(tt_input_tensor, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)
        tt_output_tensor = ttnn.permute(tt_padded_tensor, permute_dims)
    else:
        tt_output_tensor = ttnn.permute(tt_input_tensor, permute_dims)
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        if op_name == "kvpe_permute":
            tt_padded_tensor = ttnn.pad(tt_input_tensor, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)
            tt_output_tensor = ttnn.permute(tt_padded_tensor, permute_dims)
        else:
            tt_output_tensor = ttnn.permute(tt_input_tensor, permute_dims)
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        if op_name == "kvpe_permute":
            tt_padded_tensor = ttnn.pad(tt_input_tensor, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)
            tt_output_tensor = ttnn.permute(tt_padded_tensor, permute_dims)
        else:
            tt_output_tensor = ttnn.permute(tt_input_tensor, permute_dims)
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
