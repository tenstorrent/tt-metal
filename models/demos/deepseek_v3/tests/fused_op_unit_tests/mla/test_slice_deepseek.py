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
@pytest.mark.parametrize("q_lora_rank", [1536])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "slice_type, start_offset, output_size",
    [
        ("q_slice", 0, 1536),  # tt_q: slice [0:1536]
        ("kv_nope_slice", 1536, 512),  # tt_kv_nope: slice [1536:2048]
        ("kv_rope_slice", 2048, 64),  # tt_kv_rope: slice [2048:2112]
    ],
    ids=["q_slice", "kv_nope_slice", "kv_rope_slice"],
)
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
def test_deepseek_v3_mla_slice_trace_mode(
    device,
    batch_size,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_head_dim,
    warmup_iters,
    num_iters,
    slice_type,
    start_offset,
    output_size,
):
    """
    Test the slice operations from lines 1117, 1118, and 1121 of mla1d.py with trace mode.

    After the all-gather and fast_reduce_nc operations, the tensor tt_q_kv is sliced into three parts:
    1. tt_q: [0, 0, 0, 0] to [1, 1, bsz, 1536] - query lora features
    2. tt_kv_nope: [0, 0, 0, 1536] to [1, 1, bsz, 2048] - key-value nope features
    3. tt_kv_rope: [0, 0, 0, 2048] to [1, 1, bsz, 2112] - key-value rope features

    Configuration:
    - Input shape: [1, 1, 32, 2112]
    - Output shapes vary by slice type:
      - q_slice: [1, 1, 32, 1536]
      - kv_nope_slice: [1, 1, 32, 512]
      - kv_rope_slice: [1, 1, 32, 64]
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Memory: L1 interleaved
    """
    torch.manual_seed(0)

    total_hidden_size = q_lora_rank + kv_lora_rank + qk_rope_head_dim  # 2112
    input_shape = [1, 1, batch_size, total_hidden_size]
    output_shape = [1, 1, batch_size, output_size]

    # Create random tensor for input
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    # Golden output: slice along last dimension
    end_offset = start_offset + output_size
    torch_output_tensor = torch_input_tensor[:, :, :, start_offset:end_offset]

    # Create ttnn tensor
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Configure slice operation matching mla1d.py lines 1117-1126
    start_indices = [0, 0, 0, start_offset]
    end_indices = [1, 1, batch_size, start_offset + output_size]
    slice_config = {
        "memory_config": ttnn.L1_MEMORY_CONFIG,
    }

    # Compile run
    logger.info(f"Compiling slice operation: {slice_type}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Start indices: {start_indices}")
    logger.info(f"  End indices: {end_indices}")
    logger.info(f"  Output shape: {output_shape}")

    tt_output_tensor = ttnn.slice(tt_input_tensor, start_indices, end_indices, **slice_config)
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.slice(tt_input_tensor, start_indices, end_indices, **slice_config)
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.slice(tt_input_tensor, start_indices, end_indices, **slice_config)
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

    logger.info(f"✓ Trace mode {slice_type} test passed with correct output")
