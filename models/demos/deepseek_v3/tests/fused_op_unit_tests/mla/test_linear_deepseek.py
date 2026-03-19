# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "op_name, input_shape, weight_shape, input_memory_config, output_memory_config",
    [
        (
            "mla_linear_wq_kv_a",
            [1, 1, 32, 896],  # Input shape
            [896, 2112],  # Weight shape: hidden_size x (q_lora_rank + kv_lora_rank + qk_rope_head_dim)
            "width_sharded_7x4",  # WIDTH_SHARDED 7x4 grid, shard [32, 32]
            "width_sharded",  # Output: WIDTH_SHARDED
        ),
        (
            "mla_linear_wq_b",
            [1, 1, 32, 1536],  # Input shape (q_lora_rank)
            [1536, 3072],  # Weight shape: q_lora_rank x (num_heads * qk_head_dim) = 1536 x 3072
            "width_sharded_8x2",  # WIDTH_SHARDED 8x2 grid, shard [32, 96]
            "interleaved",  # Output: L1 interleaved
        ),
        (
            "mla_linear_wkv_b1",
            [1, 16, 32, 128],  # Input shape: [1, num_heads_local, bsz, qk_nope_head_dim]
            [128, 512],  # Weight shape: qk_nope_head_dim x kv_lora_rank = 128 x 512
            "interleaved",  # L1 interleaved
            "interleaved",  # Output: L1 interleaved
        ),
        (
            "mla_linear_wkv_b2",
            [1, 128, 4, 512],  # Input shape: [1, num_heads, bsz_per_device, kv_lora_rank]
            [512, 128],  # Weight shape: kv_lora_rank x v_head_dim = 512 x 128
            "interleaved",  # L1 interleaved
            "interleaved",  # Output: L1 interleaved
        ),
        (
            "mla_linear_wo",
            [1, 1, 32, 16384],  # Input shape: [1, 1, bsz, num_heads * v_head_dim]
            [16384, 896],  # Weight shape: (num_heads * v_head_dim) x hidden_size = 16384 x 896
            "interleaved",  # L1 interleaved
            "width_sharded_7x4",  # Output: WIDTH_SHARDED 7x4 grid, shard [32, 32]
        ),
    ],
    ids=["wq_kv_a", "wq_b", "wkv_b1", "wkv_b2", "wo"],
)
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 704600,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_linear_trace_mode(
    device,
    batch_size,
    op_name,
    input_shape,
    weight_shape,
    input_memory_config,
    output_memory_config,
    warmup_iters,
    num_iters,
):
    """
    Test all decode linear operations from mla1d.py with trace mode.

    Linear operations tested:
    1. wq_kv_a (line 1104): [1, 1, 32, 896] x [896, 2112] - WIDTH_SHARDED input/output
    2. wq_b (line 1225): [1, 1, 32, 1536] x [1536, 3072] - WIDTH_SHARDED input, interleaved output
    TODO: need to port to batch 16 in future pr
    3. wkv_b1 (line 1240): [1, 16, 32, 192] x [192, 512] - Interleaved input/output
    TODO: need to port to batch 128 in future pr
    4. wkv_b2 (line 1306): [1, 128, 4, 512] x [512, 128] - Interleaved input/output
    5. wo (line 1334): [1, 1, 32, 16384] x [16384, 896] - Interleaved input, WIDTH_SHARDED output

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    """
    torch.manual_seed(0)

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
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Setup input memory config based on specification
    if input_memory_config == "width_sharded_7x4":
        # For wq_kv_a: WIDTH_SHARDED 7x4 grid, shard [32, 32]
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))})
        shard_spec = ttnn.ShardSpec(shard_grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
        tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, mem_config)
    elif input_memory_config == "width_sharded_8x2":
        # For wq_b: WIDTH_SHARDED 8x2 grid, shard [32, 96]
        grid_size = device.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(16, grid_size, row_wise=True)
        shard_spec = ttnn.ShardSpec(shard_grid, [32, 96], ttnn.ShardOrientation.ROW_MAJOR)
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
        tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, mem_config)
    # else: interleaved - already in L1_MEMORY_CONFIG

    # Setup output memory config
    if output_memory_config == "width_sharded":
        output_mem_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    elif output_memory_config == "width_sharded_7x4":
        # For wo: WIDTH_SHARDED 7x4 grid, shard [32, 32]
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))})
        shard_spec = ttnn.ShardSpec(shard_grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
        output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    elif output_memory_config == "dram_interleaved":
        output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    else:  # interleaved
        output_mem_config = ttnn.L1_MEMORY_CONFIG

    # Compile run
    logger.info(f"Compiling linear operation: {op_name}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Weight shape: {weight_shape}")
    logger.info(f"  Output shape: {list(torch_output_tensor.shape)}")

    tt_output_tensor = ttnn.linear(
        tt_input_tensor,
        tt_weight_tensor,
        memory_config=output_mem_config,
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
            memory_config=output_mem_config,
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
            memory_config=output_mem_config,
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

    # Verify correctness
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    torch_output_from_tt = ttnn.to_torch(tt_output_tensor)

    assert torch_output_from_tt.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, torch_output_from_tt, 0.99)

    logger.info(f"✓ Trace mode {op_name} test passed with correct output")
