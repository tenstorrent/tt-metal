# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_wkv_b2_sequence_with_trace(
    device,
    tt_attn_out,
    wkv_b2_weight,
    num_iter=100,
    warmup_iters=10,
    profiler=BenchmarkProfiler(),
    num_heads=4,
    batch_size=32,
    kv_lora_rank=512,
    v_head_dim=128,
):
    """Run wkv_b2 sequence with trace mode for performance measurement."""
    # Compile Run
    logger.info("Compiling wkv_b2 sequence")

    # Operation 1: Reshard from HEIGHT sharded to L1 interleaved
    tt_attn_out_interleaved = ttnn.to_memory_config(tt_attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Operation 2: Permute (0, 2, 1, 3) - [1, num_heads, bsz, kv_lora_rank] -> [1, bsz, num_heads, kv_lora_rank]
    tt_attn_out_permuted = ttnn.permute(tt_attn_out_interleaved, (0, 2, 1, 3))

    # Operation 3: Linear (wkv_b2) - [1, bsz, num_heads, kv_lora_rank] -> [1, bsz, num_heads, v_head_dim]
    tt_v_out = ttnn.linear(
        tt_attn_out_permuted,
        wkv_b2_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    # Operation 4: Permute (0, 2, 1, 3) - [1, bsz, num_heads, v_head_dim] -> [1, num_heads, bsz, v_head_dim]
    tt_v_out_permuted = ttnn.permute(tt_v_out, (0, 2, 1, 3))

    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_attn_out_interleaved = ttnn.to_memory_config(tt_attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        tt_attn_out_permuted = ttnn.permute(tt_attn_out_interleaved, (0, 2, 1, 3))
        tt_v_out = ttnn.linear(
            tt_attn_out_permuted,
            wkv_b2_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        tt_v_out_permuted = ttnn.permute(tt_v_out, (0, 2, 1, 3))
        tt_attn_out_interleaved.deallocate(True)
        tt_attn_out_permuted.deallocate(True)
        tt_v_out.deallocate(True)
        tt_v_out_permuted.deallocate(True)
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iter} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iter):
        tt_attn_out_interleaved = ttnn.to_memory_config(tt_attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        tt_attn_out_permuted = ttnn.permute(tt_attn_out_interleaved, (0, 2, 1, 3))
        tt_v_out = ttnn.linear(
            tt_attn_out_permuted,
            wkv_b2_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        tt_v_out_permuted = ttnn.permute(tt_v_out, (0, 2, 1, 3))
        if i != num_iter - 1:
            tt_attn_out_interleaved.deallocate(True)
            tt_attn_out_permuted.deallocate(True)
            tt_v_out.deallocate(True)
            tt_v_out_permuted.deallocate(True)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler.start("wkv-b2-sequence-warmup")
    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)
    profiler.end("wkv-b2-sequence-warmup")

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("wkv-b2-sequence")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    profiler.end("wkv-b2-sequence")
    signpost("stop")

    return tt_v_out_permuted


@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize(
    "op_name, input_shape, output_shape",
    [
        (
            "wkv_b2_sequence",
            [1, 4, 128, 512],  # attn_out input: [1, num_heads_local, bsz_after_alltoall, kv_lora_rank]
            [1, 4, 128, 128],  # v_out output: [1, num_heads_local, bsz_after_alltoall, v_head_dim]
        ),
    ],
    ids=["wkv_b2_sequence"],
)
@pytest.mark.parametrize("warmup_iters, num_iters", [(10, 100)])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 4202496,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_wkv_b2_sequence_trace_mode(
    device,
    op_name,
    batch_size,
    input_shape,
    output_shape,
    trace_mode,
    warmup_iters,
    num_iters,
):
    """
    Test the complete _fwd_decode_wkv_b2 sequence from mla1d.py (lines 1331-1348).

    This test captures the entire operation sequence:
    1. to_memory_config: HEIGHT sharded -> L1 interleaved - 3.86 µs
    2. Permute: [1, 4, 128, 512] -> [1, 128, 4, 512] - 71 µs
    3. Linear (wkv_b2): [1, 128, 4, 512] -> [1, 128, 4, 128] - 142 µs
    4. Permute: [1, 128, 4, 128] -> [1, 4, 128, 128] - 8 µs

    Total expected: ~225 µs

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Multi-device shape with batch_size=128 (after all-to-all)
    - Single device test using 8x9 grid (72 cores) HEIGHT sharded
    """
    torch.manual_seed(0)

    num_heads = input_shape[1]
    kv_lora_rank = input_shape[3]
    v_head_dim = output_shape[3]

    logger.info(f"Running wkv_b2 sequence test: {op_name}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Output shape: {output_shape}")

    # Create golden reference tensor
    torch_attn_out = torch.randn(input_shape, dtype=torch.bfloat16)

    # Create wkv_b2 weight: [kv_lora_rank, v_head_dim]
    torch_wkv_b2_weight = torch.randn(kv_lora_rank, v_head_dim, dtype=torch.bfloat16)

    # Create HEIGHT_SHARDED memory config for input (matching flash_mla output)
    # For multi-device shape with batch=128 and 4 heads, total height = 1 * 4 * 128 = 512
    # Use 16 cores (4x4 grid) with shard height 32, matching model's pattern
    attn_out_core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}  # 4x4 grid = 16 cores
    )
    attn_out_shard_shape = [32, kv_lora_rank]
    attn_out_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=attn_out_shard_shape,
        core_grid=attn_out_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Convert input to ttnn with HEIGHT sharding
    tt_attn_out = ttnn.from_torch(
        torch_attn_out,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_attn_out = ttnn.to_memory_config(tt_attn_out, attn_out_sharded_mem_config)

    # Convert weight to ttnn
    # Weight must be in L1 to match standalone test performance (142 µs)
    # DRAM weight causes ~48 µs slowdown due to fetch latency
    tt_wkv_b2_weight = ttnn.from_torch(
        torch_wkv_b2_weight,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # pytorch reference
    torch_attn_out = torch_attn_out.permute(0, 2, 1, 3)
    torch_v_out = torch.matmul(torch_attn_out, torch_wkv_b2_weight)
    torch_v_out = torch_v_out.permute(0, 2, 1, 3)

    profiler = BenchmarkProfiler()

    try:
        if trace_mode:
            # Run sequence with trace
            tt_v_out = run_wkv_b2_sequence_with_trace(
                device,
                tt_attn_out,
                tt_wkv_b2_weight,
                num_iter=num_iters,
                warmup_iters=warmup_iters,
                profiler=profiler,
                num_heads=num_heads,
                batch_size=batch_size,
                kv_lora_rank=kv_lora_rank,
                v_head_dim=v_head_dim,
            )
        else:
            pytest.skip("Non-trace mode not implemented for this test")

        # Verify output shape
        logger.info("Verifying output shape")

        tt_v_output = tt_v_out.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

        logger.info(f"V output shape: {list(tt_v_output.shape)}")

        assert list(tt_v_output.shape) == output_shape, f"V shape mismatch: {list(tt_v_output.shape)} != {output_shape}"

        assert_with_pcc(torch_v_out, tt_v_output, 0.9999)
        logger.info("✓ wkv_b2 sequence trace mode test passed with correct output shape")

    finally:
        # Clean up
        pass
