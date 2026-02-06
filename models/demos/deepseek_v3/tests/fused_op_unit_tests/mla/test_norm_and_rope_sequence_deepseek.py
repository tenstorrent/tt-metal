# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3.tests.fused_op_unit_tests.mla.test_rope_deepseek import apply_rotary_pos_emb_torch
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_rope_tensors(device, qk_rope_head_dim=64, max_seq_len=2048, bsz=32):
    """Create RoPE tensors for testing.

    RoPE requires HEIGHT sharded cos/sin/trans matrices matching the input tensor sharding.
    For decode mode with batch size 32 on 32 cores (4x8 grid), each core gets one batch element.
    Matching the model's pattern from rope.py lines 101-115.
    """
    # Create cos and sin matrices for RoPE
    # Shape: [1, bsz, 1, qk_rope_head_dim] matching the actual decode mode shape
    cos_matrix = torch.randn(1, bsz, 1, qk_rope_head_dim, dtype=torch.bfloat16) * 0.1
    sin_matrix = torch.randn(1, bsz, 1, qk_rope_head_dim, dtype=torch.bfloat16) * 0.1

    # Transformation matrix for RoPE - repeat across batch dimension
    # Matching rope.py lines 94-100
    trans_matrix = torch.zeros(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE, dtype=torch.bfloat16)
    trans_matrix[..., torch.arange(0, ttnn.TILE_SIZE, 2), torch.arange(1, ttnn.TILE_SIZE, 2)] = 1
    trans_matrix[..., torch.arange(1, ttnn.TILE_SIZE, 2), torch.arange(0, ttnn.TILE_SIZE, 2)] = -1
    trans_matrix = trans_matrix.repeat(1, 1, bsz, 1)

    # Create HEIGHT sharded memory config matching the kv_rope input sharding
    # 32 cores in 4x8 grid, each gets [32, 64] shard for cos/sin
    rope_core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))}  # 4x8 grid = 32 cores
    )
    rope_shard_shape = [32, 64]
    rope_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=rope_shard_shape,
        core_grid=rope_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Trans matrix uses [32, 32] shard shape (one tile)
    trans_shard_shape = [32, 32]
    trans_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=trans_shard_shape,
        core_grid=rope_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Convert to ttnn with HEIGHT sharding directly in from_torch (matching rope.py pattern)
    tt_cos = ttnn.from_torch(
        cos_matrix,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=rope_sharded_mem_config,
    )

    tt_sin = ttnn.from_torch(
        sin_matrix,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=rope_sharded_mem_config,
    )

    tt_trans = ttnn.from_torch(
        trans_matrix,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=trans_sharded_mem_config,
    )

    return {
        "cos_matrix": tt_cos,
        "sin_matrix": tt_sin,
        "trans_matrix": tt_trans,
        "torch_cos": cos_matrix,
        "torch_sin": sin_matrix,
        "torch_trans": trans_matrix,
    }


def run_norm_and_rope_sequence_with_trace(
    device,
    tt_q,
    tt_kv_nope,
    tt_kv_rope,
    rope_tensors,
    q_norm_gamma,
    kv_norm_gamma,
    num_iter=100,
    warmup_iters=10,
    profiler=BenchmarkProfiler(),
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    bsz=32,
):
    """Run norm_and_rope sequence with trace mode for performance measurement."""
    # Compile Run
    logger.info("Compiling norm_and_rope sequence")

    # Q Norm
    tt_q_out = ttnn.rms_norm(tt_q, weight=q_norm_gamma, epsilon=1e-6)

    # KV Norm
    tt_kv_nope_out = ttnn.rms_norm(tt_kv_nope, weight=kv_norm_gamma, epsilon=1e-6)
    tt_kv_nope_out = ttnn.to_memory_config(tt_kv_nope_out, memory_config=ttnn.L1_MEMORY_CONFIG)

    # KV RoPE
    tt_kv_rope_out = ttnn.permute(tt_kv_rope, (0, 2, 1, 3))

    # Reshard for RoPE - HEIGHT sharded with 32 cores (4x8 grid) matching model config
    # After permute, shape is [1, 32, 1, 64] where dim 1 (batch) is distributed across 32 cores
    # Each core gets [32, 64] shard (padded height of dim 2, full width of dim 3)
    rope_core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))}  # 4x8 grid = 32 cores
    )
    rope_shard_shape = [32, 64]
    rope_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=rope_shard_shape,
        core_grid=rope_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_kv_rope_out = ttnn.to_memory_config(tt_kv_rope_out, rope_sharded_mem_config)

    # Apply RoPE
    tt_kv_rope_out = ttnn.experimental.rotary_embedding_llama(
        tt_kv_rope_out,
        rope_tensors["cos_matrix"],
        rope_tensors["sin_matrix"],
        rope_tensors["trans_matrix"],
        is_decode_mode=True,
    )

    # Reshard back to interleaved
    tt_kv_rope_out = ttnn.to_memory_config(tt_kv_rope_out, ttnn.L1_MEMORY_CONFIG)
    tt_kv_rope_out = ttnn.permute(tt_kv_rope_out, (0, 2, 1, 3))

    # Concat
    tt_kvpe = ttnn.concat([tt_kv_nope_out, tt_kv_rope_out], dim=-1)

    # Pad
    tt_kvpe = ttnn.pad(tt_kvpe, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)

    # Permute
    tt_kvpe = ttnn.permute(tt_kvpe, (0, 2, 1, 3))

    # Note: mesh_partition is multi-device only, skip for single device test
    # For single device, shape after permute is [1, 32, 32, 576] (no mesh_partition)
    # The model uses kvpe_num_cores = kvpe_shape[1] = USERS_PER_ROW / mesh_devices
    # For single device: 32 / 1 = 32 cores (matching batch size)

    # Final reshard to height sharded - 32 cores (4x8 grid) matching the batch dimension
    kvpe_core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))}  # 4x8 grid = 32 cores
    )
    kvpe_shard_shape = [32, 576]
    kvpe_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=kvpe_shard_shape,
        core_grid=kvpe_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_kvpe = ttnn.to_memory_config(tt_kvpe, kvpe_sharded_mem_config)

    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        # Q Norm
        tt_q_out = ttnn.rms_norm(tt_q, weight=q_norm_gamma, epsilon=1e-6)

        # KV Norm
        tt_kv_nope_out = ttnn.rms_norm(tt_kv_nope, weight=kv_norm_gamma, epsilon=1e-6)
        tt_kv_nope_out = ttnn.to_memory_config(tt_kv_nope_out, memory_config=ttnn.L1_MEMORY_CONFIG)

        # KV RoPE
        tt_kv_rope_out = ttnn.permute(tt_kv_rope, (0, 2, 1, 3))
        tt_kv_rope_out = ttnn.to_memory_config(tt_kv_rope_out, rope_sharded_mem_config)

        tt_kv_rope_out = ttnn.experimental.rotary_embedding_llama(
            tt_kv_rope_out,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=True,
        )

        tt_kv_rope_out = ttnn.to_memory_config(tt_kv_rope_out, ttnn.L1_MEMORY_CONFIG)
        tt_kv_rope_out = ttnn.permute(tt_kv_rope_out, (0, 2, 1, 3))

        # Concat
        tt_kvpe = ttnn.concat([tt_kv_nope_out, tt_kv_rope_out], dim=-1)

        # Pad
        tt_kvpe = ttnn.pad(tt_kvpe, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)

        # Permute
        tt_kvpe = ttnn.permute(tt_kvpe, (0, 2, 1, 3))

        # Final reshard
        tt_kvpe = ttnn.to_memory_config(tt_kvpe, kvpe_sharded_mem_config)

        tt_q_out.deallocate(True)
        tt_kv_nope_out.deallocate(True)
        tt_kv_rope_out.deallocate(True)
        tt_kvpe.deallocate(True)
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iter} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iter):
        # Q Norm
        tt_q_out = ttnn.rms_norm(tt_q, weight=q_norm_gamma, epsilon=1e-6)

        # KV Norm
        tt_kv_nope_out = ttnn.rms_norm(tt_kv_nope, weight=kv_norm_gamma, epsilon=1e-6)
        tt_kv_nope_out = ttnn.to_memory_config(tt_kv_nope_out, memory_config=ttnn.L1_MEMORY_CONFIG)

        # KV RoPE
        tt_kv_rope_out = ttnn.permute(tt_kv_rope, (0, 2, 1, 3))
        tt_kv_rope_out = ttnn.to_memory_config(tt_kv_rope_out, rope_sharded_mem_config)

        tt_kv_rope_out = ttnn.experimental.rotary_embedding_llama(
            tt_kv_rope_out,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=True,
        )

        tt_kv_rope_out = ttnn.to_memory_config(tt_kv_rope_out, ttnn.L1_MEMORY_CONFIG)
        tt_kv_rope_out = ttnn.permute(tt_kv_rope_out, (0, 2, 1, 3))

        # Concat
        tt_kvpe = ttnn.concat([tt_kv_nope_out, tt_kv_rope_out], dim=-1)

        # Pad
        tt_kvpe = ttnn.pad(tt_kvpe, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)

        # Permute
        tt_kvpe = ttnn.permute(tt_kvpe, (0, 2, 1, 3))

        # Final reshard
        tt_kvpe = ttnn.to_memory_config(tt_kvpe, kvpe_sharded_mem_config)

        if i != num_iter - 1:
            tt_q_out.deallocate(True)
            tt_kv_nope_out.deallocate(True)
            tt_kv_rope_out.deallocate(True)
            tt_kvpe.deallocate(True)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler.start("norm-and-rope-sequence-warmup")
    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)
    profiler.end("norm-and-rope-sequence-warmup")

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("norm-and-rope-sequence")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    profiler.end("norm-and-rope-sequence")
    signpost("stop")

    return tt_q_out, tt_kvpe


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "op_name, q_input_shape, kv_nope_input_shape, kv_rope_input_shape, q_output_shape, kvpe_output_shape",
    [
        (
            "norm_and_rope_sequence",
            [1, 1, 32, 1536],  # Q input: [1, 1, bsz, q_lora_rank]
            [1, 1, 32, 512],  # KV nope input: [1, 1, bsz, kv_lora_rank]
            [1, 1, 32, 64],  # KV rope input: [1, 1, bsz, qk_rope_head_dim]
            [1, 1, 32, 1536],  # Q output (after norm): [1, 1, bsz, q_lora_rank]
            [1, 32, 32, 576],  # KVPE output: [1, bsz_padded, bsz_padded, kv_lora_rank + qk_rope_head_dim]
        ),
    ],
    ids=["norm_and_rope_sequence"],
)
@pytest.mark.parametrize("warmup_iters, num_iters", [(10, 100)])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 4218900,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_norm_and_rope_sequence_trace_mode(
    device,
    op_name,
    batch_size,
    q_input_shape,
    kv_nope_input_shape,
    kv_rope_input_shape,
    q_output_shape,
    kvpe_output_shape,
    trace_mode,
    warmup_iters,
    num_iters,
):
    """
    Test the complete _fwd_decode_norm_and_rope sequence from mla1d.py (lines 1136-1210).

    This test captures the entire operation sequence:
    1. RMS Norm Q: [1, 1, 32, 1536] WIDTH_SHARDED 8x2 [32, 96] - 7.34 µs
    2. RMS Norm KV: [1, 1, 32, 512] WIDTH_SHARDED 8x2 [32, 32] - 6.68 µs
    3. to_memory_config (kv_nope): to L1 interleaved - 0.75 µs
    4. Permute kv_rope: [1, 1, 32, 64] -> [1, 32, 1, 64] - 7.6 µs
    5. Reshard kv_rope: 4x8 [32, 64] HEIGHT_SHARDED - 1.04 µs
    6. RoPE: rotary_embedding_llama - 4.5 µs
    7. Reshard kv_rope out: to L1 interleaved - 1.19 µs
    8. Permute kv_rope back: [1, 32, 1, 64] -> [1, 1, 32, 64] - 1.5 µs
    9. Concat: [kv_nope, kv_rope] -> [1, 1, 32, 576] - 1.36 µs
    10. Pad: [1, 1, 32, 576] -> [1, 32, 32, 576] - 45 µs
    11. Permute: (0, 2, 1, 3) - 24 µs
    12. Reshard to HEIGHT_SHARDED: 1x4 [32, 576] - 2.8 µs

    Total expected: 103.76 µs (excluding mesh_partition which is multi-device only)

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Single device test
    """
    torch.manual_seed(0)

    # Create input tensors
    logger.info(f"Running norm_and_rope sequence test: {op_name}")
    logger.info(f"Q input shape: {q_input_shape}")
    logger.info(f"KV nope input shape: {kv_nope_input_shape}")
    logger.info(f"KV rope input shape: {kv_rope_input_shape}")
    logger.info(f"Q output shape: {q_output_shape}")
    logger.info(f"KVPE output shape: {kvpe_output_shape}")

    # Create golden reference tensors
    torch_q = torch.randn(q_input_shape, dtype=torch.bfloat16)
    torch_kv_nope = torch.randn(kv_nope_input_shape, dtype=torch.bfloat16)
    torch_kv_rope = torch.randn(kv_rope_input_shape, dtype=torch.bfloat16)

    # Create RMS norm gammas
    q_norm_gamma = torch.ones(q_input_shape[-1], dtype=torch.bfloat16)
    kv_norm_gamma = torch.ones(kv_nope_input_shape[-1], dtype=torch.bfloat16)

    # Q: WIDTH_SHARDED 8x2 [32, 96]
    q_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 7))})  # 8x2 grid
    q_shard_shape = [32, 96]
    q_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=q_shard_shape,
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # KV: WIDTH_SHARDED 8x2 [32, 32]
    kv_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 7))})  # 8x2 grid
    kv_shard_shape = [32, 32]
    kv_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=kv_shard_shape,
        core_grid=kv_core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # KV rope: starts as interleaved (as per model: "interleaved since it goes through permute/reshard")
    # Sharding happens after the first permute

    # Convert inputs to ttnn
    tt_q = ttnn.from_torch(
        torch_q,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_q = ttnn.to_memory_config(tt_q, q_sharded_mem_config)

    tt_kv_nope = ttnn.from_torch(
        torch_kv_nope,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_kv_nope = ttnn.to_memory_config(tt_kv_nope, kv_sharded_mem_config)

    tt_kv_rope = ttnn.from_torch(
        torch_kv_rope,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    # Keep kv_rope as interleaved - sharding happens after first permute

    # Convert norm gammas to ttnn
    tt_q_norm_gamma = ttnn.from_torch(
        q_norm_gamma,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_kv_norm_gamma = ttnn.from_torch(
        kv_norm_gamma,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create RoPE tensors
    rope_tensors = create_rope_tensors(device, qk_rope_head_dim=64, bsz=batch_size)

    # pytorch reference
    torch_variance = torch_q.pow(2).mean(-1, keepdim=True)
    torch_q *= torch.rsqrt(torch_variance + 1e-6) * q_norm_gamma
    torch_q_out = torch_q
    torch_variance = torch_kv_nope.pow(2).mean(-1, keepdim=True)
    torch_kv_nope *= torch.rsqrt(torch_variance + 1e-6) * kv_norm_gamma
    torch_kv_nope_out = torch_kv_nope
    torch_kv_rope_out = torch_kv_rope.permute(0, 2, 1, 3)
    torch_trans_mat_2d = rope_tensors["torch_trans"][:, :, 0:32, :]
    torch_kv_rope_out = apply_rotary_pos_emb_torch(
        torch_kv_rope_out, rope_tensors["torch_cos"], rope_tensors["torch_sin"], torch_trans_mat_2d
    )
    torch_kv_rope_out = torch_kv_rope_out.permute(0, 2, 1, 3)
    torch_kvpe = torch.cat([torch_kv_nope_out, torch_kv_rope_out], dim=-1)
    # Match ttnn.pad(tt_kvpe, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0) — pad dim 1 on the right
    torch_kvpe = torch.nn.functional.pad(torch_kvpe, (0, 0, 0, 0, 0, ttnn.TILE_SIZE - 1, 0, 0), "constant", 0)
    torch_kvpe = torch_kvpe.permute(0, 2, 1, 3)

    profiler = BenchmarkProfiler()

    try:
        if trace_mode:
            # Run sequence with trace
            tt_q_out, tt_kvpe = run_norm_and_rope_sequence_with_trace(
                device,
                tt_q,
                tt_kv_nope,
                tt_kv_rope,
                rope_tensors,
                tt_q_norm_gamma,
                tt_kv_norm_gamma,
                num_iter=num_iters,
                warmup_iters=warmup_iters,
                profiler=profiler,
                q_lora_rank=1536,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                bsz=batch_size,
            )
        else:
            pytest.skip("Non-trace mode not implemented for this test")

        # Verify output shapes
        logger.info("Verifying output shapes")

        tt_q_output = tt_q_out.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        tt_kvpe_output = tt_kvpe.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

        logger.info(f"Q output shape: {list(tt_q_output.shape)}")
        logger.info(f"KVPE output shape: {list(tt_kvpe_output.shape)}")

        assert (
            list(tt_q_output.shape) == q_output_shape
        ), f"Q shape mismatch: {list(tt_q_output.shape)} != {q_output_shape}"
        assert (
            list(tt_kvpe_output.shape) == kvpe_output_shape
        ), f"KVPE shape mismatch: {list(tt_kvpe_output.shape)} != {kvpe_output_shape}"
        assert_with_pcc(torch_q_out, tt_q_output, 0.9999)
        assert_with_pcc(torch_kvpe, tt_kvpe_output, 0.9999)

        logger.info("✓ norm_and_rope sequence trace mode test passed with correct output shapes")

    finally:
        # Clean up
        pass
