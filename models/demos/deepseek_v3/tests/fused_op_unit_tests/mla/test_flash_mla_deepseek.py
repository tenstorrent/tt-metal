# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler


def create_paged_kvpe_cache(device, num_users, max_seq_len, head_dim, num_blocks, block_size):
    """Create a paged KVPE cache for testing."""
    # Cache is organized as [num_users, 1, num_blocks * block_size, head_dim]
    cache_shape = (num_users, 1, num_blocks * block_size, head_dim)
    cache = torch.randn(cache_shape, dtype=torch.bfloat16) * 0.1

    # Convert to ttnn with DRAM memory (matching model configuration)
    tt_cache = ttnn.from_torch(
        cache,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return tt_cache, cache


def create_page_table(device, num_users, num_blocks):
    """Create a page table mapping logical to physical blocks."""
    # Page table shape: [num_users, max_num_blocks_per_user]
    # For simplicity, use identity mapping
    blocks_per_user = num_blocks // num_users
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(num_users, blocks_per_user)

    # Convert to ttnn
    tt_page_table = ttnn.from_torch(
        page_table,
        dtype=ttnn.int32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return tt_page_table, page_table


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "op_name, q_shape, cache_params, output_shape, shard_shape, num_cores",
    [
        (
            "flash_mla_decode",
            [1, 4, 128, 576],  # Q shape after all-to-all: [1, bsz_local, num_heads, kv_lora_rank + qk_rope_head_dim]
            {
                "num_users": 4,  # Per device: 32 users / 8 devices = 4
                "max_seq_len": 2048,
                "head_dim": 576,  # kv_lora_rank + qk_rope_head_dim
                "num_blocks": 64,
                "block_size": 32,
                "kv_lora_rank": 512,
            },
            [1, 4, 128, 512],  # Output: [1, bsz_local, num_heads, kv_lora_rank]
            [32, 576],  # HEIGHT_SHARDED shard shape for input Q
            64,  # (32/8) * 16 = 4 * 16 = 64 cores
        ),
    ],
    ids=["flash_mla_decode"],
)
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 4112384,  # Larger trace region for flash attention
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_flash_mla_trace_mode(
    device,
    batch_size,
    op_name,
    q_shape,
    cache_params,
    output_shape,
    shard_shape,
    num_cores,
    warmup_iters,
    num_iters,
):
    """
    Test the paged_flash_multi_latent_attention_decode operation from mla1d.py with trace mode.

    This operation performs paged flash attention for MLA (Multi-head Latent Attention):
    - flash_mla_decode (line 1287): Paged flash MLA decode operation
      Input Q: [1, 4, 128, 576] height sharded
      Output: [1, 4, 128, 512] height sharded

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - HEIGHT_SHARDED memory layout
    - k_chunk_size: 128
    - Scale: (qk_head_dim)**-0.5
    """
    torch.manual_seed(0)

    num_users = cache_params["num_users"]
    max_seq_len = cache_params["max_seq_len"]
    head_dim = cache_params["head_dim"]
    num_blocks = cache_params["num_blocks"]
    block_size = cache_params["block_size"]
    kv_lora_rank = cache_params["kv_lora_rank"]

    # Create Q tensor
    torch_q = torch.randn(q_shape, dtype=torch.bfloat16) * 0.1

    # Create paged KVPE cache
    tt_kvpe_cache, torch_kvpe_cache = create_paged_kvpe_cache(
        device, num_users, max_seq_len, head_dim, num_blocks, block_size
    )

    # Create page table
    tt_page_table, torch_page_table = create_page_table(device, num_users, num_blocks)

    # Create position indices (current position for each user)
    cache_size = num_blocks * block_size
    position_idxs = torch.randint(1, cache_size - 1, (num_users,), dtype=torch.int32)

    # Convert Q to ttnn with L1 interleaved first
    tt_q = ttnn.from_torch(
        torch_q,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Create HEIGHT_SHARDED memory config for Q
    grid_size = device.compute_with_storage_grid_size()
    q_core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)

    q_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    tt_q = ttnn.to_memory_config(tt_q, q_sharded_mem_config)

    # Convert position indices to ttnn
    tt_position_idxs = ttnn.from_torch(
        position_idxs,
        dtype=ttnn.int32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Flash MLA configuration matching mla1d.py decode config
    qk_nope_head_dim = 192
    qk_rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 256
    scale = qk_head_dim**-0.5

    # Output memory config - height sharded with kv_lora_rank width
    output_shard_shape = [shard_shape[0], kv_lora_rank]  # [32, 512]
    output_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=output_shard_shape,
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Program config
    q_chunk_size = 0  # Unused in decode mode
    k_chunk_size = 128
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    # Compute kernel config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Compile run
    logger.info(f"Compiling flash MLA operation: {op_name}")
    logger.info(f"  Q shape: {q_shape}")
    logger.info(f"  Cache shape: [{num_users}, 1, {num_blocks * block_size}, {head_dim}]")
    logger.info(f"  Output shape: {output_shape}")
    logger.info(f"  Q shard shape: {shard_shape}")
    logger.info(f"  Output shard shape: {output_shard_shape}")
    logger.info(f"  Num cores: {num_cores}")
    logger.info(f"  Scale: {scale}")

    tt_output = ttnn.transformer.paged_flash_multi_latent_attention_decode(
        tt_q,
        tt_kvpe_cache,
        page_table_tensor=tt_page_table,
        cur_pos_tensor=tt_position_idxs,
        head_dim_v=kv_lora_rank,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=output_sharded_mem_config,
    )
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            tt_q,
            tt_kvpe_cache,
            page_table_tensor=tt_page_table,
            cur_pos_tensor=tt_position_idxs,
            head_dim_v=kv_lora_rank,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=output_sharded_mem_config,
        )
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            tt_q,
            tt_kvpe_cache,
            page_table_tensor=tt_page_table,
            cur_pos_tensor=tt_position_idxs,
            head_dim_v=kv_lora_rank,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=output_sharded_mem_config,
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

    # Verify output shape
    tt_output = ttnn.from_device(tt_output)
    torch_output = ttnn.to_torch(tt_output)

    assert list(torch_output.shape) == output_shape, f"Shape mismatch: {list(torch_output.shape)} != {output_shape}"

    # Note: Full PCC check would require a reference PyTorch flash attention implementation
    # For now, we verify the operation runs successfully and produces output of correct shape
    logger.info(f"✓ Trace mode {op_name} test passed with correct output shape")
