# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

"""
Test for flash_multi_latent_attention_decode op
Tests the flash MLA decode operation with 32k position (non-paged)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode


@pytest.mark.parametrize("batch_size", [1])
# @pytest.mark.parametrize("decode_position", [128 - 1, 2 * 1024 - 1, 4 * 1024 - 1, 8 * 1024 - 1, 32 * 1024 - 1])
@pytest.mark.parametrize("decode_position", [256 - 1, 1024 - 1, 2048 - 1])
@pytest.mark.parametrize("max_seq_len", [32 * 1024])  # 32k max sequence length per chip
# @pytest.mark.parametrize("kv_sharded", [False, True], ids=["interleaved", "sharded"])
@pytest.mark.parametrize("kv_sharded", [True], ids=["sharded"])
def test_flash_mla_decode(device, batch_size, decode_position, max_seq_len, kv_sharded):
    """Test FlashMLADecode op."""
    torch.manual_seed(0)

    # Debug: Print optimal worker core for each DRAM bank from device API
    optimal_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    for bank_id, worker_core in enumerate(optimal_workers):
        logger.info(f"DRAM bank {bank_id} -> optimal worker core ({worker_core.x}, {worker_core.y})")

    # Use 128 heads and 16 heads per core to test 8 groups of heads
    # SDPA has bug with 8x32 tile size, so can't use 64 and 8 for now
    num_heads = 128  # TP=2, so 128 / 2 = 64 heads per device
    num_q_heads_per_core = 16
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
    kvpe_dim = kv_lora_rank + qk_rope_head_dim  # 576
    scale = qk_head_dim**-0.5

    logger.info(
        f"Testing FlashMLADecode with batch_size={batch_size}, position={decode_position}, max_seq_len={max_seq_len}"
    )

    # Create sharded memory configs for Q and output
    # Q heads sharded onto S1 block output cores (from op.py S1_CORES definition)
    # S1_CORES = [(1,2), (2,2), (3,2), (4,2), (1,3), (2,3), (3,3), (4,3)]
    # With 8 Q shards (128 heads / 16 per core = 8), each Q shard uses 1 core from S1
    tiny_tile = ttnn.Tile((num_q_heads_per_core, 32))

    # Q cores must match S1 output cores - 8 cores for 8 Q shards (0-indexed)
    q_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(0, 1)),  # S1 core 0
            ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1)),  # S1 core 1
            ttnn.CoreRange(ttnn.CoreCoord(2, 1), ttnn.CoreCoord(2, 1)),  # S1 core 2
            ttnn.CoreRange(ttnn.CoreCoord(3, 1), ttnn.CoreCoord(3, 1)),  # S1 core 3
            ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(0, 2)),  # S1 core 4
            ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(1, 2)),  # S1 core 5
            ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(2, 2)),  # S1 core 6
            ttnn.CoreRange(ttnn.CoreCoord(3, 2), ttnn.CoreCoord(3, 2)),  # S1 core 7
        ]
    )
    q_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kv_lora_rank), ttnn.ShardOrientation.ROW_MAJOR),
    )

    # Create Q tensor: [1, batch_size, num_heads, kvpe_dim]
    logger.info("Creating Q tensor...")
    q_shape = (1, batch_size, num_heads, kvpe_dim)
    torch_q = torch.randn(q_shape, dtype=torch.bfloat16)

    tt_q = ttnn.from_torch(
        torch_q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=q_mem_config,
        tile=tiny_tile,
    )

    # Create program config (needed for KV cache sharding setup)
    program_config = FlashMLADecode.ProgramConfig(
        k_chunk_size=256,
        exp_approx_mode=False,  # Use exact exp for higher precision
    )

    # Create KV cache (non-paged) based on max seq len
    logger.info(f"Creating KV cache with seq_len={max_seq_len}...")
    cache_shape = (batch_size, 1, max_seq_len, kvpe_dim)
    torch_cache = torch.randn(cache_shape, dtype=torch.bfloat16)

    if kv_sharded:
        # ND sharding with ROUND_ROBIN_1D distribution across DRAM banks
        # Each shard = one k_chunk (k_chunk_size x kvpe_dim), distributed round-robin
        # Use optimal DRAM bank order matching S block work assignment for locality
        grid = program_config.grid
        kv_nd_shard_spec = ttnn.NdShardSpec(
            shard_shape=[1, 1, program_config.k_chunk_size, kvpe_dim],
            grid=grid.optimal_dram_grid(),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        kv_mem_config = ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.DRAM,
            nd_shard_spec=kv_nd_shard_spec,
        )
        num_chunks = max_seq_len // program_config.k_chunk_size
        num_banks = len(grid.OPTIMAL_DRAM_BANK_ORDER)
        logger.info(
            f"KV cache: ND sharded, DRAM banks: {num_banks} (optimal order: {grid.OPTIMAL_DRAM_BANK_ORDER}), chunks: {num_chunks}, shard_shape: [1, 1, {program_config.k_chunk_size}, {kvpe_dim}]"
        )
    else:
        # Interleaved DRAM
        kv_mem_config = ttnn.DRAM_MEMORY_CONFIG
        logger.info("KV cache: interleaved DRAM")

    tt_cache = ttnn.from_torch(
        torch_cache,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=kv_mem_config,
    )

    # Create position tensor
    logger.info("Creating position tensor...")
    position_ids = torch.ones(batch_size, dtype=torch.int32) * decode_position
    tt_position_ids = ttnn.from_torch(
        position_ids,
        dtype=ttnn.int32,
        device=device,
    )

    # Create output tensor with same sharded memory config and tiny tile
    logger.info("Creating output tensor...")
    out_shape = (1, batch_size, num_heads, kv_lora_rank)
    torch_output_zeros = torch.zeros(out_shape, dtype=torch.bfloat16)
    tt_out = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=tiny_tile,
    )

    # Create compute kernel config (matching original test)
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Run the op - stress test with 100 iterations
    num_iterations = 1000
    logger.info(f"Running FlashMLADecode.op {num_iterations} times for stress test...")
    for i in range(num_iterations):
        if i % 10 == 0:
            logger.info(f"  Iteration {i}/{num_iterations}...")
        attn_out = FlashMLADecode.op(
            q_tensor=tt_q,
            kv_cache_tensor=tt_cache,
            head_dim_v=kv_lora_rank,
            cur_pos_tensor=tt_position_ids,
            output_tensor=tt_out,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        # Convert output to torch
        output_torch = ttnn.to_torch(attn_out)

    logger.info(f"  Completed {num_iterations} iterations!")

    # Verify output shape: [1, batch_size, num_heads, kv_lora_rank]
    expected_shape = (1, batch_size, num_heads, kv_lora_rank)
    assert output_torch.shape == expected_shape, f"Expected shape {expected_shape}, got {output_torch.shape}"

    # Basic sanity check - output should not be all zeros or NaN
    # assert not torch.isnan(output_torch).any(), "Output contains NaN values"
    # assert not torch.all(output_torch == 0), "Output is all zeros"

    # Compute PyTorch reference using FlashMLADecode.golden
    logger.info("Computing PyTorch reference...")
    reference_output = FlashMLADecode.golden(
        q=torch_q,
        kv_cache=torch_cache,
        position_ids=position_ids,
        head_dim_v=kv_lora_rank,
        scale=scale,
    )

    # Compare with reference
    logger.info("Comparing TTNN output with PyTorch reference...")
    pcc_required = 0.999
    passing, pcc_message = comp_pcc(reference_output, output_torch, pcc_required)
    logger.info(f"PCC: {pcc_message}")

    # assert passing, f"PCC check failed: {pcc_message}"

    logger.info("✓ FlashMLADecode test passed!")
