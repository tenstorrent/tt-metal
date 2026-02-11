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
from models.common.utility_functions import comp_pcc, is_blackhole, is_watcher_enabled
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_chunks", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17])
@pytest.mark.parametrize(
    "k_chunk_size", [128]
)  # Chunk size 256 support can be added by consolidating tensix sem incs since cap is 15 but we have 16 tiles
@pytest.mark.parametrize("max_seq_len", [32 * 1024])  # 32k max sequence length per chip
def test_flash_mla_decode(device, batch_size, num_chunks, k_chunk_size, max_seq_len):
    """Test FlashMLADecode op."""
    if is_blackhole() and is_watcher_enabled():
        pytest.skip("Skipping test on Blackhole with watcher enabled, see issue #37631")

    # Calculate decode_position from num_chunks and k_chunk_size
    decode_position = num_chunks * k_chunk_size - 1
    torch.manual_seed(0)

    # Debug: Print optimal worker core for each DRAM bank from device API
    optimal_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    for bank_id, worker_core in enumerate(optimal_workers):
        logger.info(f"DRAM bank {bank_id} -> optimal worker core ({worker_core.x}, {worker_core.y})")

    # Use 128 heads and 16 heads per core to test 8 groups of heads
    # SDPA has bug with 8x32 tile size, so can't use 64 and 8 for now
    num_heads = 64  # TP=2, so 128 / 2 = 64 heads per device
    num_q_heads_per_core = 8
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
    kvpe_dim = kv_lora_rank + qk_rope_head_dim  # 576
    scale = qk_head_dim**-0.5

    logger.info(
        f"Testing FlashMLADecode with batch_size={batch_size}, k_chunk_size={k_chunk_size}, num_chunks={num_chunks}, "
        f"decode_position={decode_position}, max_seq_len={max_seq_len}"
    )

    # Create sharded memory configs for Q and output
    # Q heads sharded onto S1 block output cores (from op.py Grid.BLOCKS definition)
    # With 8 Q shards (128 heads / 16 per core = 8), each Q shard uses 1 core from S1
    tiny_tile = ttnn.Tile((num_q_heads_per_core, 32))

    # Q cores must match S1 output cores - use BLOCKS definition from op.py
    # BLOCKS order: S1, S2, S3, S4, S5, S6, S7, S8 - so S1 is at index 0
    s1_cores, _ = FlashMLADecode.ProgramConfig.grid.BLOCKS[0]
    q_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in s1_cores])
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
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,  # Use exact exp for higher precision
    )

    # Create KV cache (non-paged) based on max seq len
    logger.info(f"Creating KV cache with seq_len={max_seq_len}...")
    cache_shape = (batch_size, 1, max_seq_len, kvpe_dim)
    torch_cache = torch.randn(cache_shape, dtype=torch.bfloat16)

    # ND sharding with ROUND_ROBIN_1D distribution across DRAM banks (required)
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
    num_chunks_total = max_seq_len // program_config.k_chunk_size
    num_banks = len(grid.OPTIMAL_DRAM_BANK_ORDER)
    logger.info(
        f"KV cache: ND sharded (required), DRAM banks: {num_banks} (optimal order: {grid.OPTIMAL_DRAM_BANK_ORDER}), "
        f"chunks: {num_chunks_total}, shard_shape: [1, 1, {program_config.k_chunk_size}, {kvpe_dim}]"
    )

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
    compute_kernel_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Compute PyTorch reference using FlashMLADecode.golden_dummy (matches simplified compute kernel)
    logger.info("Computing PyTorch reference...")
    reference_output = FlashMLADecode.golden(
        q=torch_q,
        kv_cache=torch_cache,
        position_ids=position_ids,
        head_dim_v=kv_lora_rank,
        scale=scale,
    )

    # Run the op - stress test with multiple iterations
    num_iterations = 100
    first_output = None
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

        # Verify output shape: [1, batch_size, num_heads, kv_lora_rank]
        expected_shape = (1, batch_size, num_heads, kv_lora_rank)
        assert output_torch.shape == expected_shape, f"Expected shape {expected_shape}, got {output_torch.shape}"

        # Basic sanity check - output should not be all zeros or NaN
        assert not torch.isnan(output_torch).any(), f"Iteration {i}: Output contains NaN values"
        assert not torch.all(output_torch == 0), f"Iteration {i}: Output is all zeros"

        if i == 0:
            # First iteration: compare with golden reference and store result
            out_max_diff = torch.max(torch.abs(output_torch - reference_output)).item()
            out_mean_diff = torch.mean(torch.abs(output_torch - reference_output)).item()
            logger.info(f"Out Max absolute difference: {out_max_diff}")
            logger.info(f"Out Mean absolute difference: {out_mean_diff}")
            pcc_required = 0.998
            passing, pcc_message = comp_pcc(reference_output, output_torch, pcc_required)
            assert passing, f"Iteration {i}: PCC check failed vs golden: {pcc_message}"
            logger.info(f"    PCC vs golden: {pcc_message}")
            first_output = output_torch.clone()
        else:
            # Subsequent iterations: must be identical to first iteration
            assert torch.equal(output_torch, first_output), (
                f"Iteration {i}: Output differs from first iteration! "
                f"Max diff: {(output_torch - first_output).abs().max().item()}"
            )

    logger.info(f"  Completed {num_iterations} iterations!")
    logger.info("✓ FlashMLADecode test passed!")
