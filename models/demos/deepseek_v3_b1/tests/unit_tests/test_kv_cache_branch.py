# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN KV Cache Branch Test
Tests KV cache branch fused operation
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.tt.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_b1.fused_ops.kv_cache_branch.op import KVCacheBranch
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode
from models.demos.deepseek_v3_b1.micro_ops.kv_cache_update.op import KVCacheUpdate


@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("position_id", [0, 1, 5, 7])
def test_kv_cache_branch(device, epsilon, use_fp32, position_id):
    """Test TTNN KV cache branch fused operation"""

    max_seq_len = 8192
    batch = 1

    # Input tensor shapes
    input_shape = (1, 7168)
    W_dkv_rope_shape = (7168, 576)

    # Rope config
    rope_head_dim = 64
    rope_num_heads = 1

    # Create input PyTorch tensors
    torch.manual_seed(position_id)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_W_dkv_rope = torch.randn(W_dkv_rope_shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn((1, 512), dtype=torch.bfloat16)

    # ROPE
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_head_dim, 2, dtype=torch.float32) / rope_head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)

    # Meta-style: stack [cos(t), cos(t)] interleaved
    torch_cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)  # [max_seq_len, head_dim]
    torch_sin = torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2)  # [max_seq_len, head_dim]
    position_ids = torch.tensor([position_id])  # positions 0, 1, 2, ...
    position_ids_expanded = position_ids.unsqueeze(1)  # [batch, 1]

    logger.info(f"Done creating torch tensors.")

    # TT setup
    # Grid configuration
    spec_start_offset = (0, 8)  # Offset for the operation grid
    spec_grid = (9, 2)  # Grid dimensions for the operation

    spec_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(spec_start_offset[0], spec_start_offset[1]),
                ttnn.CoreCoord(spec_grid[0] + spec_start_offset[0] - 1, spec_grid[1] + spec_start_offset[1] - 1),
            )
        }
    )
    rms_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(spec_start_offset[0], spec_start_offset[1]),
                ttnn.CoreCoord(spec_start_offset[0], spec_start_offset[1]),
            )
        }
    )
    rope_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(8 + spec_start_offset[0], spec_start_offset[1]),
                ttnn.CoreCoord(8 + spec_start_offset[0], 1 + spec_start_offset[1]),
            )
        }
    )

    # Validate grid fits within device
    device_grid_size = device.compute_with_storage_grid_size()
    assert (
        spec_grid[0] + spec_start_offset[0] <= device_grid_size.x
    ), f"spec_grid.x ({spec_grid[0]}) + spec_start_offset.x ({spec_start_offset[0]}) must be <= device_grid_size.x ({device_grid_size.x})"
    assert (
        spec_grid[1] + spec_start_offset[1] <= device_grid_size.y
    ), f"spec_grid.y ({spec_grid[1]}) + spec_start_offset.y ({spec_start_offset[1]}) must be <= device_grid_size.y ({device_grid_size.y})"

    tile = ttnn.Tile([1, 32])

    num_cores = spec_crs.num_cores()
    torch_input_replicated = torch_input.expand(num_cores, -1).contiguous()

    input_shard_spec = ttnn.ShardSpec(
        spec_crs,
        (1, 7168),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    # Create input tensor sharded on spec grid
    ttnn_input = ttnn.from_torch(
        torch_input_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile,
    )

    # DKV Matmul
    shard_width = W_dkv_rope_shape[1] // num_cores
    assert (
        W_dkv_rope_shape[1] % num_cores == 0
    ), f"W_dkv_rope_shape[1] ({W_dkv_rope_shape[1]}) must be divisible by grid size ({num_cores})"
    assert shard_width == 32, f"Expected shard width of 32, got {shard_width}"

    W_dkv_rope_shard_shape = (W_dkv_rope_shape[0], shard_width)
    W_dkv_rope_shard_spec = ttnn.ShardSpec(
        spec_crs,
        W_dkv_rope_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    W_dkv_rope_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, W_dkv_rope_shard_spec
    )

    # Shuffle torch_W_dkv_rope to match the krope/nope core setup
    # Row 0: N N N N N N N N R  (cores 0-7 are N, core 8 is R)
    # Row 1: N N N N N N N N R  (cores 9-16 are N, core 17 is R)
    # We want the last 64 columns (512-575) to be on R cores (8 and 17)
    # Original shard order: [0, 1, ..., 7, 8, 9, ..., 15, 16, 17]
    # New shard order:      [0, 1, ..., 7, 16, 8, 9, ..., 15, 17]
    # This puts shard 16 (cols 512-543) at position 8 (R core in row 0)
    num_shards = 18
    shard_width = 32
    new_shard_order = [0, 1, 2, 3, 4, 5, 6, 7, 16, 8, 9, 10, 11, 12, 13, 14, 15, 17]
    torch_W_dkv_rope_shards = torch_W_dkv_rope.reshape(7168, num_shards, shard_width)
    torch_W_dkv_rope_shuffled = torch_W_dkv_rope_shards[:, new_shard_order, :].reshape(7168, 576)

    ttnn_W_dkv_rope = ttnn.from_torch(
        torch_W_dkv_rope_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=W_dkv_rope_mem_config,
    )

    # GAMMA
    gamma_shard_spec = ttnn.ShardSpec(
        rms_crs,
        (1, 512),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gamma_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gamma_shard_spec)

    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gamma_mem_config,
        tile=tile,
    )

    # ROPE
    # Cos/sin indexed by position: [1, batch, 1, head_dim]
    # Shape stays [1, 1, 1, head_dim] - broadcast multiply will use row 0
    rope_tile = ttnn.Tile((rope_num_heads, ttnn.TILE_SIZE))
    trans_tile = ttnn.Tile((ttnn.TILE_SIZE, ttnn.TILE_SIZE))
    cos_selected = torch_cos[position_ids].unsqueeze(0).unsqueeze(2)
    sin_selected = torch_sin[position_ids].unsqueeze(0).unsqueeze(2)

    # Use same tiny tile as input - data in row 0, rows 1+ are padding
    cos_sin_shard_spec = ttnn.ShardSpec(
        rope_crs,
        (rope_num_heads, rope_head_dim // 2),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    cos_sin_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, cos_sin_shard_spec
    )

    tt_cos = ttnn.from_torch(
        cos_selected,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cos_sin_mem_config,
        tile=rope_tile,
    )
    tt_sin = ttnn.from_torch(
        sin_selected,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cos_sin_mem_config,
        tile=rope_tile,
    )

    # Transformation matrix - replicate 32x32 for each rope core
    trans_mat = get_rot_transformation_mat()  # (1, 1, 32, 32)
    num_rope_cores = rope_crs.num_cores()
    trans_mat_replicated = trans_mat.repeat(1, 1, batch, num_rope_cores)  # (1, 1, batch*32, 32*num_rope_cores)
    trans_shard_spec = ttnn.ShardSpec(
        rope_crs,
        (ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    trans_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, trans_shard_spec)

    tt_trans_replicated = ttnn.from_torch(
        trans_mat_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=trans_mem_config,
        tile=trans_tile,
    )

    # Create output tensor
    output_shape = (1, 512)
    output_shard_shape = (1, 512)
    output_shard_spec = ttnn.ShardSpec(
        rms_crs,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    output_tile = ttnn.Tile([1, 32])
    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    # KV Cache tensor in DRAM (interleaved)
    # Shape: [max_seq_len, kv_dim] where kv_dim = 512 (nope) + 64 (rope) = 576
    dram_grid_size = device.dram_grid_size()
    kv_cache_seq_len = (
        dram_grid_size.x * dram_grid_size.y
    )  # For simplicity, just test up to number of DRAM banks, with one shard per bank
    assert (
        position_id < kv_cache_seq_len
    ), f"Position ID {position_id} must be less than KV cache sequence length {kv_cache_seq_len}"
    kv_cache_dim = 576  # 512 (nope) + 64 (rope)
    kv_cache_shape = (1, 1, kv_cache_seq_len, kv_cache_dim)
    torch_kv_cache = torch.zeros(kv_cache_shape, dtype=torch.bfloat16)
    # Get device DRAM grid size
    kv_cache_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [1, kv_cache_dim],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    kv_cache_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, kv_cache_shard_spec
    )
    # Create tensor with DRAM sharded memory config
    ttnn_kv_cache = ttnn.from_torch(
        torch_kv_cache,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=kv_cache_mem_config,
        tile=tile,
    )

    logger.info(f"Created KV cache tensor in DRAM with shape {kv_cache_shape}")

    logger.info(f"Created tensors sharded on single core with shard shape {output_shard_shape}")
    logger.info(f"Done creating TT tensors.")
    logger.info("Running KV cache branch operation...")
    _ = KVCacheBranch.op(
        ttnn_input,
        ttnn_W_dkv_rope,
        ttnn_gamma,
        tt_cos,
        tt_sin,
        tt_trans_replicated,
        ttnn_output,
        ttnn_kv_cache,
        kv_cache_write_index=position_id,  # Which sequence position to write to
    )

    logger.info("Running KV cache branch golden reference...")
    # Compute reference output using PyTorch
    torch_expected = KVCacheBranch.golden(
        torch_input,
        torch_W_dkv_rope,
        torch_gamma,
        torch_cos,
        torch_sin,
        position_ids_expanded,
        epsilon=epsilon,
    )

    # Read back from kv cache tensor in DRAM to check PCC
    torch_kv_cache = ttnn.to_torch(ttnn_kv_cache)
    compare_kv_cache = torch_kv_cache[:, :, position_id, :]
    max_diff = torch.max(torch.abs(torch_expected - compare_kv_cache)).item()
    mean_diff = torch.mean(torch.abs(torch_expected - compare_kv_cache)).item()
    logger.info(f"Max absolute difference: {max_diff}")
    logger.info(f"Mean absolute difference: {mean_diff}")

    passing, pcc_message = comp_pcc(compare_kv_cache, torch_expected, 0.98)
    logger.info(pcc_message)
    assert passing, pcc_message

    logger.info("✓ KV cache branch test passed!)")


@pytest.mark.parametrize("position_id", [0, 1, 34, 128, 1130])
def test_kv_cache_dram_shard(device, position_id):
    """Test KV cache shard untilize tilize operation"""
    torch.manual_seed(0)
    max_seq_len = 1136

    nope_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(0, 8))})
    rope_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(8, 8), ttnn.CoreCoord(8, 9))})

    # Input to nope kcache core
    #    torch_nope_cache = torch.randn(16, 32, dtype=torch.bfloat16)
    torch_nope_cache = torch.arange(512, dtype=torch.bfloat16)
    input_shard_spec = ttnn.ShardSpec(
        nope_core_grid,
        (1, 512),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    tile = ttnn.Tile([1, 32])
    # Create TTNN input tensor with WIDTH_SHARDED memory and tiny tile
    ttnn_nope_cache = ttnn.from_torch(
        torch_nope_cache,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile,
    )

    # Input to rope cores: 1 to 64 over the whole tensor
    torch_rope_cache = torch.randn(1, 64, dtype=torch.bfloat16) * 100
    rope_input_shard_spec = ttnn.ShardSpec(
        rope_core_grid,
        (1, 32),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    rope_input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, rope_input_shard_spec
    )
    rope_input_tile = ttnn.Tile([1, 32])
    ttnn_rope_cache = ttnn.from_torch(
        torch_rope_cache,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=rope_input_mem_config,
        tile=rope_input_tile,
    )

    # Create TTNN input tensor with WIDTH_SHARDED memory and tiny tile
    ttnn_output = ttnn.from_torch(
        torch_nope_cache,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile,
    )

    # KV Cache tensor in DRAM sharded
    # Create KV cache (non-paged) based on max seq len
    program_config = FlashMLADecode.ProgramConfig(
        k_chunk_size=128,
        exp_approx_mode=False,  # Use exact exp for higher precision
    )
    logger.info(f"Creating KV cache with seq_len={max_seq_len}...")
    kvpe_dim = 576
    cache_shape = (1, 1, max_seq_len, kvpe_dim)
    torch_kv_cache = torch.randn(cache_shape, dtype=torch.bfloat16)
    # for i in range(max_seq_len):
    #   torch_kv_cache[:, :, i, :] = torch.arange(576, dtype=torch.bfloat16).reshape(1, 1, 1, 576) * i

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

    ttnn_kv_cache = ttnn.from_torch(
        torch_kv_cache,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=kv_mem_config,
    )

    _ = KVCacheUpdate.op(ttnn_nope_cache, ttnn_rope_cache, ttnn_kv_cache, ttnn_output, position_id)

    torch_kv_cache_output = ttnn.to_torch(ttnn_kv_cache)
    compare_kv_cache = torch_kv_cache_output[:, :, position_id]
    # Split into nope (first 512 elements) and rope (last 64 elements)
    nope_dim = 512

    compare_nope = compare_kv_cache[..., :nope_dim]
    compare_rope = compare_kv_cache[..., nope_dim:]
    expected_nope = torch_nope_cache.reshape(1, 512)
    expected_rope = torch_rope_cache

    # Check nope portion
    nope_max_diff = torch.max(torch.abs(expected_nope - compare_nope)).item()
    nope_mean_diff = torch.mean(torch.abs(expected_nope - compare_nope)).item()
    logger.info(f"KV Cache NOPE absolute difference: {nope_max_diff}")
    logger.info(f"KV Cache NOPE mean absolute difference: {nope_mean_diff}")
    nope_passing, nope_pcc_message = comp_pcc(compare_nope, expected_nope, 0.98)
    logger.info(f"KV Cache NOPE PCC: {nope_pcc_message}")

    # Check rope portion
    rope_max_diff = torch.max(torch.abs(expected_rope - compare_rope)).item()
    rope_mean_diff = torch.mean(torch.abs(expected_rope - compare_rope)).item()
    logger.info(f"KV Cache ROPE absolute difference: {rope_max_diff}")
    logger.info(f"KV Cache ROPE mean absolute difference: {rope_mean_diff}")
    rope_passing, rope_pcc_message = comp_pcc(compare_rope, expected_rope, 0.98)
    logger.info(f"KV Cache ROPE PCC: {rope_pcc_message}")

    assert nope_passing, f"KV Cache NOPE verification failed: {nope_pcc_message}"
    assert rope_passing, f"KV Cache ROPE verification failed: {rope_pcc_message}"

    logger.info("✓ KV cache dram shard test passed!)")
