# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN RoPE (Rotary Position Embedding) Test
Tests rotary embedding operation in decode mode on a single core
Uses DeepSeek V3 reference implementation for validation
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3YarnRotaryEmbedding
from models.demos.deepseek_v3.tt.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_b1.micro_ops.rope.op import RopeSingleCore


@pytest.mark.parametrize(
    "batch, num_heads, head_dim, position_id",
    [
        (1, 1, 32, 5),  # Minimal single-core test
        (1, 2, 64, 5),  # DeepSeek V3 Q rope (multiple heads)
    ],
    ids=["minimal", "q_rope"],
)
@pytest.mark.parametrize("pcc", [0.999])
def test_rope_decode(device, batch, num_heads, head_dim, position_id, pcc):
    """
    Test RoPE in decode mode on a single core.

    In decode mode, we apply RoPE to a single token position per batch element.
    Uses Meta-style interleaved format: [r, i, r, i, ...]
    Uses tiny tiles (num_heads x 32) for efficient single-core execution.
    """
    torch.manual_seed(1234)

    seq_len = 1  # Decode processes one token at a time
    max_seq_len = 8192

    logger.info(f"Testing RoPE decode: batch={batch}, num_heads={num_heads}, head_dim={head_dim}")

    # Create input tensor [batch, num_heads, seq_len, head_dim]
    x = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16).float()

    # Create position IDs [batch] - use fixed position for reproducibility
    position_ids = torch.tensor([position_id])  # positions 0, 1, 2, ...

    # Create cos/sin matrices in Meta-style format
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)

    # Meta-style: stack [cos(t), cos(t)] interleaved
    cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)  # [max_seq_len, head_dim]
    sin = torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2)  # [max_seq_len, head_dim]

    # Reference output (original shape for comparison)
    position_ids_expanded = position_ids.unsqueeze(1)  # [batch, 1]
    ref_out = RopeSingleCore.golden(x, cos, sin, position_ids_expanded)

    # For TTNN decode mode, reshape input to [1, batch, num_heads, head_dim]
    x_ttnn = x.permute(2, 0, 1, 3)  # [seq_len=1, batch, num_heads, head_dim]

    # Create HEIGHT_SHARDED memory config for input
    # Use tiny tiles: tile height = num_heads (no padding to 32)
    tiny_tile = ttnn.Tile((num_heads, ttnn.TILE_SIZE))
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        (num_heads, head_dim),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # Create TTNN input tensor with HEIGHT_SHARDED memory and tiny tile
    tt_x = ttnn.from_torch(
        x_ttnn,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tiny_tile,
    )

    # For decode mode, cos/sin are indexed by position: [1, batch, 1, head_dim]
    # Shape stays [1, 1, 1, head_dim] - broadcast multiply will use row 0
    cos_selected = cos[position_ids].unsqueeze(0).unsqueeze(2)  # [1, batch, 1, head_dim]
    sin_selected = sin[position_ids].unsqueeze(0).unsqueeze(2)  # [1, batch, 1, head_dim]

    # Use same tiny tile as input - data in row 0, rows 1+ are padding
    # Broadcast multiply reads row 0 and broadcasts to all rows
    cos_sin_shard_spec = ttnn.ShardSpec(
        core_grid,
        (num_heads, head_dim),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    cos_sin_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, cos_sin_shard_spec
    )

    tt_cos = ttnn.from_torch(
        cos_selected,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cos_sin_mem_config,
        tile=tiny_tile,
    )
    tt_sin = ttnn.from_torch(
        sin_selected,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cos_sin_mem_config,
        tile=tiny_tile,
    )

    # Transformation matrix - standard 32x32 tile (rotation matrix)
    trans_mat = get_rot_transformation_mat()
    # Repeat for batch dimension
    trans_mat = trans_mat.repeat(1, 1, batch, 1)

    trans_tile = ttnn.Tile((ttnn.TILE_SIZE, ttnn.TILE_SIZE))
    trans_shard_spec = ttnn.ShardSpec(
        core_grid,
        (ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    trans_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, trans_shard_spec)

    tt_trans = ttnn.from_torch(
        trans_mat,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=trans_mem_config,
        tile=trans_tile,
    )

    # Create output tensor with same sharded memory config and tiny tile as input
    torch_output_zeros = torch.zeros_like(x_ttnn, dtype=torch.bfloat16)
    tt_out = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tiny_tile,
    )

    # Run TTNN rotary embedding in decode mode
    tt_out = RopeSingleCore.op(
        tt_x,
        tt_cos,
        tt_sin,
        tt_trans,
        tt_out,
    )

    # Original result: [seq_len, batch, num_heads, head_dim]
    # Compare results - reshape output back to [batch, num_heads, seq_len, head_dim]
    tt_out_torch = ttnn.to_torch(tt_out)
    tt_out_torch = tt_out_torch.permute(1, 2, 0, 3)  # [batch, num_heads, seq_len, head_dim]

    passing, pcc_msg = comp_pcc(ref_out, tt_out_torch, pcc)
    logger.info(pcc_msg)
    assert passing, f"PCC check failed: {pcc_msg}"

    logger.info("✓ RoPE decode test passed!")


@pytest.mark.parametrize(
    "batch, num_heads, head_dim, position_id",
    [
        (1, 1, 32, 5),  # KV rope case
        (1, 2, 64, 5),  # Q rope case
    ],
    ids=["kv_rope", "q_rope"],
)
@pytest.mark.parametrize("pcc", [0.999])
def test_rope_decode_yarn(device, batch, num_heads, head_dim, position_id, pcc):
    """
    Test RoPE using DeepSeek V3's YaRN (Yet another RoPE extensioN) implementation.

    DeepSeek V3 uses YaRN for extended context lengths with specific scaling parameters.
    This test uses the actual DeepseekV3YarnRotaryEmbedding reference.
    """
    torch.manual_seed(1234)

    # DeepSeek V3 YaRN parameters
    seq_len = 1  # Decode mode
    max_seq_len = 1024
    base = 10000.0
    scaling_factor = 40.0
    original_max_position_embeddings = 4096
    beta_fast = 32
    beta_slow = 1
    mscale = 1.0
    mscale_all_dim = 1.0

    logger.info(f"Testing DeepSeek V3 YaRN RoPE decode: batch={batch}, num_heads={num_heads}")

    # Create YaRN rotary embedding
    yarn_rope = DeepseekV3YarnRotaryEmbedding(
        dim=head_dim,
        max_position_embeddings=max_seq_len,
        base=base,
        scaling_factor=scaling_factor,
        original_max_position_embeddings=original_max_position_embeddings,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        mscale=mscale,
        mscale_all_dim=mscale_all_dim,
        meta_style=True,
    )

    # Create input tensor [batch, num_heads, 1, head_dim]
    x = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16).float()

    # Get cos/sin from YaRN embedding
    cos, sin = yarn_rope(x, seq_len=max_seq_len, meta_style=True)

    # Fixed position IDs for reproducibility
    position_ids = torch.tensor([position_id])
    position_ids_expanded = position_ids.unsqueeze(1)

    # Reference output
    ref_out = RopeSingleCore.golden(x, cos, sin, position_ids_expanded)

    # For TTNN decode mode, reshape input to [1, batch, num_heads, head_dim]
    x_ttnn = x.permute(2, 0, 1, 3)  # [seq_len=1, batch, num_heads, head_dim]

    # Create HEIGHT_SHARDED memory config with tiny tiles
    tiny_tile = ttnn.Tile((num_heads, ttnn.TILE_SIZE))
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        (num_heads, head_dim),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    tt_x = ttnn.from_torch(
        x_ttnn,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tiny_tile,
    )

    # Cos/sin indexed by position: [1, batch, 1, head_dim]
    # Shape stays [1, 1, 1, head_dim] - broadcast multiply will use row 0
    cos_selected = cos[position_ids].unsqueeze(0).unsqueeze(2)
    sin_selected = sin[position_ids].unsqueeze(0).unsqueeze(2)

    # Use same tiny tile as input - data in row 0, rows 1+ are padding
    cos_sin_shard_spec = ttnn.ShardSpec(
        core_grid,
        (num_heads, head_dim),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    cos_sin_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, cos_sin_shard_spec
    )

    tt_cos = ttnn.from_torch(
        cos_selected,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cos_sin_mem_config,
        tile=tiny_tile,
    )
    tt_sin = ttnn.from_torch(
        sin_selected,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cos_sin_mem_config,
        tile=tiny_tile,
    )

    # Transformation matrix - standard 32x32 tile
    trans_mat = get_rot_transformation_mat()
    trans_mat = trans_mat.repeat(1, 1, batch, 1)

    trans_tile = ttnn.Tile((ttnn.TILE_SIZE, ttnn.TILE_SIZE))
    trans_shard_spec = ttnn.ShardSpec(
        core_grid,
        (ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    trans_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, trans_shard_spec)

    tt_trans = ttnn.from_torch(
        trans_mat,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=trans_mem_config,
        tile=trans_tile,
    )

    # Create output tensor with same sharded memory config and tiny tile as input
    torch_output_zeros = torch.zeros_like(x_ttnn, dtype=torch.bfloat16)
    tt_out = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tiny_tile,
    )

    # Run TTNN rotary embedding
    tt_out = RopeSingleCore.op(
        tt_x,
        tt_cos,
        tt_sin,
        tt_trans,
        tt_out,
    )

    # Compare results
    tt_out_torch = ttnn.to_torch(tt_out)
    tt_out_torch = tt_out_torch.permute(1, 2, 0, 3)  # [batch, num_heads, seq_len, head_dim]

    passing, pcc_msg = comp_pcc(ref_out, tt_out_torch, pcc)
    logger.info(pcc_msg)
    assert passing, f"PCC check failed: {pcc_msg}"

    logger.info("✓ YaRN RoPE decode test passed!")


def test_rotation_matrix_structure():
    """
    Test that the rotation transformation matrix has the correct structure.

    The transformation matrix swaps adjacent pairs with sign changes:
    For input [a, b, c, d, ...], rotation gives [-b, a, -d, c, ...]

    Note: This is a pure torch test - no device required.
    """
    trans_mat = get_rot_transformation_mat()

    # Expected structure: 32x32 matrix
    assert trans_mat.shape == (1, 1, 32, 32), f"Expected shape (1, 1, 32, 32), got {trans_mat.shape}"

    # Check the pattern
    mat = trans_mat[0, 0]
    for i in range(0, 32, 2):
        assert mat[i, i + 1] == 1, f"Expected mat[{i}, {i+1}] = 1"
        assert mat[i + 1, i] == -1, f"Expected mat[{i+1}, {i}] = -1"

    logger.info("✓ Rotation matrix structure verified!")
