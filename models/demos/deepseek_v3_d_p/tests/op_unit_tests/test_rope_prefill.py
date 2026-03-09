# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.reference.modeling_deepseek import rotate_half
from models.demos.deepseek_v3.tt.rope import RotarySetup, get_cos_sin_matrix

PCC_REQUIRED = 0.99


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    ids=["4x2"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", [8 * 1024], ids=["seq8k"])
def test_rope_prefill(
    seq_len,
    request,
    mesh_device,
):
    """
    Test rotary positional embedding (RoPE) for prefill mode.

    This test:
    1. Creates rope tensors (cos, sin, trans_mat) for the given sequence length
    2. Applies RoPE using ttnn.experimental.rotary_embedding_llama
    3. Compares the result to a reference CPU implementation

    Args:
        seq_len: Sequence length to test
        use_pretrained: Whether to use pretrained weights (only affects config loading)
        request: Pytest request object for conditional fixture loading
        mesh_device: TTNN mesh device fixture
    """

    config = request.getfixturevalue("config_only")
    config.max_seq_len = seq_len  # override the max_seq_len to the test sequence length

    # Test parameters
    batch_size = 1
    num_heads = config.num_attention_heads
    head_dim = config.qk_rope_head_dim

    # Create rope setup
    batch_size_per_row = batch_size
    rope_setup = RotarySetup(
        device=mesh_device,
        batch_size_per_row=batch_size_per_row,
        hf_config=config,
    )

    # Get rope tensors for prefill (seq_len specified, no position_ids)
    # shard over the sp axis, replicate over the tp axis
    rope_tensors = rope_setup.get_rot_mats_table_shard_over_seq_len(seq_len=seq_len, sp_axis=0)

    # Create test input tensor: [batch, num_heads, seq_len, head_dim]
    torch_input = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # Convert to TTNN tensor
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    )

    # Apply RoPE using TTNN
    tt_output = ttnn.experimental.rotary_embedding_llama(
        tt_input,
        rope_tensors["cos_matrix"],
        rope_tensors["sin_matrix"],
        rope_tensors["trans_matrix"],
        is_decode_mode=False,  # Prefill mode
    )

    # Convert back to torch
    # Since input was sharded with dims=(None, 1), we need to concat along the same dimensions
    # For a 1x2 mesh, None means no sharding along row (dim 0), and 1 means sharding along col (dim 1)
    # So we concat along (0, 1) where 0 is the row mesh dim (maps to tensor dim 0) and 1 is the col mesh dim (maps to tensor dim 1)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            # tp sp
            dims=[2, 1],
        ),
    )

    # Get reference cos/sin matrices using the same function as rope_setup
    cos_matrix_torch, sin_matrix_torch = get_cos_sin_matrix(config)
    # Slice to the sequence length and reshape for reference implementation
    # Reference expects [max_seq_len, head_dim], we have [1, 1, max_seq_len, head_dim]
    cos_ref = cos_matrix_torch[0, 0, :seq_len, :]  # [seq_len, head_dim]
    sin_ref = sin_matrix_torch[0, 0, :seq_len, :]  # [seq_len, head_dim]

    # Create position_ids for prefill: [0, 1, 2, ..., seq_len-1]
    position_ids = torch.arange(seq_len, dtype=torch.long)

    # Apply reference RoPE using DeepSeek's reference implementation
    # apply_rotary_pos_emb expects q, k, cos, sin, position_ids, unsqueeze_dim, meta_style
    # The function does: cos[position_ids].unsqueeze(unsqueeze_dim)
    # This gives [seq_len, 1, head_dim] which needs to be reshaped to [1, 1, seq_len, head_dim]
    # to broadcast with q shape [1, num_heads, seq_len, head_dim]
    logger.info("Computing reference RoPE output using DeepSeek reference implementation")

    # Manually apply the same logic as apply_rotary_pos_emb but with proper reshaping
    cos_selected = cos_ref[position_ids]  # [seq_len, head_dim]
    sin_selected = sin_ref[position_ids]  # [seq_len, head_dim]

    # Reshape to [1, 1, seq_len, head_dim] to broadcast with [1, num_heads, seq_len, head_dim]
    cos_broadcast = cos_selected.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin_broadcast = sin_selected.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]

    # Apply rotary embedding formula: (q * cos) + (rotate_half(q) * sin)
    rotated_half_q = rotate_half(torch_input, meta_style=True)
    reference_output = (torch_input * cos_broadcast) + (rotated_half_q * sin_broadcast)

    # Compare outputs
    logger.info(f"Comparing outputs: TTNN shape={tt_output_torch.shape}, Reference shape={reference_output.shape}")
    passing, pcc = comp_pcc(reference_output, tt_output_torch, PCC_REQUIRED)
    logger.info(f"PCC: {pcc:.6f}, Required: {PCC_REQUIRED}")

    assert passing, f"RoPE prefill test failed: PCC {pcc:.6f} < {PCC_REQUIRED} for seq_len={seq_len}"

    logger.info(f"✓ RoPE prefill test passed for seq_len={seq_len} with PCC={pcc:.6f}")
