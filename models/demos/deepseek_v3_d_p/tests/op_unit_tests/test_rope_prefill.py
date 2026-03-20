# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.reference.modeling_deepseek import rotate_half
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup, get_cos_sin_matrix
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    create_balanced_chunk_order,
    reorder_tensor_chunks,
    reverse_reorder_tensor_chunks,
)

PCC_REQUIRED = 0.99


# sp x tp
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2), (2, 4)],
    ids=["4x2", "2x4"],
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
@pytest.mark.parametrize("seq_len", [128 * 1024], ids=["seq128k"])
@pytest.mark.parametrize("is_balanced", [False, True], ids=["unbalanced", "balanced"])
def test_rope_prefill(
    seq_len,
    is_balanced,
    request,
    mesh_device,
):
    """
    Test rotary positional embedding (RoPE) for prefill mode with SP sharding.

    This test:
    1. Creates rope tensors (cos, sin, trans_mat) for the given sequence length
    2. Applies RoPE using ttnn.experimental.rotary_embedding_llama
    3. Compares the result to a reference CPU implementation
    """

    config = request.getfixturevalue("config_only")

    sp_axis = 0
    tp_axis = 1
    production_mesh = [32, 4]
    mesh_shape = list(mesh_device.shape)

    # Scale sequence length to match device mesh (same as test_mla)
    seq_len = (seq_len // production_mesh[sp_axis]) * mesh_shape[sp_axis]
    config.max_seq_len = seq_len

    # Test parameters
    batch_size = 1
    num_heads = config.num_attention_heads
    head_dim = config.qk_rope_head_dim

    # Create rope setup using our own RotarySetup
    rope_setup = RotarySetup(
        hf_config=config,
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        is_balanced=is_balanced,
    )
    rope_tensors = rope_setup.get_rope_tensors(seq_len)

    # Create test input tensor: [batch, num_heads, seq_len, head_dim]
    torch_input = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # Reorder input for balanced ring attention
    sp_factor = mesh_shape[sp_axis]
    chunk_order = create_balanced_chunk_order(sp_factor) if is_balanced else None
    tt_input_torch = torch_input
    if is_balanced:
        tt_input_torch = reorder_tensor_chunks(torch_input, chunk_order, seq_dim=2)

    # Convert to TTNN tensor, shard over (sp_axis=seq, tp_axis=heads)
    tt_input = ttnn.from_torch(
        tt_input_torch,
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
        is_decode_mode=False,
    )

    # Convert back to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=[2, 1],
        ),
    )

    # Get reference cos/sin matrices
    cos_matrix_torch, sin_matrix_torch = get_cos_sin_matrix(config)
    cos_ref = cos_matrix_torch[0, 0, :seq_len, :]  # [seq_len, head_dim]
    sin_ref = sin_matrix_torch[0, 0, :seq_len, :]  # [seq_len, head_dim]

    position_ids = torch.arange(seq_len, dtype=torch.long)
    cos_selected = cos_ref[position_ids]
    sin_selected = sin_ref[position_ids]

    # Reshape to [1, 1, seq_len, head_dim] to broadcast with [1, num_heads, seq_len, head_dim]
    cos_broadcast = cos_selected.unsqueeze(0).unsqueeze(0)
    sin_broadcast = sin_selected.unsqueeze(0).unsqueeze(0)

    # Apply rotary embedding formula: (q * cos) + (rotate_half(q) * sin)
    rotated_half_q = rotate_half(torch_input, meta_style=True)
    reference_output = (torch_input * cos_broadcast) + (rotated_half_q * sin_broadcast)

    # Reverse-reorder output for balanced comparison
    if is_balanced:
        tt_output_torch = reverse_reorder_tensor_chunks(tt_output_torch, chunk_order, seq_dim=2)

    # Compare outputs
    logger.info(f"Comparing outputs: TTNN shape={tt_output_torch.shape}, Reference shape={reference_output.shape}")
    passing, pcc = comp_pcc(reference_output, tt_output_torch, PCC_REQUIRED)
    logger.info(f"PCC: {pcc:.6f}, Required: {PCC_REQUIRED}")

    assert passing, f"RoPE prefill test failed: PCC {pcc:.6f} < {PCC_REQUIRED} for seq_len={seq_len}"

    logger.info(f"✓ RoPE prefill test passed (balanced={is_balanced}, seq_len={seq_len}) with PCC={pcc:.6f}")
