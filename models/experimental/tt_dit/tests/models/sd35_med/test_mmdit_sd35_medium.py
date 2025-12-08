# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test file for SD3Transformer2DModel with real weights
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

# Import the transformer model
from models.experimental.tt_dit.models.transformers.sd35_med.transformer_sd35_medium import SD3Transformer2DModel
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel as DiffusersSD3Transformer2DModel


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_sd3_transformer_real_weights(device, reset_seeds):
    """
    Test SD3Transformer2DModel with real weights from SD3.5 Medium.
    """
    # Model configuration for SD3.5 Medium
    sample_size = 128
    patch_size = 2
    in_channels = 16
    num_layers = 24
    attention_head_dim = 64
    num_attention_heads = 24
    joint_attention_dim = 4096
    caption_projection_dim = 2432
    pooled_projection_dim = 2048
    out_channels = 16
    pos_embed_max_size = 192

    # Input dimensions
    batch_size = 1
    height = sample_size
    width = sample_size
    seq_len = (height * width) // (patch_size * patch_size)
    context_seq_len = 77

    # Create the transformer model
    model = SD3Transformer2DModel(
        sample_size=sample_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_layers=num_layers,
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        joint_attention_dim=joint_attention_dim,
        caption_projection_dim=caption_projection_dim,
        pooled_projection_dim=pooled_projection_dim,
        out_channels=out_channels,
        pos_embed_max_size=pos_embed_max_size,
        mesh_device=device,
    )

    # Load real weights from SD3.5 Medium model via HuggingFace
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    torch_model = DiffusersSD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    torch_model.eval()
    full_state_dict = torch_model.state_dict()

    # Load weights into TTNN model
    model.load_torch_state_dict(full_state_dict)

    # Create test inputs
    torch.manual_seed(0)

    # Hidden states: NCHW for Diffusers, NHWC for TTNN
    torch_hidden_states_nchw = torch.randn((batch_size, in_channels, height, width), dtype=torch.bfloat16)
    torch_hidden_states = torch_hidden_states_nchw.permute(0, 2, 3, 1)  # NCHW -> NHWC

    # Encoder hidden states: [batch, seq_len, features]
    torch_encoder_hidden_states = torch.randn((batch_size, context_seq_len, joint_attention_dim), dtype=torch.bfloat16)

    # Timestep: [batch]
    torch_timestep = torch.full((batch_size,), 500.0, dtype=torch.bfloat16)

    # Pooled projection: [batch, features]
    torch_pooled_projection = torch.randn((batch_size, pooled_projection_dim), dtype=torch.bfloat16)

    # Convert to TTNN tensors
    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    encoder_hidden_states = ttnn.from_torch(
        torch_encoder_hidden_states,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    timestep = ttnn.from_torch(
        torch_timestep.unsqueeze(-1),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    pooled_projection = ttnn.from_torch(
        torch_pooled_projection.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run reference model (step by step to get proj_out before unpatchify)
    with torch.no_grad():
        ref_hidden = torch_model.pos_embed(torch_hidden_states_nchw)
        ref_temb = torch_model.time_text_embed(torch_timestep, torch_pooled_projection)
        ref_ctx = torch_model.context_embedder(torch_encoder_hidden_states)

        for ref_block in torch_model.transformer_blocks:
            result = ref_block(hidden_states=ref_hidden, encoder_hidden_states=ref_ctx, temb=ref_temb)
            if isinstance(result, tuple):
                ref_ctx, ref_hidden = result
            else:
                ref_hidden = result

        ref_normed = torch_model.norm_out(ref_hidden, ref_temb)
        ref_proj = torch_model.proj_out(ref_normed)

    # Run TTNN model
    tt_hidden = model.pos_embed(hidden_states)
    tt_temb = model.time_text_embed(timestep, pooled_projection)
    tt_ctx = model.context_embedder(encoder_hidden_states)

    if len(tt_hidden.shape) == 3:
        tt_hidden = ttnn.reshape(tt_hidden, (1, tt_hidden.shape[0], tt_hidden.shape[1], tt_hidden.shape[2]))
    if len(tt_ctx.shape) == 3:
        tt_ctx = ttnn.reshape(tt_ctx, (1, tt_ctx.shape[0], tt_ctx.shape[1], tt_ctx.shape[2]))

    for tt_block in model.transformer_blocks:
        tt_hidden, tt_ctx = tt_block(
            x=tt_hidden,
            context=tt_ctx,
            conditioning=tt_temb,
            seq_len=seq_len,
            context_seq_len=context_seq_len,
        )

    tt_normed = model.norm_out(tt_hidden, tt_temb)
    tt_proj = model.proj_out(tt_normed)

    # Convert to torch and match shape
    tt_proj_torch = ttnn.to_torch(ttnn.from_device(tt_proj))
    while tt_proj_torch.dim() > ref_proj.dim():
        tt_proj_torch = tt_proj_torch.squeeze(0)
    while tt_proj_torch.dim() < ref_proj.dim():
        tt_proj_torch = tt_proj_torch.unsqueeze(0)

    # Final PCC comparison
    pcc_threshold = 0.99
    passing, pcc = assert_with_pcc(ref_proj, tt_proj_torch, pcc=pcc_threshold)

    if passing:
        logger.info(f"✓ SD3Transformer2DModel PCC: {pcc:.6f}")
    else:
        pytest.fail(f"SD3Transformer2DModel PCC test FAILED: pcc={pcc:.6f}")
