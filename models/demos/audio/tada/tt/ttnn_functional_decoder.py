# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN functional implementation of the TADA Decoder.

The decoder reconstructs waveforms from acoustic features:

    encoded_expanded (B, T, 512) -> decoder_proj (B, T, 1024)
    -> LocalAttentionEncoder (6 layers, segment attention)
    -> DACDecoder (CNN with ConvTranspose1d) -> waveform (B, 1, T*480)

The LocalAttentionEncoder runs on TT device.
The DACDecoder CNN runs on host (ConvTranspose1d not natively supported in TTNN).
"""

import torch

import ttnn
from models.demos.audio.tada.tt.ttnn_functional_common import (
    TADA_HIFI4_COMPUTE_CONFIG,
    TADA_MEMORY_CONFIG,
    local_attention_encoder,
)


def decoder_forward(
    encoded_expanded,
    token_masks,
    decoder_proj_weight,
    decoder_proj_bias,
    local_attn_parameters,
    wav_decoder_model,
    *,
    device,
    input_mesh_mapper,
    output_mesh_composer,
    block_attention="v2",
):
    """
    Full decoder forward pass.

    1. Linear projection (512 -> 1024) on TT device
    2. LocalAttentionEncoder on TT device
    3. DACDecoder CNN on host

    Args:
        encoded_expanded: (B, T, 512) acoustic features on CPU
        token_masks: (B, T) binary mask on CPU
        decoder_proj_weight: projection weight on device
        decoder_proj_bias: projection bias on device
        local_attn_parameters: preprocessed LocalAttentionEncoder parameters
        wav_decoder_model: reference DACDecoder model on CPU
        device: TT device
        input_mesh_mapper: mesh mapper
        output_mesh_composer: mesh composer
        block_attention: attention mask version
    Returns:
        (B, 1, T*480) reconstructed waveform on CPU
    """
    from models.demos.audio.tada.reference.tada_reference import create_decoder_segment_attention_mask

    batch_size, seq_len, embed_dim = encoded_expanded.shape

    # Step 1: Transfer to device and project
    x_tt = ttnn.from_torch(
        encoded_expanded.unsqueeze(1),  # (B, 1, T, 512)
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=input_mesh_mapper,
    )

    x_tt = ttnn.linear(
        x_tt,
        decoder_proj_weight,
        bias=decoder_proj_bias,
        memory_config=TADA_MEMORY_CONFIG,
        compute_kernel_config=TADA_HIFI4_COMPUTE_CONFIG,
    )

    # Step 2: Create decoder-specific segment attention mask
    # Decoder v2 uses cumsum(mask)-mask for block_ids, allows current+previous block
    attn_mask = create_decoder_segment_attention_mask(token_masks, version=block_attention)

    # Step 3: LocalAttentionEncoder on TT device
    x_tt = local_attention_encoder(
        x_tt,
        seq_len,
        attention_mask_torch=attn_mask,
        parameters=local_attn_parameters,
        device=device,
        input_mesh_mapper=input_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    # Step 4: Transfer back to host for DACDecoder
    decoded = ttnn.to_torch(x_tt, mesh_composer=output_mesh_composer)
    if len(decoded.shape) == 4:
        decoded = decoded.squeeze(1)
    decoded = decoded[:, :seq_len, :]  # Trim padding

    # Step 5: DACDecoder CNN on host
    # DACDecoder expects (B, C, T) input
    with torch.no_grad():
        x_rec = wav_decoder_model(decoded.transpose(1, 2).float())

    return x_rec


def convert_to_ttnn(model, name):
    """Convert all except DACDecoder CNN layers."""
    if "wav_decoder" in name:
        return False
    return True


def create_custom_mesh_preprocessor(weights_mesh_mapper):
    def custom_mesh_preprocessor(model, name):
        return custom_preprocessor(model, name, weights_mesh_mapper)

    return custom_mesh_preprocessor


def custom_preprocessor(torch_model, name, weights_mesh_mapper):
    """Custom preprocessor for decoder modules."""
    from models.demos.audio.tada.tt.ttnn_functional_common import custom_preprocessor as common_preprocessor

    return common_preprocessor(torch_model, name, weights_mesh_mapper)
