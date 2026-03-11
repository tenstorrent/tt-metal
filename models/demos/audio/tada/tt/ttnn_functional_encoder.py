# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN functional implementation of the TADA Encoder.

The encoder processes raw audio waveforms into 512-dim continuous acoustic tokens:

    Audio (B, 1, T) -> WavEncoder (CNN) -> (B, T/480, 1024)
    -> + pos_emb(token_masks) -> LocalAttentionEncoder (6 layers)
    -> hidden_linear -> (B, T/480, 512)
    -> gather at token positions -> token_values (B, N_tokens, 512)

The WavEncoder CNN runs on host (Snake1d + weight-normed Conv1d with various strides).
The LocalAttentionEncoder (transformer) runs on TT device.
"""

import torch

import ttnn
from models.demos.audio.tada.tt.ttnn_functional_common import TADA_MEMORY_CONFIG, local_attention_encoder


def encoder_cnn_on_host(audio, wav_encoder_model):
    """
    Run the WavEncoder CNN on host (CPU/GPU) using the reference model.

    The CNN portion involves Snake1d activations + weight-normed Conv1d with
    various strides/dilations. Kept on host as the LocalAttentionEncoder
    (transformer) is the compute bottleneck.

    Args:
        audio: (B, 1, T) raw audio at 24kHz
        wav_encoder_model: reference WavEncoder model on CPU
    Returns:
        (B, T/480, 1024) encoded audio features on CPU
    """
    with torch.no_grad():
        enc_out = wav_encoder_model(audio)  # (B, 1024, T/480)
    return enc_out.transpose(1, 2)  # (B, T/480, 1024)


def encoder_get_encoder_outputs(
    audio,
    token_masks,
    wav_encoder_model,
    pos_emb_weight,
    hidden_linear_weight,
    hidden_linear_bias,
    local_attn_parameters,
    *,
    device,
    input_mesh_mapper,
    output_mesh_composer,
    block_attention="v2",
):
    """
    Run the full encoder pipeline.

    1. WavEncoder CNN (on host)
    2. Add positional embeddings at text boundaries
    3. Create segment attention mask
    4. LocalAttentionEncoder (on TT device)
    5. Hidden linear projection

    Args:
        audio: (B, 1, T) padded audio on CPU
        token_masks: (B, T_frames) binary mask on CPU
        wav_encoder_model: reference WavEncoder model
        pos_emb_weight: (2, hidden_dim) position embedding weight on CPU (torch.Tensor)
        hidden_linear_weight: projection weight on device
        hidden_linear_bias: projection bias on device or None
        local_attn_parameters: preprocessed LocalAttentionEncoder parameters
        device: TT device
        input_mesh_mapper: mesh mapper
        block_attention: attention mask version ("v1" or "v2")
    Returns:
        (enc_out_tt, padded_token_masks) where enc_out_tt is on device
    """
    from models.demos.audio.tada.reference.tada_reference import create_segment_attention_mask

    # Step 1: CNN on host
    enc_out = encoder_cnn_on_host(audio, wav_encoder_model)  # (B, T_frames, 1024)
    seq_len = enc_out.shape[1]

    # Step 2: Pad token_masks to match encoder output length
    padded_token_masks = torch.nn.functional.pad(token_masks, (0, seq_len - token_masks.shape[1]), value=0)

    # Step 3: Add positional embeddings (on host before transfer)
    # pos_emb_weight is a CPU torch tensor (2, hidden_dim) used for indexing
    pos_embeddings = pos_emb_weight[padded_token_masks.long()]  # (B, T_frames, hidden_dim)
    enc_out = enc_out + pos_embeddings

    # Step 4: Create segment attention mask
    attn_mask = create_segment_attention_mask(padded_token_masks, version=block_attention)

    # Step 5: Transfer to device and run LocalAttentionEncoder
    enc_out_tt = ttnn.from_torch(
        enc_out.unsqueeze(1),  # (B, 1, T_frames, 1024)
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=input_mesh_mapper,
    )

    enc_out_tt = local_attention_encoder(
        enc_out_tt,
        seq_len,
        attention_mask_torch=attn_mask,
        parameters=local_attn_parameters,
        device=device,
        input_mesh_mapper=input_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    # Step 6: Hidden linear projection (1024 -> 512)
    if hidden_linear_weight is not None:
        enc_out_tt = ttnn.linear(
            enc_out_tt,
            hidden_linear_weight,
            bias=hidden_linear_bias,
            memory_config=TADA_MEMORY_CONFIG,
        )

    return enc_out_tt, padded_token_masks


def convert_to_ttnn(model, name):
    """Convert all except CNN conv layers and the WavEncoder."""
    if "wav_encoder" in name:
        return False
    if "pos_emb" in name:
        return False
    return True


def create_custom_mesh_preprocessor(weights_mesh_mapper):
    def custom_mesh_preprocessor(model, name):
        return custom_preprocessor(model, name, weights_mesh_mapper)

    return custom_mesh_preprocessor


def custom_preprocessor(torch_model, name, weights_mesh_mapper):
    """Custom preprocessor for encoder modules."""
    from models.demos.audio.tada.tt.ttnn_functional_common import custom_preprocessor as common_preprocessor

    return common_preprocessor(torch_model, name, weights_mesh_mapper)
