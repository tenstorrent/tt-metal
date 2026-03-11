# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for the TTNN TADA Encoder implementation.

Tests:
1. LocalAttentionEncoder within encoder context (with segment masks)
2. Full encoder get_encoder_outputs pipeline (WavEncoder on host + attention on TT)

PCC threshold: 0.99
"""

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight, preprocess_model_parameters

import ttnn
from models.common.utility_functions import torch_random
from models.demos.audio.tada.reference.tada_reference import (
    Encoder,
    LocalAttentionEncoder,
    create_segment_attention_mask,
)
from models.demos.audio.tada.tt import ttnn_functional_common, ttnn_functional_encoder
from models.demos.utils.common_demo_utils import get_mesh_mappers
from tests.ttnn.utils_for_testing import assert_with_pcc

TADA_L1_SMALL_SIZE = 1024


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_encoder_local_attention(mesh_device, batch_size):
    """
    Test the LocalAttentionEncoder portion of the encoder with segment masks.
    This is the main compute-heavy part that runs on TT device.
    """
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    d_model = 1024
    num_layers = 2  # Use fewer layers for faster test
    num_heads = 8
    d_ff = 4096
    seq_len = 64

    ref_model = LocalAttentionEncoder(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        activation="gelu",
        max_seq_len=8192,
    ).eval()

    torch_input = torch_random((batch_size, seq_len, d_model), -0.1, 0.1, dtype=torch.float32)

    # Create segment mask similar to encoder usage
    token_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for i in range(0, seq_len, 10):
        token_mask[:, i] = 1
    attn_mask = create_segment_attention_mask(token_mask, version="v2")

    torch_output = ref_model(torch_input, mask=attn_mask)

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_functional_common.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    ttnn_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )

    output = ttnn_functional_common.local_attention_encoder(
        ttnn_input,
        seq_len,
        attention_mask_torch=attn_mask,
        parameters=ttnn_parameters,
        device=mesh_device,
        input_mesh_mapper=input_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
    if len(output.shape) == 4:
        output = output.squeeze(1)
    output = output[:, :seq_len, :]

    _, pcc_message = assert_with_pcc(torch_output, output, pcc=0.99)
    logger.info(f"Encoder LocalAttention PCC: {pcc_message}")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_encoder_get_encoder_outputs(mesh_device, batch_size):
    """
    Test the full encoder pipeline: WavEncoder (host) + LocalAttention (TT) + projection.
    Uses a small audio input for fast testing.
    """
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    # Create reference encoder
    ref_encoder = Encoder(
        hidden_dim=1024,
        embed_dim=512,
        strides=[6, 5, 4, 4],
        num_attn_layers=2,  # Fewer layers for testing
        num_attn_heads=8,
        attn_dim_feedforward=4096,
        attn_dropout=0.0,
        block_attention="v2",
    ).eval()

    # Create small test audio: ~0.5s at 24kHz = 12000 samples
    # Must be divisible by stride product (480)
    audio_len = 480 * 25  # 12000 samples = 25 frames at 50fps
    torch_audio = torch_random((batch_size, audio_len), -0.1, 0.1, dtype=torch.float32)

    # Create token masks (positions of text token boundaries in audio frames)
    num_frames = audio_len // 480 + 2  # approximate
    token_masks = torch.zeros(batch_size, num_frames, dtype=torch.long)
    for i in range(0, num_frames, 5):
        token_masks[:, i] = 1

    # Run reference encoder's get_encoder_outputs
    padded_audio = torch.nn.functional.pad(torch_audio.unsqueeze(1), (0, 960), value=0)
    ref_enc_out = ref_encoder.wav_encoder(padded_audio).transpose(1, 2)
    seq_len = ref_enc_out.shape[1]
    padded_masks = torch.nn.functional.pad(token_masks, (0, seq_len - token_masks.shape[1]), value=0)
    ref_enc_out = ref_enc_out + ref_encoder.pos_emb(padded_masks)
    attn_mask = create_segment_attention_mask(padded_masks, version="v2")
    ref_enc_out = ref_encoder.local_attention_encoder(ref_enc_out, mask=attn_mask)
    ref_enc_out_projected = ref_encoder.hidden_linear(ref_enc_out)

    # Preprocess LocalAttentionEncoder parameters
    ttnn_local_attn_params = preprocess_model_parameters(
        initialize_model=lambda: ref_encoder.local_attention_encoder,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_functional_common.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    # Keep pos_emb weight as CPU torch tensor (used for indexing on host)
    pos_emb_weight = ref_encoder.pos_emb.weight.data.clone()

    hidden_linear_weight = preprocess_linear_weight(
        ref_encoder.hidden_linear.weight, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
    )
    hidden_linear_weight = ttnn.to_device(hidden_linear_weight, mesh_device)
    hidden_linear_bias = preprocess_linear_bias(
        ref_encoder.hidden_linear.bias, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
    )
    hidden_linear_bias = ttnn.to_device(hidden_linear_bias, mesh_device)

    # Run TTNN encoder pipeline
    enc_out_tt, padded_token_masks = ttnn_functional_encoder.encoder_get_encoder_outputs(
        padded_audio,
        token_masks,
        ref_encoder.wav_encoder,
        pos_emb_weight,
        hidden_linear_weight,
        hidden_linear_bias,
        ttnn_local_attn_params,
        device=mesh_device,
        input_mesh_mapper=input_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
        block_attention="v2",
    )
    enc_out_result = ttnn.to_torch(enc_out_tt, mesh_composer=output_mesh_composer)
    if len(enc_out_result.shape) == 4:
        enc_out_result = enc_out_result.squeeze(1)
    enc_out_result = enc_out_result[:, :seq_len, :]

    _, pcc_message = assert_with_pcc(ref_enc_out_projected, enc_out_result, pcc=0.99)
    logger.info(f"Encoder full pipeline PCC: {pcc_message}")
