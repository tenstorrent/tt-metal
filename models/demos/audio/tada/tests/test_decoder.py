# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for the TTNN TADA Decoder implementation.

Tests:
1. Decoder LocalAttentionEncoder with segment masks
2. Full decoder pipeline (projection + attention + DACDecoder on host)

PCC threshold: 0.99 for the transformer portion.
"""

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight, preprocess_model_parameters

import ttnn
from models.common.utility_functions import torch_random
from models.demos.audio.tada.reference.tada_reference import (
    Decoder,
    LocalAttentionEncoder,
    create_segment_attention_mask,
)
from models.demos.audio.tada.tt import ttnn_functional_common, ttnn_functional_decoder
from models.demos.utils.common_demo_utils import get_mesh_mappers
from tests.ttnn.utils_for_testing import assert_with_pcc

TADA_L1_SMALL_SIZE = 1024


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [64])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_decoder_local_attention(mesh_device, batch_size, seq_len):
    """Test the LocalAttentionEncoder portion of the decoder with v2 segment mask."""
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    d_model = 1024
    num_layers = 2
    num_heads = 8
    d_ff = 4096

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

    # Create decoder-style segment mask
    token_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for i in range(0, seq_len, 8):
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
    logger.info(f"Decoder LocalAttention PCC: {pcc_message}")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [64])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_decoder_full_pipeline(mesh_device, batch_size, seq_len):
    """
    Test full decoder: projection + LocalAttention (TT) + DACDecoder (host).
    Compares the transformer output (before DACDecoder) for PCC.
    """
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    # Create reference decoder with fewer layers for testing
    ref_decoder = Decoder(
        embed_dim=512,
        hidden_dim=1024,
        num_attn_layers=2,
        num_attn_heads=8,
        attn_dim_feedforward=4096,
        attn_dropout=0.0,
        block_attention="v2",
        wav_decoder_channels=1536,
        strides=[4, 4, 5, 6],
    ).eval()

    # Random input
    torch_encoded = torch_random((batch_size, seq_len, 512), -0.1, 0.1, dtype=torch.float32)
    token_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for i in range(0, seq_len, 8):
        token_mask[:, i] = 1

    # Run reference decoder - get intermediate output after transformer
    with torch.no_grad():
        decoder_input = ref_decoder.decoder_proj(torch_encoded)
        attn_mask = create_segment_attention_mask(token_mask, version="v2")
        ref_transformer_out = ref_decoder.local_attention_decoder(decoder_input, mask=attn_mask)

    # Prepare TTNN parameters for the LocalAttentionEncoder part
    ttnn_local_attn_params = preprocess_model_parameters(
        initialize_model=lambda: ref_decoder.local_attention_decoder,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_functional_common.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    # Prepare decoder projection weights
    decoder_proj_weight = preprocess_linear_weight(
        ref_decoder.decoder_proj.weight, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
    )
    decoder_proj_weight = ttnn.to_device(decoder_proj_weight, mesh_device)
    decoder_proj_bias = preprocess_linear_bias(
        ref_decoder.decoder_proj.bias, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
    )
    decoder_proj_bias = ttnn.to_device(decoder_proj_bias, mesh_device)

    # Run TTNN: projection + transformer (skip DACDecoder for PCC comparison)
    x_tt = ttnn.from_torch(
        torch_encoded.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )

    x_tt = ttnn.linear(
        x_tt,
        decoder_proj_weight,
        bias=decoder_proj_bias,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    x_tt = ttnn_functional_common.local_attention_encoder(
        x_tt,
        seq_len,
        attention_mask_torch=attn_mask,
        parameters=ttnn_local_attn_params,
        device=mesh_device,
        input_mesh_mapper=input_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    output = ttnn.to_torch(x_tt, mesh_composer=output_mesh_composer)
    if len(output.shape) == 4:
        output = output.squeeze(1)
    output = output[:, :seq_len, :]

    _, pcc_message = assert_with_pcc(ref_transformer_out, output, pcc=0.99)
    logger.info(f"Decoder full pipeline (transformer) PCC: {pcc_message}")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [64])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_decoder_with_dac(mesh_device, batch_size, seq_len):
    """
    Test full decoder including DACDecoder on host.
    Compares final waveform output against reference.
    """
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    ref_decoder = Decoder(
        embed_dim=512,
        hidden_dim=1024,
        num_attn_layers=2,
        num_attn_heads=8,
        attn_dim_feedforward=4096,
        attn_dropout=0.0,
        block_attention="v2",
        wav_decoder_channels=1536,
        strides=[4, 4, 5, 6],
    ).eval()

    torch_encoded = torch_random((batch_size, seq_len, 512), -0.1, 0.1, dtype=torch.float32)
    token_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for i in range(0, seq_len, 8):
        token_mask[:, i] = 1

    # Reference output
    with torch.no_grad():
        ref_output = ref_decoder(torch_encoded, token_mask)

    # Prepare TTNN parameters
    ttnn_local_attn_params = preprocess_model_parameters(
        initialize_model=lambda: ref_decoder.local_attention_decoder,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_functional_common.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    decoder_proj_weight = preprocess_linear_weight(
        ref_decoder.decoder_proj.weight, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
    )
    decoder_proj_weight = ttnn.to_device(decoder_proj_weight, mesh_device)
    decoder_proj_bias = preprocess_linear_bias(
        ref_decoder.decoder_proj.bias, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
    )
    decoder_proj_bias = ttnn.to_device(decoder_proj_bias, mesh_device)

    # Run TTNN decoder
    tt_output = ttnn_functional_decoder.decoder_forward(
        torch_encoded,
        token_mask,
        decoder_proj_weight,
        decoder_proj_bias,
        ttnn_local_attn_params,
        ref_decoder.wav_decoder,
        device=mesh_device,
        input_mesh_mapper=input_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
        block_attention="v2",
    )

    # Compare waveform outputs
    # Lower PCC threshold for full pipeline (includes DACDecoder with bfloat16 inputs)
    _, pcc_message = assert_with_pcc(ref_output, tt_output, pcc=0.98)
    logger.info(f"Decoder with DAC PCC: {pcc_message}")
