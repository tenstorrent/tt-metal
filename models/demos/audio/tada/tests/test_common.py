# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for TTNN common TADA modules: LocalSelfAttention, LocalAttentionEncoderLayer,
LocalAttentionEncoder.

PCC threshold: 0.99

Note: The tada source path and dac mock are set up in the root conftest.py.
"""

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import torch_random
from models.demos.audio.tada.reference.tada_reference import (
    LocalAttentionEncoder,
    LocalAttentionEncoderLayer,
    LocalSelfAttention,
    create_segment_attention_mask,
)
from models.demos.audio.tada.tt import ttnn_functional_common
from models.demos.utils.common_demo_utils import get_mesh_mappers
from tests.ttnn.utils_for_testing import assert_with_pcc

TADA_L1_SMALL_SIZE = 1024


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [64, 128])
@pytest.mark.parametrize("d_model, num_heads", [(1024, 8)])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_local_self_attention(mesh_device, batch_size, seq_len, d_model, num_heads):
    """Test LocalSelfAttention with RoPE (no segment mask)."""
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    # Create reference model
    ref_model = LocalSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.0,
        max_seq_len=8192,
        causal=False,
    ).eval()

    # Random input
    torch_input = torch_random((batch_size, seq_len, d_model), -0.1, 0.1, dtype=torch.float32)

    # Reference output (no mask = use default full attention)
    torch_output = ref_model(torch_input, mask=None)

    # Preprocess parameters
    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_functional_common.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    # Convert input to TTNN (4D for nlp ops)
    ttnn_input = ttnn.from_torch(
        torch_input.unsqueeze(1),  # (B, 1, S, D)
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )

    # Precompute RoPE (kept on CPU for interleaved rotation)
    head_dim = d_model // num_heads
    rope_freqs = ttnn_functional_common.compute_rope_freqs(head_dim, seq_len)
    rope_cos_cpu = rope_freqs[:, :, 0]  # (S, D/2)
    rope_sin_cpu = rope_freqs[:, :, 1]  # (S, D/2)

    # Run TTNN model
    output = ttnn_functional_common.local_self_attention(
        ttnn_input,
        seq_len,
        attention_mask=None,
        rope_cos_cpu=rope_cos_cpu,
        rope_sin_cpu=rope_sin_cpu,
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
    logger.info(f"LocalSelfAttention PCC: {pcc_message}")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [64])
@pytest.mark.parametrize("d_model, num_heads, d_ff", [(1024, 8, 4096)])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_local_attention_encoder_layer(mesh_device, batch_size, seq_len, d_model, num_heads, d_ff):
    """Test single LocalAttentionEncoderLayer."""
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    ref_model = LocalAttentionEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        activation="gelu",
        max_seq_len=8192,
    ).eval()

    torch_input = torch_random((batch_size, seq_len, d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output = ref_model(torch_input, mask=None)

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

    head_dim = d_model // num_heads
    rope_freqs = ttnn_functional_common.compute_rope_freqs(head_dim, seq_len)
    rope_cos_cpu = rope_freqs[:, :, 0]  # (S, D/2)
    rope_sin_cpu = rope_freqs[:, :, 1]  # (S, D/2)

    output = ttnn_functional_common.local_attention_encoder_layer(
        ttnn_input,
        seq_len,
        attention_mask=None,
        rope_cos_cpu=rope_cos_cpu,
        rope_sin_cpu=rope_sin_cpu,
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
    logger.info(f"LocalAttentionEncoderLayer PCC: {pcc_message}")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [64])
@pytest.mark.parametrize("d_model, num_layers, num_heads, d_ff", [(1024, 2, 8, 4096)])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_local_attention_encoder(mesh_device, batch_size, seq_len, d_model, num_layers, num_heads, d_ff):
    """Test full LocalAttentionEncoder (stack of layers + final norm)."""
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

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
    torch_output = ref_model(torch_input, mask=None)

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
        attention_mask_torch=None,
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
    logger.info(f"LocalAttentionEncoder PCC: {pcc_message}")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [64])
@pytest.mark.parametrize("d_model, num_layers, num_heads, d_ff", [(1024, 2, 8, 4096)])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_local_attention_encoder_with_mask(mesh_device, batch_size, seq_len, d_model, num_layers, num_heads, d_ff):
    """Test LocalAttentionEncoder with a segment attention mask."""
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    # create_segment_attention_mask already imported at module level

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

    # Create a simple segment mask: mark every 8th position as a text token boundary
    token_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for i in range(0, seq_len, 8):
        token_mask[:, i] = 1

    # Create segment attention mask
    attn_mask = create_segment_attention_mask(token_mask, version="v2")  # (B, S, S) boolean

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
    logger.info(f"LocalAttentionEncoder with mask PCC: {pcc_message}")
