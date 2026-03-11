# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for the TTNN VibeVoice diffusion head implementation.

Tests individual components (FFN, HeadLayer, FinalLayer) and the full
VibeVoice diffusion head, comparing against the reference TADA implementation.
PCC threshold: 0.99
"""

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import torch_random

# Import standalone reference models (no transformers.PreTrainedModel dependency)
from models.demos.audio.tada.reference.tada_reference import (
    FeedForwardNetwork,
    FinalLayer,
    HeadLayer,
    VibeVoiceDiffusionHead,
)
from models.demos.audio.tada.tt import ttnn_functional_vibevoice
from models.demos.utils.common_demo_utils import get_mesh_mappers
from tests.ttnn.utils_for_testing import assert_with_pcc

TADA_L1_SMALL_SIZE = 1024


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("embed_dim, ffn_dim", [(2048, 8192)])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_vibevoice_feedforward(mesh_device, batch_size, embed_dim, ffn_dim):
    """Test SwiGLU FeedForward network."""
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    # Create reference model
    ref_model = FeedForwardNetwork(embed_dim=embed_dim, ffn_dim=ffn_dim).eval()

    # Random input
    torch_input = torch_random((batch_size, embed_dim), -0.1, 0.1, dtype=torch.float32)

    # Reference output
    torch_output = ref_model(torch_input)

    # Preprocess parameters
    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_functional_vibevoice.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    # Convert input to TTNN (add seq dim for 3D)
    ttnn_input = ttnn.from_torch(
        torch_input.unsqueeze(1),  # (B, 1, D)
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )

    # Run TTNN model
    output = ttnn_functional_vibevoice.vibevoice_feedforward(ttnn_input, parameters=ttnn_parameters)
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer).squeeze(1)  # Remove seq dim

    _, pcc_message = assert_with_pcc(torch_output, output, pcc=0.99)
    logger.info(f"FeedForward PCC: {pcc_message}")


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "embed_dim, ffn_dim, cond_dim",
    [(2048, 8192, 2048)],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_vibevoice_head_layer(mesh_device, batch_size, embed_dim, ffn_dim, cond_dim):
    """Test single HeadLayer with adaLN modulation."""
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    # Create reference model
    ref_model = HeadLayer(embed_dim=embed_dim, ffn_dim=ffn_dim, cond_dim=cond_dim, norm_eps=1e-5).eval()

    # Random inputs
    torch_x = torch_random((batch_size, embed_dim), -0.1, 0.1, dtype=torch.float32)
    torch_c = torch_random((batch_size, cond_dim), -0.1, 0.1, dtype=torch.float32)

    # Reference output
    torch_output = ref_model(torch_x, torch_c)

    # Preprocess parameters
    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_functional_vibevoice.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    # Convert inputs to TTNN (add seq dim for 3D)
    ttnn_x = ttnn.from_torch(
        torch_x.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )
    ttnn_c = ttnn.from_torch(
        torch_c.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )

    # Run TTNN model
    output = ttnn_functional_vibevoice.vibevoice_head_layer(ttnn_x, ttnn_c, parameters=ttnn_parameters)
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer).squeeze(1)

    _, pcc_message = assert_with_pcc(torch_output, output, pcc=0.99)
    logger.info(f"HeadLayer PCC: {pcc_message}")


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "hidden_size, output_size, cond_size",
    [(2048, 528, 2048)],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_vibevoice_final_layer(mesh_device, batch_size, hidden_size, output_size, cond_size):
    """Test FinalLayer with non-affine RMSNorm and adaLN modulation."""
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    # Create reference model
    ref_model = FinalLayer(hidden_size=hidden_size, output_size=output_size, cond_size=cond_size, norm_eps=1e-5).eval()

    # Random inputs
    torch_x = torch_random((batch_size, hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_c = torch_random((batch_size, cond_size), -0.1, 0.1, dtype=torch.float32)

    # Reference output
    torch_output = ref_model(torch_x, torch_c)

    # Preprocess parameters
    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_functional_vibevoice.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    # Convert inputs to TTNN
    ttnn_x = ttnn.from_torch(
        torch_x.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )
    ttnn_c = ttnn.from_torch(
        torch_c.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )

    # Run TTNN model
    output = ttnn_functional_vibevoice.vibevoice_final_layer(ttnn_x, ttnn_c, parameters=ttnn_parameters)
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer).squeeze(1)

    _, pcc_message = assert_with_pcc(torch_output, output, pcc=0.99)
    logger.info(f"FinalLayer PCC: {pcc_message}")


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_vibevoice_full(mesh_device, batch_size):
    """Test full VibeVoice diffusion head end-to-end."""
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    # TADA 1B config for VibeVoice
    ref_model = VibeVoiceDiffusionHead(
        hidden_size=2048,
        head_layers=6,
        head_ffn_ratio=4.0,
        rms_norm_eps=1e-5,
        latent_size=528,
    ).eval()

    # Random inputs matching TADA 1B dimensions
    latent_size = 528
    hidden_size = 2048
    torch_noisy_images = torch_random((batch_size, latent_size), -0.1, 0.1, dtype=torch.float32)
    torch_timesteps = torch.rand(batch_size)  # (B,) in [0, 1]
    torch_condition = torch_random((batch_size, hidden_size), -0.1, 0.1, dtype=torch.float32)

    # Reference output
    torch_output = ref_model(torch_noisy_images, torch_timesteps, torch_condition)

    # Preprocess parameters
    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        convert_to_ttnn=ttnn_functional_vibevoice.convert_to_ttnn,
        custom_preprocessor=ttnn_functional_vibevoice.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    # Convert inputs to TTNN (add seq dim for 3D layout)
    ttnn_noisy_images = ttnn.from_torch(
        torch_noisy_images.unsqueeze(1),  # (B, 1, 528)
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )
    ttnn_condition = ttnn.from_torch(
        torch_condition.unsqueeze(1),  # (B, 1, 2048)
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )

    # Run TTNN model (timesteps stay on CPU for sinusoidal embedding)
    output = ttnn_functional_vibevoice.vibevoice_diffusion_head(
        ttnn_noisy_images,
        torch_timesteps,
        ttnn_condition,
        parameters=ttnn_parameters,
        frequency_embedding_size=256,
    )
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer).squeeze(1)
    output = output[:batch_size]

    _, pcc_message = assert_with_pcc(torch_output, output, pcc=0.99)
    logger.info(f"Full VibeVoice PCC: {pcc_message}")
