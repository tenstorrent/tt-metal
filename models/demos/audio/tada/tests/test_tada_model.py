# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for the TTNN TadaForCausalLM implementation.

Tests:
1. Input embedding creation (token + acoustic + time + mask embeddings)
2. VibeVoice diffusion head (full forward)
3. LM head (text logit generation)

PCC threshold: 0.99
"""

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import torch_random
from models.demos.audio.tada.tt import ttnn_functional_tada, ttnn_functional_vibevoice
from models.demos.utils.common_demo_utils import get_mesh_mappers
from tests.ttnn.utils_for_testing import assert_with_pcc

TADA_L1_SMALL_SIZE = 1024


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_tada_embed_inputs(mesh_device, batch_size):
    """
    Test the input embedding creation for TadaForCausalLM.
    Compares the sum of all embedding components against reference.
    """
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    # TADA 1B dimensions
    hidden_size = 2048
    acoustic_dim = 512
    vocab_size = 128256
    num_time_classes = 256

    # Create reference components
    embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
    acoustic_proj = torch.nn.Linear(acoustic_dim, hidden_size, bias=False)
    acoustic_mask_emb = torch.nn.Embedding(2, hidden_size)
    time_start_embed = torch.nn.Embedding(num_time_classes, hidden_size)
    time_end_embed = torch.nn.Embedding(num_time_classes, hidden_size)

    # Random inputs
    input_ids = torch.randint(0, vocab_size, (batch_size,))
    acoustic_features = torch_random((batch_size, acoustic_dim), -0.1, 0.1, dtype=torch.float32)
    acoustic_masks = torch.ones(batch_size, dtype=torch.long)
    time_before = torch.randint(0, num_time_classes, (batch_size,))
    time_after = torch.randint(0, num_time_classes, (batch_size,))

    # Reference output
    ref_output = (
        embed_tokens(input_ids)
        + acoustic_proj(acoustic_features)
        + acoustic_mask_emb(acoustic_masks)
        + time_start_embed(time_before)
        + time_end_embed(time_after)
    )

    # Create a simple module container for parameter preprocessing
    class EmbedContainer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = embed_tokens
            self.acoustic_proj = acoustic_proj
            self.acoustic_mask_emb = acoustic_mask_emb
            self.time_start_embed = time_start_embed
            self.time_end_embed = time_end_embed

    container = EmbedContainer().eval()

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: container,
        convert_to_ttnn=ttnn_functional_tada.convert_to_ttnn,
        custom_preprocessor=ttnn_functional_tada.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    # Run TTNN embedding
    output = ttnn_functional_tada.tada_embed_inputs(
        input_ids,
        acoustic_features,
        acoustic_masks,
        time_before,
        time_after,
        parameters=ttnn_parameters,
        device=mesh_device,
        input_mesh_mapper=input_mesh_mapper,
    )
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
    if len(output.shape) == 3:
        output = output.squeeze(1)
    elif len(output.shape) == 4:
        output = output.squeeze(1).squeeze(1)

    _, pcc_message = assert_with_pcc(ref_output, output, pcc=0.99)
    logger.info(f"TADA embed inputs PCC: {pcc_message}")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_tada_vibevoice_head(mesh_device, batch_size):
    """
    Test the VibeVoice diffusion head as used in TadaForCausalLM.
    """
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    from models.demos.audio.tada.reference.tada_reference import VibeVoiceDiffusionHead

    ref_model = VibeVoiceDiffusionHead(
        hidden_size=2048,
        head_layers=6,
        head_ffn_ratio=4.0,
        rms_norm_eps=1e-5,
        latent_size=528,
    ).eval()

    # Inputs
    latent_size = 528
    hidden_size = 2048
    torch_noisy = torch_random((batch_size, latent_size), -0.1, 0.1, dtype=torch.float32)
    torch_t = torch.rand(batch_size)
    torch_cond = torch_random((batch_size, hidden_size), -0.1, 0.1, dtype=torch.float32)

    # Reference
    torch_output = ref_model(torch_noisy, torch_t, torch_cond)

    # TTNN
    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        convert_to_ttnn=ttnn_functional_vibevoice.convert_to_ttnn,
        custom_preprocessor=ttnn_functional_vibevoice.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    ttnn_noisy = ttnn.from_torch(
        torch_noisy.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )
    ttnn_cond = ttnn.from_torch(
        torch_cond.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )

    output = ttnn_functional_vibevoice.vibevoice_diffusion_head(
        ttnn_noisy, torch_t, ttnn_cond, parameters=ttnn_parameters
    )
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer).squeeze(1)

    _, pcc_message = assert_with_pcc(torch_output, output, pcc=0.99)
    logger.info(f"TADA VibeVoice head PCC: {pcc_message}")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_tada_lm_head(mesh_device, batch_size):
    """Test the LM head (text logit generation)."""
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    hidden_size = 2048
    vocab_size = 128256

    # Reference LM head (tied with embed_tokens in TADA)
    lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    torch_hidden = torch_random((batch_size, 1, hidden_size), -0.1, 0.1, dtype=torch.float32)

    ref_output = lm_head(torch_hidden)

    # Create container for parameter preprocessing
    class LMContainer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = lm_head

    container = LMContainer().eval()

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: container,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_functional_tada.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    ttnn_hidden = ttnn.from_torch(
        torch_hidden.unsqueeze(1) if torch_hidden.dim() == 3 else torch_hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )

    output = ttnn_functional_tada.tada_lm_head(ttnn_hidden, parameters=ttnn_parameters)
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
    if len(output.shape) == 4:
        output = output.squeeze(1)

    _, pcc_message = assert_with_pcc(ref_output, output, pcc=0.99)
    logger.info(f"TADA LM head PCC: {pcc_message}")
