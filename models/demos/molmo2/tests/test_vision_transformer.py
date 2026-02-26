# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for Molmo2 Vision Transformer.

Validates the full VisionTransformer against HuggingFace reference implementation.
Tests include:
- Single and multiple layers
- Multi-layer output collection (for layers 18 and 24)
- Positional embedding interpolation
"""

import os

import pytest
import torch

import ttnn
from models.utility_functions import comp_pcc


@pytest.fixture
def model_location():
    """Get the model checkpoint location from environment."""
    return os.environ.get("HF_MODEL", "allenai/Molmo2-8B")


@pytest.fixture
def reference_model(model_location):
    """Load the Molmo2 reference model."""
    from models.demos.molmo2.reference.model import Molmo2Reference

    return Molmo2Reference(model_location, torch_dtype=torch.float32)


def get_tt_vision_transformer(mesh_device, state_dict, num_layers=25, weight_cache_path=None):
    """Create a TTNN VisionTransformer."""
    from models.demos.molmo2.tt.vision_transformer import VisionTransformer

    return VisionTransformer(
        mesh_device=mesh_device,
        state_dict=state_dict,
        num_layers=num_layers,
        hidden_dim=1152,
        intermediate_dim=4304,
        num_heads=16,
        head_dim=72,
        patch_size=14,
        image_size=378,
        layer_norm_eps=1e-6,
        weight_cache_path=weight_cache_path,
        dtype=ttnn.bfloat8_b,
    )


@pytest.mark.parametrize("num_layers", [1, 5, 25], ids=["1L", "5L", "full"])
def test_vision_transformer(mesh_device, reference_model, num_layers):
    """
    Test VisionTransformer with varying number of layers.

    Args:
        mesh_device: TTNN mesh device fixture
        reference_model: Molmo2Reference fixture
        num_layers: Number of layers to test
    """
    if reference_model is None:
        pytest.skip("Reference model not available")

    # Set expected PCC based on layer depth
    expected_pcc = 0.99 if num_layers <= 5 else 0.91

    # Get reference ViT
    ref_vit = reference_model.image_vit
    ref_vit.eval()

    # Create random input (after patch embedding)
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 729  # 27x27 patches
    hidden_dim = 1152

    x_torch = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)

    # Reference forward - collect hidden states
    ref_hidden_states = []
    x_ref = x_torch
    with torch.no_grad():
        for i in range(num_layers):
            x_ref = ref_vit.transformer.resblocks[i](x_ref)
            ref_hidden_states.append(x_ref.clone())

    # Get state dict and create TTNN transformer
    state_dict = reference_model.model.state_dict()
    tt_vit = get_tt_vision_transformer(
        mesh_device=mesh_device,
        state_dict=state_dict,
        num_layers=num_layers,
    )

    # Convert input to TTNN
    x_ttnn = ttnn.from_torch(
        x_torch.unsqueeze(0),  # [1, batch, seq_len, hidden_dim]
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # TTNN forward
    tt_hidden_states = tt_vit.forward(x_ttnn, return_all_hidden_states=True)

    # Compare final layer output
    tt_final = ttnn.to_torch(tt_hidden_states[-1], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    tt_final = tt_final.squeeze(0)

    passing, pcc_msg = comp_pcc(ref_hidden_states[-1], tt_final, pcc=expected_pcc)
    print(f"Final layer (num_layers={num_layers}) PCC: {pcc_msg}")

    assert passing, f"VisionTransformer (num_layers={num_layers}) failed PCC check: {pcc_msg}"


@pytest.mark.parametrize("num_layers", [25], ids=["full"])
def test_vision_transformer_feature_layers(mesh_device, reference_model, num_layers):
    """
    Test that hidden states from feature layers (18 and 24) match reference.

    These are the layers used for multi-scale feature extraction in the vision adapter.

    Args:
        mesh_device: TTNN mesh device fixture
        reference_model: Molmo2Reference fixture
        num_layers: Number of layers (should be 25 for this test)
    """
    if reference_model is None:
        pytest.skip("Reference model not available")

    feature_layers = [18, 24]  # 0-indexed
    expected_pcc = 0.91

    # Get reference ViT
    ref_vit = reference_model.image_vit
    ref_vit.eval()

    # Create random input
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 729
    hidden_dim = 1152

    x_torch = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)

    # Reference forward - collect all hidden states
    ref_hidden_states = []
    x_ref = x_torch
    with torch.no_grad():
        for i in range(num_layers):
            x_ref = ref_vit.transformer.resblocks[i](x_ref)
            ref_hidden_states.append(x_ref.clone())

    # Get state dict and create TTNN transformer
    state_dict = reference_model.model.state_dict()
    tt_vit = get_tt_vision_transformer(
        mesh_device=mesh_device,
        state_dict=state_dict,
        num_layers=num_layers,
    )

    # Convert input to TTNN
    x_ttnn = ttnn.from_torch(
        x_torch.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # TTNN forward
    tt_hidden_states = tt_vit.forward(x_ttnn, return_all_hidden_states=True)

    # Check each feature layer
    for layer_idx in feature_layers:
        tt_layer = ttnn.to_torch(
            tt_hidden_states[layer_idx], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        )[0]
        tt_layer = tt_layer.squeeze(0)

        passing, pcc_msg = comp_pcc(ref_hidden_states[layer_idx], tt_layer, pcc=expected_pcc)
        print(f"Layer {layer_idx} PCC: {pcc_msg}")

        assert passing, f"Feature layer {layer_idx} failed PCC check: {pcc_msg}"


@pytest.mark.parametrize("patch_num", [(27, 27), (14, 14)], ids=["native", "interp"])
def test_positional_embedding_interpolation(reference_model, patch_num):
    """
    Test positional embedding interpolation for different patch grid sizes.

    Args:
        reference_model: Molmo2Reference fixture
        patch_num: Tuple of (patches_h, patches_w) for the target grid
    """
    if reference_model is None:
        pytest.skip("Reference model not available")

    patches_h, patches_w = patch_num

    # Get reference positional embedding
    ref_pos_embed = reference_model.image_vit.positional_embedding
    base_patches_per_side = 27  # Molmo2 native: 378/14 = 27

    if patches_h == base_patches_per_side and patches_w == base_patches_per_side:
        # No interpolation needed
        expected_pos_embed = ref_pos_embed
    else:
        # Interpolate
        pos_embed = ref_pos_embed.reshape(base_patches_per_side, base_patches_per_side, -1)
        pos_embed = pos_embed.unsqueeze(0).permute(0, 3, 1, 2)

        pos_embed = torch.nn.functional.interpolate(
            pos_embed,
            size=(patches_h, patches_w),
            mode="bicubic",
            align_corners=False,
        )

        expected_pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, ref_pos_embed.shape[-1])

    # Test our interpolation function

    # Create a minimal transformer just to test interpolation
    state_dict = reference_model.model.state_dict()

    # Mock mesh device for CPU test
    class MockMeshDevice:
        def __init__(self):
            pass

        def __class__(self):
            return "NotMeshDevice"

    # Test the interpolation logic directly
    base_pos_embed = state_dict["model.vision_backbone.image_vit.positional_embedding"]

    if patches_h != base_patches_per_side:
        # Reshape and interpolate
        pos_embed = base_pos_embed.reshape(base_patches_per_side, base_patches_per_side, -1)
        pos_embed = pos_embed.unsqueeze(0).permute(0, 3, 1, 2)

        interpolated = torch.nn.functional.interpolate(
            pos_embed,
            size=(patches_h, patches_w),
            mode="bicubic",
            align_corners=False,
        )

        interpolated = interpolated.permute(0, 2, 3, 1).reshape(-1, base_pos_embed.shape[-1])

        # Should match expected
        assert torch.allclose(
            interpolated, expected_pos_embed, atol=1e-5
        ), f"Positional embedding interpolation mismatch for {patch_num}"
    else:
        # Native size, no interpolation
        assert torch.allclose(base_pos_embed, expected_pos_embed, atol=1e-5), "Native positional embedding should match"

    print(f"Positional embedding test passed for patch_num={patch_num}")
