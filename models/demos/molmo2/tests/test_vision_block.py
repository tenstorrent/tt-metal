# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for Molmo2 Vision Transformer block.

Validates VisionBlock against HuggingFace reference implementation.
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc


@pytest.fixture
def model_location():
    """Get the model checkpoint location from environment."""
    return os.environ.get("HF_MODEL", "allenai/Molmo2-8B")


@pytest.fixture
def reference_model(model_location):
    """Load the Molmo2 reference model."""
    from models.demos.molmo2.reference.model import Molmo2Reference

    return Molmo2Reference(model_location, torch_dtype=torch.float32)


def get_tt_vision_block(mesh_device, state_dict, layer_num, weight_cache_path=None):
    """Create a TTNN VisionBlock."""
    from models.demos.molmo2.tt.vision_block import VisionBlock

    return VisionBlock(
        mesh_device=mesh_device,
        state_dict=state_dict,
        layer_num=layer_num,
        hidden_dim=1152,
        intermediate_dim=4304,
        num_heads=16,
        head_dim=72,
        layer_norm_eps=1e-6,
        weight_cache_path=weight_cache_path,
        dtype=ttnn.bfloat8_b,
    )


@pytest.mark.parametrize("layer_num", [0, 12, 24])
def test_vision_block(mesh_device, reference_model, layer_num):
    """
    Test single VisionBlock against HuggingFace reference.

    Args:
        mesh_device: TTNN mesh device fixture
        reference_model: Molmo2Reference fixture
        layer_num: Which layer to test (0, 12, or 24)
    """
    # Skip if model not available
    if reference_model is None:
        pytest.skip("Reference model not available")

    # Get reference block
    ref_block = reference_model.get_vit_block(layer_num)
    ref_block.eval()

    # Create random input
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 729  # 27x27 patches
    hidden_dim = 1152

    x_torch = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)

    # Reference forward
    with torch.no_grad():
        ref_output = ref_block(x_torch)

    # Get state dict for TTNN block
    state_dict = reference_model.model.state_dict()

    # Create TTNN block
    tt_block = get_tt_vision_block(
        mesh_device=mesh_device,
        state_dict=state_dict,
        layer_num=layer_num,
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
    tt_output = tt_block(x_ttnn)

    # Convert back to torch
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    tt_output_torch = tt_output_torch.squeeze(0)  # Remove batch dim added for TTNN

    # Compare with PCC
    passing, pcc_msg = comp_pcc(ref_output, tt_output_torch, pcc=0.99)
    print(f"Layer {layer_num} PCC: {pcc_msg}")

    assert passing, f"Vision block layer {layer_num} failed PCC check: {pcc_msg}"


@pytest.mark.parametrize("seq_len", [128, 729, 1024])
def test_vision_block_seq_lengths(mesh_device, reference_model, seq_len):
    """
    Test VisionBlock with different sequence lengths.

    Args:
        mesh_device: TTNN mesh device fixture
        reference_model: Molmo2Reference fixture
        seq_len: Sequence length to test
    """
    if reference_model is None:
        pytest.skip("Reference model not available")

    layer_num = 0
    ref_block = reference_model.get_vit_block(layer_num)
    ref_block.eval()

    # Create random input
    torch.manual_seed(42)
    batch_size = 1
    hidden_dim = 1152

    x_torch = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)

    # Reference forward
    with torch.no_grad():
        ref_output = ref_block(x_torch)

    # Get state dict and create TTNN block
    state_dict = reference_model.model.state_dict()
    tt_block = get_tt_vision_block(
        mesh_device=mesh_device,
        state_dict=state_dict,
        layer_num=layer_num,
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
    tt_output = tt_block(x_ttnn)

    # Convert back to torch
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    tt_output_torch = tt_output_torch.squeeze(0)

    # Compare with PCC
    passing, pcc_msg = comp_pcc(ref_output, tt_output_torch, pcc=0.99)
    print(f"Seq len {seq_len} PCC: {pcc_msg}")

    assert passing, f"Vision block seq_len {seq_len} failed PCC check: {pcc_msg}"
