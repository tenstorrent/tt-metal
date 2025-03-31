# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
"""Tests for functional implementations of Qwen2.5-VL modules."""

import os
import json
import pytest
import torch
import torch.nn.functional as torch_F
from glob import glob
from scipy.stats import pearsonr
import numpy as np
import importlib
import sys


def get_script_dir():
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def vision_weights():
    """Load the converted vision weights."""
    script_dir = get_script_dir()
    weights_path = os.path.join(script_dir, "weights/vision_weights.pt")
    if not os.path.exists(weights_path):
        pytest.skip(f"Vision weights not found at {weights_path}. Run convert.py first.")
    return torch.load(weights_path, weights_only=False)


@pytest.fixture(params=["functional", "functional_ttnn"], ids=["torch", "ttnn"])
def implementation(request, mesh_device):
    """Fixture to run tests with both PyTorch and TTNN implementations."""
    module_name = request.param

    if module_name == "functional_ttnn":
        try:
            import ttnn
        except ImportError:
            pytest.skip("TTNN not available")

    try:
        module = importlib.import_module(f"models.demos.qwen25_vl.reference.{module_name}")

        # Set the mesh_device for the TTNN implementation
        if module_name == "functional_ttnn" and hasattr(module, "set_mesh_device"):
            module.set_mesh_device(mesh_device)

        return module
    except ImportError:
        pytest.skip(f"{module_name} implementation not available")


def load_earliest_run(module_name):
    """Load the earliest recorded run for a given module."""
    script_dir = get_script_dir()
    pattern = os.path.join(script_dir, f"module_io_data/{module_name}_*")
    runs = sorted(glob(pattern))  # Default sort will put earliest run first
    if not runs:
        pytest.skip(f"No recorded runs found for {module_name}")
    earliest_run = runs[0]  # Take first run instead of last

    # Load metadata
    with open(os.path.join(earliest_run, "metadata.json")) as f:
        metadata = json.load(f)

    # Load tensors
    def load_tensor_data(data):
        """Recursively load tensor data from saved files."""
        if data == "None":  # Handle string "None" in metadata
            return None
        if isinstance(data, dict):
            if data.get("type") == "tensor" and "path" in data:
                return torch.load(os.path.join(script_dir, data["path"]))
            return {k: load_tensor_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            loaded = [load_tensor_data(x) for x in data]
            # If all items are tensors and it's in kwargs['position_embeddings'],
            # convert to tuple as that's what the model expects
            if all(isinstance(x, torch.Tensor) for x in loaded):
                return tuple(loaded)
            # Try to convert a list of strings to integers if they all look like integers
            if all(isinstance(x, str) and x.isdigit() for x in loaded):
                return [int(x) for x in loaded]
            return loaded
        # Try to convert string to int if it represents a number
        if isinstance(data, str) and data.isdigit():
            return int(data)
        return data

    inputs = load_tensor_data(metadata["inputs"])
    outputs = load_tensor_data(metadata["outputs"])
    settings = metadata["settings"]  # Settings usually don't contain tensors

    return inputs, outputs, settings


def pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient between two tensors."""
    x_flat = x.detach().float().cpu().numpy().flatten()
    y_flat = y.detach().float().cpu().numpy().flatten()
    return pearsonr(x_flat, y_flat)[0]


def test_qwen2_rms_norm(vision_weights, implementation):
    """Test RMSNorm functional implementation."""
    inputs, outputs, _ = load_earliest_run("Qwen2RMSNorm")

    # Extract inputs and state dict
    hidden_states = inputs["args"][0]
    norm1_weights = vision_weights["blocks"]["0"]["norm1"]

    # Run functional implementation
    result = implementation.qwen2_rms_norm(hidden_states, norm1_weights)

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def test_qwen2_mlp(vision_weights, implementation):
    """Test MLP functional implementation."""
    inputs, outputs, _ = load_earliest_run("Qwen2_5_VLMLP")  # Load vision MLP inputs

    # Extract inputs and state dict from vision weights
    hidden_states = inputs["args"][0]
    block_weights = vision_weights["blocks"]["0"]["mlp"]  # Use first block's MLP weights

    # Run functional implementation
    result = implementation.qwen2_5_vl_mlp(hidden_states, block_weights)

    # Reshape to match expected output if needed
    if result.shape != outputs.shape:
        if result.numel() == outputs.numel():
            result = result.reshape(outputs.shape)
            print(f"Reshaped to: {result.shape}")
        else:
            print(
                f"ERROR: Cannot reshape {result.numel()} elements to shape {outputs.shape} with {outputs.numel()} elements"
            )

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def test_qwen2_5_vl_vision_sdpa_attention(vision_weights, implementation):
    """Test Vision SDPA Attention functional implementation."""
    inputs, outputs, settings = load_earliest_run("Qwen2_5_VLVisionSdpaAttention")

    # Extract inputs and state dict from vision weights
    hidden_states = inputs["args"][0]
    cu_seqlens = inputs["kwargs"].get("cu_seqlens")
    rotary_pos_emb = inputs["kwargs"].get("rotary_pos_emb")  # This should now be None instead of "None"
    position_embeddings = inputs["kwargs"].get("position_embeddings")  # This is already a tuple of tensors

    # Get attention weights from vision weights
    attn_weights = vision_weights["blocks"]["0"]["attn"]

    # Run functional implementation with num_heads from settings, ensuring it's an int
    result = implementation.qwen2_5_vl_vision_sdpa_attention(
        hidden_states, cu_seqlens, attn_weights, int(settings["num_heads"]), rotary_pos_emb, position_embeddings
    )

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def test_qwen2_5_vl_patch_merger(vision_weights, implementation):
    """Test Patch Merger functional implementation."""
    inputs, outputs, _ = load_earliest_run("Qwen2_5_VLPatchMerger")

    # Extract inputs and state dict from vision weights
    x = inputs["args"][0]
    merger_weights = vision_weights["merger"]

    # Run functional implementation
    result = implementation.qwen2_5_vl_patch_merger(x, merger_weights)

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def test_qwen2_5_vision_patch_embed(vision_weights, implementation):
    """Test Vision Patch Embed functional implementation."""
    inputs, outputs, _ = load_earliest_run("Qwen2_5_VisionPatchEmbed")
    model_settings = load_model_settings()

    # Extract inputs and state dict from vision weights
    hidden_states = inputs["args"][0]
    patch_embed_weights = vision_weights["patch_embed"]
    patch_size = model_settings["patch_size"]
    temporal_patch_size = model_settings["temporal_patch_size"]

    # Run functional implementation
    result = implementation.qwen2_5_vision_patch_embed(
        hidden_states, patch_embed_weights, patch_size, temporal_patch_size
    )

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def load_model_settings():
    """Load settings from the model's metadata file."""
    script_dir = get_script_dir()
    pattern = os.path.join(script_dir, f"module_io_data/Qwen2_5_VisionTransformerPretrainedModel_*")
    runs = sorted(glob(pattern))
    if not runs:
        pytest.skip("No recorded runs found for VisionTransformerPretrainedModel")
    model_run = runs[0]

    with open(os.path.join(model_run, "metadata.json")) as f:
        metadata = json.load(f)
    return metadata["settings"]


def test_qwen2_5_vl_vision_block(vision_weights, implementation):
    """Test Vision Block functional implementation."""
    inputs, outputs, _ = load_earliest_run("Qwen2_5_VLVisionBlock")
    model_settings = load_model_settings()  # Load settings from model metadata

    # Extract inputs
    hidden_states = inputs["args"][0]
    position_embeddings = inputs["kwargs"].get("position_embeddings")
    rotary_pos_emb = inputs["kwargs"].get("rotary_pos_emb")
    cu_seqlens = inputs["kwargs"].get("cu_seqlens")

    # Run functional implementation with first block's weights
    result = implementation.qwen2_5_vl_vision_block(
        hidden_states,
        cu_seqlens,
        vision_weights["blocks"]["0"],
        model_settings["num_heads"],
        rotary_pos_emb,
        position_embeddings,
    )

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def test_qwen2_5_vision_rotary_embedding(implementation):
    """Test the vision rotary embedding functional implementation."""
    inputs, outputs, _ = load_earliest_run("Qwen2_5_VisionRotaryEmbedding")
    model_settings = load_model_settings()  # Load settings from model metadata

    # Extract inputs
    seqlen = inputs["args"][0]
    head_dim = model_settings["hidden_size"] // model_settings["num_heads"]

    result = implementation.qwen2_5_vision_rotary_embedding(seqlen, dim=head_dim // 2, device=outputs.device)

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def test_qwen2_5_vl_rot_pos_emb(vision_weights, implementation):
    """Test rotary position embedding function."""
    inputs, outputs, _ = load_earliest_run("Qwen2_5_VisionTransformerPretrainedModel_rot_pos_emb")
    model_settings = load_model_settings()  # Load settings from model metadata

    # Extract inputs
    grid_thw = inputs["args"][0]
    spatial_merge_size = model_settings["spatial_merge_size"]
    head_dim = model_settings["hidden_size"] // model_settings["num_heads"]

    # Run functional implementation
    result = implementation.qwen2_5_vl_rot_pos_emb(grid_thw, spatial_merge_size, head_dim)

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def test_qwen2_5_vl_get_window_index(vision_weights, implementation):
    """Test window index generation function."""
    inputs, outputs, _ = load_earliest_run("Qwen2_5_VisionTransformerPretrainedModel_get_window_index")
    model_settings = load_model_settings()  # Load settings from model metadata

    # Extract inputs
    grid_thw = inputs["args"][0]
    window_size = model_settings["window_size"]
    spatial_merge_size = model_settings["spatial_merge_size"]
    patch_size = model_settings["patch_size"]

    # Run functional implementation
    window_index, cu_window_seqlens = implementation.qwen2_5_vl_get_window_index(
        grid_thw, window_size, spatial_merge_size, patch_size
    )

    # Check correlation with recorded output (first element of tuple is window_index)
    pcc = pearson_correlation(window_index, outputs[0])
    assert pcc > 0.999, f"PCC {pcc} below threshold"

    # Check cu_window_seqlens (second element of tuple)
    expected_cu_seqlens = outputs[1]
    assert cu_window_seqlens == expected_cu_seqlens, "cu_window_seqlens do not match"


def test_qwen2_5_vision_transformer(vision_weights, implementation):
    """Test Vision Transformer functional implementation."""
    inputs, outputs, model_settings = load_earliest_run("Qwen2_5_VisionTransformerPretrainedModel")

    # Extract inputs
    hidden_states = inputs["args"][0]
    grid_thw = inputs["kwargs"]["grid_thw"]

    # Calculate head_dim from the hidden size and num_heads
    num_heads = model_settings["num_heads"]
    head_dim = model_settings["hidden_size"] // model_settings["num_heads"]
    spatial_merge_size = model_settings["spatial_merge_size"]
    window_size = model_settings["window_size"]
    patch_size = model_settings["patch_size"]
    temporal_patch_size = model_settings["temporal_patch_size"]

    # Get fullatt_block_indexes
    fullatt_block_indexes = model_settings["fullatt_block_indexes"]

    # Ensure fullatt_block_indexes is a list of integers
    if isinstance(fullatt_block_indexes, str):
        # Handle case where it might be stored as a string representation of a list
        import ast

        fullatt_block_indexes = ast.literal_eval(fullatt_block_indexes)
    elif not isinstance(fullatt_block_indexes, list):
        fullatt_block_indexes = []

    # Limit to first n blocks
    n_blocks = 1
    vision_weights["blocks"] = {k: v for k, v in vision_weights["blocks"].items() if int(k) < n_blocks}

    result = implementation.qwen2_5_vision_transformer(
        hidden_states,
        vision_weights,
        grid_thw,
        num_heads=num_heads,
        head_dim=head_dim,
        spatial_merge_size=spatial_merge_size,
        window_size=window_size,
        fullatt_block_indexes=fullatt_block_indexes,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
    )

    # Compare with both model output and recorded output
    pcc_recorded = pearson_correlation(result, outputs)
    assert pcc_recorded > 0.999, f"PCC {pcc_recorded} below threshold"
