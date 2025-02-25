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

from functional import (
    qwen2_rms_norm,
    qwen2_5_vl_mlp,
    qwen2_5_vl_vision_sdpa_attention,
    qwen2_5_vl_patch_merger,
    qwen2_5_vision_patch_embed,
    qwen2_5_vl_vision_block,
    qwen2_5_vision_transformer,
)
from model import Qwen2_5_VLVisionConfig, Qwen2_5_VisionTransformerPretrainedModel


@pytest.fixture
def vision_weights():
    """Load the converted vision weights."""
    weights_path = "weights/vision_weights.pt"
    if not os.path.exists(weights_path):
        pytest.skip(f"Vision weights not found at {weights_path}. Run convert.py first.")
    return torch.load(weights_path, weights_only=False)


def load_earliest_run(module_name):
    """Load the earliest recorded run for a given module."""
    pattern = f"module_io_data/{module_name}_*"
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
                try:
                    return torch.load(data["path"])
                except Exception as e:
                    print(f"Error loading tensor from {data['path']}: {e}")
                    return None
            return {k: load_tensor_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            loaded = [load_tensor_data(x) for x in data]
            # If all items are tensors and it's in kwargs['position_embeddings'],
            # convert to tuple as that's what the model expects
            if all(isinstance(x, torch.Tensor) for x in loaded):
                return tuple(loaded)
            return loaded
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


def test_qwen2_rms_norm(vision_weights):
    """Test RMSNorm functional implementation."""
    inputs, outputs, _ = load_earliest_run("Qwen2RMSNorm")

    # Extract inputs and state dict
    hidden_states = inputs["args"][0]
    norm1_weights = vision_weights["blocks"]["0"]["norm1"]

    # Run functional implementation
    result = qwen2_rms_norm(hidden_states, norm1_weights)

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def test_qwen2_mlp(vision_weights):
    """Test MLP functional implementation."""
    inputs, outputs, _ = load_earliest_run("Qwen2_5_VLMLP")  # Load vision MLP inputs

    # Extract inputs and state dict from vision weights
    hidden_states = inputs["args"][0]
    block_weights = vision_weights["blocks"]["0"]["mlp"]  # Use first block's MLP weights

    # Run functional implementation
    result = qwen2_5_vl_mlp(hidden_states, block_weights)  # Use vision MLP implementation

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def test_qwen2_5_vl_vision_sdpa_attention(vision_weights):
    """Test Vision SDPA Attention functional implementation."""
    inputs, outputs, settings = load_earliest_run("Qwen2_5_VLVisionSdpaAttention")

    # Debug: Print input shapes and types
    print("\nInput shapes and types:")
    for i, arg in enumerate(inputs["args"]):
        if arg is not None:
            print(f"Arg {i}: shape={arg.shape if hasattr(arg, 'shape') else 'no shape'}, type={type(arg)}")
    print("\nKwargs:")
    for k, v in inputs["kwargs"].items():
        if v is not None:
            print(f"{k}: shape={v.shape if hasattr(v, 'shape') else 'no shape'}, type={type(v)}")

    # Extract inputs and state dict from vision weights
    hidden_states = inputs["args"][0]
    cu_seqlens = inputs["kwargs"].get("cu_seqlens")
    rotary_pos_emb = inputs["kwargs"].get("rotary_pos_emb")  # This should now be None instead of "None"
    position_embeddings = inputs["kwargs"].get("position_embeddings")  # This is already a tuple of tensors

    # Debug: Print state dict keys and shapes
    print("\nAttention weights:")
    attn_weights = vision_weights["blocks"]["0"]["attn"]
    for k, v in attn_weights.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if hasattr(v2, "shape"):
                    print(f"{k}.{k2}: shape={v2.shape}")
        elif hasattr(v, "shape"):
            print(f"{k}: shape={v.shape}")

    # Run functional implementation with num_heads from settings, ensuring it's an int
    result = qwen2_5_vl_vision_sdpa_attention(
        hidden_states, cu_seqlens, attn_weights, int(settings["num_heads"]), rotary_pos_emb, position_embeddings
    )

    # Debug: Print shapes of result and expected output
    print("\nOutput shapes:")
    print(f"Result shape: {result.shape}")
    print(f"Expected shape: {outputs.shape}")

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    print(f"\nPearson correlation: {pcc}")
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def test_qwen2_5_vl_patch_merger(vision_weights):
    """Test Patch Merger functional implementation."""
    inputs, outputs, _ = load_earliest_run("Qwen2_5_VLPatchMerger")

    # Extract inputs and state dict from vision weights
    x = inputs["args"][0]
    merger_weights = vision_weights["merger"]

    # Run functional implementation
    result = qwen2_5_vl_patch_merger(x, merger_weights)

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def test_qwen2_5_vision_patch_embed(vision_weights):
    """Test Vision Patch Embed functional implementation."""
    inputs, outputs, _ = load_earliest_run("Qwen2_5_VisionPatchEmbed")

    # Extract inputs and state dict from vision weights
    hidden_states = inputs["args"][0]
    patch_embed_weights = vision_weights["patch_embed"]

    print(f"hidden_states shape: {hidden_states.shape}")

    # Run functional implementation
    result = qwen2_5_vision_patch_embed(hidden_states, patch_embed_weights, patch_size=14, temporal_patch_size=2)

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def load_model_settings():
    """Load settings from the model's metadata file."""
    pattern = f"module_io_data/Qwen2_5_VisionTransformerPretrainedModel_*"
    runs = sorted(glob(pattern))
    if not runs:
        pytest.skip("No recorded runs found for VisionTransformerPretrainedModel")
    model_run = runs[0]

    with open(os.path.join(model_run, "metadata.json")) as f:
        metadata = json.load(f)
    return metadata["settings"]


def test_qwen2_5_vl_vision_block(vision_weights):
    """Test Vision Block functional implementation."""
    inputs, outputs, _ = load_earliest_run("Qwen2_5_VLVisionBlock")
    model_settings = load_model_settings()  # Load settings from model metadata

    # Extract inputs and state dict from vision weights
    hidden_states = inputs["args"][0]
    block_weights = vision_weights["blocks"]["0"]  # Use first block's weights

    # Get position embeddings from the recorded inputs if available
    position_embeddings = inputs["kwargs"].get("position_embeddings")
    rotary_pos_emb = inputs["kwargs"].get("rotary_pos_emb")
    cu_seqlens = inputs["kwargs"].get("cu_seqlens")

    # Run functional implementation with position embeddings
    result = qwen2_5_vl_vision_block(
        hidden_states, cu_seqlens, block_weights, int(model_settings["num_heads"]), rotary_pos_emb, position_embeddings
    )

    # Check correlation with recorded output
    pcc = pearson_correlation(result, outputs)
    assert pcc > 0.999, f"PCC {pcc} below threshold"


def test_qwen2_5_vision_transformer(vision_weights):
    """Test Vision Transformer functional implementation."""
    inputs, outputs, settings = load_earliest_run("Qwen2_5_VisionTransformerPretrainedModel")

    # Extract inputs
    hidden_states = inputs["args"][0]
    cu_seqlens = inputs["kwargs"].get("cu_seqlens")
    position_embeddings = inputs["kwargs"].get("position_embeddings")
    rotary_pos_emb = inputs["kwargs"].get("rotary_pos_emb")

    # Get window indices and window-level cumulative sequence lengths
    grid_thw = inputs["kwargs"].get("grid_thw")
    if grid_thw is None:
        pytest.skip("grid_thw not found in recorded inputs")

    # Create a model instance to compute window indices and record intermediate values
    config = Qwen2_5_VLVisionConfig(**settings)
    model = Qwen2_5_VisionTransformerPretrainedModel(config)

    print("\nModel forward pass for reference values:")
    model_output = model(hidden_states, grid_thw)

    # Now get window indices
    window_index, cu_window_seqlens = model.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    # Add required fields to state dict
    vision_weights["window_index"] = window_index
    vision_weights["cu_window_seqlens"] = cu_window_seqlens
    vision_weights["spatial_merge_unit"] = config.spatial_merge_size * config.spatial_merge_size
    vision_weights["fullatt_block_indexes"] = config.fullatt_block_indexes

    print("\nInput shapes:")
    print(f"hidden_states: {hidden_states.shape}")
    print(f"cu_seqlens: {cu_seqlens.shape if cu_seqlens is not None else None}")
    print(f"window_index: {window_index.shape}")
    print(f"cu_window_seqlens: {cu_window_seqlens.shape}")
    if rotary_pos_emb is not None:
        print(f"rotary_pos_emb: {rotary_pos_emb.shape}")
    if position_embeddings is not None:
        print(f"position_embeddings: {[p.shape for p in position_embeddings]}")

    print("\nConfig values:")
    print(f"spatial_merge_size: {config.spatial_merge_size}")
    print(f"fullatt_block_indexes: {config.fullatt_block_indexes}")
    print(f"num_heads: {config.num_heads}")

    # Run functional implementation
    print("\nFunctional implementation forward pass:")
    result = qwen2_5_vision_transformer(
        hidden_states,
        cu_seqlens,
        vision_weights,
        int(settings["num_heads"]),
        patch_size=14,
        temporal_patch_size=2,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
    )

    print("\nOutput shapes:")
    print(f"result: {result.shape}")
    print(f"model_output: {model_output.shape}")
    print(f"expected output: {outputs.shape}")

    # Compare with both model output and recorded output
    pcc_model = pearson_correlation(result, model_output)
    pcc_recorded = pearson_correlation(result, outputs)
    print(f"\nCorrelations:")
    print(f"PCC with model output: {pcc_model}")
    print(f"PCC with recorded output: {pcc_recorded}")

    assert pcc_recorded > 0.999, f"PCC {pcc_recorded} below threshold"
