# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional
from pathlib import Path


# Local preprocessing functions to avoid import issues
def preprocess_linear_weight(weight, *, dtype, layout=ttnn.TILE_LAYOUT, weights_mesh_mapper=None):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype, layout=layout, mesh_mapper=weights_mesh_mapper)
    return weight


def preprocess_linear_bias(bias, *, dtype, layout=ttnn.TILE_LAYOUT, weights_mesh_mapper=None):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype, layout=layout, mesh_mapper=weights_mesh_mapper)
    return bias


def preprocess_ms_deformable_attention_parameters(
    torch_model,
    *,
    device,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    weights_mesh_mapper=None,
    cache_file_name: Optional[str] = None,
):
    """
    Preprocesses multi-scale deformable attention model parameters from PyTorch to ttnn format.

    Args:
        torch_model: PyTorch MultiScaleDeformableAttention model
        device: ttnn device
        dtype: Target data type for ttnn tensors
        layout: Target layout for ttnn tensors
        weights_mesh_mapper: Optional mesh mapper for distributed weights
        cache_file_name: Optional cache file name for parameter caching

    Returns:
        ParameterDict containing preprocessed ttnn tensors
    """

    # Check if cached parameters exist
    if cache_file_name is not None:
        cache_path = Path(cache_file_name)
        if cache_path.exists():
            try:
                parameters = ttnn.load_parameters(device, cache_path)
                print(f"Loaded cached parameters from {cache_path}")
                return parameters
            except Exception as e:
                print(f"Failed to load cached parameters: {e}")

    print("Preprocessing multi-scale deformable attention parameters...")

    parameters = {}

    # Preprocess value projection layer
    if hasattr(torch_model, "value_proj"):
        parameters["value_proj"] = {}
        parameters["value_proj"]["weight"] = ttnn.to_device(
            preprocess_linear_weight(
                torch_model.value_proj.weight,
                dtype=dtype,
                layout=layout,
                weights_mesh_mapper=weights_mesh_mapper,
            ),
            device,
        )
        parameters["value_proj"]["bias"] = ttnn.to_device(
            preprocess_linear_bias(
                torch_model.value_proj.bias,
                dtype=dtype,
                layout=layout,
                weights_mesh_mapper=weights_mesh_mapper,
            ),
            device,
        )

    # Preprocess sampling offsets layer
    if hasattr(torch_model, "sampling_offsets"):
        parameters["sampling_offsets"] = {}
        parameters["sampling_offsets"]["weight"] = ttnn.to_device(
            preprocess_linear_weight(
                torch_model.sampling_offsets.weight,
                dtype=dtype,
                layout=layout,
                weights_mesh_mapper=weights_mesh_mapper,
            ),
            device,
        )
        parameters["sampling_offsets"]["bias"] = ttnn.to_device(
            preprocess_linear_bias(
                torch_model.sampling_offsets.bias,
                dtype=dtype,
                layout=layout,
                weights_mesh_mapper=weights_mesh_mapper,
            ),
            device,
        )

    # Preprocess attention weights layer
    if hasattr(torch_model, "attention_weights"):
        parameters["attention_weights"] = {}
        parameters["attention_weights"]["weight"] = ttnn.to_device(
            preprocess_linear_weight(
                torch_model.attention_weights.weight,
                dtype=dtype,
                layout=layout,
                weights_mesh_mapper=weights_mesh_mapper,
            ),
            device,
        )
        parameters["attention_weights"]["bias"] = ttnn.to_device(
            preprocess_linear_bias(
                torch_model.attention_weights.bias,
                dtype=dtype,
                layout=layout,
                weights_mesh_mapper=weights_mesh_mapper,
            ),
            device,
        )

    # Preprocess output projection layer
    if hasattr(torch_model, "output_proj"):
        parameters["output_proj"] = {}
        parameters["output_proj"]["weight"] = ttnn.to_device(
            preprocess_linear_weight(
                torch_model.output_proj.weight,
                dtype=dtype,
                layout=layout,
                weights_mesh_mapper=weights_mesh_mapper,
            ),
            device,
        )
        parameters["output_proj"]["bias"] = ttnn.to_device(
            preprocess_linear_bias(
                torch_model.output_proj.bias,
                dtype=dtype,
                layout=layout,
                weights_mesh_mapper=weights_mesh_mapper,
            ),
            device,
        )

    # Save parameters to cache if specified
    if cache_file_name is not None:
        cache_path = Path(cache_file_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ttnn.save_parameters(device, parameters, cache_path)
        print(f"Saved parameters to cache: {cache_path}")

    print("Parameter preprocessing completed.")
    return parameters


def create_ms_deformable_attention_parameters(
    torch_model_path: Optional[str] = None,
    torch_model: Optional[torch.nn.Module] = None,
    *,
    device,
    config,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    weights_mesh_mapper=None,
    cache_file_name: Optional[str] = None,
):
    """
    Creates preprocessed parameters for multi-scale deformable attention model.

    Args:
        torch_model_path: Path to saved PyTorch model (optional)
        torch_model: PyTorch model instance (optional)
        device: ttnn device
        config: DeformableAttentionConfig instance
        dtype: Target data type for ttnn tensors
        layout: Target layout for ttnn tensors
        weights_mesh_mapper: Optional mesh mapper for distributed weights
        cache_file_name: Optional cache file name for parameter caching

    Returns:
        ParameterDict containing preprocessed ttnn tensors
    """

    # Load or use provided PyTorch model
    if torch_model is None:
        if torch_model_path is None:
            # Create a fresh PyTorch model for parameter extraction
            from ..pytorch.deformable_attention import PyTorchMultiScaleDeformableAttention

            torch_model = PyTorchMultiScaleDeformableAttention(config)
            print("Created fresh PyTorch model for parameter extraction")
        else:
            # Load from file
            torch_model = torch.load(torch_model_path, map_location="cpu")
            print(f"Loaded PyTorch model from {torch_model_path}")
    else:
        print("Using provided PyTorch model")

    # Ensure model is in evaluation mode
    torch_model.eval()

    # Preprocess parameters
    parameters = preprocess_ms_deformable_attention_parameters(
        torch_model,
        device=device,
        dtype=dtype,
        layout=layout,
        weights_mesh_mapper=weights_mesh_mapper,
        cache_file_name=cache_file_name,
    )

    return parameters


def create_ms_deformable_attention_model_parameters(
    config,
    device,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    weights_mesh_mapper=None,
    model_location_generator=None,
):
    """
    High-level function to create model parameters following tt-metal patterns.

    Args:
        config: DeformableAttentionConfig instance
        device: ttnn device
        dtype: Target data type for ttnn tensors
        layout: Target layout for ttnn tensors
        weights_mesh_mapper: Optional mesh mapper for distributed weights
        model_location_generator: Function to generate model file paths

    Returns:
        ParameterDict containing preprocessed ttnn tensors
    """

    # Generate cache file name
    cache_file_name = None
    if model_location_generator is not None:
        cache_file_name = model_location_generator(
            "ms_deformable_attention", f"embed_dims_{config.embed_dims}_heads_{config.num_heads}"
        )

    # Create parameters
    parameters = create_ms_deformable_attention_parameters(
        torch_model=None,  # Will create fresh model
        device=device,
        config=config,
        dtype=dtype,
        layout=layout,
        weights_mesh_mapper=weights_mesh_mapper,
        cache_file_name=cache_file_name,
    )

    return parameters


def convert_torch_deformable_attention_to_ttnn_parameters(
    torch_model,
    device,
    dtype: ttnn.DataType = ttnn.bfloat16,
):
    """
    Utility function to quickly convert a PyTorch deformable attention model to ttnn parameters.

    Args:
        torch_model: PyTorch MultiScaleDeformableAttention model
        device: ttnn device
        dtype: Target data type for ttnn tensors

    Returns:
        ParameterDict containing preprocessed ttnn tensors
    """
    return preprocess_ms_deformable_attention_parameters(
        torch_model,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )
