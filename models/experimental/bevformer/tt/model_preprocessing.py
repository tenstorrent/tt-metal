# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional
from pathlib import Path


# Get default layout and dtype with fallback
try:
    DEFAULT_LAYOUT = ttnn.TILE_LAYOUT
except AttributeError:
    # Fallback for different ttnn versions
    try:
        DEFAULT_LAYOUT = ttnn.Layout.TILE
    except AttributeError:
        # Ultimate fallback - use None and let ttnn decide
        DEFAULT_LAYOUT = None

try:
    DEFAULT_DTYPE = ttnn.bfloat16
except AttributeError:
    try:
        DEFAULT_DTYPE = ttnn.DataType.BFLOAT16
    except AttributeError:
        # Ultimate fallback
        DEFAULT_DTYPE = None


def convert_parameterdict_to_object(param_dict):
    """Convert ParameterDict to object with attribute access"""
    params_obj = type("Params", (), {})()

    for layer_name, layer_params in param_dict.items():
        layer_obj = type("Layer", (), {})()
        for param_name, param_tensor in layer_params.items():
            setattr(layer_obj, param_name, param_tensor)
        setattr(params_obj, layer_name, layer_obj)

    return params_obj


# Helper functions for common preprocessing operations
def _build_ttnn_kwargs(dtype=None, layout=None, weights_mesh_mapper=None):
    """Build kwargs dict for ttnn.from_torch based on available parameters"""
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if layout is not None:
        kwargs["layout"] = layout
    if weights_mesh_mapper is not None:
        kwargs["mesh_mapper"] = weights_mesh_mapper
    return kwargs


def _process_linear_layer(layer, device, dtype=None, layout=None, weights_mesh_mapper=None):
    """Process a PyTorch linear layer into ttnn format with weight and bias"""
    if dtype is None:
        dtype = DEFAULT_DTYPE
    if layout is None:
        layout = DEFAULT_LAYOUT

    layer_params = {}
    layer_params["weight"] = ttnn.to_device(
        preprocess_linear_weight(
            layer.weight,
            dtype=dtype,
            layout=layout,
            weights_mesh_mapper=weights_mesh_mapper,
        ),
        device,
    )
    layer_params["bias"] = ttnn.to_device(
        preprocess_linear_bias(
            layer.bias,
            dtype=dtype,
            layout=layout,
            weights_mesh_mapper=weights_mesh_mapper,
        ),
        device,
    )
    return layer_params


def _manage_cache_load(cache_file_name, device):
    """Attempt to load parameters from cache"""
    if cache_file_name is None:
        return None

    cache_path = Path(cache_file_name)
    if not cache_path.exists():
        return None

    try:
        parameters = ttnn.load_parameters(device, cache_path)
        print(f"Loaded cached parameters from {cache_path}")
        return parameters
    except Exception as e:
        print(f"Failed to load cached parameters: {e}")
        return None


def _manage_cache_save(parameters, cache_file_name, device, cache_type=""):
    """Save parameters to cache if specified"""
    if cache_file_name is None:
        return

    cache_path = Path(cache_file_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ttnn.save_parameters(device, parameters, cache_path)
    print(f"Saved {cache_type}parameters to cache: {cache_path}")


# Local preprocessing functions to avoid import issues
def preprocess_linear_weight(weight, *, dtype=None, layout=None, weights_mesh_mapper=None):
    if dtype is None:
        dtype = DEFAULT_DTYPE
    if layout is None:
        layout = DEFAULT_LAYOUT
    weight = weight.T.contiguous()

    kwargs = _build_ttnn_kwargs(dtype=dtype, layout=layout, weights_mesh_mapper=weights_mesh_mapper)
    weight = ttnn.from_torch(weight, **kwargs)
    return weight


def preprocess_linear_bias(bias, *, dtype=None, layout=None, weights_mesh_mapper=None):
    if dtype is None:
        dtype = DEFAULT_DTYPE
    if layout is None:
        layout = DEFAULT_LAYOUT
    bias = bias.reshape((1, -1))

    kwargs = _build_ttnn_kwargs(dtype=dtype, layout=layout, weights_mesh_mapper=weights_mesh_mapper)
    bias = ttnn.from_torch(bias, **kwargs)
    return bias


def preprocess_ms_deformable_attention_parameters(
    torch_model,
    *,
    device,
    dtype=None,
    layout=None,
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

    # Try to load from cache first
    cached_params = _manage_cache_load(cache_file_name, device)
    if cached_params is not None:
        return cached_params

    print("Preprocessing multi-scale deformable attention parameters...")
    parameters = {}

    # Process all linear layers using helper function
    layer_names = ["value_proj", "sampling_offsets", "attention_weights", "output_proj"]
    for layer_name in layer_names:
        if hasattr(torch_model, layer_name):
            layer = getattr(torch_model, layer_name)
            parameters[layer_name] = _process_linear_layer(
                layer, device, dtype=dtype, layout=layout, weights_mesh_mapper=weights_mesh_mapper
            )

    # Save to cache and return
    _manage_cache_save(parameters, cache_file_name, device, "MS deformable attention ")
    print("Parameter preprocessing completed.")
    return parameters


def create_ms_deformable_attention_parameters(
    torch_model_path: Optional[str] = None,
    torch_model: Optional[torch.nn.Module] = None,
    *,
    device,
    config,
    dtype=None,
    layout=None,
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

    # Get or create the PyTorch model
    if torch_model is None:
        if torch_model_path is not None:
            torch_model = torch.load(torch_model_path, map_location="cpu")
            print(f"Loaded PyTorch model from {torch_model_path}")
        else:
            # Create a fresh PyTorch model for parameter extraction
            from ..pytorch.deformable_attention import PyTorchMultiScaleDeformableAttention

            torch_model = PyTorchMultiScaleDeformableAttention(config)
            print("Created fresh PyTorch model for parameter extraction")
    else:
        print("Using provided PyTorch model")

    torch_model.eval()

    return preprocess_ms_deformable_attention_parameters(
        torch_model,
        device=device,
        dtype=dtype,
        layout=layout,
        weights_mesh_mapper=weights_mesh_mapper,
        cache_file_name=cache_file_name,
    )


def preprocess_spatial_cross_attention_parameters(
    torch_model,
    *,
    device,
    dtype=None,
    layout=None,
    weights_mesh_mapper=None,
    cache_file_name: Optional[str] = None,
):
    """
    Preprocesses spatial cross attention model parameters from PyTorch to ttnn format.

    Args:
        torch_model: PyTorch SpatialCrossAttention model
        device: ttnn device
        dtype: Target data type for ttnn tensors
        layout: Target layout for ttnn tensors
        weights_mesh_mapper: Optional mesh mapper for distributed weights
        cache_file_name: Optional cache file name for parameter caching

    Returns:
        ParameterDict containing preprocessed ttnn tensors
    """

    # Try to load from cache first
    cached_params = _manage_cache_load(cache_file_name, device)
    if cached_params is not None:
        return cached_params

    print("Preprocessing spatial cross attention parameters...")
    parameters = {}

    # SCA has its own output projection layer
    if hasattr(torch_model, "output_proj"):
        parameters["output_proj"] = _process_linear_layer(
            torch_model.output_proj, device, dtype=dtype, layout=layout, weights_mesh_mapper=weights_mesh_mapper
        )

    # Extract deformable attention parameters from the nested module
    if hasattr(torch_model, "deformable_attention"):
        deform_attn = torch_model.deformable_attention

        # Process deformable attention layers
        deform_layer_names = ["value_proj", "sampling_offsets", "attention_weights"]
        for layer_name in deform_layer_names:
            if hasattr(deform_attn, layer_name):
                layer = getattr(deform_attn, layer_name)
                parameters[layer_name] = _process_linear_layer(
                    layer, device, dtype=dtype, layout=layout, weights_mesh_mapper=weights_mesh_mapper
                )

        # Handle deformable attention output projection (if separate from main output_proj)
        if hasattr(deform_attn, "output_proj") and not hasattr(torch_model, "output_proj"):
            parameters["output_proj"] = _process_linear_layer(
                deform_attn.output_proj, device, dtype=dtype, layout=layout, weights_mesh_mapper=weights_mesh_mapper
            )

    # Save to cache and return
    _manage_cache_save(parameters, cache_file_name, device, "SCA ")
    print("SCA parameter preprocessing completed.")
    return parameters


def preprocess_temporal_self_attention_parameters(
    torch_model,
    *,
    device,
    dtype=None,
    layout=None,
    weights_mesh_mapper=None,
    cache_file_name: Optional[str] = None,
):
    """
    Preprocesses temporal self attention model parameters from PyTorch to ttnn format.

    Args:
        torch_model: PyTorch TemporalSelfAttention model
        device: ttnn device
        dtype: Target data type for ttnn tensors
        layout: Target layout for ttnn tensors
        weights_mesh_mapper: Optional mesh mapper for distributed weights
        cache_file_name: Optional cache file name for parameter caching

    Returns:
        ParameterDict containing preprocessed ttnn tensors
    """

    # Try to load from cache first
    cached_params = _manage_cache_load(cache_file_name, device)
    if cached_params is not None:
        return cached_params

    print("Preprocessing temporal self attention parameters...")
    parameters = {}

    # TSA has a query projection layer for temporal features
    if hasattr(torch_model, "query_proj"):
        parameters["query_proj"] = _process_linear_layer(
            torch_model.query_proj, device, dtype=dtype, layout=layout, weights_mesh_mapper=weights_mesh_mapper
        )

    # Extract deformable attention parameters from the nested module
    if hasattr(torch_model, "deformable_attention"):
        deform_attn = torch_model.deformable_attention

        # Process deformable attention layers
        deform_layer_names = ["value_proj", "sampling_offsets", "attention_weights"]
        for layer_name in deform_layer_names:
            if hasattr(deform_attn, layer_name):
                layer = getattr(deform_attn, layer_name)
                parameters[layer_name] = _process_linear_layer(
                    layer, device, dtype=dtype, layout=layout, weights_mesh_mapper=weights_mesh_mapper
                )

        # Deformable attention output projection
        if hasattr(deform_attn, "output_proj"):
            parameters["output_proj"] = _process_linear_layer(
                deform_attn.output_proj, device, dtype=dtype, layout=layout, weights_mesh_mapper=weights_mesh_mapper
            )

    # Save to cache and return
    _manage_cache_save(parameters, cache_file_name, device, "TSA ")
    print("TSA parameter preprocessing completed.")
    return parameters


def create_spatial_cross_attention_parameters(
    torch_model_path: Optional[str] = None,
    torch_model: Optional[torch.nn.Module] = None,
    *,
    device,
    dtype=None,
    layout=None,
    weights_mesh_mapper=None,
    cache_file_name: Optional[str] = None,
):
    """Creates preprocessed parameters for spatial cross attention model."""

    # Get the PyTorch model
    if torch_model is None:
        if torch_model_path is None:
            raise ValueError("Either torch_model or torch_model_path must be provided for SCA")
        torch_model = torch.load(torch_model_path, map_location="cpu")
        print(f"Loaded PyTorch SCA model from {torch_model_path}")
    else:
        print("Using provided PyTorch SCA model")

    torch_model.eval()

    return preprocess_spatial_cross_attention_parameters(
        torch_model,
        device=device,
        dtype=dtype,
        layout=layout,
        weights_mesh_mapper=weights_mesh_mapper,
        cache_file_name=cache_file_name,
    )


def create_temporal_self_attention_parameters(
    torch_model_path: Optional[str] = None,
    torch_model: Optional[torch.nn.Module] = None,
    *,
    device,
    dtype=None,
    layout=None,
    weights_mesh_mapper=None,
    cache_file_name: Optional[str] = None,
):
    """Creates preprocessed parameters for temporal self attention model."""

    # Get the PyTorch model
    if torch_model is None:
        if torch_model_path is None:
            raise ValueError("Either torch_model or torch_model_path must be provided for TSA")
        torch_model = torch.load(torch_model_path, map_location="cpu")
        print(f"Loaded PyTorch TSA model from {torch_model_path}")
    else:
        print("Using provided PyTorch TSA model")

    torch_model.eval()

    return preprocess_temporal_self_attention_parameters(
        torch_model,
        device=device,
        dtype=dtype,
        layout=layout,
        weights_mesh_mapper=weights_mesh_mapper,
        cache_file_name=cache_file_name,
    )
