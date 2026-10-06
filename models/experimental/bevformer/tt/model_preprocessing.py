# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
BEVFormer model parameter preprocessing utilities for TTNN.
"""

from types import SimpleNamespace
from typing import Optional

import torch

import ttnn

# Get default layout and dtype
DEFAULT_LAYOUT = ttnn.TILE_LAYOUT
DEFAULT_DTYPE = ttnn.bfloat16


def convert_parameterdict_to_object(param_dict):
    """
    Convert parameter dictionary to object with dot-notation attribute access.
    """
    params_obj = type("Params", (), {})()

    for layer_name, layer_params in param_dict.items():
        layer_obj = type("Layer", (), {})()

        # Handle case where layer_params might already be an object (not a dict)
        if hasattr(layer_params, "items"):
            # layer_params is a dictionary
            for param_name, param_tensor in layer_params.items():
                setattr(layer_obj, param_name, param_tensor)
        else:
            # layer_params is already an object, copy its attributes
            if hasattr(layer_params, "__dict__"):
                for param_name, param_tensor in layer_params.__dict__.items():
                    setattr(layer_obj, param_name, param_tensor)
            else:
                # layer_params is a single tensor/value, store it directly
                setattr(params_obj, layer_name, layer_params)
                continue

        setattr(params_obj, layer_name, layer_obj)

    return params_obj


def _build_ttnn_kwargs(dtype=None, layout=None, weights_mesh_mapper=None, device=None):
    """Build kwargs dict for ttnn.from_torch based on available parameters"""
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if layout is not None:
        kwargs["layout"] = layout
    if weights_mesh_mapper is not None:
        kwargs["mesh_mapper"] = weights_mesh_mapper
    if device is not None:
        kwargs["device"] = device
    return kwargs


def _process_linear_layer(layer, device, dtype=None, layout=None, weights_mesh_mapper=None):
    """Process a PyTorch linear layer into ttnn format with weight and bias"""
    if dtype is None:
        dtype = DEFAULT_DTYPE
    if layout is None:
        layout = DEFAULT_LAYOUT

    layer_params = {}

    # Process weights and biases - preprocess functions already handle ttnn.from_torch
    if device is not None:
        processed_weight = preprocess_linear_weight(
            layer.weight,
            dtype=dtype,
            layout=layout,
            weights_mesh_mapper=weights_mesh_mapper,
            device=device,
        )
        # processed_weight is already a ttnn tensor with device placement from preprocess function
        layer_params["weight"] = processed_weight

        if layer.bias is not None:
            processed_bias = preprocess_linear_bias(
                layer.bias,
                dtype=dtype,
                layout=layout,
                weights_mesh_mapper=weights_mesh_mapper,
                device=device,
            )
            # processed_bias is already a ttnn tensor with device placement from preprocess function
            layer_params["bias"] = processed_bias
        else:
            layer_params["bias"] = None
    else:
        # For None device (testing), just store the original torch tensors
        layer_params["weight"] = layer.weight.clone().detach()
        if layer.bias is not None:
            layer_params["bias"] = layer.bias.clone().detach()
        else:
            layer_params["bias"] = None
    return layer_params


# Local preprocessing functions to avoid import issues
def preprocess_linear_weight(weight, *, dtype=None, layout=None, weights_mesh_mapper=None, device=None):
    """
    Preprocess linear layer weight for TTNN (transpose and convert).
    """
    if dtype is None:
        dtype = DEFAULT_DTYPE
    if layout is None:
        layout = DEFAULT_LAYOUT
    weight = weight.T.contiguous()

    kwargs = _build_ttnn_kwargs(dtype=dtype, layout=layout, weights_mesh_mapper=weights_mesh_mapper, device=device)
    weight = ttnn.from_torch(weight, **kwargs)
    return weight


def preprocess_linear_bias(bias, *, dtype=None, layout=None, weights_mesh_mapper=None, device=None):
    """
    Preprocess linear layer bias for TTNN (reshape and convert).
    """
    if dtype is None:
        dtype = DEFAULT_DTYPE
    if layout is None:
        layout = DEFAULT_LAYOUT
    bias = bias.reshape((1, -1))

    kwargs = _build_ttnn_kwargs(dtype=dtype, layout=layout, weights_mesh_mapper=weights_mesh_mapper, device=device)
    bias = ttnn.from_torch(bias, **kwargs)
    return bias


def linear_params(weight, bias, device, dtype):
    """A Linear's ``weight`` and ``bias`` as ``ttnn.linear`` takes them, on ``device``."""
    return SimpleNamespace(
        weight=preprocess_linear_weight(weight, dtype=dtype, device=device),
        bias=preprocess_linear_bias(bias, dtype=dtype, device=device),
    )


def preprocess_ms_deformable_attention_parameters(
    torch_model,
    *,
    device,
    dtype=None,
    layout=None,
    weights_mesh_mapper=None,
):
    """
    Preprocesses multi-scale deformable attention model parameters from PyTorch to ttnn format.

    Args:
        torch_model: PyTorch MultiScaleDeformableAttention model
        device: ttnn device
        dtype: Target data type for ttnn tensors
        layout: Target layout for ttnn tensors
        weights_mesh_mapper: Optional mesh mapper for distributed weights

    Returns:
        ParameterDict containing preprocessed ttnn tensors
    """

    parameters = {}

    # Process all linear layers using helper function
    layer_names = ["value_proj", "sampling_offsets", "attention_weights", "output_proj"]
    for layer_name in layer_names:
        if hasattr(torch_model, layer_name):
            layer = getattr(torch_model, layer_name)
            parameters[layer_name] = _process_linear_layer(
                layer, device, dtype=dtype, layout=layout, weights_mesh_mapper=weights_mesh_mapper
            )

    # Convert flat dictionary to object structure for dot notation access
    params_obj = convert_parameterdict_to_object(parameters)

    return params_obj


def create_ms_deformable_attention_parameters(
    torch_model_path: Optional[str] = None,
    torch_model: Optional[torch.nn.Module] = None,
    *,
    device,
    config,
    dtype=None,
    layout=None,
    weights_mesh_mapper=None,
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

    Returns:
        ParameterDict containing preprocessed ttnn tensors
    """

    # Get or create the PyTorch model
    if torch_model is None:
        if torch_model_path is not None:
            torch_model = torch.load(torch_model_path, map_location="cpu")
        else:
            from ..reference.ms_deformable_attention import MSDeformableAttention

            torch_model = MSDeformableAttention(config)
    torch_model.eval()

    return preprocess_ms_deformable_attention_parameters(
        torch_model,
        device=device,
        dtype=dtype,
        layout=layout,
        weights_mesh_mapper=weights_mesh_mapper,
    )


def create_spatial_cross_attention_parameters(sca, device, dtype=DEFAULT_DTYPE):
    """``reference.spatial_cross_attention.SpatialCrossAttention`` as TTSpatialCrossAttention
    takes it: the merged-slots ``output_proj`` and the nested ``MSDeformableAttention3D``'s
    value, offset and attention Linears (it has no output projection)."""
    deform = sca.deformable_attention
    return SimpleNamespace(
        output_proj=linear_params(sca.output_proj.weight, sca.output_proj.bias, device, dtype),
        deformable_attention=SimpleNamespace(
            **{
                name: linear_params(getattr(deform, name).weight, getattr(deform, name).bias, device, dtype)
                for name in ("value_proj", "sampling_offsets", "attention_weights")
            }
        ),
    )


def create_temporal_self_attention_parameters(tsa, device, dtype=DEFAULT_DTYPE):
    """``reference.temporal_self_attention.TemporalSelfAttention`` as TTTemporalSelfAttention
    takes it. The offset and attention Linears emit channels head-major,
    (head, queue, level, point[, xy]); they are reordered queue-major, so each stacked map's
    channels are one contiguous half of the row."""
    heads, queue = tsa.num_heads, tsa.num_bev_queue
    per_head = tsa.num_levels * tsa.num_points

    def queue_major(linear, width):
        order = torch.arange(heads * queue * per_head * width).view(heads, queue, per_head * width)
        order = order.permute(1, 0, 2).reshape(-1)
        return linear_params(linear.weight[order], linear.bias[order], device, dtype)

    return SimpleNamespace(
        value_proj=linear_params(tsa.value_proj.weight, tsa.value_proj.bias, device, dtype),
        sampling_offsets=queue_major(tsa.sampling_offsets, 2),
        attention_weights=queue_major(tsa.attention_weights, 1),
        output_proj=linear_params(tsa.output_proj.weight, tsa.output_proj.bias, device, dtype),
    )


def preprocess_layer_norm_parameters(layer_norm, *, device, dtype=None, layout=None, weights_mesh_mapper=None):
    """
    Process LayerNorm parameters for TTNN.
    """
    if dtype is None:
        dtype = DEFAULT_DTYPE
    if layout is None:
        layout = DEFAULT_LAYOUT

    # ttnn.layer_norm defaults to epsilon=1e-12; carry the module's own.
    layer_params = {"eps": layer_norm.eps}

    # Weight (gamma)
    kwargs = _build_ttnn_kwargs(dtype=dtype, layout=layout, weights_mesh_mapper=weights_mesh_mapper, device=device)
    if device is not None:
        layer_params["weight"] = ttnn.from_torch(layer_norm.weight, **kwargs)
        if layer_norm.bias is not None:
            layer_params["bias"] = ttnn.from_torch(layer_norm.bias, **kwargs)
        else:
            layer_params["bias"] = None
    else:
        # For testing with None device
        layer_params["weight"] = layer_norm.weight.clone().detach()
        if layer_norm.bias is not None:
            layer_params["bias"] = layer_norm.bias.clone().detach()
        else:
            layer_params["bias"] = None

    return layer_params


def create_bevformer_layer_parameters(layer, device, dtype=DEFAULT_DTYPE):
    """``reference.encoder.BEVFormerLayer`` as TTBEVFormerLayer takes it."""
    ffn = layer.ffns[0].layers
    return SimpleNamespace(
        tsa=create_temporal_self_attention_parameters(layer.attentions[0], device, dtype),
        sca=create_spatial_cross_attention_parameters(layer.attentions[1], device, dtype),
        ffn=SimpleNamespace(
            linear1=linear_params(ffn[0][0].weight, ffn[0][0].bias, device, dtype),
            linear2=linear_params(ffn[1].weight, ffn[1].bias, device, dtype),
        ),
        norms=[
            SimpleNamespace(**preprocess_layer_norm_parameters(norm, device=device, dtype=dtype))
            for norm in layer.norms
        ],
    )


def create_bevformer_encoder_parameters(encoder, device, dtype=DEFAULT_DTYPE):
    """``reference.encoder.BEVFormerEncoder`` as TTBEVFormerEncoder takes it, one entry per layer."""
    return SimpleNamespace(layers=[create_bevformer_layer_parameters(layer, device, dtype) for layer in encoder.layers])
