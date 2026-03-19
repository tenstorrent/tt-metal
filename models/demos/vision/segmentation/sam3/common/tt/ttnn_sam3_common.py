# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def tt_mlp_block(x, weight1, bias1, weight2, bias2, activation="gelu"):
    """Standard MLP block: Linear -> activation -> Linear.

    Args:
        x: Input ttnn tensor, expected in TILE_LAYOUT on device.
        weight1: Preprocessed weight for first linear (already transposed to [in, out]).
        bias1: Preprocessed bias for first linear (shape [1, 1, out]).
        weight2: Preprocessed weight for second linear (already transposed to [in, out]).
        bias2: Preprocessed bias for second linear (shape [1, 1, out]).
        activation: Activation function name. Only "gelu" is currently supported.

    Returns:
        Output ttnn tensor.
    """
    # First linear
    hidden = ttnn.linear(x, weight1, bias=bias1)

    # Activation
    if activation == "gelu":
        hidden = ttnn.gelu(hidden)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    # Second linear
    output = ttnn.linear(hidden, weight2, bias=bias2)
    return output


def tt_layer_norm(x, weight, bias, eps=1e-6):
    """Wrapper around ttnn.layer_norm.

    Args:
        x: Input ttnn tensor.
        weight: Layer norm scale parameter (gamma).
        bias: Layer norm shift parameter (beta).
        eps: Epsilon for numerical stability.

    Returns:
        Normalized ttnn tensor.
    """
    return ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=eps)


def preprocess_linear_weight(weight, dtype=ttnn.bfloat16):
    """Transpose PyTorch linear weight for ttnn matmul.

    PyTorch stores linear weights as [out_features, in_features].
    ttnn matmul expects [in_features, out_features], so we transpose.

    Args:
        weight: PyTorch weight tensor of shape [out_features, in_features].
        dtype: Target ttnn dtype.

    Returns:
        ttnn tensor of shape [in_features, out_features] in TILE_LAYOUT.
    """
    weight = weight.T.contiguous()
    return ttnn.from_torch(weight, dtype=dtype, layout=ttnn.TILE_LAYOUT)


def preprocess_linear_bias(bias, dtype=ttnn.bfloat16):
    """Reshape bias to (1, 1, N) for broadcasting with ttnn.linear.

    Args:
        bias: PyTorch bias tensor of shape [N].
        dtype: Target ttnn dtype.

    Returns:
        ttnn tensor of shape [1, 1, N] in TILE_LAYOUT.
    """
    bias = bias.reshape(1, 1, -1)
    return ttnn.from_torch(bias, dtype=dtype, layout=ttnn.TILE_LAYOUT)


def preprocess_conv2d_weight(weight, dtype=ttnn.bfloat16):
    """Convert PyTorch conv2d weight to ttnn format.

    PyTorch conv2d weight shape: [out_channels, in_channels, kH, kW].
    Permuted to [kH, kW, in_channels, out_channels] for ttnn conv2d.

    Args:
        weight: PyTorch conv2d weight tensor of shape [out_channels, in_channels, kH, kW].
        dtype: Target ttnn dtype.

    Returns:
        ttnn tensor of shape [1, 1, kH*kW*in_channels, out_channels] in ROW_MAJOR_LAYOUT.
    """
    # Permute from [out_C, in_C, kH, kW] -> [kH, kW, in_C, out_C]
    weight = weight.permute(2, 3, 1, 0).contiguous()
    out_channels = weight.shape[-1]
    # Flatten spatial+in_channel dims: [kH*kW*in_C, out_C] then unsqueeze to [1, 1, kH*kW*in_C, out_C]
    weight = weight.reshape(1, 1, -1, out_channels)
    return ttnn.from_torch(weight, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
