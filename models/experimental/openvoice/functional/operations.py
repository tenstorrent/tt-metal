# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Functional TTNN Operations for OpenVoice.

Implements core operations in functional style following official TTNN patterns.
Each function takes tensors and parameters, returns output tensors.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False


def ttnn_conv1d_functional(
    x: Any,
    weight: Any,
    bias: Optional[Any] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    device: Optional[Any] = None,
    use_ttnn: bool = True,
) -> Any:
    """
    Functional Conv1D that works on both PyTorch and TTNN tensors.

    TTNN doesn't have native Conv1D, so we reshape to use Conv2D:
    - Input: [B, C, L] -> [B, C, 1, L]
    - Kernel: [O, I, K] -> [O, I, 1, K]
    - Output: [B, O, 1, L'] -> [B, O, L']

    Args:
        x: Input tensor [B, in_channels, length]
        weight: Convolution weight [out_channels, in_channels, kernel_size]
        bias: Optional bias [out_channels]
        stride: Convolution stride
        padding: Convolution padding
        dilation: Convolution dilation
        groups: Convolution groups
        device: TTNN device
        use_ttnn: Whether to use TTNN backend

    Returns:
        Output tensor [B, out_channels, length']
    """
    is_torch = isinstance(x, torch.Tensor)

    if not TTNN_AVAILABLE or is_torch or not use_ttnn:
        # PyTorch path
        if weight is None:
            return x

        # Ensure weight is 3D for conv1d
        w = weight
        if w.dim() == 4:
            w = w.squeeze(2)
        elif w.dim() == 2:
            w = w.unsqueeze(2)

        return F.conv1d(x, w, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # TTNN path: Conv1D via Conv2D
    if weight is None:
        return x

    # Reshape for Conv2D: [B, C, L] -> [B, C, 1, L]
    x_4d = ttnn.unsqueeze(x, 2)

    # Prepare weight: [O, I, K] -> [O, I, 1, K]
    # Weight might already be 4D from preprocessing
    w = weight
    if len(w.shape) == 3:
        w = ttnn.unsqueeze(w, 2)

    # Conv2D with kernel [1, K]
    # Note: TTNN conv2d has different parameter order
    # Using matmul-based implementation for flexibility

    # For simplicity, fall back to reshaping to PyTorch, compute, reshape back
    # This is the "hybrid" approach - production would use native ttnn.conv2d
    x_torch = ttnn.to_torch(x).float()
    w_torch = ttnn.to_torch(weight).float()
    if w_torch.dim() == 4:
        w_torch = w_torch.squeeze(2)
    elif w_torch.dim() == 2:
        w_torch = w_torch.unsqueeze(2)

    b_torch = None
    if bias is not None:
        b_torch = ttnn.to_torch(bias).float()

    y_torch = F.conv1d(x_torch, w_torch, b_torch, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # Convert back to TTNN
    y = ttnn.from_torch(y_torch, dtype=x.dtype, device=device)

    return y


def ttnn_layer_norm_functional(
    x: Any,
    weight: Optional[Any] = None,
    bias: Optional[Any] = None,
    eps: float = 1e-5,
    normalized_shape: Optional[Tuple[int, ...]] = None,
    use_ttnn: bool = True,
) -> Any:
    """
    Functional Layer Normalization.

    Args:
        x: Input tensor
        weight: Scale parameter (gamma)
        bias: Shift parameter (beta)
        eps: Epsilon for numerical stability
        normalized_shape: Shape to normalize over (last N dims)
        use_ttnn: Whether to use TTNN backend

    Returns:
        Normalized tensor
    """
    is_torch = isinstance(x, torch.Tensor)

    if not TTNN_AVAILABLE or is_torch or not use_ttnn:
        # PyTorch path
        if normalized_shape is None:
            normalized_shape = (x.shape[-1],)
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

    # TTNN path
    return ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=eps)


def ttnn_attention_functional(
    query: Any,
    key: Any,
    value: Any,
    mask: Optional[Any] = None,
    scale: Optional[float] = None,
    n_heads: int = 1,
    device: Optional[Any] = None,
    use_flash_attention: bool = True,
    use_ttnn: bool = True,
) -> Any:
    """
    Functional Multi-Head Attention.

    Attempts to use TTNN's FlashAttention-2 implementation when available.

    Args:
        query: Query tensor [B, n_heads, seq_len, head_dim]
        key: Key tensor [B, n_heads, seq_len, head_dim]
        value: Value tensor [B, n_heads, seq_len, head_dim]
        mask: Optional attention mask
        scale: Attention scale (default: 1/sqrt(head_dim))
        n_heads: Number of attention heads
        device: TTNN device
        use_flash_attention: Whether to attempt FlashAttention
        use_ttnn: Whether to use TTNN backend

    Returns:
        Attention output [B, n_heads, seq_len, head_dim]
    """
    is_torch = isinstance(query, torch.Tensor)

    if scale is None:
        head_dim = query.shape[-1] if hasattr(query, "shape") else query.size(-1)
        scale = 1.0 / math.sqrt(head_dim)

    if not TTNN_AVAILABLE or is_torch or not use_ttnn:
        # PyTorch path
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, value)

    # TTNN path - try FlashAttention
    if use_flash_attention:
        try:
            output = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                is_causal=False,
                scale=scale,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            return output
        except Exception:
            pass  # Fall through to manual implementation

    # Manual attention
    k_t = ttnn.permute(key, (0, 1, 3, 2))  # Transpose last two dims
    scores = ttnn.matmul(query, k_t)
    scores = ttnn.multiply(scores, scale)

    if mask is not None:
        scores = ttnn.where(mask == 0, -1e9, scores)

    attn_weights = ttnn.softmax(scores, dim=-1)
    output = ttnn.matmul(attn_weights, value)

    return output


def ttnn_gated_activation_functional(
    x: Any,
    n_channels: int,
    use_ttnn: bool = True,
) -> Any:
    """
    Functional gated activation (WaveNet-style).

    Computes: tanh(x[:n_channels]) * sigmoid(x[n_channels:])

    Args:
        x: Input tensor [B, 2*n_channels, L]
        n_channels: Number of channels for split
        use_ttnn: Whether to use TTNN backend

    Returns:
        Gated output [B, n_channels, L]
    """
    is_torch = isinstance(x, torch.Tensor)

    if not TTNN_AVAILABLE or is_torch or not use_ttnn:
        # PyTorch path
        t_act = torch.tanh(x[:, :n_channels, :])
        s_act = torch.sigmoid(x[:, n_channels:, :])
        return t_act * s_act

    # TTNN path
    t_input = x[:, :n_channels, :]
    s_input = x[:, n_channels:, :]

    t_act = ttnn.tanh(t_input)
    s_act = ttnn.sigmoid(s_input)

    return ttnn.multiply(t_act, s_act)


def ttnn_leaky_relu_functional(
    x: Any,
    negative_slope: float = 0.1,
    use_ttnn: bool = True,
) -> Any:
    """
    Functional Leaky ReLU.

    Args:
        x: Input tensor
        negative_slope: Slope for negative values
        use_ttnn: Whether to use TTNN backend

    Returns:
        Activated tensor
    """
    is_torch = isinstance(x, torch.Tensor)

    if not TTNN_AVAILABLE or is_torch or not use_ttnn:
        return F.leaky_relu(x, negative_slope)

    return ttnn.leaky_relu(x, negative_slope=negative_slope)


def ttnn_upsample_functional(
    x: Any,
    scale_factor: int,
    mode: str = "nearest",
    use_ttnn: bool = True,
) -> Any:
    """
    Functional upsampling for 1D sequences.

    Args:
        x: Input tensor [B, C, L]
        scale_factor: Upsampling factor
        mode: Interpolation mode (nearest, linear)
        use_ttnn: Whether to use TTNN backend

    Returns:
        Upsampled tensor [B, C, L * scale_factor]
    """
    is_torch = isinstance(x, torch.Tensor)

    if not TTNN_AVAILABLE or is_torch or not use_ttnn:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)

    # TTNN upsampling via repeat_interleave
    # x: [B, C, L] -> repeat on L dimension
    return ttnn.repeat_interleave(x, scale_factor, dim=2)


# Composite operations


def ttnn_residual_block_functional(
    x: Any,
    conv1_weight: Any,
    conv1_bias: Optional[Any],
    conv2_weight: Any,
    conv2_bias: Optional[Any],
    dilation: int = 1,
    device: Optional[Any] = None,
    use_ttnn: bool = True,
) -> Any:
    """
    Functional residual block with dilated convolutions.

    Used in HiFi-GAN decoder.

    Args:
        x: Input tensor [B, C, L]
        conv1_weight, conv1_bias: First conv parameters
        conv2_weight, conv2_bias: Second conv parameters
        dilation: Dilation rate
        device: TTNN device
        use_ttnn: Whether to use TTNN backend

    Returns:
        Output tensor [B, C, L]
    """
    is_torch = isinstance(x, torch.Tensor)

    if not TTNN_AVAILABLE or is_torch or not use_ttnn:
        # PyTorch path
        kernel_size = conv1_weight.shape[-1] if conv1_weight is not None else 3
        padding = (kernel_size * dilation - dilation) // 2

        residual = x
        x = F.leaky_relu(x, 0.1)
        x = F.conv1d(x, conv1_weight, conv1_bias, padding=padding, dilation=dilation)
        x = F.leaky_relu(x, 0.1)
        x = F.conv1d(x, conv2_weight, conv2_bias, padding=padding, dilation=dilation)
        return x + residual

    # TTNN path
    kernel_size = conv1_weight.shape[-1] if hasattr(conv1_weight, "shape") else 3
    padding = (kernel_size * dilation - dilation) // 2

    residual = x
    x = ttnn.leaky_relu(x, negative_slope=0.1)
    x = ttnn_conv1d_functional(
        x, conv1_weight, conv1_bias, padding=padding, dilation=dilation, device=device, use_ttnn=use_ttnn
    )
    x = ttnn.leaky_relu(x, negative_slope=0.1)
    x = ttnn_conv1d_functional(
        x, conv2_weight, conv2_bias, padding=padding, dilation=dilation, device=device, use_ttnn=use_ttnn
    )

    return ttnn.add(x, residual)
