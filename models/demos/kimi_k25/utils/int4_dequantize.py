# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
INT4 group-32 weight dequantizer for Kimi K2.5.

Handles W4A16 symmetric INT4 quantization as used by the moonshotai/Kimi-K2.5
model (compressed_tensors format). Only routed expert linear weights are
quantized; attention, shared experts, and lm_head remain BF16.

Format details:
- 2 INT4 values packed per byte (little-endian nibbles: low=even, high=odd)
- Per-group scale: shape (out_features, in_features // group_size), dtype BF16
- Symmetric range: [-8, 7], stored as unsigned [0, 15] with zero-point 8
- group_size: 32 (Kimi K2.5 default)

Usage::

    from models.demos.kimi_k25.utils.int4_dequantize import dequantize_int4_weight

    # packed: uint8 tensor from safetensors, shape (out, in//2)
    # scales: bfloat16 tensor, shape (out, in//group_size)
    weight_bf16 = dequantize_int4_weight(packed, scales)
"""

from __future__ import annotations

import torch


_DEFAULT_GROUP_SIZE = 32  # Kimi K2.5 routed-expert group size


def unpack_int4_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack byte-packed INT4 values into a signed int8 tensor.

    Each byte holds two 4-bit values:
      - low nibble  (bits 0-3) → element at even column index
      - high nibble (bits 4-7) → element at odd column index

    The returned tensor has dtype int8 and shape ``(*packed.shape[:-1], packed.shape[-1] * 2)``.
    Values are in the range [-8, 7] (symmetric INT4, zero-point 8).

    Args:
        packed: uint8 tensor of arbitrary leading dims, last dim = in_features // 2.

    Returns:
        Signed int8 tensor with last dim doubled (= in_features).
    """
    low = (packed & 0x0F).to(torch.int8)           # even column indices
    high = ((packed >> 4) & 0x0F).to(torch.int8)   # odd column indices

    # Subtract zero-point to recover signed [-8, 7]
    low = low - 8
    high = high - 8

    # Interleave: stack along a new last dim → (..., cols_packed, 2) then flatten
    # → (..., cols_packed * 2) = (..., in_features), preserving element order:
    #   output[..., 2*j] = low[..., j]   (originally at even index 2*j)
    #   output[..., 2*j+1] = high[..., j] (originally at odd index 2*j+1)
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def dequantize_int4_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = _DEFAULT_GROUP_SIZE,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a W4A16 symmetric INT4 weight tensor to floating point.

    Implements::

        W_fp = int4_weight * scale_per_group

    where ``int4_weight ∈ [-8, 7]`` and scales are broadcast over ``group_size``
    consecutive input-feature positions.

    Args:
        packed: uint8 tensor, shape ``(out_features, in_features // 2)``.
            Two INT4 values packed per byte (low nibble first).
        scales: floating-point tensor, shape ``(out_features, in_features // group_size)``.
            Per-group dequantization scales. Typically BF16 in Kimi K2.5
            safetensors checkpoint, but any float dtype is accepted.
        group_size: number of weight elements sharing one scale value.
            Must divide ``in_features`` evenly. Default: 32.
        output_dtype: output tensor dtype. Default: ``torch.bfloat16``.

    Returns:
        Dequantized weight tensor, shape ``(out_features, in_features)``,
        dtype ``output_dtype``.

    Raises:
        ValueError: if shapes are inconsistent with the expected INT4 layout.
    """
    if packed.ndim != 2:
        raise ValueError(f"Expected 2D packed tensor, got shape {tuple(packed.shape)}")
    if scales.ndim != 2:
        raise ValueError(f"Expected 2D scales tensor, got shape {tuple(scales.shape)}")

    out_features, cols_packed = packed.shape
    in_features = cols_packed * 2

    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )

    n_groups = in_features // group_size
    expected_scale_shape = (out_features, n_groups)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"Scale shape mismatch: expected {expected_scale_shape}, got {tuple(scales.shape)}"
        )

    # Unpack to signed int8, cast to float32 for arithmetic
    int4_vals = unpack_int4_nibbles(packed)  # (out_features, in_features), int8

    # Reshape into groups, apply broadcast scales, then flatten
    #   int4_vals grouped: (out_features, n_groups, group_size)
    #   scales expanded:   (out_features, n_groups, 1)  ← broadcast over group_size
    weights_grouped = int4_vals.float().reshape(out_features, n_groups, group_size)
    scales_expanded = scales.float().unsqueeze(-1)  # (out_features, n_groups, 1)
    dequantized = (weights_grouped * scales_expanded).reshape(out_features, in_features)

    return dequantized.to(output_dtype)


def dequantize_int4_weight_from_state_dict(
    state_dict: dict,
    weight_key: str,
    scale_key: str,
    group_size: int = _DEFAULT_GROUP_SIZE,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Convenience wrapper: dequantize a named weight from a safetensors state dict.

    The Kimi K2.5 checkpoint (compressed_tensors format) stores routed-expert
    weights under keys like::

        model.layers.0.mlp.experts.0.w1.weight          ← uint8 packed
        model.layers.0.mlp.experts.0.w1.weight_scale    ← bfloat16 per-group scales

    Args:
        state_dict: dict mapping key → tensor (e.g., from ``safetensors.torch.load_file``).
        weight_key: key for the packed uint8 weight tensor.
        scale_key:  key for the per-group scale tensor.
        group_size: INT4 group size (default 32).
        output_dtype: output dtype (default bfloat16).

    Returns:
        Dequantized weight tensor, shape ``(out_features, in_features)``.
    """
    packed = state_dict[weight_key]
    scales = state_dict[scale_key]
    return dequantize_int4_weight(packed, scales, group_size=group_size, output_dtype=output_dtype)
