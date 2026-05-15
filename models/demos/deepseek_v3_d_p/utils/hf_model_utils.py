# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
FP8 dequantization utilities for DeepSeek-V3 HuggingFace checkpoints.

Provides two dequantization paths:
- ``dequantize_state_dict``: TT-style (float32 intermediates, in-place mul)
- ``dequantize_state_dict_hf``: HF-style (matches ``transformers.Fp8Dequantize``)
"""

from collections.abc import Mapping
from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig


def _get_weight_block_shape_from_quant_config(quantization_config: Any) -> tuple[int, ...]:
    if not isinstance(quantization_config, dict):
        raise ValueError(
            "Missing DeepSeek quantization_config.weight_block_size. "
            "The source checkpoint config must retain the original quantization metadata."
        )
    block_shape = quantization_config.get("weight_block_size")
    if not isinstance(block_shape, (list, tuple)) or not block_shape:
        raise ValueError(
            "Missing DeepSeek quantization_config.weight_block_size. "
            "The source checkpoint config must retain the original quantization metadata."
        )
    return tuple(int(dim) for dim in block_shape)


def get_weight_block_shape(hf_config: PretrainedConfig) -> tuple[int, ...]:
    return _get_weight_block_shape_from_quant_config(getattr(hf_config, "quantization_config", None))


def dequantize_weight_tensor(
    tensor: torch.Tensor,
    inv_scale: torch.Tensor,
    block_shape: tuple[int, ...] | list[int],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if tensor.ndim != inv_scale.ndim:
        raise ValueError(f"Tensor and inverse scale must have same ndim, got {tensor.ndim} and {inv_scale.ndim}")
    if len(block_shape) != tensor.ndim:
        raise ValueError(
            f"Block shape rank mismatch, got len(block_shape)={len(block_shape)} and tensor.ndim={tensor.ndim}"
        )
    if any(inv_scale.shape[i] * block_shape[i] < tensor.shape[i] for i in range(tensor.ndim)):
        raise ValueError(
            "Inverse scale shape does not cover tensor shape: "
            f"tensor={tuple(tensor.shape)}, inv_scale={tuple(inv_scale.shape)}, block_shape={tuple(block_shape)}"
        )

    original_shape = tuple(tensor.shape)
    padded_shape = tuple(inv_scale.shape[i] * int(block_shape[i]) for i in range(tensor.ndim))
    original_slices = tuple(slice(0, size) for size in original_shape)

    out = tensor.float()
    out = out.clone() if out.data_ptr() == tensor.data_ptr() else out
    if padded_shape != original_shape:
        padded = torch.zeros(padded_shape, dtype=out.dtype)
        padded[original_slices] = out
        out = padded

    interleaved_shape: list[int] = []
    scale_broadcast_shape: list[int] = []
    for dim, block_dim in enumerate(block_shape):
        blocks = inv_scale.shape[dim]
        interleaved_shape.extend([blocks, int(block_dim)])
        scale_broadcast_shape.extend([blocks, 1])

    out_view = out.reshape(*interleaved_shape)
    out_view.mul_(inv_scale.float().reshape(*scale_broadcast_shape))
    out = out_view.reshape(*padded_shape)
    return out[original_slices].to(dtype).contiguous()


def dequantize_weight_tensor_hf(
    tensor: torch.Tensor,
    inv_scale: torch.Tensor,
    block_shape: tuple[int, ...] | list[int],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """HuggingFace-style FP8 dequantization matching ``transformers.Fp8Dequantize``.

    Replicates the exact computation order used by HuggingFace's
    ``transformers.integrations.finegrained_fp8.Fp8Dequantize.convert()``
    (including the padded fallback for non-block-aligned shapes) so that the
    resulting bfloat16 weights are bit-identical to those produced by
    ``AutoModelForCausalLM.from_pretrained`` with FP8 quantized checkpoints.

    Key differences from ``dequantize_weight_tensor``:
      * Converts FP8 to ``inv_scale.dtype`` (not explicitly float32).
      * Uses ``F.pad`` for non-aligned shapes.
      * Out-of-place multiply (``*``) instead of in-place ``mul_``.
    """
    if tensor.ndim != 2:
        raise ValueError(f"HF-style dequant expects 2-D weight tensors, got ndim={tensor.ndim}")
    if len(block_shape) != 2:
        raise ValueError(f"block_shape must have length 2, got {len(block_shape)}")

    orig_rows, cols = tensor.shape
    block_m, block_n = int(block_shape[0]), int(block_shape[1])

    compute_dtype = inv_scale.dtype
    quantized = tensor.to(compute_dtype)

    pad_rows = (block_m - orig_rows % block_m) % block_m
    pad_cols = (block_n - cols % block_n) % block_n

    if pad_rows > 0 or pad_cols > 0:
        quantized = torch.nn.functional.pad(quantized, (0, pad_cols, 0, pad_rows))

    padded_rows, padded_cols = quantized.shape[-2:]

    reshaped = quantized.reshape(-1, padded_rows // block_m, block_m, padded_cols // block_n, block_n)
    expanded_scales = inv_scale.reshape(-1, padded_rows // block_m, padded_cols // block_n)
    expanded_scales = expanded_scales.unsqueeze(-1).unsqueeze(2)

    dequantized = reshaped * expanded_scales
    dequantized = dequantized.reshape(padded_rows, padded_cols)

    if pad_rows > 0 or pad_cols > 0:
        dequantized = dequantized[:orig_rows, :cols]

    return dequantized.to(dtype).contiguous()


def dequantize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    hf_config: PretrainedConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """TT-style FP8 dequantization (float32 intermediates, in-place mul)."""
    dequantized_state_dict: dict[str, torch.Tensor] = {}
    block_shape = get_weight_block_shape(hf_config)

    for name in sorted(key for key in state_dict.keys() if not key.endswith("_scale_inv")):
        tensor = state_dict[name]
        if tensor is None:
            raise ValueError(f"Expected tensor {name} to exist in state_dict but it was None")

        scale_name = f"{name}_scale_inv"
        if scale_name in state_dict:
            dequantized_state_dict[name] = dequantize_weight_tensor(
                tensor, state_dict[scale_name], block_shape, dtype=dtype
            )
            continue

        if tensor.dtype == torch.float8_e4m3fn:
            raise ValueError(f"Found float8 tensor '{name}' without matching inverse scale '{scale_name}'.")
        dequantized_state_dict[name] = tensor.to(dtype).contiguous() if tensor.is_floating_point() else tensor.clone()

    return dequantized_state_dict


def dequantize_state_dict_hf(
    state_dict: Mapping[str, torch.Tensor],
    hf_config: PretrainedConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """HF-style FP8 dequantization matching ``transformers.Fp8Dequantize``."""
    dequantized_state_dict: dict[str, torch.Tensor] = {}
    block_shape = get_weight_block_shape(hf_config)

    for name in sorted(key for key in state_dict.keys() if not key.endswith("_scale_inv")):
        tensor = state_dict[name]
        if tensor is None:
            raise ValueError(f"Expected tensor {name} to exist in state_dict but it was None")

        scale_name = f"{name}_scale_inv"
        if scale_name in state_dict:
            dequantized_state_dict[name] = dequantize_weight_tensor_hf(
                tensor, state_dict[scale_name], block_shape, dtype=dtype
            )
            continue

        if tensor.dtype == torch.float8_e4m3fn:
            raise ValueError(f"Found float8 tensor '{name}' without matching inverse scale '{scale_name}'.")
        dequantized_state_dict[name] = tensor.to(dtype).contiguous() if tensor.is_floating_point() else tensor.clone()

    return dequantized_state_dict
