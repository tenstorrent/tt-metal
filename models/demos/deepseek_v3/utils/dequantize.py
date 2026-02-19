# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Sequence

import torch


def dequantize_tensor(tensor: torch.Tensor, inv_scale: torch.Tensor, block_shape: Sequence[int]) -> torch.Tensor:
    """
    Dequantize a tensor using block-wise inverse scales.

    This implementation avoids materializing a fully expanded inverse-scale tensor
    via repeat_interleave and instead applies scales with broadcasted block views.
    """
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
    padded_shape = tuple(inv_scale.shape[i] * block_shape[i] for i in range(tensor.ndim))
    original_slices = tuple(slice(0, size) for size in original_shape)

    out = tensor.float()
    if padded_shape != original_shape:
        padded = torch.zeros(padded_shape, dtype=out.dtype)
        padded[original_slices] = out
        out = padded

    interleaved_shape: list[int] = []
    scale_broadcast_shape: list[int] = []
    for dim, block_dim in enumerate(block_shape):
        blocks = inv_scale.shape[dim]
        interleaved_shape.extend([blocks, block_dim])
        scale_broadcast_shape.extend([blocks, 1])

    out_view = out.reshape(*interleaved_shape)
    out_view.mul_(inv_scale.float().reshape(*scale_broadcast_shape))
    out = out_view.reshape(*padded_shape)
    return out[original_slices]
