# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Sequence

import torch


def dequantize_tensor(tensor: torch.Tensor, inv_scale: torch.Tensor, block_shape: Sequence[int]) -> torch.Tensor:
    """
    Dequantize a tensor using block-wise inverse scales.

    Performance notes:
    - Avoids `repeat_interleave` over `inv_scale`, which would materialize a huge
      expanded scale tensor and significantly increase peak memory.
    - Uses reshape+broadcast to apply scales per block, so scale expansion stays
      virtual and the only dense payload is the output tensor itself.
    - Uses in-place multiply on a view (`mul_`) to reduce temporary allocations
      and memory bandwidth pressure.
    - Pads only when needed for non-divisible shapes, then slices back to the
      original shape.
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

    # When input is already float32, `tensor.float()` may alias the original storage. We clone in
    # that case before in-place math to avoid mutating caller-owned tensors (this impacted DeepSeek
    # module tests that pass tensors from `reference_model.state_dict()`).
    out = tensor.float()
    out = out.clone() if out.data_ptr() == tensor.data_ptr() else out

    # Only allocate a padded buffer when block coverage extends past the
    # original tensor shape (tail blocks on non-divisible dimensions).
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

    # Reshape into [blocks, block_size, ...] and apply broadcasted scales in
    # place, avoiding explicit per-element scale expansion.
    out_view = out.reshape(*interleaved_shape)
    out_view.mul_(inv_scale.float().reshape(*scale_broadcast_shape))
    out = out_view.reshape(*padded_shape)
    return out[original_slices]
