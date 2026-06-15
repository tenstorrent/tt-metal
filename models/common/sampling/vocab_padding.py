# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch


def build_invalid_vocab_mask(
    vocab_size: int,
    padded_vocab_size: int,
    max_batch_size: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor | None:
    """Build an additive logits mask for LM-head vocabulary padding.

    LM heads may pad output weights so the sharded matmul has legal tile/device
    dimensions. Those padded columns produce logits, but they are not real token
    IDs and must be masked before argmax or top-k sampling.
    """
    if vocab_size < 0:
        raise ValueError(f"vocab_size must be non-negative, got {vocab_size}")
    if padded_vocab_size < vocab_size:
        raise ValueError(f"padded_vocab_size ({padded_vocab_size}) must be >= vocab_size ({vocab_size})")
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
    if vocab_size == padded_vocab_size:
        return None

    mask = torch.zeros(1, 1, max_batch_size, padded_vocab_size, dtype=dtype)
    mask[..., vocab_size:] = torch.finfo(dtype).min
    return mask


def get_vocab_shard_dims(
    cluster_shape: tuple[int, int] | list[int],
    sampling_all_gather_axis: int = 0,
) -> tuple[int | None, int | None]:
    """Return the 2D mesh mapper dims for sharding vocab over the sampling TP axis."""
    if len(cluster_shape) != 2:
        raise ValueError(f"cluster_shape must have two dimensions, got {cluster_shape}")

    rows, cols = int(cluster_shape[0]), int(cluster_shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"cluster_shape dimensions must be positive, got {cluster_shape}")

    if rows == 1 and cols == 1:
        return (None, None)
    if rows == 1:
        return (None, 3)
    if cols == 1:
        return (3, None)
    if sampling_all_gather_axis == 0:
        return (3, None)
    if sampling_all_gather_axis == 1:
        return (None, 3)
    raise ValueError(f"sampling_all_gather_axis must be 0 or 1, got {sampling_all_gather_axis}")
