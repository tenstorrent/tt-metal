# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InvalidVocabTailMask:
    """A compact additive mask for the tile-aligned invalid tail of the final vocab shard."""

    mask: torch.Tensor
    tail_width: int
    shard_width: int
    num_vocab_shards: int


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

    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"dtype must be a floating point torch dtype, got {dtype}")

    mask = torch.zeros(1, 1, max_batch_size, padded_vocab_size, dtype=dtype)
    mask[..., vocab_size:] = torch.finfo(dtype).min
    return mask


def _validate_cluster_shape(cluster_shape: tuple[int, int] | list[int]) -> tuple[int, int]:
    cluster_shape = tuple(cluster_shape)
    if len(cluster_shape) != 2:
        raise ValueError(f"cluster_shape must have two dimensions, got {cluster_shape}")

    rows, cols = int(cluster_shape[0]), int(cluster_shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"cluster_shape dimensions must be positive, got {cluster_shape}")
    return rows, cols


def get_vocab_num_shards(
    cluster_shape: tuple[int, int] | list[int],
    sampling_all_gather_axis: int = 0,
) -> int:
    """Return how many mesh partitions own contiguous slices of the vocab dimension."""
    rows, cols = _validate_cluster_shape(cluster_shape)

    if rows == 1 and cols == 1:
        return 1
    if rows == 1:
        return cols
    if cols == 1:
        return rows
    if sampling_all_gather_axis == 0:
        return rows
    if sampling_all_gather_axis == 1:
        return cols
    raise ValueError(f"sampling_all_gather_axis must be 0 or 1, got {sampling_all_gather_axis}")


def build_tail_invalid_vocab_mask(
    vocab_size: int,
    padded_vocab_size: int,
    max_batch_size: int,
    cluster_shape: tuple[int, int] | list[int],
    sampling_all_gather_axis: int = 0,
    *,
    dtype: torch.dtype = torch.bfloat16,
    tile_size: int = 32,
) -> InvalidVocabTailMask | None:
    """Build a compact mask for padding that lives only at the final shard tail.

    The sampling logits are sharded into equal local vocab widths. For model
    shapes like Qwen3-32B on T3K, all invalid IDs are a small tile-aligned suffix
    of the last local shard. In that case callers can mask only the local tail
    slice instead of adding a full-vocab all-zero mask on every device.

    Returns ``None`` when the invalid range is not a tile-aligned final-shard
    suffix; callers should use ``build_invalid_vocab_mask`` as the correctness
    fallback.
    """
    if vocab_size < 0:
        raise ValueError(f"vocab_size must be non-negative, got {vocab_size}")
    if padded_vocab_size < vocab_size:
        raise ValueError(f"padded_vocab_size ({padded_vocab_size}) must be >= vocab_size ({vocab_size})")
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if vocab_size == padded_vocab_size:
        return None
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"dtype must be a floating point torch dtype, got {dtype}")

    num_vocab_shards = get_vocab_num_shards(cluster_shape, sampling_all_gather_axis)
    if padded_vocab_size % num_vocab_shards != 0:
        return None

    shard_width = padded_vocab_size // num_vocab_shards
    tail_width = padded_vocab_size - vocab_size
    if tail_width > shard_width:
        return None
    if tail_width % tile_size != 0 or (shard_width - tail_width) % tile_size != 0:
        return None

    mask = torch.zeros(1, 1, max_batch_size, tail_width * num_vocab_shards, dtype=dtype)
    final_tail_start = tail_width * (num_vocab_shards - 1)
    mask[..., final_tail_start:] = torch.finfo(dtype).min
    return InvalidVocabTailMask(
        mask=mask,
        tail_width=tail_width,
        shard_width=shard_width,
        num_vocab_shards=num_vocab_shards,
    )


def get_vocab_shard_dims(
    cluster_shape: tuple[int, int] | list[int],
    sampling_all_gather_axis: int = 0,
) -> tuple[int | None, int | None]:
    """Return the 2D mesh mapper dims for sharding vocab over the sampling TP axis."""
    rows, cols = _validate_cluster_shape(cluster_shape)

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
