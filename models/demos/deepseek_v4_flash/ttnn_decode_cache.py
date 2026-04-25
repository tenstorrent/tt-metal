# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Batch1DecodeLayerCache:
    """Host-owned cache state for the tiny batch-1 decode stepping stone.

    ``current_position`` is the absolute position of the next token to decode,
    equal to the number of tokens represented by ``attention_input_history``.
    The cache stores compressed KV equivalents because the exact DeepSeek V4
    sparse decode tensor-cache structure is not yet part of this scaffold.
    """

    layer_id: int
    current_position: int
    batch_size: int
    hidden_size: int
    compress_ratio: int
    head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    attention_input_history: torch.Tensor
    compressed_kv: torch.Tensor
    index_compressed_kv: torch.Tensor
    last_topk_idxs: torch.Tensor | None = None

    def __post_init__(self) -> None:
        validate_batch1_decode_layer_cache(self)

    @property
    def compressed_cache_length(self) -> int:
        return int(self.compressed_kv.shape[1])

    @property
    def index_cache_length(self) -> int:
        return int(self.index_compressed_kv.shape[1])


@dataclass(frozen=True)
class Batch1DecodeCache:
    """Host-owned model cache for a stack of tiny decoder layers."""

    layer_caches: tuple[Batch1DecodeLayerCache, ...]

    def __post_init__(self) -> None:
        validate_batch1_decode_cache(self)

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def current_position(self) -> int:
        return int(self.layer_caches[0].current_position)

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(cache.layer_id for cache in self.layer_caches)


def advance_batch1_decode_layer_cache(
    cache: Batch1DecodeLayerCache,
    *,
    attention_input_token: torch.Tensor,
    compressed_kv: torch.Tensor,
    index_compressed_kv: torch.Tensor,
    last_topk_idxs: torch.Tensor,
) -> Batch1DecodeLayerCache:
    """Return the next cache state after appending one decoded token."""

    validate_decode_attention_input_token(
        attention_input_token,
        batch_size=cache.batch_size,
        hidden_size=cache.hidden_size,
    )
    next_history = torch.cat(
        [cache.attention_input_history, attention_input_token.to(cache.attention_input_history.dtype)],
        dim=1,
    ).contiguous()
    return Batch1DecodeLayerCache(
        layer_id=cache.layer_id,
        current_position=cache.current_position + 1,
        batch_size=cache.batch_size,
        hidden_size=cache.hidden_size,
        compress_ratio=cache.compress_ratio,
        head_dim=cache.head_dim,
        index_n_heads=cache.index_n_heads,
        index_head_dim=cache.index_head_dim,
        index_topk=cache.index_topk,
        attention_input_history=next_history,
        compressed_kv=compressed_kv.contiguous(),
        index_compressed_kv=index_compressed_kv.contiguous(),
        last_topk_idxs=last_topk_idxs.contiguous(),
    )


def validate_batch1_decode_cache(cache: Batch1DecodeCache) -> None:
    if not isinstance(cache.layer_caches, tuple):
        raise TypeError("layer_caches must be a tuple")
    if len(cache.layer_caches) == 0:
        raise ValueError("layer_caches must be non-empty")
    positions = {int(layer_cache.current_position) for layer_cache in cache.layer_caches}
    if len(positions) != 1:
        raise ValueError(f"all layer caches must have the same current_position, got {sorted(positions)}")
    layer_ids = tuple(layer_cache.layer_id for layer_cache in cache.layer_caches)
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError(f"layer caches must not contain duplicate layer ids, got {layer_ids}")
    for layer_cache in cache.layer_caches:
        if layer_cache.batch_size != 1:
            raise ValueError(f"batch-1 decode cache requires batch_size=1, got {layer_cache.batch_size}")


def validate_batch1_decode_layer_cache(cache: Batch1DecodeLayerCache) -> None:
    _validate_nonnegative_int(cache.layer_id, "layer_id")
    _validate_positive_int(cache.current_position, "current_position")
    if cache.batch_size != 1:
        raise ValueError(f"batch-1 decode cache requires batch_size=1, got {cache.batch_size}")
    _validate_positive_int(cache.hidden_size, "hidden_size")
    _validate_positive_int(cache.compress_ratio, "compress_ratio")
    _validate_positive_int(cache.head_dim, "head_dim")
    _validate_positive_int(cache.index_n_heads, "index_n_heads")
    _validate_positive_int(cache.index_head_dim, "index_head_dim")
    _validate_positive_int(cache.index_topk, "index_topk")

    expected_history_shape = (1, cache.current_position, cache.hidden_size)
    if tuple(cache.attention_input_history.shape) != expected_history_shape:
        raise ValueError(
            f"attention_input_history must have shape {expected_history_shape}, "
            f"got {tuple(cache.attention_input_history.shape)}"
        )

    expected_cache_len = cache.current_position // cache.compress_ratio
    expected_kv_shape = (1, expected_cache_len, cache.head_dim)
    if tuple(cache.compressed_kv.shape) != expected_kv_shape:
        raise ValueError(f"compressed_kv must have shape {expected_kv_shape}, got {tuple(cache.compressed_kv.shape)}")
    expected_index_shape = (1, expected_cache_len, cache.index_head_dim)
    if tuple(cache.index_compressed_kv.shape) != expected_index_shape:
        raise ValueError(
            f"index_compressed_kv must have shape {expected_index_shape}, "
            f"got {tuple(cache.index_compressed_kv.shape)}"
        )

    if cache.last_topk_idxs is not None:
        validate_decode_topk_idxs(cache.last_topk_idxs, batch_size=1)


def validate_decode_attention_input_token(
    attention_input_token: torch.Tensor,
    *,
    batch_size: int,
    hidden_size: int,
) -> None:
    expected_shape = (batch_size, 1, hidden_size)
    if tuple(attention_input_token.shape) != expected_shape:
        raise ValueError(
            f"attention_input_token must have shape {expected_shape}, got {tuple(attention_input_token.shape)}"
        )


def validate_decode_topk_idxs(topk_idxs: torch.Tensor, *, batch_size: int) -> None:
    if topk_idxs.ndim != 3:
        raise ValueError(f"last_topk_idxs must have shape [batch, 1, topk], got {tuple(topk_idxs.shape)}")
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"last_topk_idxs dtype must be int32 or int64, got {topk_idxs.dtype}")
    if tuple(topk_idxs.shape[:2]) != (batch_size, 1):
        raise ValueError(
            f"last_topk_idxs batch/token shape must be {(batch_size, 1)}, got {tuple(topk_idxs.shape[:2])}"
        )


def _validate_positive_int(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{label} must be positive, got {value}")


def _validate_nonnegative_int(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value}")
