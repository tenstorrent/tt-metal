# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover - transformers compatibility
    DynamicCache = None


@dataclass
class DecodeState:
    """Decode state for incremental generation with KV cache."""

    past_key_values: object
    next_token_logits: torch.Tensor
    last_hidden_state: torch.Tensor
    multi_layer_hidden: torch.Tensor | None = None


def resolve_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = dtype_name.lower()
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def normalize_past_key_values(past_key_values: object) -> object:
    if DynamicCache is not None and isinstance(past_key_values, tuple):
        return DynamicCache.from_legacy_cache(past_key_values)
    return past_key_values


def clone_past_key_values(past_key_values: object) -> object:
    if isinstance(past_key_values, tuple):
        return tuple(tuple(tensor.clone() for tensor in layer) for layer in past_key_values)
    if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
        # DynamicCache is mutable; branches must not share the same instance.
        legacy_cache = past_key_values.to_legacy_cache()
        cloned_legacy = tuple(tuple(tensor.clone() for tensor in layer) for layer in legacy_cache)
        return DynamicCache.from_legacy_cache(cloned_legacy)
    return past_key_values


def cache_seq_len(past_key_values: object) -> int:
    if past_key_values is None:
        return 0
    if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
        if hasattr(past_key_values, "get_seq_length"):
            try:
                return int(past_key_values.get_seq_length())
            except Exception:
                pass
        legacy = past_key_values.to_legacy_cache()
        if legacy and legacy[0]:
            return int(legacy[0][0].shape[-2])
        return 0
    if isinstance(past_key_values, tuple) and past_key_values and past_key_values[0]:
        return int(past_key_values[0][0].shape[-2])
    return 0

