# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""KV cache wrapper for the Qwen3 GRPO completer.

Thin Python wrapper around the C++ ``ttml.models.KvCache`` that exposes the
interface the ttml Qwen3 attention layers consume:

  - ``update(layer_idx, key_states, value_states)`` -> ``(k, v)`` autograd tensors
  - ``get_seq_length()`` -> int

This is a self-contained port of ``examples/qwen3/utils/kv_cache.py`` without the
``utils.tensor_utils`` dependency (mask construction lives in the completer).
Cache allocation and slice-write logic are handled entirely on the C++ side and
the cache is lazily sized from the first ``update`` call, so it automatically
matches the per-device batch under DDP / FSDP batch sharding.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import ttml


class Qwen3KVCache:
    """Thin Python wrapper around the C++ ``ttml.models.KvCache``."""

    def __init__(self, num_layers: int, max_seq_len: int) -> None:
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self._cpp_cache: Optional[Any] = None

    def _lazy_init(self, batch_size: int, num_kv_heads: int, head_dim: int) -> None:
        """Create the C++ KvCache on the first update call (shape is known)."""
        self._cpp_cache = ttml.models.KvCache(
            ttml.models.KvCacheConfig(self.num_layers, batch_size, num_kv_heads, self.max_seq_len, head_dim)
        )

    def update(
        self,
        layer_idx: int,
        key_states: "ttml.autograd.Tensor",
        value_states: "ttml.autograd.Tensor",
    ) -> Tuple["ttml.autograd.Tensor", "ttml.autograd.Tensor"]:
        """Write new K/V into the cache; return ``(full_k, full_v)`` autograd tensors.

        During prefill (cache empty) the inputs are returned unchanged so the
        square-causal SDPA path is used. During decode the full cached K/V
        (``[B, num_kv_heads, max_seq_len, head_dim]``) are returned.
        """
        k_val = key_states.get_value()
        v_val = value_states.get_value()

        k_shape = list(k_val.shape)
        B, H, new_seq, hdim = k_shape[0], k_shape[1], k_shape[2], k_shape[3]

        if self._cpp_cache is None:
            self._lazy_init(B, H, hdim)

        is_prefill = self._cpp_cache.get_cache_position() == 0

        self._cpp_cache.update(layer_idx, k_val, v_val, new_seq)

        if is_prefill:
            return key_states, value_states

        k_out = self._cpp_cache.get_k_cache(layer_idx)
        v_out = self._cpp_cache.get_v_cache(layer_idx)
        return (
            ttml.autograd.create_tensor(k_out, requires_grad=False),
            ttml.autograd.create_tensor(v_out, requires_grad=False),
        )

    def get_seq_length(self) -> int:
        """Return the number of tokens currently stored in the cache."""
        if self._cpp_cache is None:
            return 0
        return self._cpp_cache.get_cache_position()

    def reset(self) -> None:
        """Reset the cache position for a new sequence (keeps buffers allocated)."""
        if self._cpp_cache is not None:
            self._cpp_cache.reset()

    def clear(self) -> None:
        """Free the cache's device buffers.

        Unlike :meth:`reset` (which only rewinds the position and keeps the K/V
        DRAM resident for reuse), this destroys the cached tensors so the memory
        is returned before the next, unrelated forward pass (e.g. the training
        ``compute_nlog_probs`` that runs right after generation).
        """
        if self._cpp_cache is not None:
            self._cpp_cache.clear()
            self._cpp_cache = None
