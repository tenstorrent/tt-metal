# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 KV cache and attention-mask helpers for autoregressive inference.

Wraps the C++ ``ttml.models.KvCache`` for cache storage/updates while exposing
the Python interface consumed by the Qwen3 attention layers. This is the single
source of truth shared by the ``examples/qwen3`` and ``examples/grpo`` examples;
it lives beside the Qwen3 model implementation (``ttml.models.qwen3``).

Imported lazily (not from ``ttml.models.qwen3.__init__``) so the heavy
``torch`` / ``ttnn`` module-level imports only happen when the cache is used.
"""

from __future__ import annotations

import torch
import ttnn

import ttml


# =====================================================================
# KV Cache wrapper around C++ ttml.models.KvCache
# =====================================================================


class KVCache:
    """Thin Python wrapper around the C++ ``ttml.models.KvCache``.

    Exposes the interface consumed by the Qwen3 attention layers:
      - ``update(layer_idx, key_states, value_states)`` -> ``(k, v)`` autograd tensors
      - ``get_attn_mask(seq_len)`` -> causal mask autograd tensor
      - ``get_seq_length()`` -> int

    Cache allocation and slice-write logic is handled entirely on the C++ side
    and the cache is lazily sized from the first ``update`` call, so it
    automatically matches the per-device batch under DDP / FSDP batch sharding.
    """

    def __init__(self, num_layers: int, max_seq_len: int, *, compressed: bool = False) -> None:
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.compressed = compressed
        self._cpp_cache: "ttml.models.KvCache | None" = None
        self._full_mask = None

    def _lazy_init(self, batch_size: int, num_kv_heads: int, head_dim: int) -> None:
        """Create the C++ KvCache on the first update call (shape is known)."""
        self._cpp_cache = ttml.models.KvCache(
            ttml.models.KvCacheConfig(self.num_layers, batch_size, num_kv_heads, self.max_seq_len, head_dim)
        )

    def _ensure_mask(self):
        if self._full_mask is not None:
            return
        device = ttml.autograd.AutoContext.get_instance().get_device()
        mask_torch = torch.tril(torch.ones(1, 1, self.max_seq_len, self.max_seq_len, dtype=torch.bfloat16))
        host = ttnn.from_torch(mask_torch, dtype=ttnn.bfloat16)
        dev = ttnn.to_device(host, device)
        self._full_mask = ttnn.tilize_with_zero_padding(dev)

    def get_attn_mask(self, seq_len=None):
        """Return a slice of the pre-allocated causal mask for the current step.

        During prefill (cache_position == 0): pass seq_len = prompt length.
        Returns [1, 1, seq_len, seq_len] square causal mask.

        During decode: returns [1, 1, 1, kv_len] where kv_len = max_seq_len.
        """
        self._ensure_mask()
        cache_pos = self.get_seq_length()
        if cache_pos == 0 and seq_len is not None:
            mask_slice = ttnn.slice(self._full_mask, [0, 0, 0, 0], [1, 1, seq_len, seq_len])
        else:
            mask_slice = ttnn.slice(
                self._full_mask,
                [0, 0, cache_pos, 0],
                [1, 1, cache_pos + 1, self.max_seq_len],
            )
        return ttml.autograd.create_tensor(mask_slice, requires_grad=False)

    def update(
        self,
        layer_idx: int,
        key_states: "ttml.autograd.Tensor",
        value_states: "ttml.autograd.Tensor",
    ):
        """Write new K/V into the cache; return ``(full_k, full_v)`` autograd tensors.

        During prefill (cache empty) the inputs are returned unchanged so the
        square-causal SDPA path is used. During decode the full cached K/V
        (``[B, num_kv_heads, max_seq_len, head_dim]``) are returned. Delegates
        storage and position tracking to the C++ KvCache.
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
        ``compute_nlog_probs`` that runs right after generation in GRPO).
        """
        if self._cpp_cache is not None:
            self._cpp_cache.clear()
            self._cpp_cache = None
