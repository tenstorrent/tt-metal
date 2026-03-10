# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""KV cache and related attention mask utilities for autoregressive inference."""

import torch
import ttnn
import ttml

from utils.tensor_utils import get_device


# =====================================================================
# Tensor creation helper
# =====================================================================


def _to_device_tiled(tensor_torch, device, dtype=ttnn.bfloat16):
    """Host torch tensor -> device tilized ttml tensor (no grad)."""
    host = ttnn.from_torch(tensor_torch, dtype=dtype)
    dev = ttnn.to_device(host, device)
    tiled = ttnn.tilize_with_zero_padding(dev)
    return ttml.autograd.create_tensor(tiled, requires_grad=False)


# =====================================================================
# Attention mask helpers
# =====================================================================


def _causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, dtype=torch.bfloat16))
    return _to_device_tiled(mask, device)


def _step_attn_mask(step, prefill_seq_len, device):
    """Return the attention mask for the current step."""

    # Prefil mask
    if step == 0 and prefill_seq_len is not None:
        return _causal_mask(prefill_seq_len, device)
    # In decode mode we can run without mask since we attend to all tokens in the cache
    # and the kv_cache is init with zeros - even with
    return _causal_mask(prefill_seq_len + step, device)


# =====================================================================
# KV Cache (static pre-allocated, inference-only)
# =====================================================================


class KVCache:
    """Static pre-allocated KV cache for autoregressive inference.

    Pre-allocates fixed-size DRAM tensors [B, num_kv_heads, max_seq_len, head_dim]
    on the first call and uses ttnn.experimental.slice_write for O(1) in-place updates.
    Also pre-allocates a full [1, 1, max_seq_len, max_seq_len] causal mask and
    returns slices of it via get_attn_mask() — no per-step mask creation.

    Not differentiable — use only with GradMode.DISABLED.
    """

    _TILE = 32

    def __init__(self, num_layers: int, max_seq_len: int, *, compressed: bool = False):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.compressed = False
        self._cache_position: int = 0
        self._initialized: bool = False
        self.key_cache: list = [None] * num_layers
        self.value_cache: list = [None] * num_layers
        self._full_mask = None

    def _lazy_init(self, batch_size: int, num_kv_heads: int, head_dim: int) -> None:
        """Pre-allocate DRAM cache buffers and causal mask on first update call."""
        device = get_device()
        dram_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM
        )
        cache_shape = [batch_size, num_kv_heads, self.max_seq_len, head_dim]
        for i in range(self.num_layers):
            self.key_cache[i] = ttnn.zeros(
                cache_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram_config
            )
            self.value_cache[i] = ttnn.zeros(
                cache_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram_config
            )

        self._initialized = True

    def _ensure_mask(self):
        if self._full_mask is not None:
            return
        device = get_device()
        mask_torch = torch.tril(
            torch.ones(1, 1, self.max_seq_len, self.max_seq_len, dtype=torch.bfloat16)
        )
        host = ttnn.from_torch(mask_torch, dtype=ttnn.bfloat16)
        dev = ttnn.to_device(host, device)
        self._full_mask = ttnn.tilize_with_zero_padding(dev)

    def _kv_seq_len(self):
        """K/V sequence dimension that update() will return for the current decode step."""
        if not self.compressed:
            return self.max_seq_len
        full_end = self._cache_position + 1
        return min(
            ((full_end + self._TILE - 1) // self._TILE) * self._TILE,
            self.max_seq_len,
        )

    def get_attn_mask(self, seq_len=None):
        """Return a slice of the pre-allocated causal mask for the current step.

        During prefill (cache_position == 0): pass seq_len = prompt length.
        Returns [1, 1, seq_len, seq_len] square causal mask.

        During decode: returns [1, 1, 1, kv_len] where kv_len matches the K/V
        tensors returned by update() — max_seq_len in full mode, tile-aligned
        cache window in compressed mode.
        """
        self._ensure_mask()
        if self._cache_position == 0 and seq_len is not None:
            mask_slice = ttnn.slice(
                self._full_mask, [0, 0, 0, 0], [1, 1, seq_len, seq_len]
            )
        else:
            pos = self._cache_position
            kv_len = self._kv_seq_len()
            mask_slice = ttnn.slice(
                self._full_mask,
                [0, 0, pos, 0],
                [1, 1, pos + 1, kv_len],
            )
        return ttml.autograd.create_tensor(mask_slice, requires_grad=False)

    def update(self, layer_idx: int, key_states, value_states):
        """Write new K/V in-place via experimental.slice_write; return (full_k, full_v) ttml tensors.

        During prefill (_cache_position == 0 at entry): writes to cache and returns
        the original tensors directly — avoids a redundant read-back from DRAM.

        During decode: writes the new token at _cache_position and returns a slice
        of the pre-allocated cache covering positions 0..(_cache_position + new_seq).
        _cache_position is advanced only after the last layer to keep all layers
        in sync during a single forward pass (mirrors the C++ KvCache behaviour).
        """
        k_val = key_states.get_value()
        v_val = value_states.get_value()

        k_shape = list(k_val.shape)
        B, H, new_seq, hdim = k_shape[0], k_shape[1], k_shape[2], k_shape[3]

        if not self._initialized:
            self._lazy_init(B, H, hdim)

        is_prefill = self._cache_position == 0
        pos = self._cache_position
        step = [1, 1, 1, 1]
        begins = [0, 0, pos, 0]
        ends = [B, H, pos + new_seq, hdim]

        ttnn.experimental.slice_write(
            k_val, self.key_cache[layer_idx], begins, ends, step
        )
        ttnn.experimental.slice_write(
            v_val, self.value_cache[layer_idx], begins, ends, step
        )

        # Advance position only after the last layer so all layers in a single
        # forward pass write to the same cache offset.
        if layer_idx == self.num_layers - 1:
            self._cache_position += new_seq

        if is_prefill:
            # Return the original tensors — no need to read back what we just wrote.
            return key_states, value_states

        if self.compressed:
            full_end = pos + new_seq
            full_end_padded = min(
                ((full_end + self._TILE - 1) // self._TILE) * self._TILE,
                self.max_seq_len,
            )
            k_out = ttnn.slice(
                self.key_cache[layer_idx], [0, 0, 0, 0], [B, H, full_end_padded, hdim]
            )
            v_out = ttnn.slice(
                self.value_cache[layer_idx], [0, 0, 0, 0], [B, H, full_end_padded, hdim]
            )
        else:
            k_out = self.key_cache[layer_idx]
            v_out = self.value_cache[layer_idx]
        return (
            ttml.autograd.create_tensor(k_out, requires_grad=False),
            ttml.autograd.create_tensor(v_out, requires_grad=False),
        )

    def get_seq_length(self) -> int:
        """Return the number of tokens currently stored in the cache."""
        return self._cache_position
