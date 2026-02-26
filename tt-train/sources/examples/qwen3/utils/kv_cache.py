# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""KV cache and related attention mask utilities for autoregressive inference."""

import torch
import ttnn
import ttml

from utils.tensor_utils import get_device as _get_device


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


def _decode_mask(valid_kv_len, device):
    """Create decode attention mask [1, 1, 1, padded_kv] for a single query token.

    Width is rounded up to the next multiple of 32 (tile boundary) so the
    shape only changes every 32 decode steps.  Positions beyond valid_kv_len
    are set to 0 so zero-initialised cache slots are masked out.
    """
    _TILE = 32
    padded_kv = ((valid_kv_len + _TILE - 1) // _TILE) * _TILE
    mask = torch.zeros(1, 1, 1, padded_kv, dtype=torch.bfloat16)
    mask[:, :, :, :valid_kv_len] = 1.0
    return _to_device_tiled(mask, device)


def _step_attn_mask(step, kv_cache, past_kv, prefill_seq_len, causal_mask, device):
    """Return the attention mask for the current step."""
    if not kv_cache:
        return causal_mask
    if step == 0:
        return _causal_mask(prefill_seq_len, device)
    total_seq = past_kv.get_seq_length() + 1
    return _decode_mask(total_seq, device)


# =====================================================================
# KV Cache (static pre-allocated, inference-only)
# =====================================================================


class KVCache:
    """Static pre-allocated KV cache for autoregressive inference.

    Pre-allocates fixed-size DRAM tensors [B, num_kv_heads, max_seq_len, head_dim]
    on the first call and uses ttnn.slice_write for O(1) in-place updates.

    With pre-allocated buffers:
      - ttnn.slice_write writes only the new token(s) in-place (O(1) write)
      - Fixed-size cache tensors avoid per-step DRAM allocation
      - Returned slice shapes still change per step, but the heavy allocation is gone

    Not differentiable — use only with GradMode.DISABLED.
    """

    def __init__(self, num_layers: int, max_seq_len: int):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self._cache_position: int = 0
        self._initialized: bool = False
        self.key_cache: list = [None] * num_layers
        self.value_cache: list = [None] * num_layers

    def _lazy_init(self, batch_size: int, num_kv_heads: int, head_dim: int) -> None:
        """Pre-allocate zeroed DRAM cache buffers on first update call."""
        device = _get_device()
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

    def update(self, layer_idx: int, key_states, value_states):
        """Write new K/V in-place via slice_write; return (full_k, full_v) ttml tensors.

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

        ttnn.slice_write(k_val, self.key_cache[layer_idx], begins, ends, step)
        ttnn.slice_write(v_val, self.value_cache[layer_idx], begins, ends, step)

        # Advance position only after the last layer so all layers in a single
        # forward pass write to the same cache offset.
        if layer_idx == self.num_layers - 1:
            self._cache_position += new_seq

        if is_prefill:
            # Return the original tensors — no need to read back what we just wrote.
            return key_states, value_states

        # Decode: return the valid cache window rounded up to the next tile
        # boundary (32).  This matches the C++ grouped_query_attention which
        # slices to mask.logical_shape()[-1] — a value that is already
        # tile-aligned.  Without rounding, full_end grows by 1 every step and
        # every decode step gets a unique K/V tensor shape, forcing TTNN to
        # recompile all downstream kernels on every token.
        full_end = pos + new_seq
        _TILE = 32
        full_end_padded = min(
            ((full_end + _TILE - 1) // _TILE) * _TILE, self.max_seq_len
        )
        k_full = ttnn.slice(
            self.key_cache[layer_idx], [0, 0, 0, 0], [B, H, full_end_padded, hdim]
        )
        v_full = ttnn.slice(
            self.value_cache[layer_idx], [0, 0, 0, 0], [B, H, full_end_padded, hdim]
        )
        return (
            ttml.autograd.create_tensor(k_full, requires_grad=False),
            ttml.autograd.create_tensor(v_full, requires_grad=False),
        )

    def get_seq_length(self) -> int:
        """Return the number of tokens currently stored in the cache."""
        return self._cache_position
