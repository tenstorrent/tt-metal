# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""KV-cache infrastructure for the SeamlessM4T-v2 text decoder.

The text decoder has two kinds of attention per layer:

* **Self-attention** is causal over decoder tokens. During autoregressive
  decode, the cache grows by one token per step. We use
  ``ttnn.update_cache(cache, k_new, update_idx=pos)`` to write the new
  per-step K/V into the pre-allocated slot.
* **Cross-attention** projects K/V from the encoder output, which is
  fixed for the whole generation. The cross-attn cache is populated
  ONCE right after encoder forward (per layer) and reused on every decode
  step.

Both caches store tensors of shape ``[batch, num_heads, seq_len, head_dim]``
in DRAM TILE_LAYOUT bfloat16, mirroring how the non-cached MHA path
already produces them (the projection is followed by a ``reshape ->
transpose(1, 2)`` to land in this same layout before SDPA).

The pattern is copied from
``models/demos/qwen3_tts/tt/kv_cache.py`` and adapted for SeamlessM4T-v2
(no GQA — num_kv_heads == num_heads — and two cache classes instead of
one to make the static-vs-incremental distinction explicit).
"""

from __future__ import annotations

from typing import Tuple

import torch

import ttnn


class SelfAttentionKVCache:
    """Per-layer pre-allocated decoder self-attention KV cache.

    Shape per K and V: ``[batch, num_heads, max_seq_len, head_dim]``.
    Stored in DRAM TILE_LAYOUT (default bfloat16). Updated in-place via
    ``ttnn.update_cache(cache, new_kv, update_idx=pos)`` for one token at
    a time during decode.

    Args:
        device: ttnn device or mesh device.
        num_layers: number of decoder layers (one cache slot per layer).
        batch: max batch size to support.
        num_heads: number of attention heads.
        max_seq_len: cache capacity in tokens. Pad to a tile multiple (32)
            so the cache is well-formed for SDPA reads.
        head_dim: per-head dim.
        dtype: cache storage dtype (default bfloat16).
    """

    def __init__(
        self,
        device,
        num_layers: int,
        batch: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.num_layers = int(num_layers)
        self.batch = int(batch)
        self.num_heads = int(num_heads)
        self.max_seq_len = int(max_seq_len)
        self.head_dim = int(head_dim)
        self.dtype = dtype

        # Pre-allocate one [B, num_heads, max_seq, head_dim] K and V cache
        # per layer, zero-initialised in DRAM TILE_LAYOUT.
        self.k_caches = []
        self.v_caches = []
        for _ in range(self.num_layers):
            zeros = torch.zeros(self.batch, self.num_heads, self.max_seq_len, self.head_dim, dtype=torch.bfloat16)
            self.k_caches.append(
                ttnn.from_torch(
                    zeros,
                    device=device,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            self.v_caches.append(
                ttnn.from_torch(
                    zeros,
                    device=device,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )

    def update(self, layer_idx: int, k_new: ttnn.Tensor, v_new: ttnn.Tensor, pos: int) -> None:
        """Write a single token's K/V into the layer's cache at ``pos``.

        Args:
            layer_idx: which decoder layer's cache to update.
            k_new: ``[B, num_heads, 1, head_dim]`` TILE_LAYOUT.
            v_new: ``[B, num_heads, 1, head_dim]`` TILE_LAYOUT.
            pos: target position along the sequence dimension.
        """
        ttnn.update_cache(self.k_caches[layer_idx], k_new, update_idx=int(pos))
        ttnn.update_cache(self.v_caches[layer_idx], v_new, update_idx=int(pos))

    def read(self, layer_idx: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return the full ``[B, num_heads, max_seq, head_dim]`` K and V caches.

        SDPA reads the whole buffer; the attention mask handles the
        "future positions" (positions > current ``pos``) which still hold
        their zero-initialised contents.
        """
        return self.k_caches[layer_idx], self.v_caches[layer_idx]

    def reset(self) -> None:
        """Zero every layer's cache (clear all stored values).

        Uses an in-place multiply by 0.0 to avoid reallocating the DRAM
        buffer (which would change the tensor handle the caller may have
        captured in a trace).
        """
        for i in range(self.num_layers):
            self.k_caches[i] = ttnn.mul(self.k_caches[i], 0.0, output_tensor=self.k_caches[i])
            self.v_caches[i] = ttnn.mul(self.v_caches[i], 0.0, output_tensor=self.v_caches[i])


class CrossAttentionKVCache:
    """Per-layer encoder K/V cache. Static after prefill.

    Shape per K and V: ``[batch, num_heads, encoder_seq_len, head_dim]``.
    Populated by calling :meth:`populate` once per layer right after the
    encoder runs (the per-layer cross-attention K/V projections are
    applied to the encoder hidden states and the result is stored here).
    Reused for every decode step by reading the cached K/V directly via
    :meth:`read` — cross-attention skips the K/V projection in cached
    mode.

    Args:
        device: ttnn device or mesh device.
        num_layers: number of decoder layers.
        batch: max batch size to support.
        num_heads: number of cross-attention heads (matches self-attn
            for SeamlessM4T-v2).
        encoder_seq_len: encoder output sequence length (pre-padded to
            a tile multiple by the caller).
        head_dim: per-head dim.
        dtype: cache storage dtype (default bfloat16).
    """

    def __init__(
        self,
        device,
        num_layers: int,
        batch: int,
        num_heads: int,
        encoder_seq_len: int,
        head_dim: int,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.num_layers = int(num_layers)
        self.batch = int(batch)
        self.num_heads = int(num_heads)
        self.encoder_seq_len = int(encoder_seq_len)
        self.head_dim = int(head_dim)
        self.dtype = dtype

        # Per-layer placeholders. Filled by populate() right after encoder forward.
        self.k_caches: list = [None] * self.num_layers
        self.v_caches: list = [None] * self.num_layers
        self._populated = [False] * self.num_layers

    def populate(self, layer_idx: int, k: ttnn.Tensor, v: ttnn.Tensor) -> None:
        """Store the encoder K and V tensors for the given layer.

        The caller already projected ``encoder_hidden_states`` through the
        layer's cross-attention K and V weights and reshaped to
        ``[B, num_heads, src_len, head_dim]``. We retain the tensors as-is
        — they will be re-used by every decode step.
        """
        self.k_caches[layer_idx] = k
        self.v_caches[layer_idx] = v
        self._populated[layer_idx] = True

    def is_populated(self, layer_idx: int) -> bool:
        return self._populated[layer_idx]

    def read(self, layer_idx: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return the layer's encoder K/V cache (must be populated first)."""
        if not self._populated[layer_idx]:
            raise RuntimeError(f"CrossAttentionKVCache layer {layer_idx} has not been populated.")
        return self.k_caches[layer_idx], self.v_caches[layer_idx]

    def reset(self) -> None:
        """Drop all stored encoder K/V tensors (e.g. between utterances)."""
        for i in range(self.num_layers):
            if self.k_caches[i] is not None:
                ttnn.deallocate(self.k_caches[i])
                self.k_caches[i] = None
            if self.v_caches[i] is not None:
                ttnn.deallocate(self.v_caches[i])
                self.v_caches[i] = None
            self._populated[i] = False


class TextDecoderKVCache:
    """Bundle of (self_attn, cross_attn) caches for the whole text decoder.

    Acts as the single ``past_key_values`` argument threaded through
    :class:`TextDecoder` and :class:`TextDecoderLayer` during autoregressive
    decode.

    Args:
        device: ttnn device.
        num_layers: number of decoder layers.
        batch: batch size.
        num_heads: attention head count (16 for v2-Large).
        head_dim: per-head dim (64 for v2-Large).
        max_decode_seq_len: cache capacity in *decoder* tokens.
        encoder_seq_len: encoder sequence length (already tile-padded
            by the caller of the decoder).
        dtype: cache storage dtype (default bfloat16).
    """

    def __init__(
        self,
        device,
        num_layers: int,
        batch: int,
        num_heads: int,
        head_dim: int,
        max_decode_seq_len: int,
        encoder_seq_len: int,
        dtype=ttnn.bfloat16,
    ):
        self.self_attn = SelfAttentionKVCache(
            device=device,
            num_layers=num_layers,
            batch=batch,
            num_heads=num_heads,
            max_seq_len=max_decode_seq_len,
            head_dim=head_dim,
            dtype=dtype,
        )
        self.cross_attn = CrossAttentionKVCache(
            device=device,
            num_layers=num_layers,
            batch=batch,
            num_heads=num_heads,
            encoder_seq_len=encoder_seq_len,
            head_dim=head_dim,
            dtype=dtype,
        )
        self.num_layers = int(num_layers)
        self.batch = int(batch)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.max_decode_seq_len = int(max_decode_seq_len)
        self.encoder_seq_len = int(encoder_seq_len)

    def reset(self) -> None:
        self.self_attn.reset()
        self.cross_attn.reset()
