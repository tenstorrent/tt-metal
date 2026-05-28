# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""KV-cache infrastructure for the SeamlessM4T-v2 text decoder.

The text decoder has two kinds of attention per layer:

* **Self-attention** is causal over decoder tokens. During autoregressive
  decode, the cache grows by one token per step. We use
  ``ttnn.experimental.paged_update_cache(cache, kv, update_idxs_tensor=pos_tt)``
  to write the new per-step K/V into the pre-allocated slot. The
  position is passed as a 1-element int32 tensor (a persistent buffer
  that we ``copy_host_to_device_tensor`` into per step). This allows a
  SINGLE metal trace to be captured for all decode positions — the
  trace replays correctly across positions because the position is
  read from device memory at execution time, not baked into the kernel
  args at capture time.
* **Cross-attention** projects K/V from the encoder output, which is
  fixed for the whole generation. The cross-attn cache is populated
  ONCE right after encoder forward (per layer) and reused on every decode
  step.

Both caches store tensors of shape ``[batch, num_heads, seq_len, head_dim]``
in DRAM TILE_LAYOUT bfloat16, mirroring how the non-cached MHA path
already produces them (the projection is followed by a ``reshape ->
transpose(1, 2)`` to land in this same layout before SDPA).

Compared with the qwen3_tts pattern, we use ``paged_update_cache`` with
a *single* logical page per cache (i.e. no actual paging — the cache is
just a single contiguous tensor of shape ``[batch, num_heads, max_seq,
head_dim]``). This is the minimum-friction switch from the int-baked
``ttnn.update_cache`` while keeping the SDPA read side completely
unchanged.

The pattern is copied from
``models/demos/qwen3_tts/tt/kv_cache.py`` and adapted for SeamlessM4T-v2
(no GQA — num_kv_heads == num_heads — and two cache classes instead of
one to make the static-vs-incremental distinction explicit).
"""

from __future__ import annotations

from typing import Tuple

import torch

import ttnn
from models.common.utility_functions import nearest_y


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
        zeros = torch.zeros(self.batch, self.num_heads, self.max_seq_len, self.head_dim, dtype=torch.bfloat16)
        for _ in range(self.num_layers):
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
        # Pre-build a host-side zeros tensor for fast H2D zero-resets
        # (avoids allocating during reset()). Same dtype/layout as the
        # cache so copy_host_to_device_tensor can stream it in.
        self._zero_host = ttnn.from_torch(
            zeros,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        )

        # Pre-build the sharded-memory-config for the K/V "new token"
        # input to ``paged_update_cache``. The op requires:
        #   * input.shape = [1, batch, num_heads, head_dim]
        #   * input HEIGHT_SHARDED on L1
        #   * shard width == head_dim (the last dim)
        #   * shard height == ceil(num_heads / TILE) * TILE
        # Per-batch one core. For batch=1 we just pin one core.
        grid_size = device.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(self.batch, grid_size, row_wise=True)
        shard_shape = (nearest_y(self.num_heads, ttnn.TILE_SIZE), self.head_dim)
        self._update_input_mem_cfg = ttnn.create_sharded_memory_config(
            shape=shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        # Persistent device-side position-index buffer used when the
        # caller passes a Python int. Stable address so a captured trace
        # can read it directly. Shape ``[batch]``, dtype int32, ROW_MAJOR
        # DRAM (as required by paged_update_cache's update_idxs_tensor).
        self._persistent_pos_tt = ttnn.from_torch(
            torch.zeros(self.batch, dtype=torch.int32),
            device=device,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def get_persistent_pos_buffer(self) -> ttnn.Tensor:
        """Return the persistent position-index buffer.

        Callers that want to drive ``update()`` via a tensor (e.g. for
        trace capture, where the position must vary across replays
        without re-compiling) should write into this buffer via
        ``ttnn.copy_host_to_device_tensor(host, get_persistent_pos_buffer())``
        and then call ``update(..., pos=get_persistent_pos_buffer())``.
        """
        return self._persistent_pos_tt

    def update(self, layer_idx: int, k_new: ttnn.Tensor, v_new: ttnn.Tensor, pos) -> None:
        """Write a single token's K/V into the layer's cache at ``pos``.

        Args:
            layer_idx: which decoder layer's cache to update.
            k_new: ``[B, num_heads, 1, head_dim]`` TILE_LAYOUT. Internally
                reshaped to ``[1, B, num_heads, head_dim]`` and converted
                to HEIGHT_SHARDED L1 before the ``paged_update_cache`` op.
            v_new: ``[B, num_heads, 1, head_dim]`` TILE_LAYOUT.
            pos: target position along the sequence dimension. May be:
                * a Python int — we write it into the persistent device
                  position buffer (one H2D copy of a 1-int tensor) and
                  use that as the ``update_idxs_tensor``. Safe for
                  trace capture because the persistent buffer's address
                  is stable across calls.
                * a ``ttnn.Tensor`` — used directly as the
                  ``update_idxs_tensor``. Caller is responsible for
                  ensuring it's a 1-element int32 ROW_MAJOR DRAM tensor
                  and that its address is stable across trace replays.
        """
        # ---- 1. Resolve position into a device tensor. ----
        if isinstance(pos, ttnn.Tensor):
            pos_tt = pos
        else:
            # Write the int into the persistent position buffer.
            host = ttnn.from_torch(
                torch.tensor([int(pos)] * self.batch, dtype=torch.int32),
                dtype=ttnn.int32,
            )
            ttnn.copy_host_to_device_tensor(host, self._persistent_pos_tt)
            pos_tt = self._persistent_pos_tt

        # ---- 2. Reshape K/V from [B, NH, 1, HD] to [1, B, NH, HD] sharded. ----
        # NB: the caller produced the tensors in [B, NH, 1, HD] for
        # convenience of the broader code path. paged_update_cache wants
        # [1, B, NH, HD] HEIGHT_SHARDED on L1. We do reshape + i2s here.
        k_resh = ttnn.reshape(k_new, (1, self.batch, self.num_heads, self.head_dim))
        v_resh = ttnn.reshape(v_new, (1, self.batch, self.num_heads, self.head_dim))
        k_sharded = ttnn.interleaved_to_sharded(k_resh, self._update_input_mem_cfg)
        v_sharded = ttnn.interleaved_to_sharded(v_resh, self._update_input_mem_cfg)
        ttnn.deallocate(k_resh)
        ttnn.deallocate(v_resh)

        # ---- 3. paged_update_cache (no page table — single contiguous cache). ----
        ttnn.experimental.paged_update_cache(self.k_caches[layer_idx], k_sharded, update_idxs_tensor=pos_tt)
        ttnn.experimental.paged_update_cache(self.v_caches[layer_idx], v_sharded, update_idxs_tensor=pos_tt)
        ttnn.deallocate(k_sharded)
        ttnn.deallocate(v_sharded)

    def read(self, layer_idx: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return the full ``[B, num_heads, max_seq, head_dim]`` K and V caches.

        SDPA reads the whole buffer; the attention mask handles the
        "future positions" (positions > current ``pos``) which still hold
        their zero-initialised contents.
        """
        return self.k_caches[layer_idx], self.v_caches[layer_idx]

    def reset(self) -> None:
        """Zero every layer's cache (clear all stored values).

        Streams a host-side zero tensor into each cache buffer via
        ``copy_host_to_device_tensor``. This preserves the device tensor
        handles (captured into traces) while overwriting their contents
        -- matching the qwen3_tts cross-call trace-reuse pattern.
        """
        for i in range(self.num_layers):
            ttnn.copy_host_to_device_tensor(self._zero_host, self.k_caches[i])
            ttnn.copy_host_to_device_tensor(self._zero_host, self.v_caches[i])


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

        # Pre-allocate persistent K/V buffers per layer so the device
        # addresses stay stable across generate() calls. Captured decode
        # traces hold a pointer to these buffers; on a second
        # generate() we OVERWRITE them via the populate() call below
        # rather than reallocating. Shape per K and V:
        # [batch, num_heads, encoder_seq_len, head_dim].
        self.k_caches: list = []
        self.v_caches: list = []
        zeros = torch.zeros(self.batch, self.num_heads, self.encoder_seq_len, self.head_dim, dtype=torch.bfloat16)
        for _ in range(self.num_layers):
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
        self._populated = [False] * self.num_layers

    def populate(self, layer_idx: int, k: ttnn.Tensor, v: ttnn.Tensor) -> None:
        """Store the encoder K and V tensors for the given layer.

        The caller already projected ``encoder_hidden_states`` through the
        layer's cross-attention K and V weights and reshaped to
        ``[B, num_heads, src_len, head_dim]``. We copy them into the
        pre-allocated persistent buffers (constructed once at __init__)
        rather than retaining the new pointers -- this keeps the device
        addresses stable across generate() calls so any captured decode
        trace that reads from the cross-attn cache remains valid.
        """
        # ``ttnn.fill_cache(cache, input, batch_idx)`` writes the whole
        # input tensor along the seq dim into cache[batch_idx, ...]. With
        # batch=1, this overwrites slot 0 with the freshly projected K/V.
        # K/V have shape ``[1, num_heads, encoder_seq_len, head_dim]``
        # (already tile-padded to match the cache slot exactly).
        ttnn.fill_cache(self.k_caches[layer_idx], k, 0)
        ttnn.fill_cache(self.v_caches[layer_idx], v, 0)
        # The new k/v device tensors aren't needed any more -- their
        # data has been copied into the persistent cache buffers.
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        self._populated[layer_idx] = True

    def is_populated(self, layer_idx: int) -> bool:
        return self._populated[layer_idx]

    def read(self, layer_idx: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return the layer's encoder K/V cache (must be populated first)."""
        if not self._populated[layer_idx]:
            raise RuntimeError(f"CrossAttentionKVCache layer {layer_idx} has not been populated.")
        return self.k_caches[layer_idx], self.v_caches[layer_idx]

    def reset(self) -> None:
        """Mark every layer's encoder K/V cache as un-populated (zero in-place).

        Does NOT deallocate the buffers -- they're persistent so captured
        decode traces stay valid across generate() calls. The next
        populate() call overwrites the buffer contents via update_cache.
        """
        for i in range(self.num_layers):
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
