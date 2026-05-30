# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Self-attention KV cache for the dots.ocr Qwen2 LM decode path.

The dots.ocr language model is decoder-only (no cross-attention), so a single
:class:`SelfAttentionKVCache` covers the whole AR loop. It mirrors the
SeamlessM4T-v2 reference (``models/demos/facebook_seamless_m4t_v2_large/tt/
kv_cache.py``) but is specialised for Qwen2 GQA: the cache stores ONLY the
``num_kv_heads`` (=2) K/V heads per layer — the GQA 2->12 expansion is performed
inside :func:`ttnn.transformer.scaled_dot_product_attention_decode`, which takes
the query at the full head count and the cache at the KV head count.

Cache layout per K and V: ``[batch, num_kv_heads, max_seq_len, head_dim]`` in
DRAM TILE_LAYOUT bfloat16.

The single-token update uses
``ttnn.experimental.paged_update_cache(cache, kv, update_idxs_tensor=pos_tt)``
where ``pos_tt`` is a persistent 1-element int32 device tensor. Reading the
position from device memory (rather than baking a Python int into the kernel
args at trace-capture time) is what makes a SINGLE captured metal trace replay
correctly across every decode position — see ``skills/perf/SKILL.md`` "The trace
pitfall".

The model dir name (rednote_hilab_dots.ocr) contains a dot, so this module is
imported by file path with importlib by its consumers (the project convention);
it has no sibling-import needs of its own.
"""
from __future__ import annotations

from typing import Tuple

import torch

import ttnn
from models.common.utility_functions import nearest_y


class SelfAttentionKVCache:
    """Per-layer pre-allocated decoder self-attention KV cache (Qwen2 GQA).

    Shape per K and V: ``[batch, num_kv_heads, max_seq_len, head_dim]`` in DRAM
    TILE_LAYOUT (default bfloat16). Updated one token at a time during decode via
    :func:`ttnn.experimental.paged_update_cache`.

    Args:
        device: ttnn device or mesh device.
        num_layers: number of decoder layers (one K and one V slot per layer).
        batch: max batch size (1 for greedy OCR decode).
        num_kv_heads: number of KV heads stored (2 for Qwen2 GQA here).
        max_seq_len: cache capacity in tokens. Padded to a tile multiple (32).
        head_dim: per-head dim (128).
        dtype: cache storage dtype (default bfloat16).
    """

    def __init__(
        self,
        device,
        num_layers: int,
        batch: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.num_layers = int(num_layers)
        self.batch = int(batch)
        self.num_kv_heads = int(num_kv_heads)
        # Pad the cache capacity to a multiple of 512 (NOT just a tile=32). The
        # flash decode kernel (scaled_dot_product_attention_decode, auto k_chunk)
        # produces NaN when the zeroed tail beyond cur_pos forms a fully-masked
        # trailing k-chunk -- a degenerate softmax (0/0) in the streaming
        # accumulation. This happens at tile-but-not-512-aligned capacities (e.g.
        # 4960, 6944) but NOT at 512-multiples (5120, 5632, 6144, 6656, 7168 all
        # verified NaN-free for cur_pos across the prompt). Rounding to 512 keeps
        # the auto-chunk clean for any over-allocated cache, which is what lets a
        # long generate-to-EOS run (large max_seq_len) decode correctly.
        self.max_seq_len = int(nearest_y(max_seq_len, 512))
        self.head_dim = int(head_dim)
        self.dtype = dtype

        zeros = torch.zeros(self.batch, self.num_kv_heads, self.max_seq_len, self.head_dim, dtype=torch.bfloat16)
        self.k_caches = []
        self.v_caches = []
        for _ in range(self.num_layers):
            self.k_caches.append(
                ttnn.from_torch(
                    zeros, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            )
            self.v_caches.append(
                ttnn.from_torch(
                    zeros, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            )

        # Host-side zeros for fast in-place reset (no realloc -> trace stays valid).
        self._zero_host = ttnn.from_torch(zeros, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        # Sharded mem-config for the per-step K/V input to paged_update_cache:
        #   input.shape = [1, batch, num_kv_heads, head_dim], HEIGHT_SHARDED on L1,
        #   shard width == head_dim, shard height == ceil(num_kv_heads/TILE)*TILE.
        grid_size = device.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(self.batch, grid_size, row_wise=True)
        shard_shape = (nearest_y(self.num_kv_heads, ttnn.TILE_SIZE), self.head_dim)
        self._update_input_mem_cfg = ttnn.create_sharded_memory_config(
            shape=shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        # Persistent int32 position buffer (stable address for trace replay).
        self._persistent_pos_tt = ttnn.from_torch(
            torch.zeros(self.batch, dtype=torch.int32),
            device=device,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def get_persistent_pos_buffer(self) -> ttnn.Tensor:
        """Return the persistent ``[batch]`` int32 position buffer (stable addr)."""
        return self._persistent_pos_tt

    def write_pos(self, pos: int) -> ttnn.Tensor:
        """Stream a Python int into the persistent position buffer; return it."""
        host = ttnn.from_torch(torch.tensor([int(pos)] * self.batch, dtype=torch.int32), dtype=ttnn.int32)
        ttnn.copy_host_to_device_tensor(host, self._persistent_pos_tt)
        return self._persistent_pos_tt

    def update(self, layer_idx: int, k_new: ttnn.Tensor, v_new: ttnn.Tensor, pos) -> None:
        """Write one token's K/V into ``layer_idx`` at ``pos``.

        Args:
            layer_idx: which layer's cache to update.
            k_new / v_new: ``[batch, num_kv_heads, 1, head_dim]`` TILE_LAYOUT.
                Reshaped to ``[1, batch, num_kv_heads, head_dim]`` HEIGHT_SHARDED
                L1 before the op (paged_update_cache's required input layout).
            pos: int (written into the persistent buffer) or a ttnn.Tensor
                (used directly as ``update_idxs_tensor``; caller owns its layout).
        """
        if isinstance(pos, ttnn.Tensor):
            pos_tt = pos
        else:
            pos_tt = self.write_pos(int(pos))

        k_resh = ttnn.reshape(k_new, (1, self.batch, self.num_kv_heads, self.head_dim))
        v_resh = ttnn.reshape(v_new, (1, self.batch, self.num_kv_heads, self.head_dim))
        k_sharded = ttnn.interleaved_to_sharded(k_resh, self._update_input_mem_cfg)
        v_sharded = ttnn.interleaved_to_sharded(v_resh, self._update_input_mem_cfg)
        ttnn.deallocate(k_resh)
        ttnn.deallocate(v_resh)

        ttnn.experimental.paged_update_cache(self.k_caches[layer_idx], k_sharded, update_idxs_tensor=pos_tt)
        ttnn.experimental.paged_update_cache(self.v_caches[layer_idx], v_sharded, update_idxs_tensor=pos_tt)
        ttnn.deallocate(k_sharded)
        ttnn.deallocate(v_sharded)

    def fill_prefill(self, layer_idx: int, k: ttnn.Tensor, v: ttnn.Tensor) -> None:
        """Bulk-populate the cache for the whole prompt during prefill.

        Args:
            layer_idx: which layer's cache to fill.
            k / v: ``[batch, num_kv_heads, prompt_len, head_dim]`` TILE_LAYOUT.
                ``ttnn.fill_cache`` writes the input along the seq dim starting at
                slot 0 of batch index 0; the remaining (future) slots keep their
                zero contents and are masked out by SDPA's ``cur_pos``.
        """
        ttnn.fill_cache(self.k_caches[layer_idx], k, 0)
        ttnn.fill_cache(self.v_caches[layer_idx], v, 0)

    def read(self, layer_idx: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return the full ``[batch, num_kv_heads, max_seq, head_dim]`` K and V."""
        return self.k_caches[layer_idx], self.v_caches[layer_idx]

    def reset(self) -> None:
        """Zero every layer's cache in place (preserves device handles)."""
        for i in range(self.num_layers):
            ttnn.copy_host_to_device_tensor(self._zero_host, self.k_caches[i])
            ttnn.copy_host_to_device_tensor(self._zero_host, self.v_caches[i])
