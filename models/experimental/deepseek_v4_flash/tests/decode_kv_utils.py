# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""KV state for driving one attention layer's decode outside the model.

The model builds its KV buffers, block pools and page tables in
:meth:`DeepSeekV4Model.prepare_static_decode` and generates the per-step SDPA bounds on
device. The layer-level tests build the same state here for a single layer, and the same
bounds on host.
"""

from __future__ import annotations

import torch

import ttnn
from models.experimental.deepseek_v4_flash.tt.attention import (
    PAGED_KV_LAYER_TYPES,
    build_static_layer_cache,
    dense_kv_context_limit,
    dense_kv_rows,
    int32_pos_tensor,
)
from models.experimental.deepseek_v4_flash.tt.common import _MASK_NEG
from models.experimental.deepseek_v4_flash.tt.paged_cache import (
    PagedKVManager,
    PagedLayerView,
    build_groups,
    plan_pool_blocks,
    round_context,
)

PAGE_BLOCK_SIZE = 32


class DecodeLayerKV:
    """One layer's KV and compressor window buffers, laid out the way the model lays them out.

    ``cache`` is the layer's :class:`_StaticLayerCache`. Sliding and CSA layers keep their
    KV in its dense ``kv`` buffer (batch 1 only) and ``view`` is ``None``. HCA layers get a
    private block pool and page table instead, one session per user, their page-table rows
    stacked into the ``[B, logical_blocks]`` tensor the paged ops index by user, and
    compressed blocks handed out as windows close; ``view`` is what
    :meth:`DeepSeekV4Attention.decode` reads and writes that KV through.
    """

    def __init__(self, cfg, layer_type: str, seq_len: int, batch: int, device, block_size: int = PAGE_BLOCK_SIZE):
        self.layer_type = layer_type
        self.batch = batch
        self.device = device
        self.sliding_window = cfg.sliding_window
        self.compress_rate = None if layer_type == "sliding_attention" else cfg.compress_rates[layer_type]
        self.paged = layer_type in PAGED_KV_LAYER_TYPES
        self.dense_rows = dense_kv_rows(layer_type, cfg.sliding_window)
        rates = [] if self.compress_rate is None else [self.compress_rate]
        # Every group's entry count has to tile into whole blocks, which is the same
        # rounding the model applies before it sizes its pools.
        self.max_seq = round_context(seq_len, rates, block_size)
        limit = dense_kv_context_limit([layer_type], cfg.compress_rates)
        if limit is not None and seq_len > limit:
            raise ValueError(f"{layer_type} holds at most {limit} tokens of context, got seq_len={seq_len}")
        self.cache = build_static_layer_cache(
            device, layer_type, cfg.head_dim, self.max_seq, cfg.compress_rates, cfg.sliding_window, batch=batch
        )
        self.view = None
        if not self.paged:
            return

        groups = build_groups([layer_type], cfg.compress_rates, cfg.sliding_window, self.max_seq, block_size=block_size)
        self.group = groups[layer_type]
        self.manager = PagedKVManager(groups, plan_pool_blocks(groups, batch, batch * self.max_seq))
        self.sids = [self.manager.open_session() for _ in range(batch)]
        pool = ttnn.from_torch(
            torch.zeros(self.manager.pools[layer_type].num_blocks, 1, self.group.block_size, cfg.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.table = ttnn.from_torch(
            torch.zeros(batch, self.group.logical_blocks, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.view = PagedLayerView(pool, self.table, self.group.position_modulo)
        self._write_table()

    def _write_table(self) -> None:
        """Refresh the persistent table with one row per user, in slot order."""
        rows = torch.cat([self.manager.page_row(sid, self.layer_type) for sid in self.sids])
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(rows, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), self.table
        )

    def step(self, pos: int) -> None:
        """Give every user blocks for the rows a step at ``pos`` touches (paged layers only)."""
        if not self.paged:
            return
        # Every session is grown before the table is rewritten -- short-circuiting on the
        # first session that moved would leave the rest of the batch unmapped.
        grown = [self.manager.ensure_capacity(sid, pos) for sid in self.sids]
        if any(grown):
            self._write_table()

    def bounds(self, pos: int) -> tuple[ttnn.Tensor | None, ttnn.Tensor | None]:
        """``(mask, sdpa_cur_pos)`` for a step at ``pos``, the host twin of the model's
        on-device bounds.

        A sliding ring is always a contiguous prefix, ``[0, min(pos, W - 1)]``, so it is
        causal. A compressor layer's ``[sliding ring | compressed]`` axis is a contiguous
        valid prefix before the first window closes and once the ring is full, so it is
        causal there. In between the ring still has unwritten slots, and the additive mask
        ``[1, 1, 1, kv_len]`` covers the whole axis: the dense buffer's rows, or the paged
        axis rounded up to whole blocks.
        """
        cr, w = self.compress_rate, self.sliding_window
        if cr is None:
            return None, int32_pos_tensor(min(pos, w - 1), self.device, self.batch)
        closed = (pos + 1) // cr
        if closed == 0 or pos + 1 >= w:
            cur = pos if closed == 0 else w + closed - 1
            return None, int32_pos_tensor(cur, self.device, self.batch)
        kv_len = self.group.kv_len if self.paged else self.dense_rows
        n_entries = self.max_seq // cr if self.paged else self.dense_rows - w
        idx = torch.arange(kv_len)
        ring = idx < w
        compressed = (idx >= w) & (idx < w + n_entries)
        invalid = torch.ones(kv_len, dtype=torch.bool)
        invalid[ring] = idx[ring] > pos
        invalid[compressed] = (idx[compressed] - w) >= closed
        mask = torch.zeros(1, 1, 1, kv_len, dtype=torch.float32)
        mask.masked_fill_(invalid.view(1, 1, 1, -1), _MASK_NEG)
        return ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device), None
