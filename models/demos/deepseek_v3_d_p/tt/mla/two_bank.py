# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 loop pass-2 attention support: the window ring and the joint index list (A2-A6).

Pass-2 layer L attends, under one softmax, over

* bank A: the pass-1 KVPE of layer L at pass-1 layer L's top-k (A5), and
* bank B: pass 2's own KVPE of positions p-W+1 .. p, the sliding window (A3).

Both banks live in the layer's slot of the same KVPE cache, which ``sparse_sdpa`` needs (one tensor, one format):

    positions [0, ring_base)                       bank A: pass 1's rows (G2's copy of G1's cache); pass 2 never
                                                   writes them
    positions [ring_base, ring_base + RING)        the window ring: pass 2's own rows, RING = 5 chunks

Pass 2 writes each chunk's KV with the pass-1 cache write (``update_padded_kv_cache``, which places the chunk's rotated
SP rows), twice: at virtual start ``ring_base + (start mod P)`` and ``P`` further, with period ``P = 2 * chunk``.
Then position q (of this or any earlier chunk of the slot) is always at the canonical row
``ring_id(q) = ring_base + P + (q mod P)``:

* the copy that lands in ``[ring_base + P, ring_base + 2P)`` is exactly position q's canonical row, and every write
  puts one copy of each of its positions there (the two copies differ by P);
* rows outside that band are scratch nobody reads;
* a window needs at most ``chunk + W - 1 <= P`` consecutive positions, so no two of them share a canonical row.

``RING = 2P + chunk`` holds the second copy of a chunk starting at ``start mod P = P - 32``. ``ring_base`` and P are
whole chunks, so the virtual starts rotate the chunk exactly as the pass-1 write of ``start`` does. The carry across
chunks, per user slot and per layer, is the ring itself: it is part of the slot.

Attention is the GLM-5.3 shared-layer path, the sparse KV push of one natural index list per query row,
``[window ids | pass-1 top-k]`` (``W + top-k`` wide). The window ids are ``ring_id(min(p - W + 1 + j, end - 1))``:
the clamp only touches padding rows (p >= end), which then read rows their own chunk wrote rather than ring rows
an earlier chunk or request left, so their (unused) outputs do not depend on history. ``index_list``
computes them on device from the chunk start (metadata[1]), together with each query row's position p in the rotated
chunk layout (``rotated_chunk_positions``), so one captured trace serves every start. ``sparse_sdpa`` stops a row at
its first sentinel, so sentinels must be one tail; that holds when every window slot is real (``p >= W - 1``) because
bank A's own sentinels are already a tail. A row with ``p < W - 1`` has ``W - 1 - p`` leading window slots before
position 0: a gather through ``perm = min(c + max(0, W - 1 - p), sentinel column)`` slides the row left by that many
slots and fills the end with the sentinel column.

Window format (A2, ``loop_window_format``): ``scaled_fp8`` (default) or ``bf16``. The ring is part of the bank-A
cache, so the window format is the cache format: a bf16 window means a bf16 cache, whose bank A must reach G2 as bf16
(that producer is outside this module).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.indexer import indexer_layer_is_reused
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

WINDOW_FORMATS = {"scaled_fp8": MlaKvCacheFormat.SCALED_FP8, "bf16": MlaKvCacheFormat.BF16_RM}
RING_PERIOD_CHUNKS = 2  # P = 2 chunks: a window (<= chunk + W - 1 positions) never wraps onto itself
RING_CHUNKS = 2 * RING_PERIOD_CHUNKS + 1  # the second copy of a chunk at start mod P = P - 32 ends below 2P + chunk


def window_kv_format(config) -> MlaKvCacheFormat:
    """The window's (and so the cache's) format, from ``config.loop_window_format``."""
    try:
        return WINDOW_FORMATS[config.loop_window_format]
    except KeyError as error:
        raise ValueError(
            f"loop_window_format must be one of {sorted(WINDOW_FORMATS)}, got {config.loop_window_format!r}"
        ) from error


def ring_rows(chunk_global: int) -> int:
    """Cache positions a loop layer's slot holds past bank A for the window ring, at chunks of ``chunk_global``."""
    return RING_CHUNKS * chunk_global


def loop_pass2_layers(config) -> range:
    """The layers that run loop pass 2 (GLM-5.4 loop layers when the loop runs at least twice), else empty."""
    if getattr(config, "loop_count", 1) < 2:
        return range(0)
    return range(config.loop_first_layer, config.loop_last_layer + 1)


def pass2_kv_cache_seq_len(max_seq_len: int, chunk_global: int) -> int:
    """Per-slot length of the KVPE cache of a stage that runs loop pass 2: ``max_seq_len`` pass-1 positions, then the
    window ring. The one place cache allocators and the model agree on it."""
    if max_seq_len % chunk_global:
        raise ValueError(f"max_seq_len {max_seq_len} must be whole chunks of {chunk_global} to hold a window ring")
    return max_seq_len + ring_rows(chunk_global)


def ring_id(position: int, ring_base: int, chunk_global: int) -> int:
    """Cache row (natural position in the slot) holding pass 2's KV of ``position``. Host mirror of the device ids."""
    period = RING_PERIOD_CHUNKS * chunk_global
    return ring_base + period + position % period


def pass2_indexer_types(config) -> list[str]:
    """G2's indexer map: every layer 'shared', so pass-2 layers bind no indexer and take injected top-k (A5)."""
    return ["shared"] * len(config.indexer_types)


def pass1_index_set(config, layer: int) -> int:
    """Which of the loop's pass-1 top-k sets (the full layers of the loop, in layer order, as
    ``tt.loop_handoff`` sends them) pass-2 layer ``layer`` reuses: its pass-1 reuse group's (A5). ``config`` is the
    model config, whose ``indexer_types`` is pass 1's map (not G2's ``pass2_indexer_types``)."""
    first, last = config.loop_first_layer, config.loop_last_layer
    if not first <= layer <= last:
        raise ValueError(f"layer {layer} is outside the loop {first}..{last}")
    if indexer_layer_is_reused(config, first):
        raise ValueError(f"the loop's first layer {first} must own an indexer: its group's set is not sent")
    return sum(not indexer_layer_is_reused(config, i) for i in range(first, layer + 1)) - 1


class TwoBankIndex:
    """Per-mesh constants and device math of pass 2's index list and ring writes; pass-2 layers run serially and
    share it. Built once (outside any trace capture); every per-chunk input is read on device."""

    def __init__(
        self,
        mesh_device,
        *,
        ring_base: int,
        window: int,
        chunk_global: int,
        query_rows: int,
        topk: int,
        sp_axis: int,
        tp_sequence_sharded: bool,
    ):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        sp, tp = mesh_device.shape[sp_axis], mesh_device.shape[1 - sp_axis]
        self.chunk_local = chunk_global // sp
        if chunk_global % (sp * 32) or self.chunk_local % query_rows:
            raise ValueError(f"chunk {chunk_global} must split into 32-row SP shards of whole query-row blocks")
        if ring_base % chunk_global or window - 1 > chunk_global:
            raise ValueError(f"ring base {ring_base} must be whole chunks and window {window} at most a chunk + 1")
        self.mesh_device = mesh_device
        self.ring_base = ring_base
        self.window = window
        self.chunk_global = chunk_global
        self.period = RING_PERIOD_CHUNKS * chunk_global
        self.width = window + topk  # index list width
        if self.width % 32:
            raise ValueError(f"index list width {self.width} must be a multiple of 32 for sparse_sdpa's k chunks")

        # Per chip: chip (s, t) holds row block [s, t] of a [sp, tp, ...] host tensor.
        dims = [None, None]
        dims[sp_axis], dims[1 - sp_axis] = 0, 1
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims))

        def per_chip(host, dtype, layout):
            return ttnn.from_torch(
                host.to(torch.float32 if dtype == ttnn.float32 else torch.int32),
                device=mesh_device,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        # Query row r of chip (s, t) is SP-shard row i = t * query_rows + r (TP-sequence-sharded queries) or r.
        s_idx = torch.arange(sp).view(sp, 1, 1, 1)
        t_idx = (
            torch.arange(tp).view(1, tp, 1, 1) if tp_sequence_sharded else torch.zeros(1, tp, 1, 1, dtype=torch.int64)
        )
        r_idx = torch.arange(query_rows).view(1, 1, query_rows, 1)
        shape = (sp, tp, query_rows, self.width)
        # fp32 is exact for every value here (< 2**24); see index_list.
        self._shard_base = per_chip((s_idx * self.chunk_local).expand(shape), ttnn.float32, ttnn.TILE_LAYOUT)
        self._shard_row = per_chip((t_idx * query_rows + r_idx).expand(shape), ttnn.float32, ttnn.TILE_LAYOUT)
        self._column = per_chip(
            torch.arange(self.width).view(1, 1, 1, -1).expand(shape), ttnn.float32, ttnn.TILE_LAYOUT
        )
        # Source of the compacting gather: [window ids | top-k | sentinel column], ids written per chunk.
        source = torch.zeros(sp, tp, query_rows, self.width + 1, dtype=torch.int64)
        source[..., -1] = -1  # 0xFFFFFFFF through int32
        self._source = per_chip(source, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def ring_writes(self, start: int, end: int) -> list[tuple[int, int]]:
        """Host form of the ring writes of the chunk [start, end): (virtual start, virtual end) per copy."""
        first = self.ring_base + start % self.period
        return [(v, v + end - start) for v in (first, first + self.period)]

    def ring_write_metadata(self, metadata) -> list[list[ttnn.Tensor]]:
        """Device form of ``ring_writes`` from the chunk metadata (slot, start, end): one metadata triple per copy,
        whose start and end are the copy's virtual ones. New tensors."""
        start = _scalar_f32(metadata[1])
        length = ttnn.subtract(_scalar_f32(metadata[2]), start)
        first = ttnn.add(ttnn.subtract(start, _floor_mul(start, self.period, 16.0)), float(self.ring_base))
        out = []
        for v in (first, ttnn.add(first, float(self.period))):
            out.append([metadata[0], _scalar_u32(v), _scalar_u32(ttnn.add(v, length))])
        return out

    def index_list(self, topk_indices: ttnn.Tensor, chunk_start: ttnn.Tensor, chunk_end: ttnn.Tensor) -> ttnn.Tensor:
        """This chunk's natural index list ``[1, 1, query_rows, W + top-k]`` uint32: ring ids of each query row's
        window, then its pass-1 top-k, compacted for rows before position W - 1. ``chunk_start`` is the chunk's start
        as a ``[1, 1, 1, 1]`` uint32 device tensor (metadata[1]) and ``chunk_end`` its end (metadata[2]); everything
        derived from them is computed here on device, in float32, exact below 2**24 (floors are taken half a step
        above an integer quotient). A new tensor.

        Per query row (chip s, shard row i) and column c (window slot j = c):
            sigma_s = start mod chunk; S, off = the boundary chip's block base and the start's offset in it
            a = i + off [s's block base == S]  (mod chunk_local)      -- the row's position in its block
            o = (s * chunk_local + a - sigma_s) mod chunk              -- its position p minus start
            q = min(start + o - (W - 1) + j, end - 1)                 -- window slot j's position; padding rows
                                                                         (p >= end) read only rows this chunk wrote,
                                                                         so they do not depend on earlier requests
            id = ring_base + P + (q mod P)                            -- its ring row (q < 0 is compacted away)
            perm = min(c + max(0, W - 1 - start - o), W + top-k)      -- slide rows with p < W - 1 left
        """
        L, W, chunk = self.chunk_local, self.window, self.chunk_global
        start = _scalar_f32(chunk_start)  # [1, 1, 1, 1]
        sigma_s = ttnn.subtract(start, _floor_mul(start, chunk, 16.0))  # starts are 32-aligned
        block_s = _floor_mul(sigma_s, L, 16.0)
        off = ttnn.subtract(sigma_s, block_s)
        a = ttnn.add(self._shard_row, ttnn.multiply(ttnn.eq(self._shard_base, block_s), off))
        a = _wrap_above(a, L)
        o = _wrap_below(ttnn.subtract(ttnn.add(self._shard_base, a), sigma_s), chunk)
        q = ttnn.add(ttnn.add(start, ttnn.subtract(o, float(W - 1))), self._column)
        q = ttnn.minimum(q, ttnn.subtract(_scalar_f32(chunk_end), 1.0))
        q = ttnn.relu(q)  # slots before position 0 are compacted away below; keep their arithmetic in range
        ids = ttnn.add(ttnn.subtract(q, _floor_mul(q, self.period, 0.5)), float(self.ring_base + self.period))
        shift = ttnn.relu(ttnn.subtract(ttnn.rsub(o, float(W - 1)), start))
        perm = ttnn.minimum(ttnn.add(self._column, shift), float(self.width))

        def to_rm_u32(x, cols):
            x = ttnn.slice(x, [0, 0, 0, 0], [1, 1, x.shape[2], cols]) if cols != x.shape[3] else x
            return ttnn.to_layout(ttnn.typecast(x, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)

        rows = self._source.shape[2]
        ttnn.experimental.slice_write(
            to_rm_u32(ids, W), self._source, start=[0, 0, 0, 0], end=[1, 1, rows, W], step=[1, 1, 1, 1]
        )
        ttnn.experimental.slice_write(
            topk_indices, self._source, start=[0, 0, 0, W], end=[1, 1, rows, self.width], step=[1, 1, 1, 1]
        )
        return ttnn.gather(self._source, -1, to_rm_u32(perm, self.width))


def _scalar_f32(t: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.typecast(ttnn.to_layout(t, ttnn.TILE_LAYOUT), ttnn.float32)


def _scalar_u32(t: ttnn.Tensor) -> ttnn.Tensor:
    """A [1, 1, 1, 1] fp32 TILE value as the 1-element uint32 ROW_MAJOR tensor the chunked ops read."""
    return ttnn.to_layout(ttnn.typecast(t, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)


def _floor_mul(x, step, nudge):
    """step * floor(x / step) for integer-valued x; nudge is below the distance from x to the next multiple."""
    return ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(x, nudge), 1.0 / step)), float(step))


def _wrap_below(x, bound):  # x + bound where x < 0
    return ttnn.add(x, ttnn.multiply(ttnn.lt(x, 0.0), float(bound)))


def _wrap_above(x, bound):  # x - bound where x >= bound
    return ttnn.subtract(x, ttnn.multiply(ttnn.ge(x, float(bound)), float(bound)))
