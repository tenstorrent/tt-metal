# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 loop pass-2 attention support: the two-bank joint key table and its index list (A2-A6).

Pass-2 layer L attends, under one softmax, over

* bank A: the pass-1 KVPE of layer L (G2's copy of G1's cache) at pass-1 layer L's top-k (A5), and
* bank B: pass 2's own KVPE of positions p-W+1 .. p, the sliding window (A3).

``sparse_sdpa`` reads one ROW_MAJOR tensor in one format, so both banks share one table per mesh, in the bank-A
cache's format (``MlaKvCache.format``; the window format IS that format, see ``window_kv_format``):

    [0, push_rows)                          bank A: the sparse KV push's compact buffer (the rows this chip's top-k
                                            names, pushed by their owner chips; addressed by the plan's compact ids)
    [window_base, window_base + W - 1)      the window carry: pass-2 rows of positions start-W+1 .. start-1
    [chunk_base, chunk_base + chunk)        this chunk's pass-2 rows, all SP shards (the halo: SP all-gathered)

with ``window_base = push_rows`` and ``chunk_base = window_base + W - 1``. The push validates only that its output
holds at least its own rows, so the window rows sit behind them in the same buffer, and the push never writes them.

The index list of a query at chunk row ``g`` is ``[window ids | bank-A compact ids]``, ``W + top-k`` wide. Its window
id for slot ``j`` (position ``start + g - W + 1 + j``) is ``window_base + g + j``: affine, independent of the chunk
start, so it is uploaded once. ``sparse_sdpa`` stops a row at its first sentinel, so sentinels must be one tail; that
holds when every window slot is real (``p >= W - 1``) because bank A's own sentinels are already a tail. A row with
``p < W - 1`` has ``W - 1 - p`` leading window slots before position 0: a gather through
``perm = min(max(c + W - 1 - g - start, c), sentinel column)`` slides the row left by that many slots and fills the end
with the sentinel column, reading ``start`` on device so one captured trace serves every chunk (any 32-aligned start).

Window format (A2, ``loop_window_format``): ``scaled_fp8`` (default) or ``bf16``. The table has one format, so the
window format is the bank-A cache format: a bf16 window means a bf16 table, whose bank A must reach G2 as bf16
(that producer is outside this module). Chunk rows enter the table exactly as the pass-1 cache write would store them
(scaled FP8: per-token FP8 latent with power-of-two scales and BF16 RoPE; bf16: BF16 rows).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.indexer import indexer_layer_is_reused
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCache, MlaKvCacheFormat

_SENTINEL = 0xFFFFFFFF  # sparse_sdpa's "no key"; uploaded as int32 -1 and read back through uint32 (same bits)

WINDOW_FORMATS = {"scaled_fp8": MlaKvCacheFormat.SCALED_FP8, "bf16": MlaKvCacheFormat.BF16_RM}


def window_kv_format(config) -> MlaKvCacheFormat:
    """The joint table's (and the window's) cache format, from ``config.loop_window_format``."""
    try:
        return WINDOW_FORMATS[config.loop_window_format]
    except KeyError as error:
        raise ValueError(
            f"loop_window_format must be one of {sorted(WINDOW_FORMATS)}, got {config.loop_window_format!r}"
        ) from error


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


def chunk_row(position: int, start: int, sp: int, chunk_local: int) -> int:
    """Row of ``position`` (in [start, start + sp * chunk_local)) in the SP all-gather of a chunk at ``start``: the
    gather stacks the SP shards, and chip s's row i holds the position ``rotated_chunk_positions`` gives it (the cache
    writer's layout: chip s owns the slab blocks s, s + sp, ..., so a chunk starting mid-slab rotates the boundary
    chip's rows by the start's offset in its block)."""
    chunk = sp * chunk_local
    if not start <= position < start + chunk:
        raise ValueError(f"position {position} outside the chunk [{start}, {start + chunk})")
    slab_pos, start_slab = position % chunk, start % chunk
    s = slab_pos // chunk_local
    off = start_slab % chunk_local if s == start_slab // chunk_local else 0
    return s * chunk_local + (slab_pos % chunk_local - off) % chunk_local


class TwoBankTable:
    """The joint table, its persistent buffers and index constants, for one mesh; pass-2 layers run serially and
    share it. Built once (outside any trace capture); every per-chunk input is read on device."""

    def __init__(
        self,
        mesh_device,
        *,
        kv_format: MlaKvCacheFormat,
        geometry,
        push_rows: int,
        window: int,
        chunk_global: int,
        query_rows: int,
        topk: int,
        sp_axis: int,
        tp_sequence_sharded: bool,
        num_links: int,
    ):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.mesh_device = mesh_device
        self.kv_format = kv_format
        self.geometry = geometry
        self.window = window
        self.chunk_global = chunk_global
        self.sp_axis = sp_axis
        self.num_links = num_links
        sp, tp = mesh_device.shape[sp_axis], mesh_device.shape[1 - sp_axis]
        self.row_width = kv_format.storage_width(geometry)
        self.window_base = push_rows
        self.chunk_base = push_rows + window - 1
        self.width = window + topk  # index list width
        if self.width % 32:
            raise ValueError(f"index list width {self.width} must be a multiple of 32 for sparse_sdpa's k chunks")

        def rows_tensor(rows):
            return ttnn.allocate_tensor_on_device(
                ttnn.TensorSpec(
                    shape=ttnn.Shape([1, 1, rows, self.row_width]),
                    dtype=kv_format.storage_dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    buffer_type=ttnn.BufferType.DRAM,
                ),
                mesh_device,
            )

        # Contents undefined until written: the push fills bank A's rows, the chunk write and the carry the window's,
        # and no index names a row that neither wrote (carry rows of positions < 0 are compacted away).
        self.storage = rows_tensor(self.chunk_base + chunk_global)
        self._gathered = rows_tensor(chunk_global)  # the SP all-gather's output, copied behind the carry

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

        self.sp, self.chunk_local = sp, chunk_global // sp
        if chunk_global % (sp * 32) or self.chunk_local % query_rows:
            raise ValueError(f"chunk {chunk_global} must split into 32-row SP shards of whole query-row blocks")
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
        # Source of the gather: [window ids | bank-A ids | sentinel column], ids written per chunk.
        source = torch.zeros(sp, tp, query_rows, self.width + 1, dtype=torch.int64)
        source[..., -1] = -1
        self._source = per_chip(source, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def table(self) -> MlaKvCache:
        return MlaKvCache(format=self.kv_format, storage=self.storage, geometry=self.geometry)

    def write_window(self, rows: ttnn.Tensor, window_carry: ttnn.Tensor | None) -> None:
        """Place this chunk's pass-2 rows (this SP shard's ``[1, 1, chunk/sp, row]``, TP-replicated, in the table
        format) behind the carry, gathered over SP so every chip holds the whole chunk (the shard halo), and the carry
        (``[1, 1, W - 1, row]``, replicated; None leaves the region as it is) in front of it."""
        gathered = ttnn.experimental.high_bw_all_gather(
            rows, dim=2, output_tensor=self._gathered, cluster_axis=self.sp_axis, num_links=self.num_links
        )
        self._write_rows(gathered, self.chunk_base)
        if window_carry is not None:
            if tuple(window_carry.shape) != (1, 1, self.window - 1, self.row_width):
                raise ValueError(
                    f"window carry must be [1, 1, {self.window - 1}, {self.row_width}], got {window_carry.shape}"
                )
            self._write_rows(window_carry, self.window_base)

    def _write_rows(self, rows: ttnn.Tensor, start: int) -> None:
        # The interleaved row-major slice_write compiles only the last-dim start (0), so fixed offsets stay trace-safe.
        end = [1, 1, start + rows.shape[2], self.row_width]
        ttnn.experimental.slice_write(rows, self.storage, start=[0, 0, start, 0], end=end, step=[1, 1, 1, 1])

    def table_row(self, position: int, start: int) -> int:
        """Table row of window position ``position`` (in [start - W + 1, start + chunk)) for a chunk at ``start``: the
        carry region in position order, then the chunk region in the SP gather's (rotated) order. Host mirror of the
        ids ``index_list`` computes on device."""
        if position < start:
            if position < start - (self.window - 1):
                raise ValueError(f"position {position} is before the window of a chunk at {start}")
            return self.chunk_base - (start - position)
        return self.chunk_base + chunk_row(position, start, self.sp, self.chunk_local)

    def window_rows(self) -> ttnn.Tensor:
        """The window region, [1, 1, W - 1 + chunk, row]: the carry, then this chunk's pass-2 rows in gather order
        (``table_row(p, start) - window_base`` is position p's row). A new tensor."""
        return ttnn.slice(
            self.storage, [0, 0, self.window_base, 0], [1, 1, self.chunk_base + self.chunk_global, self.row_width]
        )

    def window_carry(self, start: int, end: int) -> ttnn.Tensor:
        """After the chunk [start, end) (end = its last real token + 1), the carry the next chunk (at ``end``) passes
        back: positions end-W+1 .. end-1 in position order, [1, 1, W - 1, row]; rows of positions < 0 are
        unspecified. Eager (host offsets, a few slices); a new tensor."""
        if not start < end <= start + self.chunk_global:
            raise ValueError(f"chunk [{start}, {end}) is not a non-empty chunk of at most {self.chunk_global} tokens")
        first = end - (self.window - 1)
        rows = [self.table_row(max(p, start - (self.window - 1)), start) for p in range(first, end)]
        pieces, lo = [], 0
        for k in range(1, len(rows) + 1):  # runs of consecutive table rows
            if k == len(rows) or rows[k] != rows[k - 1] + 1:
                pieces.append(ttnn.slice(self.storage, [0, 0, rows[lo], 0], [1, 1, rows[k - 1] + 1, self.row_width]))
                lo = k
        if len(pieces) == 1:
            return pieces[0]
        carry = ttnn.concat(pieces, dim=2)
        for piece in pieces:
            ttnn.deallocate(piece)
        return carry

    def index_list(self, bank_a_ids: ttnn.Tensor, chunk_start: ttnn.Tensor) -> ttnn.Tensor:
        """This chunk's index list ``[1, 1, query_rows, W + top-k]`` uint32: the window ids and ``bank_a_ids`` (the
        push plan's compact ids), compacted for rows before position W - 1. ``chunk_start`` is the chunk's start as a
        ``[1, 1, 1, 1]`` uint32 device tensor (metadata[1]); everything derived from it is computed here on device, in
        float32, exact below 2**24 (floors are taken half a step above an integer quotient). A new tensor.

        Per query row (chip s, shard row i) and column c (window slot j = c):
            sigma_s = start mod chunk; S, off = the boundary chip's block base and the start's offset in it
            a = i + off [s's block base == S]  (mod chunk_local)      -- the row's position in its block
            o = (s * chunk_local + a - sigma_s) mod chunk              -- its position minus start
            u = o - (W - 1) + j                                       -- window slot j's position minus start
            id = u < 0 ? chunk_base + u                               -- carry region, position order
                       : chunk_base + chunk_row(start + u)            -- the gather's rotated order
            perm = min(c + max(0, W - 1 - start - o), W + top-k)      -- slide rows with positions < W - 1 left
        """
        L, W, chunk = self.chunk_local, self.window, self.chunk_global
        f32 = ttnn.float32
        start = ttnn.typecast(ttnn.to_layout(chunk_start, ttnn.TILE_LAYOUT), f32)  # [1, 1, 1, 1]

        def floor_mul(x, step, nudge):  # step * floor(x / step) for integer x, nudge < distance to the next multiple
            return ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(x, nudge), 1.0 / step)), float(step))

        def wrap_below(x, bound):  # x + bound where x < 0
            return ttnn.add(x, ttnn.multiply(ttnn.lt(x, 0.0), float(bound)))

        def wrap_above(x, bound):  # x - bound where x >= bound
            return ttnn.subtract(x, ttnn.multiply(ttnn.ge(x, float(bound)), float(bound)))

        sigma_s = ttnn.subtract(start, floor_mul(start, chunk, 16.0))  # starts are 32-aligned
        block_s = floor_mul(sigma_s, L, 16.0)
        off = ttnn.subtract(sigma_s, block_s)
        a = ttnn.add(self._shard_row, ttnn.multiply(ttnn.eq(self._shard_base, block_s), off))
        a = wrap_above(a, L)
        o = wrap_below(ttnn.subtract(ttnn.add(self._shard_base, a), sigma_s), chunk)
        u = ttnn.add(ttnn.subtract(o, float(W - 1)), self._column)
        sigma_q = wrap_above(wrap_below(ttnn.add(sigma_s, u), chunk), chunk)
        block_q = floor_mul(sigma_q, L, 0.5)
        i_q = ttnn.subtract(ttnn.subtract(sigma_q, block_q), ttnn.multiply(ttnn.eq(block_q, block_s), off))
        i_q = wrap_below(i_q, L)
        chunk_ids = ttnn.add(ttnn.add(block_q, i_q), float(self.chunk_base))
        carry_ids = ttnn.add(u, float(self.chunk_base))
        ids = ttnn.where(ttnn.lt(u, 0.0), carry_ids, chunk_ids)
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
            bank_a_ids, self._source, start=[0, 0, 0, W], end=[1, 1, rows, self.width], step=[1, 1, 1, 1]
        )
        return ttnn.gather(self._source, -1, to_rm_u32(perm, self.width))


def _row_major_dram(t: ttnn.Tensor) -> ttnn.Tensor:
    if t.layout != ttnn.ROW_MAJOR_LAYOUT:
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if t.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return t


def pack_window_rows(kv_format: MlaKvCacheFormat, fields) -> ttnn.Tensor:
    """This chip's chunk rows in the table format, from the sparse KV stem's fields (it returns them unpacked for
    the fused cache write): scaled FP8 (latent FP8, scales, RoPE) or BF16 (latent, RoPE). Consumes ``fields``. A single
    tensor is already packed (the stem packs when it also returns KV intermediates)."""
    if not isinstance(fields, tuple):
        return fields
    if kv_format == MlaKvCacheFormat.SCALED_FP8:
        # The pack op reads ROW_MAJOR DRAM-interleaved fields; the stem leaves RoPE tiled (and fields may sit in L1).
        dram = [_row_major_dram(t) for t in fields]
        rows = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(*dram)
        for t in {id(x): x for x in (*fields, *dram)}.values():
            ttnn.deallocate(t)
        return rows
    latent, rope = fields
    joined = ttnn.concat([latent, rope], dim=-1)
    ttnn.deallocate(latent)
    ttnn.deallocate(rope)
    rows = ttnn.to_layout(joined, ttnn.ROW_MAJOR_LAYOUT)
    if rows is not joined:
        ttnn.deallocate(joined)
    return ttnn.typecast(rows, ttnn.bfloat16) if rows.dtype != ttnn.bfloat16 else rows
