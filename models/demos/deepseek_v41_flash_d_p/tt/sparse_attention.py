# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 attention of a compressed (ratio 1 / 2) layer (reference/ds41_ref.py:_attention with r > 0;
inference Attention + sparse_attn) on the SP x TP mesh: the window of tt/attention.py plus the layer's top-k
compressed rows of its KV source, in ONE sparse_sdpa per chunk.

    keys of position p = window rows [p - 127, p] (as tt/attention.TtWindowAttention) + compressed rows topk[p]
                         (natural group ids t < (p + 1) // r, -1 = none); MQA: keys = values, 512 wide
    attn_out = fp8_linear(grouped_wo_a(fp8_qdq(unrope(softmax_sink(q . keys) keys))), wo_b)

Everything but the table and the index list is TtWindowAttention's chain (q: bf16 cast -> wq_b (heads over TP) ->
RoPE -> x 512^-0.5 -> bf16 -> head -> seq all_to_all; stock ttnn.transformer.sparse_sdpa(attention_sink); back,
unrope, bf16, grouped wo_a, bf16, wo_b, TP sum; the released code's fp8 quant-dequant steps are not emulated, FIX.8). Difference: q and the table keep the rope tail in the reference's interleaved order (``INTERLEAVED``):
the compressed rows are stored that way, so no per-chunk transform of the (up to 1 GiB) compressed table is needed;
q is re-interleaved after its fp32 RoPE and the output de-interleaved before the inverse RoPE (data movement only,
exact). The scores are the same dot products (summed in another order).

The table (one per chunk, ROW_MAJOR bf16, replicated):
    [window region (128 carry + S chunk rows, tt/attention._table: writes the chunk's window KV into the served
     0_window_kv cache first) | compressed rows]
    compressed rows = the KV source's served compressed cache (tt/compressor.TtServedRows, compressed rows, TP-dedup:
    token g on owner chip q = (g / stripe) % N, local row (g / (N stripe)) stripe + g mod stripe), the local rows of
    the first nb = ceil((start + S) / S) chunk periods (every group a query of this chunk can pick; the compressor
    wrote this chunk's groups before), viewed as compressed rows [nb stripe / r, 512], tiled, all_gather over axis 1
    then axis 0 (chip-linear order q = r TP + c) -> [N nb stripe / r, 512] ROW_MAJOR: the gather's mesh order.
    Group j -> table row 128 + S + q (nb hs) + b hs + (j mod hs), hs = stripe / r, s = j div hs, q = s mod N,
    b = s div N: computed on the device from the uint32 top-k with fp32 integer arithmetic (exact below 2^24;
    floor((x + 1/2) / d) is exact for any divisor d), -1 kept as the sentinel (K-18: fp32 -> int32 -> bitcast).
The index list per query [window 128 | top-k 512] (640 = 5 k-chunks of 128) uint32 ROW_MAJOR: sparse_sdpa stops at
the first sentinel (a contiguous tail, sparse_sdpa_reader.cpp), so for starts < 128 (a window with < 128 real rows)
the list is compacted by ttnn.gather through a load-time permutation per start class (TtCSA.index_tables' idea):
the real window ids, then the top-k (its own sentinel tail), the rest the window's sentinel. The window part is
TtWindowAttention's load-time list; the top-k's sentinels are already a tail (indexer: descending score, -inf ->
0xFFFFFFFF; the reference: ascending ids, -1 past the visible ones).

No host work per chunk: the per-chunk inputs are integers (start, end) picking load-time tables and scalars.

STATIC_TABLES=on (P.21, the default): the table above is not rebuilt per chunk. Each KV source keeps ONE persistent
table per slot (tt/compressor.AttnTable, made with its served compressed cache): [the history in natural order, group
j at row j, max_seq / r rows | window region 128 + S at base = max_seq / r]. The compressor appends the chunk's
groups with one slice_write (the replicated natural-order rows it writes into the served cache), a host load of the
served cache writes the same rows; each compressed layer slice_writes its [carry | chunk] into the window region (the
source's layers run in order on one queue and reuse it). The index list is [window ids + base (load-time) | the top-k
ids as they are]: mesh_partition + concat, + the same compaction gather for starts < 128 (INDEXER_TOPK_LOCAL=on, P.8:
the indexer's top-k is already the chip's own rows, so no mesh_partition). No history gather, no
[window | history] concat, no fp32 id remap, and the table has one shape in every chunk.
"""

from __future__ import annotations

import torch

import ttnn

from . import ccl
from .attention import WINDOW, TtWindowAttention, TtWindowCache, WindowTables, _upload, attention_weights
from .ccl import dealloc
from .common import DRAM, SP_AXIS, TP_AXIS
from .compressor import AttnTable, static_tables
from .rope import RopeTable

TOPK_LIMIT = 1 << 24  # ids (and table rows) are exact in fp32 below this


class ChunkMemo:
    """INDEXER_GLUE=on (P.16, glue report item 7): device tensors several layers of a chunk read from one source,
    computed by the first and reused by the rest, one entry per slot:

        ("comp", kv source)    the KV source's compressed history gathered over the mesh (``_compressed``) and nb:
                               the source layer's own attention and every reader of it (L3 of L2; L21, L24 of L20)
        ("index", idx source)  the index list of a shared top-k ([window | top-k rows], ``_index``): L3 = L2's, L21 =
                               L20's (same top-k, same KV source, same start)
        ("keys", kv source)    the tiled index keys of a served index-key cache (tt/indexer.py ``_keys``): L24 =
                               L20's

    Each entry carries the key it was made for (the chunk start, the source cache's write version, its identity);
    a lookup with another key misses, and a new entry frees the slot's old one. tt/model.py drops a slot after its
    last reader layer of the chunk (``TtDS41Model.memo_last``) and the top-k store drops the "index" slot of a layer
    whose top-k is replaced; ``TtDS41Model.begin_chunk`` empties it at every chunk start (FIX.5), so nothing is
    reused across chunks or across re-runs of one chunk. Per slot (per DS41DeviceState)."""

    def __init__(self):
        self._d: dict = {}

    def get(self, slot, key):
        hit = self._d.get(slot)
        if hit is None or hit[0] != key:
            return None
        return hit[1]

    def put(self, slot, key, value) -> None:
        self.drop(slot)
        self._d[slot] = (key, value)

    def drop(self, slot) -> None:
        hit = self._d.pop(slot, None)
        if hit is None:
            return
        v = hit[1]
        for t in v if isinstance(v, tuple) else (v,):
            if isinstance(t, ttnn.Tensor) and t.is_allocated():
                dealloc(t)

    def clear(self) -> None:
        for slot in list(self._d):
            self.drop(slot)


class SparseIndexTables:
    """Load-time compaction permutations of the [window 128 | top-k K] index list, per start class < 128 (the class
    of start >= 128 is the identity): [SP, TP, q_rows, 128 + K] uint32, chip (r, c) its own query rows."""

    def __init__(self, mesh, tables: WindowTables, topk: int):
        g = tables.geo
        self.topk, self.width = topk, WINDOW + topk
        d = g.query_offsets().unsqueeze(-1)  # [SP, TP, q, 1] natural chunk offsets of the chip's queries
        i = torch.arange(self.width)
        self.perm = {}
        for cls in range(0, WINDOW, 32):
            w = torch.clamp(cls + d + 1, max=WINDOW)  # real window rows of position start + d (start = cls)
            dump = WINDOW - 1  # a sentinel of the window part (it has one whenever w < 128)
            perm = torch.where(i < w, i, torch.where(i < w + topk, WINDOW + i - w, torch.full_like(i, dump)))
            self.perm[cls] = _upload(mesh, perm.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (0, 1))


class TtSparseAttention(TtWindowAttention):
    """attention of a compressed layer. ``__call__(q_resid, kv, start, end, cache, ckv_cache, topk)``: q_resid
    [1, 1, S/SP, 1280] fp32 and kv [1, 1, S/SP, 512] bf16 (row-split, TP-replicated, natural blocks of
    positions [start, start + S)), end = actual_end, ``cache`` the layer's TtWindowCache (written), ``ckv_cache``
    the KV source's served compressed cache (tt/compressor.TtServedRows, read; already holding this chunk's groups
    when the layer is its own source), topk [1, 1, S/SP, K] uint32 ROW_MAJOR (row-split, TP-replicated; group ids,
    0xFFFFFFFF = none) or the chip's own rows [1, 1, q_rows, K] (INDEXER_TOPK_LOCAL=on, P.8) -> attn_out [1, 1, S/SP, H/TP] fp32. Does not consume its inputs.
    """

    INTERLEAVED = True

    def __init__(self, mesh, cfg, w: dict, rope: RopeTable, tables: WindowTables, sparse: SparseIndexTables):
        super().__init__(mesh, cfg, w, rope, tables)
        self.sparse = sparse
        self.static = static_tables()  # STATIC_TABLES=on (P.21): the KV source's persistent table (``_static``)

    # ------------------------------------------------------------------ parts
    def _gather_mesh(self, t: ttnn.Tensor, dedup: bool) -> ttnn.Tensor:
        """[1, 1, n, w] TILE per chip -> its copies gathered in chip-linear order (TP then SP when ``dedup``, else
        over SP only)."""
        g = self.geo
        out = t
        if dedup and g.tp > 1:
            out = ccl.all_gather(out, 2, TP_AXIS)
        if g.sp > 1:
            nxt = ccl.all_gather(out, 2, SP_AXIS)
            if out is not t:
                dealloc(out)
            out = nxt
        return out

    def _compressed(self, ckv_cache, start: int) -> tuple[ttnn.Tensor, int]:
        """The KV source's compressed rows of the first nb chunk periods in the gather's mesh order:
        [1, 1, owners nb hs, 512] bf16 ROW_MAJOR, replicated; returns (rows, nb)."""
        g = self.geo
        r, stripe = ckv_cache.ratio, ckv_cache.stripe
        assert ckv_cache.width == self.d and ckv_cache.tok_width * r == self.d, (ckv_cache.width, self.d)
        nb = -(-(start + g.chunk) // g.chunk)
        loc = nb * stripe
        assert loc <= ckv_cache.rows, f"chunk [{start}, {start + g.chunk}) past the compressed cache ({ckv_cache.rows})"
        sl = ckv_cache.slab.slice_rows(0, loc // r)  # [1, 1, loc / r, 512] compressed rows
        tl = ttnn.to_layout(sl, ttnn.TILE_LAYOUT, memory_config=DRAM)
        dealloc(sl)
        gat = self._gather_mesh(tl, ckv_cache.tp_axis is not None)
        if gat is not tl:
            dealloc(tl)
        rm = ttnn.to_layout(gat, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        dealloc(gat)
        return rm, nb

    def _topk_rows(self, topk: ttnn.Tensor, ckv_cache, nb: int, offset: int) -> ttnn.Tensor:
        """The chip's queries' top-k group ids [1, 1, q_rows, K] uint32 ROW_MAJOR -> table rows (same shape and
        dtype; 0xFFFFFFFF kept): offset + q (nb hs) + b hs + j mod hs, s = j div hs, q = s mod N, b = s div N."""
        hs = ckv_cache.stripe // ckv_cache.ratio  # compressed rows per owner stripe
        n_own = ckv_cache.owners
        assert offset + n_own * nb * hs < TOPK_LIMIT

        f32 = dict(dtype=ttnn.float32, memory_config=DRAM)

        def op(fn, *args):  # fn(*args) fp32, consuming the tensor arguments made here
            out = fn(*args, **f32)
            for t in args:
                if isinstance(t, ttnn.Tensor) and t is not x:
                    dealloc(t)
            return out

        def floor_div(v, d):  # floor((v + 1/2) / d): exact for integers v >= 0 and any divisor d
            y = op(ttnn.multiply, ttnn.add(v, 0.5, **f32), 1.0 / d)
            out = ttnn.floor(y, memory_config=DRAM)
            dealloc(y)
            return out

        tiled = topk.layout != ttnn.TILE_LAYOUT  # K-1: to_layout to the same layout returns its input (not ours)
        tl = ttnn.to_layout(topk, ttnn.TILE_LAYOUT, memory_config=DRAM) if tiled else topk
        x = ttnn.typecast(ttnn.bitcast(tl, ttnn.int32), ttnn.float32, memory_config=DRAM)  # none -> -1.0
        if tiled:
            dealloc(tl)
        s = floor_div(x, hs)
        u = op(ttnn.subtract, x, op(ttnn.multiply, ttnn.clone(s, memory_config=DRAM), float(hs)))  # j mod hs
        b = floor_div(s, n_own)
        q = op(ttnn.subtract, s, op(ttnn.multiply, ttnn.clone(b, memory_config=DRAM), float(n_own)))  # s mod N
        row = op(ttnn.add, op(ttnn.add, u, float(offset)), op(ttnn.multiply, b, float(hs)))
        row = op(ttnn.add, row, op(ttnn.multiply, q, float(nb * hs)))
        valid = ttnn.ge(x, 0.0, dtype=ttnn.float32)
        out = ttnn.where(valid, row, -1.0)
        for t in (x, row, valid):
            dealloc(t)
        i32 = ttnn.typecast(out, ttnn.int32, memory_config=DRAM)  # -1 stays -1 (K-18)
        dealloc(out)
        u32 = ttnn.bitcast(i32, ttnn.uint32)
        rm = ttnn.to_layout(u32, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        if u32 is not i32:
            dealloc(u32)
        dealloc(i32)
        return rm

    def _mine(self, topk: ttnn.Tensor) -> ttnn.Tensor:
        """The chip's own query rows [1, 1, q_rows, K] of a top-k: as given when it is already per chip
        (INDEXER_TOPK_LOCAL=on, P.8: the indexer did not gather it; or TP == 1), else mesh_partition of the
        row-split, TP-replicated [1, 1, S/SP, K] (the gathered layout; component harnesses)."""
        g = self.geo
        rows = topk.shape[-2]
        if g.tp == 1 or rows == g.q_rows:
            return topk
        assert rows == g.w, (rows, g.w, g.q_rows)
        return ttnn.mesh_partition(topk, dim=2, cluster_axis=TP_AXIS, memory_config=DRAM)

    def _index(self, topk: ttnn.Tensor, ckv_cache, nb: int, start: int) -> ttnn.Tensor:
        """The chip's index list [1, 1, q_rows, 128 + K] uint32 ROW_MAJOR: [window | top-k rows], sentinels last."""
        g = self.geo
        assert topk.shape[-1] == self.sparse.topk, (topk.shape, self.sparse.topk)
        mine = self._mine(topk)
        rows = self._topk_rows(mine, ckv_cache, nb, WINDOW + g.chunk)
        if mine is not topk:
            dealloc(mine)
        cls = g.start_class(start)
        src = ttnn.concat([self.tables.index[cls], rows], dim=-1, memory_config=DRAM)
        dealloc(rows)
        if cls == WINDOW:  # every window full: the top-k's own sentinel tail ends the list
            return src
        out = ttnn.gather(src, -1, self.sparse.perm[cls])
        dealloc(src)
        return out

    def _index_static(self, topk: ttnn.Tensor, table: AttnTable, start: int) -> ttnn.Tensor:
        """STATIC_TABLES=on: the chip's index list [1, 1, q_rows, 128 + K] uint32 ROW_MAJOR = [window ids + base |
        the top-k group ids as they are] (group j is table row j; 0xFFFFFFFF stays the sentinel), compacted for
        starts < 128 as ``_index``."""
        g = self.geo
        assert topk.shape[-1] == self.sparse.topk, (topk.shape, self.sparse.topk)
        mine = self._mine(topk)
        cls = g.start_class(start)
        src = ttnn.concat([table.index[cls], mine], dim=-1, memory_config=DRAM)
        if mine is not topk:
            dealloc(mine)
        # every window full: the top-k's own sentinel tail ends the list; CKV_FORMAT=bfp4_rows: ROW_BFP4 takes the
        # window's and the top-k's sentinel tails apart (no compaction)
        if cls == WINDOW or table.bfp4:
            return src
        out = ttnn.gather(src, -1, self.sparse.perm[cls])
        dealloc(src)
        return out

    def _static(self, q_resid, kv, start, end, cache, ckv_cache, topk, memo, sources) -> ttnn.Tensor:
        """STATIC_TABLES=on (P.21): the attention over the KV source's persistent table (tt/compressor.AttnTable):
        this layer's [carry | chunk] window rows slice_written into its window region, the history already there
        (the compressor appended this chunk's groups, a host load the prefix)."""
        g = self.geo
        table = ckv_cache.table
        assert self.INTERLEAVED, "the table holds the rows as stored (interleaved rope tail)"
        assert table is not None, "STATIC_TABLES=on: the KV source's compressed cache was built without its table"
        cos, sin = self.rope.rows(start, g.chunk)
        carry, kv_rm = self._window_rows(kv, start, end, cache)
        table.put_window(carry, kv_rm)
        self._release_window(carry, kv_rm, cache, start, end)
        shared = memo is not None and sources is not None
        ikey = (start, id(ckv_cache), ckv_cache.version)
        index = memo.get(("index", sources[1]), ikey) if shared else None
        if index is None:
            index = self._index_static(topk, table, start)
            if shared:
                memo.put(("index", sources[1]), ikey, index)
        q_rm = self._q(q_resid, cos, sin)
        o = self._sdpa(q_rm, table.buf, index, window=table.win)
        dealloc(q_rm)
        if not shared:
            dealloc(index)
        out = self._out(o, cos, sin)
        dealloc(o)
        self.rope.free(cos, sin)
        return out

    def __call__(
        self,
        q_resid,
        kv,
        start: int,
        end: int | None,
        cache: TtWindowCache,
        ckv_cache,
        topk,
        memo: ChunkMemo | None = None,
        sources: tuple[int, int] | None = None,
    ) -> ttnn.Tensor:
        """``memo`` / ``sources`` (INDEXER_GLUE=on, P.16): the chunk's ChunkMemo and this layer's (KV source, index
        source) layers: the gathered compressed history and the index list are taken from it when an earlier layer
        of the chunk made them for the same source, start and cache version."""
        g = self.geo
        s = q_resid.shape[-2] * g.sp
        assert s == g.chunk, f"chunk of {s} rows; the window tables were built for {g.chunk}"
        end = start + s if end is None else int(end)
        if self.static:
            return self._static(q_resid, kv, start, end, cache, ckv_cache, topk, memo, sources)
        cos, sin = self.rope.rows(start, s)
        window = self._table(kv, start, end, cache)  # [1, 1, 128 + S, 512], interleaved tail (INTERLEAVED)
        shared = memo is not None and sources is not None
        ckey = (start, id(ckv_cache), ckv_cache.version)
        hit = memo.get(("comp", sources[0]), ckey) if shared else None
        if hit is None:
            comp, nb = self._compressed(ckv_cache, start)
            if shared:
                memo.put(("comp", sources[0]), ckey, (comp, nb))
        else:
            comp, nb = hit
        table = ttnn.concat([window, comp], dim=2, memory_config=DRAM)
        dealloc(window)
        if not shared:
            dealloc(comp)
        index = memo.get(("index", sources[1]), ckey) if shared else None
        if index is None:
            index = self._index(topk, ckv_cache, nb, start)
            if shared:
                memo.put(("index", sources[1]), ckey, index)
        q_rm = self._q(q_resid, cos, sin)
        o = self._sdpa(q_rm, table, index)
        for t in (q_rm, table) + (() if shared else (index,)):
            dealloc(t)
        out = self._out(o, cos, sin)
        dealloc(o)
        self.rope.free(cos, sin)  # INDEXER_GLUE=on: the chunk's memoized rows stay
        return out


def build_sparse_attention(mesh, ck, cfg, layer: int, max_seq: int, chunk: int, rope=None, tables=None, sparse=None):
    """The attention of a compressed layer (its own rope table / window tables / index permutations unless shared
    ones are given)."""
    from .kv import rope_freqs

    assert cfg.ratio(layer), f"layer {layer} is window-only: tt/attention.build_attention"
    rope = rope or RopeTable(mesh, rope_freqs(cfg, layer), max_seq)
    tables = tables or WindowTables(mesh, chunk)
    sparse = sparse or SparseIndexTables(mesh, tables, cfg.index_topk)
    return TtSparseAttention(mesh, cfg, attention_weights(ck, layer), rope, tables, sparse)
