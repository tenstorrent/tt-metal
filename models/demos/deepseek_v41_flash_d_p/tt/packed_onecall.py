# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PACKED_ATTN_ONECALL=on (PB.3A, owner 2026-11-19; bringup/PACKED_DESIGN.md s5 v2): a packed chunk's attention steps
ONCE a layer over every lane, each chip computing its own lane's segment in place (no lane moves, no 4x compute).

PB.3's v1 (tt/packed_device.py, ONECALL=off) moves each lane's rows to lane 0's chips and runs today's attention
steps once per segment. Every chip computes its rows at positions start + q R + d there, so lane k's rows are right
only after the move. Here every position-dependent quantity becomes a per-chip one instead, read from the chip's lane:

    rope rows        tt/rope.LaneStarts: RopeTable / CompressRope slice each lane's rows at its own start and
                     concatenate them (the natural-order rows of the packed chunk), split over the mesh as today
    q_a, kv, q, wq_b, wo_a, wo_b, the compressor's matmuls, the indexer's gates and query: once (row-local)
    window region   each chip's own rows masked at its lane's valid end (a per-chip keep mask, made once per tuple of
                     lane ends), written into the layer's region; the 128 carry rows: from the previous chip of the
                     lane (the neighbour send, never across a lane boundary), on a lane's first chip from its own
                     copy of the slot's carry (``rtail``, written replicated per segment: a local copy, no send).
                     The ring pieces of every lane's slot go to that lane's chips (LaneSend: one prepared program)
    window index    the region lists of each chip's lane start class (a lane's first chip: its segment's class;
                     every other chip's lists are the class-128 lists, identical on every chip)
    compressor      once; each segment's served writes from the chunk's compressed rows gathered once over SP and
                     rolled so that its lane comes first (the rows v1's move produced, bit for bit)
    indexer score   per segment: one ring call over the segment's slot keys with ttnn.bringup.ring_indexer_score_dsa's
                     natural_query_lane (fork option, PB.3A): the lane's chips score at the segment's positions;
                     the top-k (and the candidate pool) of the segment kept in its slot state as today
    history table   per segment the slot's history gathered into its lane buffer (v1's per-lane buffers), then
                     copied on the lane's chips into one table buffer (LaneCopy): sparse_sdpa reads ONE table, each
                     chip its lane's slot rows
    top-k           each segment's top-k copied on its lane's chips into one buffer (LaneCopy)
    sparse_sdpa     once a layer (window and compressed layers)

Bit-identical to v1 (the loop), segment by segment: a lane's chips run the same ops on the same bytes as lane 0's
chips did in v1 (the oracle, tests/packed/test_packed_oracle.py, with the switch on). A lone segment in lane 0 is
today's chunk (PackedChunk.trivial: this module is not used). Every per-chip table here is a constant of the
descriptor's lane ends / start classes, uploaded once per distinct tuple (``_const``) and kept.
"""

from __future__ import annotations

import ttnn
from models.demos.common.bringup.testing import profiler

from . import ccl
from .attention import WINDOW
from .ccl import dealloc
from .common import DRAM
from .cores import sub_cores, to_rm
from .settings import settings

RING_PIECES = 4  # tt/window_send.RING_PIECES


def onecall_on() -> bool:
    """PACKED_ATTN_ONECALL=on (call-time, bringup/hooks.py batch_call_settings)."""
    return settings.get("PACKED_ATTN_ONECALL") == "on"


# ---------------------------------------------------------------------------------------------- per-chip constants
_CONST: dict = {}  # (id(mesh), kind, key) -> (mesh, tensor): constants of a tuple of lane values, kept for the process
_CONST_MAX = 256


def _const(mesh, kind: str, key, make):
    k = (id(mesh), kind, key)
    hit = _CONST.get(k)
    if hit is not None and hit[0] is mesh and hit[1].is_allocated():
        _CONST[k] = _CONST.pop(k)  # (most recent last: PB.7C, fresh layouts fill the set; the oldest insertion could
        # be a constant the caller fetched a moment ago for this very call, freed under it: "not allocated")
        return hit[1]
    if len(_CONST) >= _CONST_MAX:  # (a bounded set: the least recently used goes)
        old = next(iter(_CONST))
        t = _CONST.pop(old)[1]
        if t.is_allocated():
            ttnn.deallocate(t)
    t = make()
    ccl.pin(mesh, t)
    _CONST[k] = (mesh, t)
    return t


def topk_rows_lanes(att, topk: ttnn.Tensor, ckv_cache, stride_t: ttnn.Tensor) -> ttnn.Tensor:
    """tt/sparse_attention.TtSparseAttention._topk_rows (offset 0) with each chip's own owner stride (``stride_t``, a
    per-chip [1, 1, 1, 1] fp32 scalar: its lane's gather buffer's slab rows; slots whose caches differ in size).
    Integers below 2^24 throughout (exact): the rows of the keys v1 reads in that lane's buffer."""
    from .cores import sub_cores_tiled

    hs = ckv_cache.stripe // ckv_cache.ratio
    n_own = ckv_cache.owners
    f32 = dict(dtype=ttnn.float32, memory_config=DRAM)

    def op(fn, *args):
        out = fn(*args, **f32, sub_core_grids=sub_cores(x))
        for t in args:
            if isinstance(t, ttnn.Tensor) and t is not x and t is not stride_t:
                dealloc(t)
        return out

    def floor_div(v, d):
        y = op(ttnn.multiply, ttnn.add(v, 0.5, **f32, sub_core_grids=sub_cores(v)), 1.0 / d)
        out = ttnn.floor(y, memory_config=DRAM, sub_core_grids=sub_cores(y))
        dealloc(y)
        return out

    tiled = topk.layout != ttnn.TILE_LAYOUT
    tl = ttnn.to_layout(topk, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(topk)) if tiled else topk
    x = ttnn.typecast(
        ttnn.bitcast(tl, ttnn.int32, sub_core_grids=sub_cores(tl)),
        ttnn.float32,
        memory_config=DRAM,
        sub_core_grids=sub_cores(),
    )
    if tiled:
        dealloc(tl)
    s = floor_div(x, hs)
    u = op(ttnn.subtract, x, op(ttnn.multiply, ttnn.clone(s, memory_config=DRAM), float(hs)))
    b = floor_div(s, n_own)
    q = op(ttnn.subtract, s, op(ttnn.multiply, ttnn.clone(b, memory_config=DRAM), float(n_own)))
    row = op(ttnn.add, op(ttnn.add, u, 0.0), op(ttnn.multiply, b, float(hs)))
    row = op(ttnn.add, row, op(ttnn.multiply, q, stride_t))
    valid = ttnn.ge(x, 0.0, dtype=ttnn.float32, sub_core_grids=sub_cores(x))
    out = ttnn.where(valid, row, -1.0, sub_core_grids=sub_cores(valid))
    for t in (x, row, valid):
        dealloc(t)
    i32 = ttnn.typecast(out, ttnn.int32, memory_config=DRAM, sub_core_grids=sub_cores_tiled(out))
    dealloc(out)
    u32 = ttnn.bitcast(i32, ttnn.uint32, sub_core_grids=sub_cores(i32))
    rm = to_rm(u32, memory_config=DRAM)
    if u32 is not i32:
        dealloc(u32)
    dealloc(i32)
    return rm


def snapshot_carry_ok(cache, start: int) -> bool:
    """The window carry [start - 128, start) of a segment is its slot's served snapshot (WINDOW_KV_KEEP=snapshot128):
    the snapshot was written at cap == start (the slot's last chunk ended there: a continuing request) with every
    row of the carry held (no ``snap_missing`` inside it), start >= 128. The snapshot is every chip's copy of the
    ring owners' rows (TtWindowCache.write_snapshot): the bytes the ring carry gathers, read in place by the lane send
    (carry row i = snapshot row (start + i) % 128)."""
    if cache.snap is None or cache.snap_cap != start or start < WINDOW or cache.carry_masked(start):
        return False
    m = cache.snap_missing
    return m is None or m[1] <= start - WINDOW


# ---------------------------------------------------------------------------------------------- LaneCopy


_COPIES: dict = {}  # (id(mesh), lane chips, page, blocks, stride, dstride) -> LaneCopy


# ---------------------------------------------------------------------------------------------- LaneSend
# The window region's fused write per lane (tt/window_send.WindowSend.fused, ATTN_FUSE=on, bf16 regions): the copy
# kernel writes the chip's own rows into its region rows [128, 128 + q_rows) and its lane's ring pieces into its lane's
# slot ring; the send kernel sends its last 128 rows to the next chip of its lane (LANE_SEND), a lane's first chip
# copies its own copy of the slot's carry rows (the replicated ``rtail``) into its region rows [0, 128) (LANE_LOCAL),
# every other chip waits for its neighbour's rows.


_SENDS: dict = {}  # (id(mesh), lanes, q_rows) -> LaneSend


def _has_lane_kv() -> bool:
    """The sdpa fork's kv_lanes option is built (ttnn.bringup.sparse_sdpa's binding names it)."""
    return "kv_lanes" in (getattr(ttnn.bringup.sparse_sdpa, "__doc__", "") or "")


LANE_KV = [_has_lane_kv()]  # (a list: tests may switch it off for the copy path)


def _has_map_runtime() -> bool:
    """The sdpa fork's kv_lane_map_runtime option is built (PB.9G)."""
    return "kv_lane_map_runtime" in (getattr(ttnn.bringup.sparse_sdpa, "__doc__", "") or "")


def sdpa_lanes(
    att, q_rm, bufs: list, lane_of_chip: list, index, window, lane_rows=None, map_runtime: bool = False
) -> ttnn.Tensor:
    """TtSparseAttention._sdpa's ROW_BFP4 + kv_window call (HISTORY_FORMAT=bfp4) with the fork's kv_lanes (PB.7B
    ``lane_rows``: the fork's kv_lane_rows, each query row its own lane; PB.9G ``map_runtime``: the fork's
    kv_lane_map_runtime, the lane map a runtime value: chip-aligned lanes, one segment a chip)."""
    from .attention import SDPA_K_CHUNK
    from .cores import max_grid

    assert bufs[0].dtype == ttnn.uint8, "PACKED_ATTN_ONECALL=on: HISTORY_FORMAT=bfp4 (ROW_BFP4 records)"
    return ttnn.bringup.sparse_sdpa(
        q_rm,
        bufs[0],
        index,
        att.d,
        kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP4,
        scale=1.0,
        k_chunk_size=SDPA_K_CHUNK,
        compute_kernel_config=att.ckc,
        attention_sink=att.sink,
        kv_window=window,
        max_grid=max_grid(),
        kv_lanes=list(bufs),
        kv_lane_of_chip=[int(v) for v in lane_of_chip],
        **({} if lane_rows is None else {"kv_lane_rows": [int(v) for v in lane_rows]}),
        **({"kv_lane_map_runtime": True} if map_runtime else {}),
    )


def _has_one_ring() -> bool:
    """The indexer fork's one-call ring (ring_indexer_score_dsa lane_rows / k_extra) is built."""
    return "lane_rows" in (getattr(ttnn.bringup.ring_indexer_score_dsa, "__doc__", "") or "")


ONE_RING = [_has_one_ring()]  # (a list: tests may switch it off for the per-segment rings)
_KBUFS: dict = {}  # (id(mesh), capacity, j) -> (mesh, lane j's gathered-key buffer, the indexer's kbuf's spec)


def _own_extent() -> dict:
    """PACKED_LANE_BALANCE=on (PB.3M, call-time): the lanes ring gathers each lane buffer to its own lane's extent
    (ring_indexer_score_dsa lane_own_extent), not every buffer to the longest lane's; off: the fork's default."""
    return {"lane_own_extent": True} if settings.get("PACKED_LANE_BALANCE") == "on" else {}


def _kbuf(mesh, base: ttnn.Tensor, j: int) -> ttnn.Tensor:
    """Lane buffer j >= 1's ring output of the one-call indexer (the indexer's own ``_kbuf`` is j 0)."""
    key = (id(mesh), tuple(base.shape), base.dtype, j)
    hit = _KBUFS.get(key)
    if hit is not None and hit[0] is mesh and hit[1].is_allocated():
        return hit[1]
    t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(base.shape)), base.dtype, base.layout, mesh, DRAM)
    ccl.pin(mesh, t)
    _KBUFS[key] = (mesh, t)
    return t


def _out(att, o: ttnn.Tensor, cos, sin) -> ttnn.Tensor:
    """The attention phase's last step: the sdpa output (consumed) -> attn_out."""
    out = att._out(o, cos, sin)
    dealloc(o)
    return out


# ---------------------------------------------------------------------------------------------- table buffers
_TABLES: dict = {}  # (id(mesh), shape, dtype) -> (mesh, tensor): one table buffer a history size (every source)
_TOPS: dict = {}  # (id(mesh), shape) -> (mesh, tensor): the combined top-k of a packed chunk


# ---------------------------------------------------------------------------------------------- the chunk
STATS = {"chunks": 0}  # packed chunks run through OneCall since load (diagnostic)
_REFUSED: set = set()  # (layer, reason) of the per-segment served-write fallbacks logged (PB.3K, diagnostic)


class OneCall:
    """The one-call attention of a packed chunk (tt/packed_device.PackedChunk.oc)."""

    def __init__(self, pk):
        raise TypeError(
            "OneCall is the one-call path's base (PB.9G: tt/packed_glued.GluedCall runs every packed chunk)"
        )

    # ------------------------------------------------------------------ helpers
    def close(self) -> None:
        if self.hide is not None:
            self.hide.close()
        for key in list(self.memo):
            self._drop(key)

    def _drop(self, key) -> None:
        v = self.memo.pop(key, None)
        if key[0] in ("index", "top", "cands", "srv_all") and v is not None and v.is_allocated():
            dealloc(v)

    # ------------------------------------------------------------------ the layer
    def attention(self, block, x: ttnn.Tensor) -> ttnn.Tensor:
        """The layer's attention output [1, 1, q_rows, H] for every row of the packed chunk (x: attn_norm, the chips'
        own rows; not consumed)."""
        m, i = self.model, block.layer
        if self.hide is not None and block.indexer is not None and self.hide.tails_here(block):
            # PB.3H: at a KV source layer the compressor first (it reads x only), then the segments' history tails
            # (its fresh rows) gathered beside q_a, kv and the indexer's gates and query
            profiler.signpost("compressor")
            self._compress(block, x)
            ix, got = block.indexer, {}

            def q_a(_):
                got["qr"], p = block.q_a.with_proj(x)
                got["kv"] = block.kv.from_proj(p, self.starts)
                dealloc(p)

            def gates(_):
                ix.qr_twin = block.q_a.twin_out if block.q_a.twin else None
                return ix.gates(x)

            pre = self.hide.tails(block, [q_a, gates, lambda w: self._query(ix, got["qr"], w)])
            qr, kv = got["qr"], got["kv"]
            profiler.signpost("indexer")
            self._index(block, x, qr, pre=pre)
        else:
            qr, p = block.q_a.with_proj(x)
            profiler.signpost("kv")  # (the step names: the profile's phases, as the single-request path's)
            kv = block.kv.from_proj(p, self.starts)
            dealloc(p)
            if block.compressor is not None:
                profiler.signpost("compressor")
                self._compress(block, x)
            if block.indexer is not None:
                profiler.signpost("indexer")
                self._index(block, x, qr)
        if block.cfg.ratio(i):
            out = self._sparse(block, qr, kv)
        else:
            out = self._window(block, qr, kv)
        dealloc(kv)
        dealloc(qr)
        for r in self.pk.segs:  # (INDEXER_GLUE=on: as TtDS41Block.__call__'s end, every slot state)
            st = r.state
            if st.memo is not None:
                for slot in m.memo_last.get(i, ()):
                    st.memo.drop(slot)
        for slot in m.memo_last.get(i, ()):
            for key in [k for k in self.memo if k[0] == "index" and k[1] == slot[1] and slot[0] == "index"]:
                self._drop(key)
        return out

    def compress_only(self, block, x: ttnn.Tensor) -> None:
        """DECODER_SKIP (kv_only, layer 20): the compressor once, every segment's served writes."""
        self._compress(block, x)

    # ------------------------------------------------------------------ compressor

    # ------------------------------------------------------------------ indexer
    def _query(self, ix, qr: ttnn.Tensor, w):
        """The indexer's query (the chips' own rows at their lanes' positions), rounded as INDEX_Q_FORMAT says ->
        (w, q, q_kept)."""
        q = ix.query(qr, self.starts)
        q_kept = False
        if ix.q_bfp4 and q.dtype == ttnn.bfloat4_b:  # ROPE_CHAINS=on: the rope wrote it rounded
            q_kept = ix._q_kept
        elif ix.q_bfp4:
            q4, q_kept = ix._round(q, "query")
            dealloc(q)
            q = q4
        return w, q, q_kept

    def _keep(self, key, t: ttnn.Tensor) -> None:
        old = self.memo.pop(key, None)
        if old is not None and old.is_allocated():
            dealloc(old)
        self.memo[key] = t

    # ------------------------------------------------------------------ attention
    def _phase(self, i: int, *steps):
        """The attention phase's steps (wq_b + rope, sparse_sdpa, rope + wo_a + wo_b), each fn(previous result);
        PB.3H: a region beside the pending history gathers."""
        if self.hide is not None:
            return self.hide.attention(i, list(steps))
        out = None
        for step in steps:
            out = step(out)
        return out
