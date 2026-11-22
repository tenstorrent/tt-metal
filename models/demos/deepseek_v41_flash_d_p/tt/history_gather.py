# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""STATIC_TABLES=off, HISTORY_GATHER=rows (P.74, owner 2026-11-18): the KV source's compressed history gathered over
the mesh every chunk into a persistent buffer per chip, allocated with the served compressed caches (load / state
build) for their whole context and reused by every chunk, user (slot) and KV source.

The gather is ONE stock ttnn.experimental.high_bw_all_gather straight from the source's served cache (the engine's
ND-sharded [slots x layers, 1, rows, 512] tensor, tt/served.py): ``input_batch_index`` picks the (slot, layer) slab,
``gathered_dim_size`` = owners x L moves only each chip's first L local rows (the chunk periods a query of this chunk
can reach), and every owner q lands at its FIXED slot [q R, q R + L) of the output, R = the slab's local rows
(rows past L keep old bytes; no index ever points there). A TP-dedup cache (every chip its own stripes) is gathered
over the whole mesh (cluster_axis=None, row-major rank q = r TP + c: the compressed caches' chip-linear owner order;
the Galaxy's FABRIC_2D_TORUS_XY: two Hamiltonian cycles, every chip's four links); a TP-replicated one over SP.
No slice copy of the slab, no TP / SP intermediate, no allocation per chunk (P.73's ``rows_axes``: slice -> TP gather
-> SP gather, each a new tensor; ``tiles``: + tilize / untilize).

Table row of group j (tt/sparse_attention._topk_rows with stride R instead of nb hs): q R + b hs + j mod hs. The same
bf16 rows as every other path (a data move): sparse_sdpa reads the same keys per query.

The buffers are [1, 1, owners R, 512] bf16 ROW_MAJOR DRAM interleaved: ONE per distinct table size (owners x slab
rows), shared by every chunk, user and KV source of that size, held by each KV source's compressed cache
(tt/compressor.TtServedRows.hold_history) whose history some attention gathers in the model's DECODER_SKIP mode
(TtDS41Model.history_sources: under decode not layer 20's ratio-1 source, whose readers are all decoder layers) and
freed with the last of them. The owner asked for one buffer; the stock op's output
must be exactly the worst-case shape of its input (owners x the slab's rows), and ttnn has no Python view of a buffer
prefix of another volume (ttnn.view / experimental.view check the volume; a MeshBuffer view is C++ only), so a model
whose sources have two ratios (1 and 2: R and R / 2 rows) holds two: 1.5x the largest source's table. In serving
every slot's slab has the deployment's rows (the engine's tensors), so that is two buffers for every user.

One buffer serves every source of its size because a source's readers are the layers up to the next source (reference
Config.kv_source_of: the latest source at or before a layer) and the layers run in order: no reader of source A runs
after source B gathered. ``gather`` records which (source cache, start, version) the buffer holds and ``holds`` lets
the memo check it before reusing an entry (a stale hit asserts instead of reading another source's rows).

HISTORY_GATHER_HIDE=on (P.75): the history [0, c S) of a chunk at start (c = start // S: every owner's first c hs
local rows, complete before the chunk) does not depend on the chunk. ``prefetch`` gathers it at the first engram
layer's wait for its host rows (tt/model.TtDS41Model.prefetch_history; the device idles there, P.73), each source into
a buffer of its own (key (rows, source layer): the three sources' histories live at once; shared by every user).
At the source's layer ``gather`` then moves only the chunk's own periods [c hs, nb hs) (1 or 2 an owner): sliced
from the slab, gathered into a small persistent buffer [owners m, 512] (key ("tail", owners m)) and slice_written
(viewed [1, owners, m, 512] -> [1, owners, R, 512]) into rows [q R + c hs, q R + nb hs): the same bytes at the same
rows as one gather of [0, nb hs) (a data move), so the table, the index and sparse_sdpa are bit-identical. A chunk
whose prefetch did not run (no engram layer loaded, a re-run, a cache written since) gathers everything as before;
every chunk start clears the markers (``new_chunk``, FIX.39: an outside write never bumps ``version``).

HISTORY_FORMAT=bfp4 (P.77, OWNER DECISION 2026-11-18: the 4-bit compressed-KV path, as the reference's fp4 history):
the buffers hold 320 B bfloat4_b row records ([1, 1, owners R, 320] uint8, the layout of tt/compressor.AttnTable's
bfp4_rows table) instead of bf16 rows, read by the sdpa fork's ROW_BFP4 + kv_window. Every gather first packs the
chip's OWN served bf16 rows (``_pack``: slice of the slab -> tilize -> ttnn.bringup.ckv_bfp4, round to nearest even,
prepared) into a local record buffer [1, 1, R, 320] (one per slab size, shared by every source: the sources' gathers
run in order on one queue), and the gather moves the records (P.11's pack-before-gather: 320 instead of 1024 B a row
over the fabric). The records are a function of each served row alone (a row's 32 shared exponents are its own), so a
prefetch + tail gives the bytes of a full gather, and an outside-written slot gives the bytes of a self-written one
(derived from the served rows every chunk; no private copy). The served cache stays bf16 (decode reads it).
"""

from __future__ import annotations

import ttnn

from . import ccl
from .ccl import dealloc
from .common import DRAM, SP_AXIS, TP_AXIS
from .cores import max_grid, sub_cores, to_tile
from .settings import settings

WIDTH = 512  # a compressed row (bf16)
REC = 320  # P.77: a bfloat4_b row record (ttnn.bringup.ckv_bfp4; tt/compressor.BFP4_ROW_BYTES)
PACK_BUCKET = 2048  # P.77: a whole-history pack covers [0, L) rounded up to this (few prepared programs, ~R / 2048)


def bfp4_on() -> bool:
    """HISTORY_FORMAT=bfp4 (P.77): the gathered history as 4-bit row records (read at a buffer's allocation)."""
    return settings.get("HISTORY_FORMAT") == "bfp4"


class HistoryBuffer:
    """One persistent gather output of one mesh: [1, 1, rows, WIDTH] bf16 ROW_MAJOR DRAM interleaved (``bfp4``, P.77:
    [1, 1, rows, REC] uint8 row records); ``users``: the caches that acquired it."""

    def __init__(self, mesh, rows: int, owners: int | None = None, bfp4: bool = False):
        self.mesh, self.rows, self.users = mesh, int(rows), 0
        self.bfp4 = bool(bfp4)
        self.width = REC if self.bfp4 else WIDTH
        self.tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.rows, self.width]),
            ttnn.uint8 if self.bfp4 else ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh,
            DRAM,
        )
        self.owner = None  # (id(cache), start, version) of the rows it holds
        self.pre = None  # P.75: (id(cache), start, version, h) of a prefetched history (rows [q R, q R + h))
        self.view = None  # P.75: [1, owners, rows / owners, WIDTH], a view of ``tensor`` (the slice_write's operand)
        self.ready = None  # PB.3H: (id(cache), start, version, rows) of a gather completed ahead of its ``gather`` call
        # PB.HG (PACKED_HIST_OWNED=on): id(cache) -> (kind "pre" | "ready", start, version, rows) of every segment's
        # cache whose rows this buffer holds on that segment's chips (one buffer a source for every segment)
        self.own: dict = {}
        if owners is not None:
            assert self.rows % owners == 0, (self.rows, owners)
            self.view = ttnn.reshape(self.tensor, [1, owners, self.rows // owners, self.width])
            assert self.view.buffer_address() == self.tensor.buffer_address(), "P.75: the 4-D view must not copy"

    def free(self) -> None:
        if self.tensor.is_allocated():
            ttnn.deallocate(self.tensor)


class Held:
    """A gathered history that lives in a persistent buffer (tt/sparse_attention.ChunkMemo and the callers never
    free it: it is not a ttnn.Tensor)."""

    __slots__ = ("tensor",)

    def __init__(self, tensor: ttnn.Tensor):
        self.tensor = tensor


_BUF: dict = {}  # id(mesh) -> (mesh, {rows: HistoryBuffer})
LATE: list = []  # (id(cache), start) of gathers whose cache held no buffer yet (acquired then; diagnostic)


def owners_and_axis(mesh, dedup: bool):
    """(chips holding distinct rows, the gather's cluster_axis): the whole mesh (None) for a TP-dedup cache on a
    split TP axis, else SP."""
    sp, tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
    if dedup and tp > 1:
        return sp * tp, None
    assert sp > 1, "HISTORY_GATHER=rows: one chip holds the whole history (no gather)"
    return sp, SP_AXIS


def acquire(mesh, rows: int, key=None, owners: int | None = None, tails: tuple = (), bfp4: bool | None = None) -> None:
    """A KV source's compressed cache at build (load / state): the buffer of ``rows`` (made by the first, shared by
    the rest; ``key``: P.75's per-source key (rows, layer), default the size) and the gather's semaphores. ``tails``
    (P.75): the chunk-period row counts a source's tail gather moves an owner; their small buffers (owners x m rows)
    are shared by every source. HISTORY_FORMAT=bfp4 (P.77): record buffers, plus the local record buffer of the
    slab size R = rows / owners (key ("loc", R), [1, 1, R, REC]: each chip packs its own rows there)."""
    hit = _BUF.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        hit = _BUF[id(mesh)] = (mesh, {})
    key = int(rows) if key is None else key
    bfp4 = bfp4_on() if bfp4 is None else bfp4
    for k, n, own, b4 in _wanted(key, rows, owners, tails, bfp4):
        b = hit[1].get(k)
        if b is None or not b.tensor.is_allocated():
            b = hit[1][k] = HistoryBuffer(mesh, n, own, b4)
        assert b.bfp4 == b4, ("P.77: a history buffer of the other HISTORY_FORMAT is still held", k, b.bfp4)
        b.users += 1
    for axis in (None, SP_AXIS):  # (one semaphore pair a ring geometry, DS41-216; made at load, agent rule 5)
        if axis is None or mesh.shape[axis] > 1:
            ccl._sems(mesh, axis)


def _wanted(key, rows: int, owners: int | None, tails: tuple, bfp4: bool) -> list:
    """(key, rows, owners of the 4-D view or None, bfp4) of every buffer a cache holds."""
    want = [(key, int(rows), owners if tails else None, bfp4)]
    want += [(("tail", owners * m) + (("bfp4",) if bfp4 else ()), owners * m, owners, bfp4) for m in tails]
    if bfp4:
        assert owners is not None and int(rows) % owners == 0, ("P.77: the local record buffer's size", rows, owners)
        want.append((("loc", int(rows) // owners), int(rows) // owners, None, True))
    return want


def release(mesh, rows: int, key=None, owners: int | None = None, tails: tuple = (), bfp4: bool | None = None) -> None:
    """The cache freed: the buffer (and P.75's tail buffers, P.77's local record buffer) go with their last user.
    ``bfp4``: the format it was acquired with (default the current setting)."""
    hit = _BUF.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        return
    key = int(rows) if key is None else key
    bfp4 = bfp4_on() if bfp4 is None else bfp4
    for k, _, _, _ in _wanted(key, rows, owners, tails, bfp4):
        b = hit[1].get(k)
        if b is None:
            continue
        b.users -= 1
        if b.users <= 0:
            b.free()
            del hit[1][k]


def _buffer(mesh, key) -> HistoryBuffer | None:
    hit = _BUF.get(id(mesh))
    return hit[1].get(key) if hit is not None and hit[0] is mesh else None


def _key(ckv_cache):
    """The buffer key of a cache (P.75: its own (rows, layer) when hidden; else its size)."""
    k = getattr(ckv_cache, "history_key", None)
    return k if k is not None else ckv_cache.owners * ckv_cache.slab.rows


STATS = {"prefetch": 0, "tail": 0, "full": 0, "ready": 0}  # P.75: gathers by kind since load (diagnostic)
# PB.3H (diagnostic, tests/packed/test_packed_walls.py): when a list, every history gather / prep op appends
# (kind, source layer, lane, first device-operation id, end id): the profile tells its programs from the rest
TRACE: list | None = None


def _lane_src(ckv_cache) -> tuple:
    """(source layer, lane) of a cache's buffer key (P.75 (rows, layer); PB.3 lanes ((rows, layer), "lane", k))."""
    k = getattr(ckv_cache, "history_key", None)
    if isinstance(k, tuple) and len(k) == 3 and k[1] == "lane":
        return (k[0][1] if isinstance(k[0], tuple) else -1), int(k[2])
    return (k[1] if isinstance(k, tuple) else -1), 0


class _Traced:
    """``with _Traced(kind, cache):`` records the device-operation ids the block launched into TRACE (when on);
    ``lane``: the lane to book it to (default the cache's)."""

    __slots__ = ("kind", "cache", "a", "lane")

    def __init__(self, kind: str, ckv_cache, lane: int | None = None):
        self.kind, self.cache, self.lane = kind, ckv_cache, lane

    def __enter__(self):
        self.a = ttnn._ttnn.get_device_operation_id() if TRACE is not None else None
        return self

    def __exit__(self, *exc):
        if self.a is not None and TRACE is not None:
            src, lane = _lane_src(self.cache)
            lane = lane if self.lane is None else self.lane
            TRACE.append((self.kind, src, lane, self.a, ttnn._ttnn.get_device_operation_id()))
        return False


def gather(ckv_cache, local_rows: int, start: int) -> tuple[ttnn.Tensor, int]:
    """The first ``local_rows`` compressed rows of every owner of ``ckv_cache`` (tt/compressor.TtServedRows) ->
    the persistent buffer of its size (owner q at row q R), returns (the buffer [1, 1, owners R, WIDTH], R)."""
    slab = ckv_cache.slab
    mesh = ckv_cache.mesh
    owners, axis = owners_and_axis(mesh, ckv_cache.tp_axis is not None)
    assert owners == ckv_cache.owners, (owners, ckv_cache.owners)
    R = slab.rows
    assert slab.width == WIDTH and slab.tensor.dtype == ttnn.bfloat16, (slab.width, slab.tensor.dtype)
    assert R % 32 == 0, ("the slab's rows pack in whole tiles (P.77) and gather as before", R)
    assert 0 < local_rows <= R, (local_rows, R)
    buf = _buffer(mesh, _key(ckv_cache))
    assert buf is not None, (
        f"HISTORY_GATHER=rows: no history buffer {_key(ckv_cache)} (acquired by tt/compressor.TtServedRows; "
        f"held: {list((_BUF.get(id(mesh)) or (None, {}))[1])})"
    )
    if owned_active(ckv_cache):  # PB.HG: this segment's rows on its own chips only
        _owned_complete(ckv_cache, buf, local_rows, start)
        return buf.tensor, R
    ready, buf.ready = buf.ready, None
    if ready is not None and ready == (id(ckv_cache), start, ckv_cache.version, local_rows):
        buf.pre = None  # PB.3H: completed ahead (tail_finish: the same rows as the tail path below)
        buf.owner = (id(ckv_cache), start, ckv_cache.version)
        STATS["ready"] += 1
        return buf.tensor, R
    pre, buf.pre = buf.pre, None
    if (
        pre is not None
        and pre[:2] == (id(ckv_cache), start)
        and 0 <= ckv_cache.version - pre[2] <= 1  # (the chunk's own write only: rows >= start, past the history)
        and pre[3] < local_rows
    ):  # P.75: the history is there; only the chunk's own periods move
        _tail(ckv_cache, buf, pre[3], local_rows, owners, axis)
        STATS["tail"] += 1
    else:
        _full(ckv_cache, buf, local_rows, owners, axis)
        STATS["full"] += 1
    buf.owner = (id(ckv_cache), start, ckv_cache.version)
    return buf.tensor, R  # (the kept tensor, not the op's new wrapper: DS41-149)


def _full(ckv_cache, buf: HistoryBuffer, local_rows: int, owners: int, axis) -> None:
    """Every owner's first ``local_rows`` local slab rows -> its fixed slot of ``buf`` (bfp4, P.77: packed into the
    chip's local record buffer first, the records gathered)."""
    src = None
    if buf.bfp4:
        with _Traced("pack", ckv_cache):
            src = _pack(ckv_cache, 0, local_rows)
    _gather_full(ckv_cache, buf, src, local_rows, owners, axis)


def _gather_full(ckv_cache, buf: HistoryBuffer, src, local_rows: int, owners: int, axis, sub: dict | None = None):
    """The gather of ``_full``: ``src`` the packed records (bfp4) or None (the slab's bf16 rows). ``sub`` (PB.3H):
    high_bw_all_gather's subdevice_id / sub_core_grids (the gather sub-device of tt/gather_hide.Split)."""
    slab = ckv_cache.slab
    s0, s1 = ccl._sems(buf.mesh, axis)
    kw = dict(sub or {})
    with _Traced("gather", ckv_cache):
        if src is not None:
            ttnn.experimental.high_bw_all_gather(
                src,
                2,
                buf.tensor,
                cluster_axis=axis,
                num_links=ccl.AG_LINKS,
                gathered_dim_size=owners * local_rows,
                ready_semaphore=s0,
                data_valid_semaphore=s1,
                **kw,
            )
            return
        ttnn.experimental.high_bw_all_gather(
            slab.tensor,
            2,
            buf.tensor,
            cluster_axis=axis,
            num_links=ccl.AG_LINKS,
            input_batch_index=slab.batch,
            gathered_dim_size=owners * local_rows,
            ready_semaphore=s0,
            data_valid_semaphore=s1,
            **kw,
        )


def _pack(ckv_cache, lo: int, hi: int, loc: HistoryBuffer | None = None) -> ttnn.Tensor:
    """P.77 (HISTORY_FORMAT=bfp4): the chip's own served bf16 rows [lo, hi) of ``ckv_cache``'s slab -> 320 B
    bfloat4_b records at the same rows of the local record buffer (returned; persistent, never freed by the caller).
    Packs whole tiles: [lo // 32 x 32, ...) up to hi rounded to 32 rows, or, for a whole history (lo 0), to
    PACK_BUCKET rows (the prepared program is one per row count: a growing history would build one a chunk). The
    extra rows past hi are later slab rows (no index reads their records; a later pack overwrites them)."""
    slab = ckv_cache.slab
    R = slab.rows
    if loc is None:  # (PB.3H: a caller's own record buffer of the slab's rows, else the shared one)
        loc = _buffer(ckv_cache.mesh, ("loc", R))
    assert loc is not None and loc.bfp4 and loc.rows == R, ("P.77: no local record buffer", R)
    a = lo // 32 * 32
    b = min(-(-hi // (PACK_BUCKET if a == 0 else 32)) * (PACK_BUCKET if a == 0 else 32), R)
    assert 0 <= a < b <= R and (b - a) % 32 == 0, (lo, hi, a, b, R)
    from .page_copy import dyn

    if dyn("hist"):  # MB.4: one call, the rows runtime values (no program a history length): see _pack_rows
        return _pack_rows(ckv_cache, loc, a, -(-hi // 32) * 32)
    rows = slab.slice_rows(a, b)  # [1, 1, b - a, 512] bf16 ROW_MAJOR DRAM
    tl = to_tile(rows, memory_config=DRAM)
    if tl is not rows:
        dealloc(rows)
    # compact (no per-core args, row0 frozen) where row0 is always 0; the tail's row0 moves: a value slot (prepared)
    ttnn.bringup.ckv_bfp4(tl, loc.tensor, a, prepared=True, compact=a == 0, max_grid=max_grid())
    dealloc(tl)
    # (host metadata only: the full-mesh gather checks its input's dim-2 sharding topology, DS41-291)
    loc.tensor.update_tensor_topology(slab.tensor.tensor_topology())
    return loc.tensor


def _pack_rows(ckv_cache, loc: HistoryBuffer, a: int, b: int) -> ttnn.Tensor:
    """POS_DYNAMIC=on / PACKED_GLUED_DYNAMIC=on (MB.4; dyn("hist"): every single-request and every packed segment's
    pack, prefetch and tail): ``_pack``'s rows [a, b) in ONE fork call, ttnn.bringup.ckv_bfp4_rows: each core reads its
    bands' slab rows straight from the served slab (page slab.batch R + row), tilizes them on the core and packs their
    records at rows [a, b) of the local record buffer; the source row, the record row and the band count are per-run
    values, so one program serves every history length, slot and layer (rule 20), with no slice, no tilize op and no
    per-piece dispatches (MB.2's PACK_BUCKET pieces: a page copy on 4 cores + tilize + ckv_bfp4 a 2048 rows, which grew
    with the history: +2 ms device, +5.5 ms host a chunk at 1M, MB.4). The records are a function of each row alone:
    the same bytes as the off path's slice + tilize + ckv_bfp4."""
    slab = ckv_cache.slab
    ttnn.bringup.ckv_bfp4_rows(slab.tensor, loc.tensor, slab.batch * slab.rows + a, b - a, a, max_grid=max_grid())
    loc.tensor.update_tensor_topology(slab.tensor.tensor_topology())
    return loc.tensor


def _tail(ckv_cache, buf: HistoryBuffer, h: int, local_rows: int, owners: int, axis) -> None:
    """P.75: every owner's local rows [h, local_rows) -> rows [q R + h, q R + local_rows) of ``buf`` (its rows below h
    prefetched): a slice of the slab, one gather into the small buffer of that size, one slice_write.
    POS_DYNAMIC=on (MB.2): always the largest tail (2 hs rows: h moved down to local_rows - 2 hs; rows the prefetch
    gathered already, the same bytes again), as page copies: one shape for every chunk."""
    from .page_copy import dyn

    tails = tuple(getattr(ckv_cache, "_history_tails", ()) or ())
    if dyn("hist") and tails and local_rows - h < max(tails) <= local_rows:
        h = local_rows - max(tails)
    m = local_rows - h
    small = _buffer(buf.mesh, ("tail", owners * m) + (("bfp4",) if buf.bfp4 else ()))
    assert small is not None and buf.view is not None, ("P.75: no tail buffer", owners, m)
    with _Traced("pack", ckv_cache):
        part = _tail_part(ckv_cache, buf, h, local_rows)
    _tail_gather(ckv_cache, buf, part, small, axis)
    with _Traced("write", ckv_cache):
        ttnn.bringup.slice_write(
            small.view, buf.view, [0, 0, h, 0], [1, owners, local_rows, buf.width], [1, 1, 1, 1], max_grid=max_grid()
        )


def _tail_part(ckv_cache, buf: HistoryBuffer, h: int, local_rows: int, loc: HistoryBuffer | None = None):
    """The tail's input: the chip's own rows [h, local_rows) (bfp4: packed into ``loc`` / the shared record buffer
    first, then sliced) as a full-mesh gather input (the slab's topology)."""
    from .page_copy import dyn, pages_of

    if buf.bfp4:  # P.77: the chunk's own rows packed on their chip first, their records gathered
        rec = _pack(ckv_cache, h, local_rows, loc)
        if dyn("hist"):  # PB.7C / MB.2: a page copy (the slice is keyed on h)
            part = pages_of(rec, h, local_rows - h, [1, 1, local_rows - h, REC])
        else:
            part = ttnn.slice(
                rec, [0, 0, h, 0], [1, 1, local_rows, REC], memory_config=DRAM, sub_core_grids=sub_cores(rec)
            )
    elif dyn("hist"):
        slab = ckv_cache.slab
        part = pages_of(slab.tensor, slab.batch * slab.rows + h, local_rows - h, [1, 1, local_rows - h, slab.width])
    else:
        part = ckv_cache.slab.slice_rows(h, local_rows)  # [1, 1, m, 512] ROW_MAJOR DRAM interleaved
    # (host metadata only: the slice's own topology is not the slab's dim-2 sharding the full-mesh gather checks)
    part.update_tensor_topology(ckv_cache.slab.tensor.tensor_topology())
    return part


def _tail_gather(ckv_cache, buf: HistoryBuffer, part, small: HistoryBuffer, axis, sub: dict | None = None) -> None:
    s0, s1 = ccl._sems(buf.mesh, axis)
    with _Traced("tail", ckv_cache):
        ttnn.experimental.high_bw_all_gather(
            part,
            2,
            small.tensor,
            cluster_axis=axis,
            num_links=ccl.AG_LINKS,
            ready_semaphore=s0,
            data_valid_semaphore=s1,
            **dict(sub or {}),
        )
    dealloc(part)


def new_chunk() -> None:
    """FIX.39 (owner review 2026-11-18): at every chunk start (TtDS41Model.begin_chunk) no buffer holds a prefetched
    history: ``gather`` reuses one only when this chunk's ``prefetch`` made it. ``version`` counts this process's
    writes only; a write from outside (the KV Manager filling a served cache after a store load / d2p pull) never bumps
    it, so a marker left from an earlier chunk (a pipeline stage that starts mid-model: no prefetch before its sources)
    could pass the (cache, start, version) check over rows gathered before that write. Host only."""
    for _, bufs in _BUF.values():
        for b in bufs.values():
            b.pre = None
            b.ready = None
            b.own.clear()
    owned_end()


def prefetch(ckv_cache, start: int, chunk: int) -> bool:
    """P.75 (HISTORY_GATHER_HIDE=on): every owner's history rows [0, c hs) (c = start // chunk: the chunk periods
    complete before this chunk) -> the source's own buffer, before the chunk's first engram wait; ``gather`` at the
    source's layer then moves only the rest. False: nothing to prefetch (c = 0, a cache not hidden)."""
    if getattr(ckv_cache, "history_key", None) is None or not isinstance(ckv_cache.history_key, tuple):
        return False
    owners, axis = owners_and_axis(ckv_cache.mesh, ckv_cache.tp_axis is not None)
    buf = _buffer(ckv_cache.mesh, ckv_cache.history_key)
    assert buf is not None, ("P.75: the source's buffer", ckv_cache.history_key)
    if owned_active(ckv_cache):  # PB.HG without the gather sub-device: this segment's history to its chips, serially
        return _owned_prefetch_one(ckv_cache, buf, int(start), int(chunk))
    buf.pre = None
    h = (int(start) // int(chunk)) * (ckv_cache.stripe // ckv_cache.ratio)
    if h == 0:
        return False
    assert h <= ckv_cache.slab.rows, (h, ckv_cache.slab.rows)
    _full(ckv_cache, buf, h, owners, axis)
    buf.owner = None  # (rows [0, h) only until ``gather`` completes it)
    buf.pre = (id(ckv_cache), int(start), ckv_cache.version, h)
    STATS["prefetch"] += 1
    return True


# ---------------------------------------------------------------------------------------------- PB.3H split steps
def prefetch_prepare(ckv_cache, start: int, chunk: int, loc: HistoryBuffer | None = None):
    """PACKED_GATHER_HIDE=on (PB.3H, tt/gather_hide.py): ``prefetch`` up to its gather (the bfp4 pack into ``loc``, a
    record buffer of the caller's, so several prefetches can be packed before their gathers run). Returns the plan
    ``prefetch_gather`` takes, or None (nothing to prefetch: c = 0, a cache not hidden)."""
    if getattr(ckv_cache, "history_key", None) is None or not isinstance(ckv_cache.history_key, tuple):
        return None
    owners, axis = owners_and_axis(ckv_cache.mesh, ckv_cache.tp_axis is not None)
    buf = _buffer(ckv_cache.mesh, ckv_cache.history_key)
    assert buf is not None, ("P.75: the source's buffer", ckv_cache.history_key)
    buf.pre = buf.ready = None
    h = (int(start) // int(chunk)) * (ckv_cache.stripe // ckv_cache.ratio)
    if h == 0:
        return None
    assert h <= ckv_cache.slab.rows, (h, ckv_cache.slab.rows)
    src = None
    if buf.bfp4:
        with _Traced("pack", ckv_cache):
            src = _pack(ckv_cache, 0, h, loc)
    return (ckv_cache, buf, src, h, owners, axis, int(start))


def prefetch_gather(plan, sub: dict | None = None) -> None:
    """The gather of a ``prefetch_prepare`` plan (``sub``: the gather sub-device's high_bw_all_gather arguments) and
    the prefetch marker ``gather`` then takes the tail path from (as ``prefetch``)."""
    ckv_cache, buf, src, h, owners, axis, start = plan
    _gather_full(ckv_cache, buf, src, h, owners, axis, sub)
    buf.owner = None  # (rows [0, h) only until ``gather`` completes it)
    buf.pre = (id(ckv_cache), start, ckv_cache.version, h)
    STATS["prefetch"] += 1


def prefetch_mark(plan) -> None:
    """PACKED_LONG_HIDE=on (PB.6L): the prefetch marker of a ``prefetch_prepare`` plan set at its pack (host only),
    before its pieces are gathered: the tail (``tail_prepare`` / ``gather``) then moves only the chunk's own periods,
    rows the pieces never write. Every piece is gathered before the source's reader (gather_hide.Hide.flush at the
    table, after the regions' joins)."""
    ckv_cache, buf, src, h, owners, axis, start = plan
    buf.owner = None  # (rows [0, h) only until ``gather`` completes it)
    buf.pre = (id(ckv_cache), start, ckv_cache.version, h)
    STATS["prefetch"] += 1


def prefetch_gather_range(plan, a: int, b: int, sub: dict | None = None) -> None:
    """PACKED_LONG_HIDE=on (PB.6L): rows [a, b) of every owner of a ``prefetch_prepare`` plan's packed records into
    the same rows of each owner's fixed slot of the source's buffer (ttnn.bringup.high_bw_all_gather
    gathered_dim_start; a = 0: the prefix, as ``prefetch_gather``). A bf16 plan (no records) moves its whole history
    at once. ``sub``: the gather sub-device's arguments."""
    ckv_cache, buf, src, h, owners, axis, start = plan
    assert 0 <= a < b <= h, (a, b, h)
    if src is None:
        assert (a, b) == (0, h), "PB.6L: pieces of a bf16 history (the slab is ND-sharded: no range gather)"
        _gather_full(ckv_cache, buf, src, h, owners, axis, sub)
        return
    s0, s1 = ccl._sems(buf.mesh, axis)
    with _Traced("gather", ckv_cache):
        ttnn.bringup.high_bw_all_gather(
            src,
            2,
            buf.tensor,
            cluster_axis=axis,
            num_links=ccl.AG_LINKS,
            gathered_dim_size=owners * (b - a),
            gathered_dim_start=a if a > 0 else None,
            ready_semaphore=s0,
            data_valid_semaphore=s1,
            **dict(sub or {}),
        )


def tail_prepare(ckv_cache, start: int, local_rows: int, loc: HistoryBuffer | None = None):
    """PB.3H: the tail of ``gather(ckv_cache, local_rows, start)`` ahead of its call, up to its gather (pack + slice).
    None when ``gather`` would not take the tail path (no matching prefetch: it then gathers in full itself)."""
    buf = _buffer(ckv_cache.mesh, _key(ckv_cache))
    if buf is None or buf.view is None:
        return None
    pre = buf.pre
    if not (
        pre is not None
        and pre[:2] == (id(ckv_cache), start)
        and 0 <= ckv_cache.version - pre[2] <= 1
        and pre[3] < local_rows <= ckv_cache.slab.rows
    ):
        return None
    owners, axis = owners_and_axis(ckv_cache.mesh, ckv_cache.tp_axis is not None)
    h = pre[3]
    from .page_copy import dyn

    tails = tuple(getattr(ckv_cache, "_history_tails", ()) or ())
    if dyn("hist") and tails and local_rows - h < max(tails) <= local_rows:
        # PB.9X (as _tail, MB.2): always the largest tail (h moved down; rows the prefetch gathered already, the same
        # bytes again): the tails' copy / slice_write shapes do not follow the history length (PB.9X probe: a build at
        # history 8192, tails_finish's slice_write input [1, 32, 128, 320] -> [1, 32, 256, 320])
        h = local_rows - max(tails)
    with _Traced("pack", ckv_cache):
        part = _tail_part(ckv_cache, buf, h, local_rows, loc)
    return (ckv_cache, buf, part, h, int(local_rows), owners, axis, int(start))


def tail_gather(plan, small: HistoryBuffer, sub: dict | None = None) -> None:
    """The tail plan's gather into ``small`` (the caller's [owners m, width] buffer with its 4-D view)."""
    ckv_cache, buf, part, h, local_rows, owners, axis, start = plan
    assert small.rows == owners * (local_rows - h) and small.view is not None and small.bfp4 == buf.bfp4
    _tail_gather(ckv_cache, buf, part, small, axis, sub)


def tail_finish(plan, small: HistoryBuffer) -> None:
    """The tail's slice_write into the source buffer; ``gather(ckv_cache, local_rows, start)`` then finds it ready."""
    ckv_cache, buf, part, h, local_rows, owners, axis, start = plan
    with _Traced("write", ckv_cache):
        ttnn.bringup.slice_write(
            small.view, buf.view, [0, 0, h, 0], [1, owners, local_rows, buf.width], [1, 1, 1, 1], max_grid=max_grid()
        )
    buf.pre = None
    buf.ready = (id(ckv_cache), start, ckv_cache.version, local_rows)
    STATS["tail"] += 1


def tails_max_rows(plans: list) -> int:
    """PB.7C (PACKED_GLUED_DYNAMIC=on): the combined tail gather's fixed rows an owner: every segment's tail at its
    largest (two periods of the source's stripe a segment, GLUED_SEGS segments): one gather program for every layout;
    only the plans' rows move (high_bw_all_gather's gathered_dim_size, a runtime value)."""
    from .packed import GLUED_MAX

    ckv = plans[0][0]
    return 2 * GLUED_MAX * (ckv.stripe // ckv.ratio)


def tails_input(plans: list):
    """PB.3H: several tail plans' inputs (each chip's own rows of each) concatenated on dim 2 in plan order: one
    full-mesh gather input (consumes the parts). Every plan of one gather geometry (owners, axis, record width)."""
    parts = [p[2] for p in plans]
    first = plans[0]
    assert all(p[5:7] == first[5:7] and p[1].bfp4 == first[1].bfp4 for p in plans), "tails_input: one geometry"
    from .page_copy import dyn, page_copy

    if len(parts) == 1 and not dyn("hist"):
        return parts[0]
    with _Traced("pack", first[0], -1):
        if dyn("hist"):  # PB.7C: page copies into the fixed-size input (the concat is keyed on every part's rows)
            m_max = tails_max_rows(plans)
            assert sum(int(t.shape[-2]) for t in parts) <= m_max, ([t.shape for t in parts], m_max)
            x = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, m_max, int(parts[0].shape[-1])]),
                parts[0].dtype,
                parts[0].layout,
                parts[0].device(),
                DRAM,
            )
            o = 0
            for t in parts:
                page_copy(t, x, 0, o, int(t.shape[-2]))
                o += int(t.shape[-2])
        else:
            x = ttnn.concat(parts, dim=2, memory_config=DRAM, sub_core_grids=sub_cores(parts[0]))
    for t in parts:
        dealloc(t)
    x.update_tensor_topology(first[0].slab.tensor.tensor_topology())  # (host metadata, as _tail_part)
    return x


def tails_gather(plans: list, x, small: HistoryBuffer, sub: dict | None = None) -> None:
    """``tails_input``'s rows of every owner -> ``small`` ([owners M, width], M the plans' rows together; owner q's
    block holds every plan's rows in plan order): ONE gather for every segment's tail. Consumes ``x``."""
    first = plans[0]
    M = sum(p[4] - p[3] for p in plans)
    from .page_copy import dyn

    kw = {}
    if dyn("hist"):  # PB.7C: fixed shapes; only each owner's first M rows move (a runtime extent)
        assert small.rows == first[5] * tails_max_rows(plans) and small.view is not None, (small.rows, M)
        kw = dict(gathered_dim_size=first[5] * M)
    else:
        assert small.rows == first[5] * M and small.view is not None and small.bfp4 == first[1].bfp4
    s0, s1 = ccl._sems(small.mesh, first[6])
    with _Traced("tail", first[0], -1):
        ttnn.experimental.high_bw_all_gather(
            x,
            2,
            small.tensor,
            **kw,
            cluster_axis=first[6],
            num_links=ccl.AG_LINKS,
            ready_semaphore=s0,
            data_valid_semaphore=s1,
            **dict(sub or {}),
        )
    dealloc(x)


def tails_finish(plans: list, small: HistoryBuffer) -> None:
    """Each plan's rows of ``small`` (its owner-block offset) slice_written into its source buffer (as tail_finish)."""
    o = 0
    from .page_copy import dyn, page_copy

    dy = dyn("hist")
    M = tails_max_rows(plans) if dy else sum(p[4] - p[3] for p in plans)  # (an owner's block of ``small``)
    for ckv_cache, buf, part, h, local_rows, owners, axis, start in plans:
        m = local_rows - h
        with _Traced("write", ckv_cache):
            sl = small.view
            if dy:  # PB.7C: owner q's rows [o, o + m) of its block (M_max rows), one strided copy
                sl = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, owners, m, buf.width]),
                    small.view.dtype,
                    small.view.layout,
                    small.view.device(),
                    DRAM,
                )
                page_copy(small.view, sl, o, 0, m, owners, M, m)
            elif len(plans) > 1:
                sl = ttnn.slice(
                    small.view,
                    [0, 0, o, 0],
                    [1, owners, o + m, buf.width],
                    memory_config=DRAM,
                    sub_core_grids=sub_cores(),
                )
            ttnn.bringup.slice_write(
                sl, buf.view, [0, 0, h, 0], [1, owners, local_rows, buf.width], [1, 1, 1, 1], max_grid=max_grid()
            )
            if sl is not small.view:
                dealloc(sl)
        o += m
        buf.pre = None
        buf.ready = (id(ckv_cache), start, ckv_cache.version, local_rows)
        STATS["tail"] += 1


def holds(ckv_cache, start: int) -> bool:
    """The buffer of ``ckv_cache``'s size still holds its rows of this start and write version (a memo hit is
    valid: no other source of that size gathered since)."""
    buf = _buffer(ckv_cache.mesh, _key(ckv_cache))
    if buf is not None and owned_active(ckv_cache):  # PB.HG: per segment's cache
        st = buf.own.get(id(ckv_cache))
        return st is not None and st[0] == "ready" and st[1:3] == (start, ckv_cache.version)
    return buf is not None and buf.owner == (id(ckv_cache), start, ckv_cache.version)


# ---------------------------------------------------------------------------------------------- PB.HG owned histories
# (tt/owned_gather.py: the program; tests/packed/test_owned_send_ops.py: its bytes on every layout, mesh only)
# PACKED_HIST_OWNED=on (OWNER 2026-11-21): in a chip-aligned packed chunk every segment's history of a KV source goes only
# to the chips that segment owns (tt/owned_gather.py), into the source's ONE buffer (no lane buffers): chip d's rows
# q R + [0, local rows) are its own segment's, the bytes the lane buffer held there (the same packs, the same rows).
_OWN_LOGGED: list = []
_OWN: dict = {}  # the chunk's context: mesh, segs (SegRun, chip order of their entries), chip -> segment index


def owned_on() -> bool:
    """PACKED_HIST_OWNED=on (call-time; a chip-aligned chunk with HISTORY_FORMAT=bfp4 only)."""
    from .settings import settings as _s

    return _s.get("PACKED_HIST_OWNED") == "on" and bfp4_on()


def owned_begin(mesh, segs: list, chip_seg: list) -> None:
    """A chip-aligned packed chunk starts (tt/packed_glued.GluedCall): ``segs`` its computing segments, ``chip_seg``
    per chip (chip-linear) the index of the segment it holds (idle chips: 0, the lane the sdpa's map reads there)."""
    _OWN.clear()
    ids = {}
    for r in segs:
        for i, c in r.state.compress.items():
            if i in r.state.shared_compress or getattr(c.ckv, "history_rows", None) is None:
                continue
            if not isinstance(getattr(c.ckv, "history_key", None), tuple):
                continue
            ids[id(c.ckv)] = (i, r)
    _OWN.update(mesh=mesh, segs=list(segs), chip=list(chip_seg), ids=ids)
    if not _OWN_LOGGED:  # (once a process: the history DRAM a chip, for the report)
        _OWN_LOGGED.append(1)
        from loguru import logger

        keys = sorted({c.ckv.history_key for r in segs for i, c in r.state.compress.items() if id(c.ckv) in ids})
        mib = {
            k: _buffer(mesh, k).rows * _buffer(mesh, k).width / 2**20 for k in keys if _buffer(mesh, k) is not None
        }
        logger.info(
            f"PB.HG owned histories: one buffer a source {{(rows, layer): MiB a chip}} {mib}, total "
            f"{sum(mib.values()):.1f} MiB a chip (off: the same a lane, x up to {len(segs)} lanes here)"
        )


def owned_end() -> None:
    _OWN.clear()


def owned_active(ckv_cache) -> bool:
    return bool(_OWN) and id(ckv_cache) in _OWN["ids"]


class OwnedGroup:
    """PB.HG: every segment's compressed cache of KV source ``s`` in an owned chunk: one gather item
    (tt/gather_hide.Hide), ``ckvs`` / ``starts`` in segment order, ``dest`` per chip its entry index."""

    def __init__(self, s: int, ckvs: list, starts: list, dest: list):
        self.s, self.ckvs, self.starts, self.dest = int(s), list(ckvs), [int(v) for v in starts], list(dest)
        c0 = self.ckvs[0]
        self.mesh, self.slab, self.stripe, self.ratio = c0.mesh, c0.slab, c0.stripe, c0.ratio
        self.owners, self.tp_axis, self.history_key = c0.owners, c0.tp_axis, c0.history_key
        self.version = c0.version


class OwnedRun:
    """The item's stand-in SegRun (gather_hide reads .start: the group's latest, its largest history; .lane)."""

    def __init__(self, start: int):
        self.start, self.lane = int(start), -1


def owned_group(s: int) -> "OwnedGroup | None":
    """Source s's group of the chunk (None: no owned cache of s)."""
    if not _OWN:
        return None
    ckvs, starts, idx = [], [], {}
    for r in _OWN["segs"]:
        c = r.state.compress.get(s)
        if c is None or id(c.ckv) not in _OWN["ids"]:
            continue
        if id(c.ckv) not in idx:
            idx[id(c.ckv)] = len(ckvs)
            ckvs.append(c.ckv)
            starts.append(int(r.start))
    if not ckvs:
        return None
    dest = []
    for j in _OWN["chip"]:
        c = _OWN["segs"][j].state.compress.get(s)
        dest.append(idx.get(id(c.ckv)) if c is not None else None)
    return OwnedGroup(s, ckvs, starts, dest)


def _owned_axis(ckv) -> int:
    _, axis = owners_and_axis(ckv.mesh, ckv.tp_axis is not None)
    return 0 if axis is None else 1


def _owned_send(buf: HistoryBuffer, R: int, ckv0, entries: list, dest: list, sub: dict | None = None) -> None:
    from .owned_gather import sender

    ents = [(t.tensor if isinstance(t, HistoryBuffer) else t, lo, hi) for t, lo, hi in entries]
    with _Traced("gather", ckv0, -1):
        sender(buf.mesh)(buf.tensor, R, _owned_axis(ckv0), ents, dest)


class OwnedPlan:
    """A group's prefetch: every entry's records [0, h) packed into its own record buffer."""

    def __init__(self, group, buf, entries, owners):
        self.group, self.buf, self.entries, self.owners = group, buf, entries, owners  # entries [(ckv, loc, h, start)]
        self.h = max(e[2] for e in entries)


def owned_prepare(group: OwnedGroup, chunk: int, locs: list) -> "OwnedPlan | None":
    """``prefetch_prepare`` of a group: every entry's history [0, c hs) packed into ``locs[e]``; None when no entry has
    a history (every segment's first chunk)."""
    buf = _buffer(group.mesh, group.history_key)
    assert buf is not None and buf.bfp4, ("PB.HG: the source's record buffer", group.history_key)
    entries = []
    for ckv, st, loc in zip(group.ckvs, group.starts, locs):
        h = (int(st) // int(chunk)) * (ckv.stripe // ckv.ratio)
        assert h <= ckv.slab.rows, (h, ckv.slab.rows)
        buf.own.pop(id(ckv), None)
        if h > 0:
            with _Traced("pack", ckv, -1):
                _pack(ckv, 0, h, loc)
        entries.append((ckv, loc, h, int(st)))
    if all(e[2] == 0 for e in entries):
        return None
    return OwnedPlan(group, buf, entries, int(group.owners))


def owned_mark(plan: OwnedPlan) -> None:
    """The prefetch markers (host only): each entry's rows [0, h) are (or will be, before its reader) in the buffer."""
    for ckv, loc, h, st in plan.entries:
        if h > 0:
            plan.buf.own[id(ckv)] = ("pre", st, ckv.version, h)
    plan.buf.owner = None
    STATS["prefetch"] += 1


def owned_range(plan: OwnedPlan, a: int, b: int | None = None, sub: dict | None = None) -> None:
    """Rows [a, b) of every entry's history (clipped to its own [0, h)) to its segment's chips: ONE program."""
    b = plan.h if b is None else b
    ents = [(loc, min(a, h), min(b, h)) for ckv, loc, h, st in plan.entries]
    R = int(plan.group.slab.rows)
    _owned_send(plan.buf, R, plan.group.ckvs[0], ents, plan.group.dest, sub)


def owned_prefetch(group: OwnedGroup, chunk: int, locs: list) -> bool:
    """Serially (a flush): the group's whole prefetch."""
    plan = owned_prepare(group, chunk, locs)
    if plan is None:
        return False
    owned_range(plan, 0)
    owned_mark(plan)
    return True


def owned_tails_prepare(group: OwnedGroup, chunk: int, locs: list):
    """At the source layer, after its compressor's writes: every entry's rows past its prefetch [h, local rows) (all
    of [0, local rows) when nothing was prefetched) packed into ``locs[e]``; returns the plan ``owned_tails_send``
    takes (None: nothing to move)."""
    buf = _buffer(group.mesh, group.history_key)
    assert buf is not None and buf.bfp4, ("PB.HG: the source's record buffer", group.history_key)
    ents = []
    for ckv, st, loc in zip(group.ckvs, group.starts, locs):
        nb = -(-(int(st) + int(chunk)) // int(chunk))
        local_rows = nb * ckv.stripe // ckv.ratio
        assert 0 < local_rows <= ckv.slab.rows, (local_rows, ckv.slab.rows)
        pre = buf.own.get(id(ckv))
        h = 0
        if pre is not None and pre[0] == "pre" and pre[1] == st and 0 <= ckv.version - pre[2] <= 1:
            h = pre[3]
        if h < local_rows:
            with _Traced("pack", ckv, -1):
                _pack(ckv, h, local_rows, loc)
        ents.append((ckv, loc, h, local_rows, st))
    return (group, buf, ents)


def owned_tails_send(plan, sub: dict | None = None) -> None:
    """Every entry's rows [h, local rows) to its chips (ONE program), then its markers ready."""
    group, buf, ents = plan
    R = int(group.slab.rows)
    _owned_send(buf, R, group.ckvs[0], [(loc, h, lr) for ckv, loc, h, lr, st in ents], group.dest, sub)
    for ckv, loc, h, lr, st in ents:
        buf.own[id(ckv)] = ("ready", st, ckv.version, lr)
        STATS["tail"] += 1
    buf.owner = None


def _owned_one(ckv, buf: HistoryBuffer, lo: int, hi: int) -> None:
    """Rows [lo, hi) of one owned cache (packed into the shared local record buffer) to its own chips."""
    s, _ = _OWN["ids"][id(ckv)]
    group = owned_group(s)
    e = next(i for i, c in enumerate(group.ckvs) if c is ckv)
    R = int(ckv.slab.rows)
    loc = _buffer(ckv.mesh, ("loc", R))
    with _Traced("pack", ckv, -1):
        _pack(ckv, lo, hi, loc)
    dest = [e if d == e else None for d in group.dest]
    ents = [(loc, lo, hi) if i == e else (None, 0, 0) for i in range(len(group.ckvs))]
    _owned_send(buf, R, ckv, ents, dest)


def _owned_prefetch_one(ckv, buf: HistoryBuffer, start: int, chunk: int) -> bool:
    h = (start // chunk) * (ckv.stripe // ckv.ratio)
    buf.own.pop(id(ckv), None)
    if h == 0:
        return False
    assert h <= ckv.slab.rows, (h, ckv.slab.rows)
    _owned_one(ckv, buf, 0, h)
    buf.own[id(ckv)] = ("pre", start, ckv.version, h)
    STATS["prefetch"] += 1
    return True


def _owned_complete(ckv, buf: HistoryBuffer, local_rows: int, start: int) -> None:
    """``gather`` of an owned cache: done when its rows are ready, else moved now (its own chips only)."""
    st = buf.own.get(id(ckv))
    if st is not None and st[0] == "ready" and st[1:] == (start, ckv.version, local_rows):
        STATS["ready"] += 1
        return
    h = 0
    if st is not None and st[0] == "pre" and st[1] == start and 0 <= ckv.version - st[2] <= 1 and st[3] < local_rows:
        h = st[3]
    _owned_one(ckv, buf, h, local_rows)
    buf.own[id(ckv)] = ("ready", start, ckv.version, local_rows)
    STATS["tail" if h else "full"] += 1
