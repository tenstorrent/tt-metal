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
"""

from __future__ import annotations

import ttnn

from . import ccl
from .ccl import dealloc
from .common import DRAM, SP_AXIS, TP_AXIS
from .cores import max_grid

WIDTH = 512  # a compressed row (bf16)


class HistoryBuffer:
    """One persistent gather output of one mesh: [1, 1, rows, WIDTH] bf16 ROW_MAJOR DRAM interleaved; ``users``: the
    caches that acquired it."""

    def __init__(self, mesh, rows: int, owners: int | None = None):
        self.mesh, self.rows, self.users = mesh, int(rows), 0
        self.tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.rows, WIDTH]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, DRAM
        )
        self.owner = None  # (id(cache), start, version) of the rows it holds
        self.pre = None  # P.75: (id(cache), start, version, h) of a prefetched history (rows [q R, q R + h))
        self.view = None  # P.75: [1, owners, rows / owners, WIDTH], a view of ``tensor`` (the slice_write's operand)
        if owners is not None:
            assert self.rows % owners == 0, (self.rows, owners)
            self.view = ttnn.reshape(self.tensor, [1, owners, self.rows // owners, WIDTH])
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


def acquire(mesh, rows: int, key=None, owners: int | None = None, tails: tuple = ()) -> None:
    """A KV source's compressed cache at build (load / state): the buffer of ``rows`` (made by the first, shared by
    the rest; ``key``: P.75's per-source key (rows, layer), default the size) and the gather's semaphores. ``tails``
    (P.75): the chunk-period row counts a source's tail gather moves an owner; their small buffers (owners x m rows)
    are shared by every source."""
    hit = _BUF.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        hit = _BUF[id(mesh)] = (mesh, {})
    key = int(rows) if key is None else key
    want = [(key, int(rows), owners if tails else None)] + [(("tail", owners * m), owners * m, owners) for m in tails]
    for k, n, own in want:
        b = hit[1].get(k)
        if b is None or not b.tensor.is_allocated():
            b = hit[1][k] = HistoryBuffer(mesh, n, own)
        b.users += 1
    for axis in (None, SP_AXIS):  # (one semaphore pair a ring geometry, DS41-216; made at load, agent rule 5)
        if axis is None or mesh.shape[axis] > 1:
            ccl._sems(mesh, axis)


def release(mesh, rows: int, key=None, owners: int | None = None, tails: tuple = ()) -> None:
    """The cache freed: the buffer (and P.75's tail buffers) go with their last user."""
    hit = _BUF.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        return
    key = int(rows) if key is None else key
    for k in [key] + [("tail", owners * m) for m in tails]:
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


STATS = {"prefetch": 0, "tail": 0, "full": 0}  # P.75: gathers by kind since load (diagnostic, tests)


def gather(ckv_cache, local_rows: int, start: int) -> tuple[ttnn.Tensor, int]:
    """The first ``local_rows`` compressed rows of every owner of ``ckv_cache`` (tt/compressor.TtServedRows) ->
    the persistent buffer of its size (owner q at row q R), returns (the buffer [1, 1, owners R, WIDTH], R)."""
    slab = ckv_cache.slab
    mesh = ckv_cache.mesh
    owners, axis = owners_and_axis(mesh, ckv_cache.tp_axis is not None)
    assert owners == ckv_cache.owners, (owners, ckv_cache.owners)
    R = slab.rows
    assert slab.width == WIDTH and slab.tensor.dtype == ttnn.bfloat16, (slab.width, slab.tensor.dtype)
    assert 0 < local_rows <= R, (local_rows, R)
    buf = _buffer(mesh, _key(ckv_cache))
    assert buf is not None, (
        f"HISTORY_GATHER=rows: no history buffer {_key(ckv_cache)} (acquired by tt/compressor.TtServedRows; "
        f"held: {list((_BUF.get(id(mesh)) or (None, {}))[1])})"
    )
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
        _full(slab, buf, owners * local_rows, axis)
        STATS["full"] += 1
    buf.owner = (id(ckv_cache), start, ckv_cache.version)
    return buf.tensor, R  # (the kept tensor, not the op's new wrapper: DS41-149)


def _full(slab, buf: HistoryBuffer, gathered: int, axis) -> None:
    """Every owner's first gathered / owners local slab rows -> its fixed slot of ``buf``."""
    s0, s1 = ccl._sems(buf.mesh, axis)
    ttnn.experimental.high_bw_all_gather(
        slab.tensor,
        2,
        buf.tensor,
        cluster_axis=axis,
        num_links=ccl.AG_LINKS,
        input_batch_index=slab.batch,
        gathered_dim_size=gathered,
        ready_semaphore=s0,
        data_valid_semaphore=s1,
    )


def _tail(ckv_cache, buf: HistoryBuffer, h: int, local_rows: int, owners: int, axis) -> None:
    """P.75: every owner's local rows [h, local_rows) -> rows [q R + h, q R + local_rows) of ``buf`` (its rows below h
    prefetched): a slice of the slab, one gather into the small buffer of that size, one slice_write."""
    m = local_rows - h
    small = _buffer(buf.mesh, ("tail", owners * m))
    assert small is not None and buf.view is not None, ("P.75: no tail buffer", owners, m)
    part = ckv_cache.slab.slice_rows(h, local_rows)  # [1, 1, m, 512] ROW_MAJOR DRAM interleaved
    # (host metadata only: the slice's own topology is not the slab's dim-2 sharding the full-mesh gather checks)
    part.update_tensor_topology(ckv_cache.slab.tensor.tensor_topology())
    s0, s1 = ccl._sems(buf.mesh, axis)
    ttnn.experimental.high_bw_all_gather(
        part, 2, small.tensor, cluster_axis=axis, num_links=ccl.AG_LINKS, ready_semaphore=s0, data_valid_semaphore=s1
    )
    dealloc(part)
    ttnn.bringup.slice_write(
        small.view, buf.view, [0, 0, h, 0], [1, owners, local_rows, WIDTH], [1, 1, 1, 1], max_grid=max_grid()
    )


def new_chunk() -> None:
    """FIX.39 (owner review 2026-11-18): at every chunk start (TtDS41Model.begin_chunk) no buffer holds a prefetched
    history: ``gather`` reuses one only when this chunk's ``prefetch`` made it. ``version`` counts this process's
    writes only; a write from outside (the KV Manager filling a served cache after a store load / d2p pull) never bumps
    it, so a marker left from an earlier chunk (a pipeline stage that starts mid-model: no prefetch before its sources)
    could pass the (cache, start, version) check over rows gathered before that write. Host only."""
    for _, bufs in _BUF.values():
        for b in bufs.values():
            b.pre = None


def prefetch(ckv_cache, start: int, chunk: int) -> bool:
    """P.75 (HISTORY_GATHER_HIDE=on): every owner's history rows [0, c hs) (c = start // chunk: the chunk periods
    complete before this chunk) -> the source's own buffer, before the chunk's first engram wait; ``gather`` at the
    source's layer then moves only the rest. False: nothing to prefetch (c = 0, a cache not hidden)."""
    if getattr(ckv_cache, "history_key", None) is None or not isinstance(ckv_cache.history_key, tuple):
        return False
    owners, axis = owners_and_axis(ckv_cache.mesh, ckv_cache.tp_axis is not None)
    buf = _buffer(ckv_cache.mesh, ckv_cache.history_key)
    assert buf is not None, ("P.75: the source's buffer", ckv_cache.history_key)
    buf.pre = None
    h = (int(start) // int(chunk)) * (ckv_cache.stripe // ckv_cache.ratio)
    if h == 0:
        return False
    assert h <= ckv_cache.slab.rows, (h, ckv_cache.slab.rows)
    _full(ckv_cache.slab, buf, owners * h, axis)
    buf.owner = None  # (rows [0, h) only until ``gather`` completes it)
    buf.pre = (id(ckv_cache), int(start), ckv_cache.version, h)
    STATS["prefetch"] += 1
    return True


def holds(ckv_cache, start: int) -> bool:
    """The buffer of ``ckv_cache``'s size still holds its rows of this start and write version (a memo hit is
    valid: no other source of that size gathered since)."""
    buf = _buffer(ckv_cache.mesh, _key(ckv_cache))
    return buf is not None and buf.owner == (id(ckv_cache), start, ckv_cache.version)
