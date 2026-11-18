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
"""

from __future__ import annotations

import ttnn

from . import ccl
from .common import DRAM, SP_AXIS, TP_AXIS

WIDTH = 512  # a compressed row (bf16)


class HistoryBuffer:
    """One persistent gather output of one mesh: [1, 1, rows, WIDTH] bf16 ROW_MAJOR DRAM interleaved; ``users``: the
    caches that acquired it."""

    def __init__(self, mesh, rows: int):
        self.mesh, self.rows, self.users = mesh, int(rows), 0
        self.tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.rows, WIDTH]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, DRAM
        )
        self.owner = None  # (id(cache), start, version) of the rows it holds

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


def acquire(mesh, rows: int) -> None:
    """A KV source's compressed cache at build (load / state): the buffer of ``rows`` (made by the first, shared by
    the rest) and the gather's semaphores."""
    hit = _BUF.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        hit = _BUF[id(mesh)] = (mesh, {})
    b = hit[1].get(int(rows))
    if b is None or not b.tensor.is_allocated():
        b = hit[1][int(rows)] = HistoryBuffer(mesh, rows)
    b.users += 1
    for axis in (None, SP_AXIS):  # (one semaphore pair a ring geometry, DS41-216; made at load, agent rule 5)
        if axis is None or mesh.shape[axis] > 1:
            ccl._sems(mesh, axis)


def release(mesh, rows: int) -> None:
    """The cache freed: the buffer goes with its last user."""
    hit = _BUF.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        return
    b = hit[1].get(int(rows))
    if b is None:
        return
    b.users -= 1
    if b.users <= 0:
        b.free()
        del hit[1][int(rows)]


def _buffer(mesh, rows: int) -> HistoryBuffer | None:
    hit = _BUF.get(id(mesh))
    return hit[1].get(rows) if hit is not None and hit[0] is mesh else None


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
    buf = _buffer(mesh, owners * R)
    assert buf is not None, (
        f"HISTORY_GATHER=rows: no history buffer of {owners * R} rows (acquired by tt/compressor.TtServedRows; "
        f"held: {sorted((_BUF.get(id(mesh)) or (None, {}))[1])})"
    )
    s0, s1 = ccl._sems(mesh, axis)
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
    )
    buf.owner = (id(ckv_cache), start, ckv_cache.version)
    return buf.tensor, R  # (the kept tensor, not the op's new wrapper: DS41-149)


def holds(ckv_cache, start: int) -> bool:
    """The buffer of ``ckv_cache``'s size still holds its rows of this start and write version (a memo hit is
    valid: no other source of that size gathered since)."""
    buf = _buffer(ckv_cache.mesh, ckv_cache.owners * ckv_cache.slab.rows)
    return buf is not None and buf.owner == (id(ckv_cache), start, ckv_cache.version)
