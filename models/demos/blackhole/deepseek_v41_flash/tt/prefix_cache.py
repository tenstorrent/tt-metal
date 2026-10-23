# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefix snapshots of the interleaved prefill (APC_CHUNKED_PREFILL_PLAN.md, M2 PROTOTYPE).

A snapshot is the state of ONE user after a prefill step that ended on a chunk boundary ``s`` (a multiple of the chunk; ``Model.pf_resume[b] == s``), copied to the host, so that
a later prompt with the same first ``s`` tokens can be restored into ANY user (any mesh row, any prefill slot) and continue with ``Model.prefill_interleaved([(b, tokens, s, e)])`` instead
of recomputing [0, s). What a continuation reads (everything else is rewritten by the chunks it runs):

  * the carried prefill state of every layer, per (mesh row, prefill slot), kept in persistent per-device tensors that the traced chunk updates under the user mask:
    ``halo`` (the 128 kv rows before the chunk), the kv table ``kvt.t`` rows [0, s / ratio) (compressed latents; the window scratch rows behind them are rewritten by every layer for every
    chunk), the column-sharded index-key FIFO (``keys_rm`` and its tile copy ``keys``: only the last s / ratio / cols slots are valid, the FIFO shifts every chunk and appends at the end) or,
    on the dense path, ``lat_buf`` (tail FIFO of the latents),
  * the compressed latents of the kv-source layers in the paged pool, rows [0, s / ratio) of every source through the user's page table (decode reads them; the sink writes them chunk by chunk),
  * the host Engram history (pure function of the tokens: re-derived, not stored).
The window ring rows and the ratio-2 ``prev_cs`` are NOT part of a snapshot: the continuation always replays the last >= 128 positions of the prompt (the lookup rule ``s <= e - 128``), the sink
rewrites every ring row and ``prev_cs`` from those chunks.

Restore writes the persistent tensors OUTSIDE the trace with the model's own idiom, a per-(mesh row, slot) 0/1 mask: the snapshot is uploaded as a mesh-sharded tensor that is zero everywhere but at
the target (row, slot) and ``ttnn.where(mask, new, old)`` + ``ttnn.copy`` put it into the persistent buffer (same address: a captured trace keeps reading it). Row-major tables whose rows
are 512 wide (the kv table, the pool latents) are written with ``paged_scatter_rows`` instead (ids = 0xFFFFFFFF everywhere but the valid rows of the target slot): no whole-table upload, no
where. NOT possible: ``ttnn.copy_host_to_device_tensor(host, ttnn.get_device_tensors(t)[k])`` addresses the WHOLE mesh buffer and broadcasts the host tensor to every device
(tests/test_apc_m2_device_write.py, variant A).
"""

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_paged as PG

SKIP = 0xFFFFFFFF


# ---- which tensors carry prefill state ---------------------------------------------------------------------------------------------------------------------
@dataclass
class CarriedSpec:
    """One persistent per-slot tensor [Up, 1, N, D] of the prefill model. ``kind``: ``full`` (everything) or ``tail`` (a FIFO that shifts and appends at the end: the valid rows are
    [end - n, end), ``end`` = N - pad_rows unless given); ``count(s)`` the valid count n after a step that ended at position s (None for ``full``).
    """

    name: str
    tensor: object
    kind: str
    ratio: int = 1
    div: int = 1  # entries per column shard (sharded key FIFO: the tail holds s / ratio / cols entries)
    pad_rows: int = (
        0  # ``keys`` (tile copy of ``keys_rm`` + 32 zero pad rows): the tail ends ``pad_rows`` before the last row
    )
    end: int = 0  # kv table: the latent FIFO ends at row LAT (window scratch rows follow); 0 = N - pad_rows
    start: int = (
        -1
    )  # >= 0: the valid rows of a ``tail`` kind START here (single-trace key slab: history entry j at row start + j / cols) instead of ending at ``end_row``
    replicated: bool = True  # identical on the 8 columns of a mesh row (everything but the column-sharded key FIFO): one host copy, one upload per mesh row

    def end_row(self, N):
        return self.end or N - self.pad_rows

    def row0(self, N, n):
        """first valid row of a ``tail`` kind holding n entries"""
        return self.start if self.start >= 0 else self.end_row(N) - n

    def count(self, s):
        return None if self.kind == "full" else s // self.ratio // self.div


def carried_specs(model):
    """-> OrderedDict name -> CarriedSpec of every persistent carried-state tensor of ``model.prefill_model`` (deduplicated: a kv table / key FIFO shared by the layers of a source group
    appears once)."""
    pm = model.prefill_model
    out, seen = OrderedDict(), set()

    def add(spec):
        if id(spec.tensor) not in seen:
            seen.add(id(spec.tensor))
            out[spec.name] = spec

    for lid, pl in pm.layers:
        pa = pl.pa
        if getattr(pa, "halo", None) is not None:
            add(CarriedSpec(f"L{lid}.halo", pa.halo, "full"))
        if getattr(pa, "lat_buf", None) is not None:
            add(CarriedSpec(f"L{lid}.lat_buf", pa.lat_buf, "tail", ratio=pa.ratio))
        sp = getattr(pa, "sparse", None)
        if sp is None or not getattr(sp, "dyn_on", False):
            continue
        if (
            sp.kvt is not None and sp.kvt.t is not None
        ):  # (paged prefill, DSV41_PF_PAGED=read: no table, the latents are the pool pages)
            add(CarriedSpec(f"L{lid}.kvt", sp.kvt.t, "tail", ratio=pa.ratio, end=sp.kvt.LAT))
        ix = sp.indexer
        if ix is not None and sp.owner and ix.key_owner is None:
            sh = getattr(ix, "sh", None)
            if sh is not None and sh.get(
                "single"
            ):  # single-trace slab (tt/prefill_single.py): history entry j at local row Cc8 + j / cols
                add(
                    CarriedSpec(
                        f"L{lid}.keys",
                        ix.keys,
                        "tail",
                        ratio=pa.ratio,
                        div=sh["cols"],
                        start=sh["Cc8"],
                        replicated=False,
                    )
                )
            elif sh is not None:  # column-sharded FIFO
                add(
                    CarriedSpec(f"L{lid}.keys_rm", ix.keys_rm, "tail", ratio=pa.ratio, div=sh["cols"], replicated=False)
                )
                add(
                    CarriedSpec(
                        f"L{lid}.keys", ix.keys, "tail", ratio=pa.ratio, div=sh["cols"], pad_rows=32, replicated=False
                    )
                )
            elif ix.keys is not None:  # replicated FIFO: appended at the end like the others
                add(CarriedSpec(f"L{lid}.keys", ix.keys, "tail", ratio=pa.ratio, end=ix.KL))
    return out


# ---- per-device shard access ---------------------------------------------------------------------------------------------------------------------------
def _dev(t, k):
    return ttnn.get_device_tensors(t)[k]


def read_shard(t, k):
    """device shard k (row-major over (mesh row, column)) of mesh tensor t -> torch."""
    return ttnn.to_torch(_dev(t, k))


SNAP_PIECE_MIB = 16  # device-side slice size of a snapshot read (see snapshot_user)
PIECE = 2048  # rows per ``paged_scatter_rows`` call: the row count is a COMPILE-time argument (one program per distinct count, ~0.4 s at first use), so every write is split into pieces of the same size (the last one padded with SKIP ids)


def scatter_rows_mesh(model, t, per_row, base=0):
    """In place: ``t[ids + base] = data`` on the mesh rows of ``per_row`` {mesh row: (ids [n] torch, data [n, D] torch)} only (the other mesh rows get SKIP ids), t a bf16 (or, for the pool, fp8: the
    kernel converts) row-major tensor whose rows are D wide; fixed-size pieces."""
    md, rows, cols = model.md, model.rows, model.cols
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    n, D = next(iter(per_row.values()))[1].shape
    for a in range(0, n, PIECE):
        m = min(PIECE, n - a)
        ids = torch.full((rows, 1, PIECE), SKIP, dtype=torch.int64)
        src = torch.zeros(rows, 1, PIECE, D, dtype=torch.bfloat16)
        for r, (ids_r, data_r) in per_row.items():
            ids[r, 0, :m] = ids_r[a : a + m].long() + base
            src[r, 0, :m] = data_r[a : a + m]
        d_ids = ttnn.from_torch(
            ids.to(torch.int32),
            device=md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        d_src = ttnn.from_torch(
            src,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        P.paged_scatter_rows(t, d_src, d_ids)
        ttnn.synchronize_device(md)  # (the device buffers are freed here)
        ttnn.deallocate(d_src)
        ttnn.deallocate(d_ids)


def _where(mask, new, old):
    """``gate_user_state`` for one tensor: ``new`` where the per-(row, slot) mask is 1, else ``old`` (row-major tensors go through tile layout like the model's gate)."""
    rm = new.layout == ttnn.ROW_MAJOR_LAYOUT
    n_, o_ = (ttnn.to_layout(new, ttnn.TILE_LAYOUT), ttnn.to_layout(old, ttnn.TILE_LAYOUT)) if rm else (new, old)
    out = ttnn.where(mask, n_, o_)
    if rm:
        ttnn.deallocate(n_)
        ttnn.deallocate(o_)
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
    return out


def slot_mask(model, targets):
    """per-(mesh row, slot) 0/1 mask [rows * Up, 1, 1, 1] sharded over the mesh rows (the ``DynCtx.umask`` layout), one-hot at every (r, i) of ``targets``"""
    rows, cols, Up, md = model.rows, model.cols, model.Up, model.md
    mask = torch.zeros(rows * Up, 1, 1, 1)
    for r, i in targets:
        mask[r * Up + i] = 1.0
    return ttnn.from_torch(
        mask,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )


def _fill(host, targets, j, rn, data, tail, N, row0, replicated):
    """put ``data`` (list [cols] of [1, 1, n, D]) of tensor j of a batch into the host slab at every target (r, i)"""
    ncol = host.shape[1]
    n = data[0].shape[2]
    for flat in targets:
        for c in range(ncol):
            lo = j * rn + (0 if tail else row0)
            host[flat, c, lo : lo + n] = data[c][0, 0]


def write_batch(model, batch, targets, mask_d):
    """Write a batch of persistent tensors that share (layout, rows-per-tensor, width, replication) into the slots ``targets`` [(r, i)] of their mesh rows with ONE host upload: ``batch`` =
    [(spec, per_col)]. ``tail`` kinds write only the valid rows (slice + where + slice_write), full tensors are replaced (rest of the slot zeros).
    """
    rows, cols, Up, md = model.rows, model.cols, model.Up, model.md
    sp0, _ = batch[0]
    t0_ = sp0.tensor
    N, D = t0_.shape[2], t0_.shape[3]
    tail = sp0.kind == "tail"
    n = batch[0][1][0].shape[2]
    rn = n if tail else N
    replicated = sp0.replicated
    ncol = 1 if replicated else cols
    host = torch.zeros(rows * Up, ncol, len(batch) * rn, D, dtype=torch.bfloat16)
    flats = [r * Up + i for r, i in targets]
    for j, (sp, per_col) in enumerate(batch):
        row0 = sp.row0(sp.tensor.shape[2], n) if tail else 0
        _fill(host, flats, j, rn, per_col, tail, N, row0, replicated)
    mesh = ttnn.ShardTensor2dMesh(md, dims=(0, None if replicated else 1), mesh_shape=(rows, cols))
    big = ttnn.from_torch(
        host, device=md, dtype=ttnn.bfloat16, layout=t0_.layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mesh
    )
    for j, (sp, per_col) in enumerate(batch):
        t = sp.tensor
        new_j = ttnn.slice(big, [0, 0, j * rn, 0], [Up, 1, (j + 1) * rn, D]) if len(batch) > 1 else big
        if tail:
            row0 = sp.row0(t.shape[2], n)
            old = ttnn.slice(t, [0, 0, row0, 0], [Up, 1, row0 + n, D])
            upd = _where(mask_d, new_j, old)
            ttnn.experimental.slice_write(upd, t, [0, 0, row0, 0], [Up, 1, row0 + n, D], [1, 1, 1, 1])
            extra = (old,)
        else:
            upd = _where(mask_d, new_j, t)
            ttnn.copy(upd, t)
            extra = ()
        for x in (upd,) + extra + ((new_j,) if len(batch) > 1 else ()):
            ttnn.deallocate(x)
    ttnn.deallocate(big)


@dataclass
class PrefixSnapshot:
    s: int
    toks: torch.Tensor  # int64 [s]
    carried: dict = field(default_factory=dict)  # name -> [cols] torch (slot dim kept, trimmed rows)
    pool: dict = field(default_factory=dict)  # src -> torch bf16 [n_entries, 512] latents of the kv-source layer
    seconds: float = 0.0

    @property
    def nbytes(self):
        seen, n = set(), 0
        for x in [x for v in self.carried.values() for x in v] + list(self.pool.values()):
            if id(x) not in seen:  # (replicated tensors and the pool latents share one host copy)
                seen.add(id(x))
                n += x.numel() * x.element_size()
        return n


def _sources(model):
    """kv-source layers present in the model (their latents live in the pool pages)."""
    return [s for s in P.SOURCES if s in model.layer_ids]


def _slot_of(model, b):
    """(mesh row, prefill slot in the row) of user b's carried state."""
    Up = model.Up
    if Up == model.U:
        return b // model.U, b % model.U
    sl = model.__dict__.get("pf_slot_of", {}).get(b)
    assert sl is not None, f"user {b} has no prefill slot"
    return sl // Up, sl % Up


class ApcUnsupported(RuntimeError):
    """the model build has no carried state the snapshot code knows (dense prefill path, fp8 kv table): prefix caching stays off"""


def read_pool_latents(model, b, src, n):
    """Paged prefill: entries [0, n) of kv-source layer ``src`` of user ``b`` -> torch bf16 [n, 512], gathered ON the device from the user's pool pages (``ttnn.embedding`` over the
    bf16 pool, row ids of the user's mesh row; pieces of 16384 rows) and read from one device of the mesh row."""
    pool, md, rows, cols = model.pool, model.md, model.rows, model.cols
    assert pool.dtype == ttnn.bfloat16, "APC on the paged prefill needs the bf16 pool"
    r = b // model.U
    phys = pool.phys_rows(b, src, torch.arange(n)).long()
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    out = []
    for a in range(0, n, 16384):
        m = min(16384, n - a)
        ids = torch.zeros(rows, 1, 1, m, dtype=torch.int64)
        ids[r, 0, 0] = phys[a : a + m]
        d_ids = ttnn.from_torch(
            ids.to(torch.int32),
            device=md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        g = ttnn.embedding(d_ids, pool.pool, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, m, 512] per device
        out.append(ttnn.to_torch(ttnn.get_device_tensors(g)[r * cols]).reshape(m, 512).clone())
        ttnn.deallocate(g)
        ttnn.deallocate(d_ids)
    return torch.cat(out) if len(out) > 1 else out[0]


def read_pool_latents_begin(model, b, src, n):
    """``read_pool_latents`` without waiting: [(rows m, host tensor handle)] of the gathers (pieces of 16384 rows), enqueued on CQ 0 and freed right after their reads (see ``snapshot_begin``)."""
    pool, md, rows, cols = model.pool, model.md, model.rows, model.cols
    assert pool.dtype == ttnn.bfloat16, "APC on the paged prefill needs the bf16 pool"
    r = b // model.U
    phys = pool.phys_rows(b, src, torch.arange(n)).long()
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    out = []
    for a in range(0, n, 16384):
        m = min(16384, n - a)
        ids = torch.zeros(rows, 1, 1, m, dtype=torch.int64)
        ids[r, 0, 0] = phys[a : a + m]
        d_ids = ttnn.from_torch(
            ids.to(torch.int32),
            device=md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        g = ttnn.embedding(d_ids, pool.pool, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, m, 512] per device
        out.append((m, ttnn.from_device(ttnn.get_device_tensors(g)[r * cols], blocking=False)))
        ttnn.deallocate(g)
        ttnn.deallocate(d_ids)
    return out


def _snapshot_nb():
    """``DSV41_APC_SNAP_NB`` (default 1): the carried-state reads of a snapshot are issued without blocking and waited for in groups (``SNAP_GROUP_MIB`` of device slices alive at once). A snapshot costs ~0.4 s whatever
    its size (measured 6.6 .. 55 MiB: 0.37 .. 0.47 s) because every carried tensor was a blocking slice + read round trip (~300 per snapshot); 6 snapshots of a 32k prompt = 2.8 s of the 28 s prefill call.
    """
    return os.environ.get("DSV41_APC_SNAP_NB", "1") == "1"


SNAP_GROUP_MIB = (
    32  # device bytes (per device) of slices kept alive between two synchronisations of the non-blocking snapshot reads
)


def snapshot_user(model, b, s, toks):
    """-> PrefixSnapshot (see ``_snapshot_user``). ``DSV41_APC_SNAP_CHECK=1`` (debug): also takes the blocking snapshot and asserts both are bit-identical."""
    nb = _snapshot_nb()
    snap = _snapshot_user(model, b, s, toks, nb)
    if nb and os.environ.get("DSV41_APC_SNAP_CHECK") == "1":
        ref = _snapshot_user(model, b, s, toks, False)
        assert snap.carried.keys() == ref.carried.keys()
        for k in ref.carried:
            assert len(snap.carried[k]) == len(ref.carried[k]) and all(
                torch.equal(x, y) for x, y in zip(snap.carried[k], ref.carried[k])
            ), f"non-blocking snapshot differs from the blocking one in {k} at {s}"
        for k in ref.pool:
            assert torch.equal(snap.pool[k], ref.pool[k]), f"pool latents of source {k} differ at {s}"
        print(
            f"prefix cache: snapshot check OK at {s} tokens ({len(ref.carried)} tensors; non-blocking {snap.seconds:.2f} s, blocking {ref.seconds:.2f} s)",
            flush=True,
        )
    return snap


def _snapshot_user(model, b, s, toks, nb):
    """State of user ``b`` right after a step that ended at position ``s`` (``model.pf_resume[b] == s``) -> PrefixSnapshot. ``toks``: the user's tokens [>= s]. The pool latents of the kv-source layers
    are NOT read from the pool (an fp8 pool, and reading it would mean the whole per-device pool): the kv table of a source layer holds the same bf16 latents (entry j = FIFO row LAT - n + j).
    """
    t0 = time.perf_counter()
    assert model.__dict__.get("pf_resume", {}).get(b) == s, f"user {b} is not resumable at {s}"
    r, i = _slot_of(model, b)
    cols = model.cols
    snap = PrefixSnapshot(s=s, toks=torch.as_tensor(toks).long().reshape(-1)[:s].clone())
    specs = carried_specs(model)
    paged = PG.read_on()
    for src in _sources(model):
        if not paged and f"L{src}.kvt" not in specs:
            raise ApcUnsupported(f"no kv table for source layer {src} (dense prefill path)")
    group_bytes = SNAP_GROUP_MIB << 20
    live, live_bytes = (
        [],
        0,
    )  # non-blocking mode: (device slice to free or None, [(list, index) slots of the pending host tensors])
    pending = []  # (list, index) slots holding a host tensor handle until the next sync
    staged = OrderedDict()

    def flush():
        nonlocal live, live_bytes, pending
        if not live and not pending:
            return
        ttnn.synchronize_device(model.md)
        for lst, k in pending:
            lst[k] = ttnn.to_torch(lst[k]).clone()
        for part in live:
            if part is not None:
                ttnn.deallocate(part)
        live, live_bytes, pending = [], 0, []

    def read(part, k, lst):
        """device shard k of mesh tensor ``part`` -> appended to ``lst`` (a torch tensor; a host handle until the next flush in non-blocking mode)"""
        if nb:
            lst.append(ttnn.from_device(_dev(part, k), blocking=False))
            pending.append((lst, len(lst) - 1))
        else:
            lst.append(read_shard(part, k).clone())

    for name, sp in specs.items():
        if sp.tensor.dtype != ttnn.bfloat16:
            raise ApcUnsupported(f"{name} is {sp.tensor.dtype}")
        n = sp.count(s)
        N, D = sp.tensor.shape[2], sp.tensor.shape[3]
        lo = sp.row0(N, n) if sp.kind == "tail" else 0
        end = lo + n if sp.kind == "tail" else N
        # slice the slot (and, for a FIFO, its valid tail) ON DEVICE first: reading a whole padded table per tensor is what made a snapshot cost ~0.5 s regardless of its size
        whole = (
            sp.tensor.shape[0] == 1 and lo == 0 and end == N
        )  # (a full slice returns the persistent tensor itself: never free it)
        pieces = [[] for _ in range(1 if sp.replicated else cols)]
        if whole:
            for c in range(len(pieces)):
                read(sp.tensor, r * cols + c, pieces[c])
            if nb:
                live.append(
                    None
                )  # (persistent tensor: read in place, nothing is enqueued that could change it before the flush)
        else:
            # row pieces of <= SNAP_PIECE_MIB: one device copy of the whole slice is 512 MiB at 524,288 tokens (1M config), more than the largest free DRAM block (OOM in the server at the 2^19 snapshot)
            step = max(1, (SNAP_PIECE_MIB << 20) // (D * sp.tensor.element_size()))
            for p0 in range(lo, end, step):
                part = ttnn.slice(sp.tensor, [i, 0, p0, 0], [i + 1, 1, min(p0 + step, end), D])
                for c in range(len(pieces)):
                    read(part, r * cols + c, pieces[c])
                if nb:
                    live.append(part)
                    live_bytes += (min(p0 + step, end) - p0) * D * sp.tensor.element_size()
                    if live_bytes >= group_bytes:
                        flush()
                else:
                    ttnn.deallocate(part)
        staged[name] = (pieces, sp)
    flush()
    for name, (pieces, sp) in staged.items():
        vals = [torch.cat(x, dim=2) if len(x) > 1 else x[0] for x in pieces]
        snap.carried[name] = vals * cols if sp.replicated else vals
    for src in _sources(model):
        if paged:  # no table: the latents are read from the user's pool pages
            snap.pool[src] = read_pool_latents(model, b, src, s // P.SRC_RATIO[src])
        else:
            snap.pool[src] = snap.carried[f"L{src}.kvt"][0][0, 0]  # [s / ratio, 512], shared with the carried copy
    snap.seconds = time.perf_counter() - t0
    return snap


class PendingSnapshot:
    """A snapshot whose device reads are ENQUEUED (``snapshot_begin``) but whose host copies are not made yet (``snapshot_finish``)."""

    def __init__(self, snap, staged, handles, pool_parts, event, t_begin):
        self.snap, self.staged, self.handles, self.pool_parts, self.event, self.t_begin = (
            snap,
            staged,
            handles,
            pool_parts,
            event,
            t_begin,
        )


def snapshot_begin(model, b, s, toks):
    """Deferred form of ``snapshot_user`` (``DSV41_APC_DEFER=1``): enqueue the device slices / gathers and the NON-blocking reads of user ``b``'s state at position ``s`` on CQ 0 and return a
    ``PendingSnapshot``; the host does not wait. Everything is ordered by the command queue: the reads precede the next replay (which may modify the state), a device slice is freed right after its
    read is enqueued (the next program that reuses the memory runs after the read), so no device memory is alive across the next replay (a trace's intermediates share the freed DRAM). The host
    copies are made by ``snapshot_finish`` after ``event_synchronize`` (GIL-free wait for the reads only, the next replay may still be running).
    """
    t0 = time.perf_counter()
    assert model.__dict__.get("pf_resume", {}).get(b) == s, f"user {b} is not resumable at {s}"
    r, i = _slot_of(model, b)
    cols = model.cols
    snap = PrefixSnapshot(s=s, toks=torch.as_tensor(toks).long().reshape(-1)[:s].clone())
    specs = carried_specs(model)
    paged = PG.read_on()
    for src in _sources(model):
        if not paged and f"L{src}.kvt" not in specs:
            raise ApcUnsupported(f"no kv table for source layer {src} (dense prefill path)")
    handles, staged = (
        [],
        OrderedDict(),
    )  # handles: (list, index) slots holding a host tensor handle until ``snapshot_finish``

    def read(part, k, lst):
        lst.append(ttnn.from_device(_dev(part, k), blocking=False))
        handles.append((lst, len(lst) - 1))

    for name, sp in specs.items():
        if sp.tensor.dtype != ttnn.bfloat16:
            raise ApcUnsupported(f"{name} is {sp.tensor.dtype}")
        n = sp.count(s)
        N, D = sp.tensor.shape[2], sp.tensor.shape[3]
        lo = sp.row0(N, n) if sp.kind == "tail" else 0
        end = lo + n if sp.kind == "tail" else N
        whole = sp.tensor.shape[0] == 1 and lo == 0 and end == N
        pieces = [[] for _ in range(1 if sp.replicated else cols)]
        if whole:
            for c in range(len(pieces)):
                read(sp.tensor, r * cols + c, pieces[c])
        else:
            step = max(1, (SNAP_PIECE_MIB << 20) // (D * sp.tensor.element_size()))
            for p0 in range(lo, end, step):
                part = ttnn.slice(sp.tensor, [i, 0, p0, 0], [i + 1, 1, min(p0 + step, end), D])
                for c in range(len(pieces)):
                    read(part, r * cols + c, pieces[c])
                ttnn.deallocate(part)
        staged[name] = (pieces, sp)
    pool_parts = {}
    for src in _sources(model):
        if paged:
            pool_parts[src] = read_pool_latents_begin(model, b, src, s // P.SRC_RATIO[src])
    event = ttnn.record_event(model.md, 0)
    return PendingSnapshot(snap, staged, handles, pool_parts, event, time.perf_counter() - t0)


def snapshot_finish(model, pend):
    """Wait for the reads of ``pend`` (not for anything enqueued after them) and make the host copies -> PrefixSnapshot (bit-identical to ``snapshot_user`` at the same position)."""
    t0 = time.perf_counter()
    ttnn.event_synchronize(pend.event)
    for lst, k in pend.handles:
        lst[k] = ttnn.to_torch(lst[k]).clone()
    snap, cols = pend.snap, model.cols
    for name, (pieces, sp) in pend.staged.items():
        vals = [torch.cat(x, dim=2) if len(x) > 1 else x[0] for x in pieces]
        snap.carried[name] = vals * cols if sp.replicated else vals
    for src in _sources(model):
        if src in pend.pool_parts:
            parts = [ttnn.to_torch(h).reshape(m, 512).clone() for m, h in pend.pool_parts[src]]
            snap.pool[src] = torch.cat(parts) if len(parts) > 1 else parts[0]
        else:
            snap.pool[src] = snap.carried[f"L{src}.kvt"][0][0, 0]
    snap.seconds = pend.t_begin + time.perf_counter() - t0
    pend.handles, pend.pool_parts, pend.staged = [], {}, OrderedDict()
    return snap


def restore_user(model, snap, b):
    return restore_users(model, snap, [b])


def restore_users(model, snap, bs):
    """Put ``snap`` into every user of ``bs`` (any mesh rows / prefill slots; users of the SAME call share one host upload and one set of device writes): pages, pool latents, carried state of their
    slots, Engram history, ``pf_resume``. Afterwards ``model.prefill_interleaved([(b, tokens, snap.s, e)], ...)`` continues the prompts. -> seconds.
    """
    t0 = time.perf_counter()
    s, cols = snap.s, model.cols
    resume = model.__dict__.setdefault("pf_resume", {})
    # 1. pages + slots
    targets = []  # (mesh row, slot) per user
    for b in bs:
        model.release_user(b)
        model.pool.release(b)
        model.pool.admit(b, s + 1)
    model.pool.sync_page_table()
    for b in bs:
        if model.Up != model.U:
            slot = model._slot_alloc([(b, None, 0, s)], resume)[b]
            targets.append((slot // model.Up, slot % model.Up))
        else:
            targets.append((b // model.U, b % model.U))
        resume[b] = s  # (protects the slot from the next user's allocation; the state follows below)
    assert len(set(targets)) == len(targets), f"restore targets collide: {targets}"
    # 2. carried state of the target (mesh row, slot)s: masked writes, outside the trace
    t_c = time.perf_counter()
    mask_d = slot_mask(model, targets)
    groups, per = OrderedDict(), {}
    for name, sp in carried_specs(model).items():
        n = sp.count(s)
        per_col = snap.carried[name]
        if (
            sp.kind == "tail"
            and sp.tensor.layout == ttnn.ROW_MAJOR_LAYOUT
            and sp.tensor.shape[3] == 512
            and sp.replicated
        ):
            t_n = time.perf_counter()
            N = sp.tensor.shape[2]
            row0 = sp.end_row(N) - n
            scatter_rows_mesh(
                model, sp.tensor, {r: (torch.arange(n) + (i * N + row0), per_col[0][0, 0]) for r, i in targets}
            )
            per[name] = round(time.perf_counter() - t_n, 4)
            continue
        key = (
            sp.kind,
            sp.tensor.layout,
            sp.tensor.shape[2],
            sp.tensor.shape[3],
            n,
            sp.replicated,
            sp.end_row(sp.tensor.shape[2]) if sp.kind == "tail" else 0,
            sp.start,
        )
        groups.setdefault(key, []).append((sp, per_col))
    for key, batch in groups.items():
        t_n = time.perf_counter()
        write_batch(model, batch, targets, mask_d)
        per[f"batch[{key[0]},{key[3]},n={key[4]}]x{len(batch)}"] = round(time.perf_counter() - t_n, 4)
    ttnn.synchronize_device(model.md)
    ttnn.deallocate(mask_d)
    t_c = time.perf_counter() - t_c
    # 3. pool latents of the kv-source layers (page tables of the target users; replicated over the columns; the kernel converts to an fp8 pool)
    t_p = time.perf_counter()
    pool = model.pool
    for src, lat in snap.pool.items():
        ar = torch.arange(lat.shape[0])
        scatter_rows_mesh(model, pool.pool, {b // model.U: (pool.phys_rows(b, src, ar).long(), lat) for b in bs})
    t_p = time.perf_counter() - t_p
    model.__dict__["_restore_timing"] = {"carried_s": round(t_c, 3), "pool_s": round(t_p, 3), "per_tensor_s": per}
    # 4. Engram history (host, pure function of the tokens)
    if model.host_rows is not None:
        st = model.hasher.st
        for b in bs:
            st.cache[b, :s] = st.token_map[snap.toks]
    return time.perf_counter() - t0


# ---- host store -------------------------------------------------------------------------------------------------------------------------------------------
class PrefixStore:
    """LRU store of snapshots under a host byte budget. Key = (signature, sha256 of the first s tokens): ``signature`` ties a snapshot to the model / chunk / context configuration
    (the generator hashes the tokens itself, not vLLM's block hashes). ``lookup(tokens, start, end, margin)`` -> the snapshot with the LARGEST s such that s % chunk == 0, s <= start and
    s <= end - margin (the last ``margin`` positions of a prompt are always replayed: ring rows, ``prev_cs`` and the drafter taps come from those chunks), or None. A miss means recompute
    from 0 (always correct)."""

    def __init__(self, chunk, budget_bytes, signature=""):
        self.chunk, self.budget, self.signature = int(chunk), int(budget_bytes), signature
        self.tag = ""  # S_pad context of the carried-state layout (several prefill contexts: a snapshot is only restored into the context it was taken in)
        self.snaps = OrderedDict()  # key -> PrefixSnapshot (oldest first)
        self.bytes = 0
        self.hits = self.misses = self.evictions = self.puts = 0

    def _chain(self, tokens, top):
        """{s: key} for every multiple s of the chunk up to ``top`` (one pass of an incremental sha256)."""
        import hashlib

        t = torch.as_tensor(tokens).long().reshape(-1)[:top].contiguous().numpy()
        h, out = hashlib.sha256(), {}
        for s in range(self.chunk, top + 1, self.chunk):
            h.update(t[s - self.chunk : s].tobytes())
            out[s] = (self.signature, self.tag, s, h.copy().hexdigest())
        return out

    def key(self, tokens, s):
        return self._chain(tokens, s)[s]

    def has(self, tokens, s):
        return self.key(tokens, s) in self.snaps

    def put(self, snap):
        key = self.key(snap.toks, snap.s)
        if key in self.snaps:
            self.snaps.move_to_end(key)
            return False
        if snap.nbytes > self.budget:
            return False
        self.snaps[key] = snap
        self.bytes += snap.nbytes
        self.puts += 1
        while self.bytes > self.budget:
            _, old = self.snaps.popitem(last=False)
            self.bytes -= old.nbytes
            self.evictions += 1
        return True

    def lookup(self, tokens, start, end, margin=128):
        tokens = torch.as_tensor(tokens).long().reshape(-1)
        top = min(int(start), int(end) - int(margin), tokens.numel()) // self.chunk * self.chunk
        if top > 0:
            chain = self._chain(tokens, top)
            for s in range(top, 0, -self.chunk):
                snap = self.snaps.get(chain[s])
                if snap is not None:
                    self.snaps.move_to_end(chain[s])
                    self.hits += 1
                    return snap
        self.misses += 1
        return None

    def clear(self):
        """``reset_prefix_cache`` / a weight reload / a re-captured prefill trace."""
        self.snaps.clear()
        self.bytes = 0


class PrefixCache:
    """The model's prefix caching: a ``PrefixStore`` plus the snapshot policy, driven from ``Model.prefill_interleaved`` (APC_CHUNKED_PREFILL_PLAN.md).

    * restore: a new prompt (no carried state) takes the deepest stored snapshot s <= end - margin (``margin`` = 128, the attention window: the continuation rewrites the last 128 positions: ring rows read by decode, ``prev_cs``, drafter taps), the
      prompt then computes [s, end) only;
    * capture: after a replay window that ended at B (a multiple of the chunk) a user that is not an image prompt is snapshotted when ``policy`` wants B (``pow2``: B / chunk a power of two, i.e.
      512, 1024, 2048, 4096 ... with the default chunk; ``all``; an integer N: the first N boundaries) and the store does not hold it yet (same-step prompts that share a prefix: the first
      user's snapshot serves the others from their next wave on).
    Disabled (``enabled = False``) after the first ``ApcUnsupported`` (dense prefill path ...) or by ``DSV41_APC_FILE`` pointing at a file that contains 0 (a live off switch for A/B runs).
    """

    def __init__(self, model, chunk, budget_bytes, policy="pow2", log=print):
        self.m, self.chunk, self.log, self.policy = model, int(chunk), log, policy
        self.store = PrefixStore(chunk, budget_bytes)
        self.enabled = True
        # the continuation must rewrite the last WINDOW (128) positions of the prompt: the ring rows decode reads, ``prev_cs`` (last token) and the drafter taps. The extra ring rows of a spec build
        # (288 rows = window + slack for speculative writes) hold positions that no attention reads, so they need not be rewritten: margin 128 for both pools.
        import os

        self.margin = int(os.environ.get("DSV41_APC_MARGIN", "128"))
        self.image_token_id = None  # set when image input is on: prompts with image placeholders bypass the cache
        self.stats = dict(restores=0, restored_tokens=0, snapshots=0, snap_s=0.0, restore_s=0.0, skipped_vision=0)

    def active(self):
        import os

        f = os.environ.get("DSV41_APC_FILE")
        if f:
            try:
                if open(f).read().strip() == "0":
                    return False
            except OSError:
                pass
        return self.enabled

    def set_signature(self, sig):
        if sig != self.store.signature:
            self.store.clear()
            self.store.signature = sig

    def clear(self):
        self.finish_pending()
        self.store.clear()

    def wants(self, B):
        k = B // self.chunk
        if self.policy == "all":
            return True
        if self.policy.isdigit():
            return k <= int(self.policy)
        return k > 0 and (k & (k - 1)) == 0

    def _has_image(self, tokens, n):
        return self.image_token_id is not None and bool(
            (torch.as_tensor(tokens).reshape(-1)[:n] == self.image_token_id).any()
        )

    def defer(self):
        """``DSV41_APC_DEFER`` (default 1): snapshots are only STARTED at their boundary (``begin_snapshot``: device reads enqueued, no host wait) and completed by ``finish_pending`` while the device
        replays the next window; the store insertion waits until then (a lookup / the end of the call finish every pending one first).
        """
        import os

        return os.environ.get("DSV41_APC_DEFER", "1") == "1"

    def begin_snapshot(self, b, B, tokens):
        if not self.active() or not self.wants(B):
            return
        if self._has_image(tokens, B):
            self.stats["skipped_vision"] += 1
            return
        key = self.store.key(tokens, B)
        pend = self.__dict__.setdefault("pending", [])
        if key in self.store.snaps or any(k == key for _, k, _ in pend):
            return
        t0 = time.perf_counter()
        try:
            p = snapshot_begin(self.m, b, B, tokens)
        except ApcUnsupported as e:
            self.enabled = False
            self.log(f"prefix caching disabled: {e}")
            return
        pend.append((p, key, (b, B, time.perf_counter() - t0)))

    def finish_pending(self):
        """Complete every started snapshot (host copies, store insertion). -> seconds spent."""
        pend = self.__dict__.get("pending")
        if not pend:
            return 0.0
        t0 = time.perf_counter()
        self.pending = []
        for p, key, (b, B, t_b) in pend:
            snap = snapshot_finish(self.m, p)
            self._store(snap, b, B, t_b, "deferred")
        return time.perf_counter() - t0

    def _store(self, snap, b, B, t_dev, how=""):
        ev0 = self.store.evictions
        stored = self.store.put(snap)
        if not stored and snap.nbytes > self.store.budget:
            self.log(
                f"prefix cache: snapshot at {B} tokens REFUSED ({snap.nbytes / 2**20:.0f} MiB > host budget {self.store.budget / 2**20:.0f} MiB; DSV41_APC_GB)"
            )
        if self.store.evictions > ev0:
            self.log(
                f"prefix cache: {self.store.evictions - ev0} older snapshot(s) EVICTED to make room for {B} tokens (store {self.store.bytes / 2**20:.0f} MiB of {self.store.budget / 2**20:.0f} MiB, {self.store.evictions} evictions so far)"
            )
        self.stats["snapshots"] += 1
        self.stats["snap_s"] += snap.seconds
        self.log(
            f"prefix cache: snapshot of user {b} at {B} tokens ({snap.nbytes / 2**20:.1f} MiB, {snap.seconds:.2f} s {how} [enqueue {t_dev:.3f}]; store {self.store.bytes / 2**20:.0f} MiB, {len(self.store.snaps)} snapshots)"
        )

    def maybe_snapshot(self, b, B, tokens):
        if not self.active() or not self.wants(B):
            return
        if self._has_image(tokens, B):
            self.stats["skipped_vision"] += 1
            return
        t_h0 = time.perf_counter()
        if self.store.has(tokens, B):
            return
        t_h1 = time.perf_counter()
        try:
            snap = snapshot_user(self.m, b, B, tokens)
        except ApcUnsupported as e:
            self.enabled = False
            self.log(f"prefix caching disabled: {e}")
            return
        ev0 = self.store.evictions
        t_p0 = time.perf_counter()
        stored = self.store.put(snap)
        t_p1 = time.perf_counter()
        if not stored and snap.nbytes > self.store.budget:
            self.log(
                f"prefix cache: snapshot at {B} tokens REFUSED ({snap.nbytes / 2**20:.0f} MiB > host budget {self.store.budget / 2**20:.0f} MiB; DSV41_APC_GB)"
            )
        if self.store.evictions > ev0:
            self.log(
                f"prefix cache: {self.store.evictions - ev0} older snapshot(s) EVICTED to make room for {B} tokens (store {self.store.bytes / 2**20:.0f} MiB of {self.store.budget / 2**20:.0f} MiB, {self.store.evictions} evictions so far)"
            )
        self.stats["snapshots"] += 1
        self.stats["snap_s"] += snap.seconds
        self.log(
            f"prefix cache: snapshot of user {b} at {B} tokens ({snap.nbytes / 2**20:.1f} MiB, {snap.seconds:.2f} s [has {t_h1 - t_h0:.3f}, device reads + host copy {t_p0 - t_h1:.3f}, put {t_p1 - t_p0:.3f}]; store {self.store.bytes / 2**20:.0f} MiB, {len(self.store.snaps)} snapshots)"
        )

    def lookup(self, b, tokens, end):
        """-> the deepest usable snapshot for user b's prompt (end = this call's end position), else None"""
        if not self.active():
            return None
        if self._has_image(tokens, end):
            self.stats["skipped_vision"] += 1
            return None
        return self.store.lookup(tokens, end, end, margin=self.margin)

    def restore_group(self, snap, bs):
        t = restore_users(self.m, snap, bs)
        self.stats["restores"] += len(bs)
        self.stats["restored_tokens"] += snap.s * len(bs)
        self.stats["restore_s"] += t
        self.log(
            f"prefix cache: users {bs} restored at {snap.s} tokens in {t:.2f} s ({self.m.__dict__.get('_restore_timing')})"
        )
        return t
