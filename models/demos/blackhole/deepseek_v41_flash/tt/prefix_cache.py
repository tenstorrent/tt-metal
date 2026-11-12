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


def scatter_rows_mesh(model, t, per_row, base=0, sync_every=1, stats=None, reuse=False):
    """In place: ``t[ids + base] = data`` on the mesh rows of ``per_row`` {mesh row: (ids [n] torch, data [n, D] torch)} only (the other mesh rows get SKIP ids), t a bf16 (or, for the pool, fp8: the
    kernel converts) row-major tensor whose rows are D wide; fixed-size pieces."""
    md, rows, cols = model.md, model.rows, model.cols
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    n, D = next(iter(per_row.values()))[1].shape
    live = (
        []
    )  # (d_src, d_ids) of the pieces enqueued since the last synchronisation (``sync_every`` > 1: DSV41_XROW_SYNC_EVERY, the host prepares the next piece while the device writes this one)
    bufs = []
    for a in range(0, n, PIECE):
        _t0 = time.perf_counter()
        m = min(PIECE, n - a)
        if (
            reuse and bufs
        ):  # (DSV41_XROW_BATCH: the host buffers are allocated once; only the rows of the target mesh rows are refilled, the padding of a short last piece re-initialised)
            ids, src = bufs
            if m < PIECE:
                ids.fill_(SKIP)
                src.zero_()
        else:
            ids = torch.full((rows, 1, PIECE), SKIP, dtype=torch.int64)
            src = torch.zeros(rows, 1, PIECE, D, dtype=torch.bfloat16)
            if reuse:
                bufs.extend((ids, src))
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
        _t1 = time.perf_counter()
        P.paged_scatter_rows(t, d_src, d_ids)
        _t2 = time.perf_counter()
        live.append((d_src, d_ids))
        if stats is not None:
            stats["prep_s"] = stats.get("prep_s", 0.0) + (_t1 - _t0)  # host: torch buffers + from_torch uploads
            stats["enq_s"] = stats.get("enq_s", 0.0) + (_t2 - _t1)  # host: the scatter launch
        if len(live) >= max(1, int(sync_every)):
            _t3 = time.perf_counter()
            ttnn.synchronize_device(md)  # (the device buffers are freed here)
            if stats is not None:
                stats["sync_s"] = stats.get("sync_s", 0.0) + (time.perf_counter() - _t3)
            for x, y in live:
                ttnn.deallocate(x)
                ttnn.deallocate(y)
            live = []
    if live:
        ttnn.synchronize_device(md)
        for x, y in live:
            ttnn.deallocate(x)
            ttnn.deallocate(y)


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
    chk = os.environ.get("DSV41_APC_ADDRCHK") == "1"
    for j, (sp, per_col) in enumerate(batch):
        t = sp.tensor
        a_before = t.buffer_address() if chk else None
        new_j = ttnn.slice(big, [0, 0, j * rn, 0], [Up, 1, (j + 1) * rn, D]) if len(batch) > 1 else big
        if tail and os.environ.get("DSV41_APC_INPLACE", "1") == "1":
            # IN PLACE: ``slice_write`` into a tile-layout carried tensor converts it to row-major and back, i.e. RE-ALLOCATES it (found by TT_METAL_TRACE_ALLOC_TRACKING: 4 buffers allocated
            # here, alive at the next replay): the captured prefill trace keeps reading / writing the OLD address of the halo / key FIFO / latent FIFO, so a restored prefix was invisible to
            # the trace (and the freed old buffer was reused). The slot's rows are rebuilt as a whole tensor and copied into the existing buffer like the non-tail kinds (``ttnn.copy``).
            row0 = sp.row0(t.shape[2], n)
            parts = []
            if row0 > 0:
                parts.append(ttnn.slice(t, [0, 0, 0, 0], [Up, 1, row0, D]))
            parts.append(new_j)
            if row0 + n < t.shape[2]:
                parts.append(ttnn.slice(t, [0, 0, row0 + n, 0], [Up, 1, t.shape[2], D]))
            full_new = ttnn.concat(parts, dim=2) if len(parts) > 1 else new_j
            upd = _where(mask_d, full_new, t)
            ttnn.copy(upd, t)
            extra = tuple(x for x in parts if x is not new_j) + ((full_new,) if full_new is not new_j else ())
        elif tail:
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
        if (
            chk
        ):  # DSV41_APC_ADDRCHK=1: a persistent carried tensor must keep its device address (the captured prefill trace reads / writes it there)
            a_after = t.buffer_address()
            print(
                f"APC-ADDR {sp.name} {'tail' if tail else 'full'} {'MOVED' if a_after != a_before else 'same'} {a_before} -> {a_after}",
                flush=True,
            )
    ttnn.deallocate(big)


@dataclass
class PrefixSnapshot:
    s: int
    toks: torch.Tensor  # int64 [s]
    carried: dict = field(default_factory=dict)  # name -> [cols] torch (slot dim kept, trimmed rows)
    pool: dict = field(default_factory=dict)  # src -> torch bf16 [n_entries, 512] latents of the kv-source layer
    seconds: float = 0.0
    # bounded replay (DSV41_PF_REPLAY): False when the user ran only layers 0..20 on the chunks before ``s`` (the halos of layers >= 21 are stale: the snapshot may only be restored when the continuation starts
    # at or before the first replay chunk of its own prompt, which ignores those halos). True: every layer ran on every chunk (the usual snapshot).
    full: bool = True
    # prefix-cache KEY tokens when the prompt has images (DSV41_APC_IMAGES): ``toks`` with the positions of every image span replaced by a value derived from the image's content hash
    # (``key_tokens``): two prompts with the same text and different images get different keys. None: the key is made from ``toks`` (text prompts: unchanged).
    ktoks: object = None
    # delta snapshots (DSV41_APC_DELTA=1, ``DeltaStore``): ``s_lo`` = the read covered only the rows of [s_lo, s) of the position-indexed tensors (the tail kinds and the pool latents); once the
    # record is in a ``DeltaStore`` those rows live in refcounted ``Block``s (``bkeys``: the block chain keys of [0, s)) and ``carried`` keeps only the ``full`` kinds (the halos).
    s_lo: int = 0
    bkeys: object = None
    tails: tuple = ()  # names of the position-indexed (``tail``) carried tensors (their rows go into blocks)
    pool_alias: bool = False  # dense prefill: ``pool`` is a view of the kv-table rows in ``carried`` (stored once)

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
    if (
        Up == model.U or model.__dict__.get("_slot_k") is not None
    ):  # (slot bucket call: slot = in-row index of the user, the tensors are the k-slot views of the Up = U state)
        return b // model.U, b % model.U
    sl = model.__dict__.get("pf_slot_of", {}).get(b)
    assert sl is not None, f"user {b} has no prefill slot"
    return sl // Up, sl % Up


class ApcUnsupported(RuntimeError):
    """the model build has no carried state the snapshot code knows (dense prefill path, fp8 kv table): prefix caching stays off"""


def fp8_pool_ok():
    """DSV41_FP8_SERVING=1: an fp8 KV pool is a supported APC format (snapshots read it with ``paged_gather_rows``; the restore writes through ``paged_scatter_rows``, which converts: an e4m3 value read back
    as bf16 converts back to the same byte, so a restored prefix equals the cold one)."""
    return os.environ.get("DSV41_FP8_SERVING", "0") == "1"


def _check_pool(pool):
    assert pool.dtype == ttnn.bfloat16 or (
        pool.dtype == ttnn.fp8_e4m3 and fp8_pool_ok()
    ), "APC on the paged prefill needs the bf16 pool (an fp8 pool with DSV41_FP8_SERVING=1)"


def _gather_pool(pool, d_ids):
    """pool rows ``d_ids`` ([1,1,1,m] uint32 per device) -> bf16 [1,1,m,512]: the embedding lookup for a bf16 pool (as shipped), the gather kernel for an fp8 pool (embedding needs a bf16 table)."""
    if pool.dtype == ttnn.bfloat16:
        return ttnn.embedding(d_ids, pool.pool, layout=ttnn.ROW_MAJOR_LAYOUT)
    return P.paged_gather_rows(pool.pool, ttnn.reshape(d_ids, [1, int(d_ids.shape[-1])]))


def read_pool_latents(model, b, src, n, lo=0):
    """Paged prefill: entries [0, n) of kv-source layer ``src`` of user ``b`` -> torch bf16 [n, 512], gathered ON the device from the user's pool pages (``ttnn.embedding`` over the
    bf16 pool, row ids of the user's mesh row; pieces of 16384 rows) and read from one device of the mesh row."""
    pool, md, rows, cols = model.pool, model.md, model.rows, model.cols
    _check_pool(pool)
    r = b // model.U
    phys = pool.phys_rows(b, src, torch.arange(lo, n)).long()  # (``lo``: delta snapshots read only the entries [lo, n))
    n = n - lo
    if n <= 0:
        return torch.zeros(0, 512, dtype=torch.bfloat16)
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
        g = _gather_pool(pool, d_ids)  # [1, 1, m, 512] bf16 per device
        out.append(ttnn.to_torch(ttnn.get_device_tensors(g)[r * cols]).reshape(m, 512).clone())
        ttnn.deallocate(g)
        ttnn.deallocate(d_ids)
    return torch.cat(out) if len(out) > 1 else out[0]


def gather_pool_rows(model, r, phys):
    """pool rows ``phys`` (torch long [n], physical row ids of mesh row ``r``) -> torch bf16 [n, 512]; pieces of 16384 rows, read from one chip of the row (an fp8 pool is converted to bf16)"""
    pool, md, rows, cols = model.pool, model.md, model.rows, model.cols
    _check_pool(pool)
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    outs = []
    for a in range(0, int(phys.numel()), 16384):
        m = min(16384, int(phys.numel()) - a)
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
        g = _gather_pool(pool, d_ids)
        outs.append(ttnn.to_torch(ttnn.get_device_tensors(g)[r * cols]).reshape(m, 512).clone())
        ttnn.deallocate(g)
        ttnn.deallocate(d_ids)
    return torch.cat(outs) if len(outs) > 1 else outs[0]


def read_pool_latents_begin(model, b, src, n, lo=0):
    """``read_pool_latents`` without waiting: [(rows m, host tensor handle)] of the gathers (pieces of 16384 rows), enqueued on CQ 0 and freed right after their reads (see ``snapshot_begin``)."""
    pool, md, rows, cols = model.pool, model.md, model.rows, model.cols
    _check_pool(pool)
    r = b // model.U
    phys = pool.phys_rows(b, src, torch.arange(lo, n)).long()
    n = n - lo
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
        g = _gather_pool(pool, d_ids)  # [1, 1, m, 512] bf16 per device
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


def snapshot_user(model, b, s, toks, s_lo=0):
    """-> PrefixSnapshot (see ``_snapshot_user``). ``DSV41_APC_SNAP_CHECK=1`` (debug): also takes the blocking snapshot and asserts both are bit-identical."""
    nb = _snapshot_nb()
    snap = _snapshot_user(model, b, s, toks, nb, s_lo)
    if nb and os.environ.get("DSV41_APC_SNAP_CHECK") == "1":
        ref = _snapshot_user(model, b, s, toks, False, s_lo)
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


def _snapshot_user(model, b, s, toks, nb, s_lo=0):
    """State of user ``b`` right after a step that ended at position ``s`` (``model.pf_resume[b] == s``) -> PrefixSnapshot. ``toks``: the user's tokens [>= s]. The pool latents of the kv-source layers
    are NOT read from the pool (an fp8 pool, and reading it would mean the whole per-device pool): the kv table of a source layer holds the same bf16 latents (entry j = FIFO row LAT - n + j).
    """
    t0 = time.perf_counter()
    assert model.__dict__.get("pf_resume", {}).get(b) == s, f"user {b} is not resumable at {s}"
    r, i = _slot_of(model, b)
    cols = model.cols
    snap = PrefixSnapshot(s=s, toks=torch.as_tensor(toks).long().reshape(-1)[:s].clone(), s_lo=s_lo)
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
        if sp.kind == "tail" and s_lo:  # delta snapshot (DSV41_APC_DELTA): only the rows of [s_lo, s)
            lo += sp.count(s_lo)
        # slice the slot (and, for a FIFO, its valid tail) ON DEVICE first: reading a whole padded table per tensor is what made a snapshot cost ~0.5 s regardless of its size
        whole = (
            sp.tensor.shape[0] == 1 and lo == 0 and end == N
        )  # (a full slice returns the persistent tensor itself: never free it)
        pieces = [[] for _ in range(1 if sp.replicated else cols)]
        if lo >= end:  # (delta snapshot with every row already held by cached blocks)
            for c in range(len(pieces)):
                pieces[c].append(torch.zeros(1, 1, 0, D, dtype=torch.bfloat16))
        elif whole:
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
        if sp.kind == "tail":
            snap.tails += (name,)
    for src in _sources(model):
        if paged:  # no table: the latents are read from the user's pool pages
            snap.pool[src] = read_pool_latents(model, b, src, s // P.SRC_RATIO[src], s_lo // P.SRC_RATIO[src])
        else:
            snap.pool[src] = snap.carried[f"L{src}.kvt"][0][0, 0]  # [s / ratio, 512], shared with the carried copy
            snap.pool_alias = True
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


def snapshot_begin(model, b, s, toks, s_lo=0):
    """Deferred form of ``snapshot_user`` (``DSV41_APC_DEFER=1``): enqueue the device slices / gathers and the NON-blocking reads of user ``b``'s state at position ``s`` on CQ 0 and return a
    ``PendingSnapshot``; the host does not wait. Everything is ordered by the command queue: the reads precede the next replay (which may modify the state), a device slice is freed right after its
    read is enqueued (the next program that reuses the memory runs after the read), so no device memory is alive across the next replay (a trace's intermediates share the freed DRAM). The host
    copies are made by ``snapshot_finish`` after ``event_synchronize`` (GIL-free wait for the reads only, the next replay may still be running).
    """
    t0 = time.perf_counter()
    assert model.__dict__.get("pf_resume", {}).get(b) == s, f"user {b} is not resumable at {s}"
    r, i = _slot_of(model, b)
    cols = model.cols
    snap = PrefixSnapshot(s=s, toks=torch.as_tensor(toks).long().reshape(-1)[:s].clone(), s_lo=s_lo)
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
        if sp.kind == "tail" and s_lo:  # delta snapshot (DSV41_APC_DELTA): only the rows of [s_lo, s)
            lo += sp.count(s_lo)
        whole = sp.tensor.shape[0] == 1 and lo == 0 and end == N
        pieces = [[] for _ in range(1 if sp.replicated else cols)]
        if lo >= end:  # (delta snapshot with every row already held by cached blocks)
            for c in range(len(pieces)):
                pieces[c].append(torch.zeros(1, 1, 0, D, dtype=torch.bfloat16))
        elif whole:
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
            pool_parts[src] = read_pool_latents_begin(model, b, src, s // P.SRC_RATIO[src], s_lo // P.SRC_RATIO[src])
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
        if sp.kind == "tail":
            snap.tails += (name,)
    for src in _sources(model):
        if src in pend.pool_parts:
            parts = [ttnn.to_torch(h).reshape(m, 512).clone() for m, h in pend.pool_parts[src]]
            snap.pool[src] = (
                torch.cat(parts) if len(parts) > 1 else parts[0] if parts else torch.zeros(0, 512, dtype=torch.bfloat16)
            )
        else:
            snap.pool[src] = snap.carried[f"L{src}.kvt"][0][0, 0]
            snap.pool_alias = True
    snap.seconds = pend.t_begin + time.perf_counter() - t0
    pend.handles, pend.pool_parts, pend.staged = [], {}, OrderedDict()
    return snap


# ---- decode-output snapshots (DSV41_APC_DECODE=1, needs the delta store) --------------------------------------------------------------------------------
# State of a user at the END of its decode, converted to the carried prefill state at a chunk multiple s <= the tokens the model holds:
#   pool latents      the same pool rows the prefill sink writes (decode appends them): ``read_pool_latents`` as for a prefill snapshot;
#   index keys        decode slab ``dec.k_cache`` [U,1,nl,128] per chip, entry e on column e % cols at slot e // cols = the single-trace prefill slab (global slot Cc + e on column p % cols): the same
#                     per-column rows, read as they are;
#   halo (kv rows of the positions s-128..s-1, 40 layers)   only the sliding-window RING survives decode, so the halo is gathered from the ring when the committed length crosses s
#                     (``decode_halo_begin``, one non-blocking gather), the latest crossing per user is kept.
def _halo_names(model):
    return [
        (n, int(n[1:].split(".")[0]))
        for n, sp in carried_specs(model).items()
        if sp.kind == "full" and n.endswith(".halo")
    ]


def decode_halo_begin(model, b, s):
    """Enqueue the gather of the 128 kv rows [s - 128, s) of every layer's ring of user ``b`` (bf16 whatever the pool dtype) and the non-blocking read of the result from one chip of its mesh row.
    -> (s, {layer id: slot into the handle}, host tensor handle [1, 1, nlayers * 128, 512], event)"""
    pool, md, rows, cols = model.pool, model.md, model.rows, model.cols
    names = _halo_names(model)
    r, u, R = b // pool.T, b % pool.T, pool.ring_rows
    pos = (s - 128 + torch.arange(128)) % R
    ids_l = []
    for _, lid in names:
        base = pool.ring_base(model.attns[lid].ring_slot) + u * R
        ids_l.append(base + pos)
    ids = torch.cat(ids_l)
    m = int(ids.numel())
    idm = torch.zeros(rows, 1, 1, m, dtype=torch.int64)
    idm[r, 0, 0] = ids
    d_ids = ttnn.from_torch(
        idm.to(torch.int32),
        device=md,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )
    g = _gather_pool(pool, d_ids)
    h = ttnn.from_device(ttnn.get_device_tensors(g)[r * cols], blocking=False)
    ttnn.deallocate(g)
    ttnn.deallocate(d_ids)
    ev = ttnn.record_event(md, 0)
    return s, {lid: i for i, (_, lid) in enumerate(names)}, h, ev


def decode_keys_rows(model, b, L, q0, q1):
    """rows [q0, q1) of the single-trace slab layout (per mesh column c: row q holds entry q * cols + c) of the keys of user ``b``, index-key owner layer ``L``, from the DECODE slab: one torch bf16
    [1, 1, q1 - q0, 128] per mesh column. A column-sharded decode slab (entry e on column e % cols, slot e // cols) is the same layout: rows are read as they are; a replicated slab (entry e at row e)
    is read from one chip and re-interleaved on the host."""
    dec = model.dec_idx[L]
    r, u, cols = b // model.U, b % model.U, model.cols
    if q1 <= q0:
        return [torch.zeros(1, 1, 0, 128, dtype=torch.bfloat16) for _ in range(cols)]
    if getattr(dec, "cs", False):
        part = ttnn.slice(dec.k_cache, [u, 0, q0, 0], [u + 1, 1, q1, 128])
        out = [
            ttnn.to_torch(ttnn.get_device_tensors(part)[r * cols + c])
            .to(torch.bfloat16)
            .reshape(1, 1, q1 - q0, 128)
            .clone()
            for c in range(cols)
        ]
        ttnn.deallocate(part)
        return out
    part = ttnn.slice(dec.k_cache, [u, 0, q0 * cols, 0], [u + 1, 1, q1 * cols, 128])
    ent = ttnn.to_torch(ttnn.get_device_tensors(part)[r * cols]).to(torch.bfloat16).reshape(-1, 128)
    ttnn.deallocate(part)
    return [ent[c::cols].reshape(1, 1, q1 - q0, 128).clone() for c in range(cols)]


def decode_unsupported(model):
    """-> why the decode state of this build cannot be converted (else '')"""
    specs = carried_specs(model)
    if not _sources(model) or not PG.read_on():
        return "needs the paged prefill read mode (latents in the pool)"
    for name, sp in specs.items():
        if sp.kind == "tail":
            if not (name.endswith(".keys") and sp.start >= 0):
                return f"{name} is not a single-trace key slab"
            L = int(name[1:].split(".")[0])
            if model.dec_idx.get(L) is None or L not in model.index_owner:
                return f"layer {L} has no decode key slab"
        elif not name.endswith(".halo"):
            return f"carried tensor {name} has no decode source"
    for _, lid in _halo_names(model):
        if getattr(getattr(model, "attns", {}).get(lid), "ring_slot", None) is None:
            return f"layer {lid} has no ring slot"
    return ""


def decode_snapshot(model, b, s, toks, s_lo, halo):
    """-> PrefixSnapshot (delta read of [s_lo, s)) of user ``b`` from its DECODE state; ``halo`` = what ``decode_halo_begin`` returned for ``s`` (the handle is read here)"""
    t0 = time.perf_counter()
    cols = model.cols
    hs, slot_of, h, ev = halo
    assert hs == s
    ttnn.event_synchronize(ev)
    hal = ttnn.to_torch(h).reshape(-1, 128, 512).clone()
    snap = PrefixSnapshot(s=s, toks=torch.as_tensor(toks).long().reshape(-1)[:s].clone(), s_lo=s_lo)
    for name, sp in carried_specs(model).items():
        if sp.kind == "full":
            lid = int(name[1:].split(".")[0])
            snap.carried[name] = [hal[slot_of[lid]].reshape(1, 1, 128, 512)] * cols
        else:
            L = int(name[1:].split(".")[0])
            q0, q1 = sp.count(s_lo), sp.count(s)
            snap.carried[name] = decode_keys_rows(model, b, L, q0, q1)
            snap.tails += (name,)
    for src in _sources(model):
        snap.pool[src] = read_pool_latents(model, b, src, s // P.SRC_RATIO[src], s_lo // P.SRC_RATIO[src])
    snap.seconds = time.perf_counter() - t0
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
        if model.Up != model.U and model.__dict__.get("_slot_k") is None:
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
        row = st.token_map[snap.toks]
        iid = getattr(getattr(model, "apc", None), "image_token_id", None)
        if (
            iid is not None and PrefixCache.images_enabled()
        ):  # image positions are cached as DEAD by the live path (``token_mask`` of the Engram hasher): no n-gram spans an image
            row = torch.where(snap.toks == iid, torch.full_like(row, int(st.DEAD)), row)
        for b in bs:
            st.cache[b, :s] = row
    return time.perf_counter() - t0


def delta_on():
    """``DSV41_APC_DELTA=1``: block / delta host snapshots (``DeltaStore``). Default 0 (whole-prefix snapshots, unchanged)."""
    return os.environ.get("DSV41_APC_DELTA", "0") == "1"


def delta_unsupported(model, chunk):
    """-> a reason string when the model's carried state cannot be stored as chunk blocks, else ''."""
    from models.demos.blackhole.deepseek_v41_flash.tt import replay_cfg

    for name, sp in carried_specs(model).items():
        if sp.kind != "tail":
            continue
        if chunk % (sp.ratio * sp.div):
            return f"{name}: chunk {chunk} is not a multiple of ratio {sp.ratio} x cols {sp.div}"
        # bounded replay: layers >= 21 do not run on the chunks before the zone, so a snapshot there is 'partial'; its position-indexed rows are shared between records only when they all come
        # from layers that run on every chunk (the kv sources and index-key owners are layers 2 / 8 / 14 / 20)
        lid = int(name[1:].split(".")[0]) if name.startswith("L") and name[1:2].isdigit() else -1
        if replay_cfg.split_enabled() and lid >= 21:
            return f"{name} belongs to a layer that the bounded replay skips on early chunks"
    for src in _sources(model):
        if chunk % P.SRC_RATIO[src]:
            return f"pool source {src}: chunk {chunk} is not a multiple of ratio {P.SRC_RATIO[src]}"
    return ""


def image_spans(vs):
    """[(start, length, content hash)] of the images of a ``VisionSeq`` (None / without keys -> None)."""
    if vs is None or getattr(vs, "keys", None) is None:
        return None
    return [(int(st), int(r.shape[0]), k) for (st, r), k in zip(vs.spans, vs.keys)]


def key_tokens(tokens, spans, image_token_id):
    """Key tokens of a prompt with images: ``tokens`` (1-D) where every image span [start, start + n) holds ONE value derived from the image's content hash (a negative int64, so it never equals
    a vocabulary id) instead of the shared placeholder id. -> (key tokens, ok); ok is False when the prompt has image placeholders that no span explains (never cache such a prompt).
    """
    import hashlib

    t = torch.as_tensor(tokens).long().reshape(-1).clone()
    covered = 0
    for st, n, k in spans or []:
        if st >= t.numel():
            continue  # (a span that starts at or after the end of the tokens given is not part of any key below)
        v = -1 - int.from_bytes(
            hashlib.sha256(bytes.fromhex(k) if len(k) % 2 == 0 else k.encode()).digest()[:7], "little"
        )
        t[
            st : st + n
        ] = v  # (a span that crosses the end: its first part: the image identity is in the key, the position in the chain)
        covered += min(n, t.numel() - st)
    ok = int((t == image_token_id).sum()) == 0 if image_token_id is not None else True
    return t, ok


def straddles(spans, B):
    """Does an image span cross position B (start < B < start + n)? No snapshot may be taken at such a position (the state would hold half an image)."""
    return any(st < B < st + n for st, n, _ in spans or [])


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

    def has(self, tokens, s, full=True):
        old = self.snaps.get(self.key(tokens, s))
        return old is not None and (old.full or not full)

    def put(self, snap):
        key = self.key(snap.toks if snap.ktoks is None else snap.ktoks, snap.s)
        if key in self.snaps:
            old = self.snaps[key]
            if old.full or not snap.full:
                self.snaps.move_to_end(key)
                return False
            self.bytes -= old.nbytes  # (a full snapshot replaces a partial one of the same prefix)
            del self.snaps[key]
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

    def lookup(self, tokens, start, end, margin=128, zone=None):
        """``zone`` (bounded replay): first replay chunk start of the prompt; a partial snapshot (``full`` False) is only usable at s <= zone."""
        tokens = torch.as_tensor(tokens).long().reshape(-1)
        top = min(int(start), int(end) - int(margin), tokens.numel()) // self.chunk * self.chunk
        if top > 0:
            chain = self._chain(tokens, top)
            for s in range(top, 0, -self.chunk):
                snap = self.snaps.get(chain[s])
                if snap is not None and not snap.full and zone is not None and s > zone:
                    snap = None
                if snap is not None:
                    self.snaps.move_to_end(chain[s])
                    self.hits += 1
                    return snap
        self.misses += 1
        return None

    def assemble(self, snap):
        """full-form snapshot of a stored one (identity here; ``DeltaStore`` concatenates its blocks)"""
        return snap

    def drop(self, key):
        """remove one snapshot (thinning); -> its record"""
        old = self.snaps.pop(key)
        self.bytes -= old.nbytes
        return old

    def clear(self):
        """``reset_prefix_cache`` / a weight reload / a re-captured prefill trace."""
        self.snaps.clear()
        self.bytes = 0


@dataclass
class Block:
    """The position-indexed state of ONE chunk (``chunk`` tokens) of a prompt: ``carried[name]`` = [cols] (or [1] when replicated) torch [1, 1, rows, D] of every ``tail`` tensor, ``pool[src]`` =
    torch [rows, 512] latents of every kv source (paged pool only). ``refs`` = number of snapshot records whose chain contains the block.
    """

    carried: dict
    pool: dict
    refs: int = 0

    @property
    def nbytes(self):
        n = 0
        for v in self.carried.values():
            n += sum(x.numel() * x.element_size() for x in v)
        return n + sum(x.numel() * x.element_size() for x in self.pool.values())


class DeltaStore(PrefixStore):
    """``PrefixStore`` whose snapshots are DELTAS (``DSV41_APC_DELTA=1``): the position-indexed state (compressed latents of the pool, index keys, kv-table rows: append-only, entry j is a function of
    the first (j + 1) * ratio tokens) is stored in content-addressed blocks of one chunk, keyed by the token chain at the block END, shared by every snapshot whose prefix contains them (refcounted);
    a snapshot record keeps only what is NOT additive: the halos (128 raw kv rows per layer, ~5 MiB), the tokens and the chain of block keys. A snapshot of a 64k prompt that extends a stored
    one by 4k tokens costs the halo plus 4k tokens of blocks instead of the whole 64k. Eviction is LRU over records; a block is freed with its last record. ``lookup`` / ``has`` / ``put`` keep the
    ``PrefixStore`` contract; ``assemble`` turns a record into the full-form ``PrefixSnapshot`` that ``restore_users`` takes (logical rows concatenated, so the layout of the device FIFO, which
    depends on s, never enters the store)."""

    def __init__(self, chunk, budget_bytes, signature=""):
        super().__init__(chunk, budget_bytes, signature)
        self.blocks = {}  # block chain key -> Block
        self.lost = 0  # records dropped at put because a block they relied on was evicted meanwhile

    def block_keys(self, tokens, s):
        ch = self._chain(tokens, s)
        return [ch[x] for x in range(self.chunk, s + 1, self.chunk)]

    def have_prefix(self, keys, extra=()):
        """largest n such that blocks 0..n-1 of the chain ``keys`` are all held (or in ``extra``) -> n"""
        n = 0
        for k in keys:
            if k in self.blocks or k in extra:
                n += 1
            else:
                break
        return n

    def _split(self, snap, keys):
        """the blocks of ``snap`` (a delta read of [s_lo, s)) not held yet -> {key: Block}, and move the rows out of ``snap.carried`` / ``snap.pool``"""
        C, nb0 = self.chunk, snap.s_lo // self.chunk
        nb = len(keys) - nb0
        out = {}
        if nb <= 0:
            return out
        parts = {}
        for name in snap.tails:
            v = snap.carried[name]
            rows = v[0].shape[2]
            assert rows % nb == 0, f"{name}: {rows} delta rows do not split into {nb} blocks"
            parts[name] = (v, rows // nb)
        pparts = {}
        if not snap.pool_alias:
            for src, lat in snap.pool.items():
                assert lat.shape[0] % nb == 0, f"pool {src}: {lat.shape[0]} delta rows do not split into {nb} blocks"
                pparts[src] = (lat, lat.shape[0] // nb)
        for j in range(nb):
            key = keys[nb0 + j]
            if key in self.blocks:
                continue
            car = {}
            for name, (v, m) in parts.items():
                if len({id(x) for x in v}) == 1:  # replicated over the columns: one host copy
                    car[name] = [v[0][:, :, j * m : (j + 1) * m].clone()]
                else:
                    car[name] = [x[:, :, j * m : (j + 1) * m].clone() for x in v]
            pool = {src: lat[j * m : (j + 1) * m].clone() for src, (lat, m) in pparts.items()}
            out[key] = Block(car, pool)
        return out

    def put(self, snap):
        key = self.key(snap.toks if snap.ktoks is None else snap.ktoks, snap.s)
        old = self.snaps.get(key)
        if old is not None and (old.full or not snap.full):
            self.snaps.move_to_end(key)
            return False
        keys = self.block_keys(snap.toks if snap.ktoks is None else snap.ktoks, snap.s)
        new = self._split(snap, keys)
        if any(k not in self.blocks and k not in new for k in keys):
            self.lost += 1  # (a block read as present at ``begin`` was evicted before ``finish``)
            return False
        for name in snap.tails:
            snap.carried.pop(name, None)
        snap.pool = {}
        own = snap.nbytes
        add = own + sum(b.nbytes for b in new.values())
        if add > self.budget:
            return False
        for k, b in new.items():
            self.blocks[k] = b
            self.bytes += b.nbytes
        for k in keys:
            self.blocks[k].refs += 1
        if old is not None:
            self.drop(key)  # (after the new refs: the blocks the two share survive)
        snap.bkeys = keys
        self.snaps[key] = snap
        self.bytes += own
        self.puts += 1
        while self.bytes > self.budget and len(self.snaps) > 1:
            k0 = next(iter(self.snaps))
            if k0 == key:
                break
            self.drop(k0)
            self.evictions += 1
        if self.bytes > self.budget:  # (the record's own chain alone does not fit)
            self.drop(key)
            self.puts -= 1
            return False
        return True

    def drop(self, key):
        old = self.snaps.pop(key)
        self.bytes -= old.nbytes
        for k in old.bkeys or ():
            b = self.blocks[k]
            b.refs -= 1
            if b.refs <= 0:
                self.bytes -= b.nbytes
                del self.blocks[k]
        return old

    def assemble(self, snap):
        """record -> full-form ``PrefixSnapshot`` (what ``restore_users`` reads); shares the halo tensors, concatenates the block rows"""
        if snap.bkeys is None:
            return snap
        blks = [self.blocks[k] for k in snap.bkeys]
        carried = dict(snap.carried)
        pool = {}
        if blks:
            for name in blks[0].carried:
                ncol = len(blks[0].carried[name])
                cat = [torch.cat([b.carried[name][c] for b in blks], dim=2) for c in range(ncol)]
                carried[name] = cat
            for src in blks[0].pool:
                pool[src] = torch.cat([b.pool[src] for b in blks])
        if snap.pool_alias:  # dense prefill: the latents ARE the rows of the kv table
            from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P_

            for src in P_.SOURCES:
                if f"L{src}.kvt" in carried:
                    pool[src] = carried[f"L{src}.kvt"][0][0, 0]
        return PrefixSnapshot(
            s=snap.s,
            toks=snap.toks,
            carried=carried,
            pool=pool,
            full=snap.full,
            ktoks=snap.ktoks,
            pool_alias=snap.pool_alias,
        )

    def clear(self):
        super().clear()
        self.blocks.clear()


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
        self.delta = (
            delta_on()
        )  # DSV41_APC_DELTA=1: block / delta snapshots (``DeltaStore``); checked against the model's carried tensors at the first snapshot
        self.store = DeltaStore(chunk, budget_bytes) if self.delta else PrefixStore(chunk, budget_bytes)
        self.delta_step = int(os.environ.get("DSV41_APC_DELTA_STEP", "0")) if self.delta else 0
        self.decode = (
            self.delta and os.environ.get("DSV41_APC_DECODE", "0") == "1"
        )  # snapshot at the end of decode (needs the delta store)
        self.dec_lag = int(
            os.environ.get("DSV41_APC_DECODE_LAG", "100")
        )  # a halo is only captured while the committed length is at most this far past the chunk multiple (ring slack)
        self.dec = {}
        self.enabled = True
        # the continuation must rewrite the last WINDOW (128) positions of the prompt: the ring rows decode reads, ``prev_cs`` (last token) and the drafter taps. The extra ring rows of a spec build
        # (288 rows = window + slack for speculative writes) hold positions that no attention reads, so they need not be rewritten: margin 128 for both pools.

        self.margin = int(os.environ.get("DSV41_APC_MARGIN", "128"))
        self.image_token_id = None  # set when image input is on: prompts with image placeholders bypass the cache
        self.stats = dict(restores=0, restored_tokens=0, snapshots=0, snap_s=0.0, restore_s=0.0, skipped_vision=0)
        self.grid = None  # APC M0' (tt/apc_grid.py): DSV41_APC_GRID=512 adds a tip / divergence snapshot per prompt
        self.plans, self.serial = {}, 0
        if os.environ.get("DSV41_APC_GRID", "0") not in ("", "0"):
            from models.demos.blackhole.deepseek_v41_flash.tt.apc_grid import GridPolicy

            g = int(os.environ["DSV41_APC_GRID"])
            if g == self.chunk and self.defer():
                self.grid = GridPolicy(self.chunk, self.margin, tips=int(os.environ.get("DSV41_APC_GRID_TIPS", "2")))
                self.grid.div_extra = int(os.environ.get("DSV41_APC_GRID_LADDER", "0"))
            else:
                self.log(
                    f"prefix cache: DSV41_APC_GRID={g} ignored (needs grid == chunk {self.chunk} and DSV41_APC_DEFER=1)"
                )

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
        if (
            self.delta_step and k > 0 and k % self.delta_step == 0
        ):  # (delta snapshots cost the halo + the new blocks: a finer grid is affordable)
            return True
        if self.policy == "all":
            return True
        if self.policy.isdigit():
            return k <= int(self.policy)
        return k > 0 and (k & (k - 1)) == 0

    @staticmethod
    def images_enabled():
        """``DSV41_APC_IMAGES=1``: prompts with images use / feed the cache (keys carry the image content hashes). Default 0: image prompts bypass the cache."""
        import os

        return os.environ.get("DSV41_APC_IMAGES", "0") == "1"

    @staticmethod
    def strict_boundaries():
        """``DSV41_APC_IMAGES_STRICT=1``: no snapshot inside an image span (default 0: a state in the middle of an image is a function of the tokens and the image like any other; without it a
        prompt of many adjacent images has almost no usable boundary, the 512-token chunk grid falls inside an image nearly everywhere).
        """
        import os

        return os.environ.get("DSV41_APC_IMAGES_STRICT", "0") == "1"

    def _keyed(self, tokens, n, vs, B=None):
        """-> the tokens [:n] to KEY with (text prompts: ``tokens`` itself), or None when this prompt must not use the cache here (image prompt with the switch off, no image keys, an image
        span crossing ``B`` (snapshot position) or placeholders no span explains)."""
        if not self._has_image(tokens, n):
            return tokens
        if not self.images_enabled():
            return None
        spans = image_spans(vs)
        if spans is None or (B is not None and self.strict_boundaries() and straddles(spans, B)):
            return None
        kt, ok = key_tokens(torch.as_tensor(tokens).reshape(-1)[:n], spans, self.image_token_id)
        return kt if ok else None

    def new_prompt(self, b, tokens, total, zone=0, vs=None):
        """APC M0': a prompt starts on user ``b``: choose its extra snapshot boundaries (``GridPolicy.plan``: tip / divergence); continuation pieces reuse the plan."""
        self.serial += 1
        gf = os.environ.get(
            "DSV41_APC_GRID_FILE"
        )  # live A/B switch: a file that contains 0 turns the grid snapshots off (the usual pow2 points stay)
        off = False
        if gf:
            try:
                off = open(gf).read().strip() == "0"
            except OSError:
                pass
        kt = None
        if self.grid is not None and not off and self.active():
            # image prompts (DSV41_APC_IMAGES=1) plan on the image-aware key chain (tokens with one content-hash value per image span), the same chain begin_snapshot keys with;
            # text-only prompts: ``_keyed`` returns ``tokens`` itself, so the plan is identical to the one without the image code
            kt = self._keyed(tokens, int(total), vs)
        if kt is not None:
            self.plans[b] = (self.serial, self.grid.plan(kt, total, zone))
        else:
            self.plans.pop(b, None)

    def wants_user(self, b, B):
        """``wants`` plus the grid points planned for user ``b``'s prompt."""
        return self.wants(B) or B in self.plans.get(b, (0, {}))[1]

    def _thin(self, snap):
        """A new tip supersedes the tips of OLDER prompts on its own token chain (the conversation moved on): they cost host RAM and are rarely hit again. Power-of-two and divergence points stay."""
        if (
            self.grid is None
            or getattr(snap, "kind", "") not in ("tip", "tip2", "zone")
            or getattr(snap, "kind", "")
            == "zone"  # (zone points are the safe hit points of same-length prompts: a longer prompt's tip must not remove them)
            or os.environ.get("DSV41_APC_GRID_THIN", "1") != "1"
        ):
            return
        chain = self.store._chain(
            getattr(snap, "ktoks", None) if getattr(snap, "ktoks", None) is not None else snap.toks, snap.s
        )  # (ktoks: image-extended key tokens of the APC-images branch)
        for key, old in list(self.store.snaps.items()):
            if (
                getattr(old, "kind", "") in ("tip", "tip2")
                and getattr(old, "serial", 0) < snap.serial
                and old.s < snap.s
                and chain.get(old.s) == key
            ):
                self.store.drop(key)
                self.stats["thinned"] = self.stats.get("thinned", 0) + 1

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

    def begin_snapshot(self, b, B, tokens, full=True, vs=None):
        if not self.active() or not self.wants_user(b, B):
            return
        kt = self._keyed(tokens, B, vs, B)
        if kt is None:
            self.stats["skipped_vision"] += 1
            return
        key = self.store.key(kt, B)
        pend = self.__dict__.setdefault("pending", [])
        if self.store.has(kt, B, full) or any(k == key and f_ >= full for _, k, (_, _, _, f_) in pend):
            return
        t0 = time.perf_counter()
        s_lo = self._delta_start(kt, B)
        try:
            p = snapshot_begin(self.m, b, B, tokens, s_lo)
            if (
                self.delta and os.environ.get("DSV41_APC_DELTA_CHECK") == "1"
            ):  # debug: the whole snapshot to compare the assembled record with (blocking read, same command queue)
                p.ref = _snapshot_user(self.m, b, B, tokens, False, 0)
        except ApcUnsupported as e:
            self.enabled = False
            self.log(f"prefix caching disabled: {e}")
            return
        if kt is not tokens:
            p.snap.ktoks = torch.as_tensor(kt).long().reshape(-1)[:B].clone()
        p.kind = (
            "pow2" if self.wants(B) else self.plans.get(b, (0, {}))[1].get(B, "")
        )  # (M0': what chose this boundary; tips of older prompts are thinned)
        p.serial = self.plans.get(b, (self.serial, {}))[0]
        pend.append((p, key, (b, B, time.perf_counter() - t0, bool(full))))
        if self.delta:
            self.__dict__.setdefault("pending_bkeys", set()).update(self.store.block_keys(kt, B)[s_lo // self.chunk :])

    def finish_pending(self):
        """Complete every started snapshot (host copies, store insertion). -> seconds spent."""
        pend = self.__dict__.get("pending")
        if not pend:
            return 0.0
        t0 = time.perf_counter()
        self.pending = []
        self.__dict__.pop("pending_bkeys", None)
        for p, key, (b, B, t_b, full) in pend:
            snap = snapshot_finish(self.m, p)
            snap.full = full
            snap.kind, snap.serial = getattr(p, "kind", ""), getattr(p, "serial", 0)
            self._store(snap, b, B, t_b, "deferred")
            if getattr(p, "ref", None) is not None:
                self._check_delta(p.ref, key, B)
            self._thin(snap)
        return time.perf_counter() - t0

    def _check_delta(self, ref, key, B):
        """DSV41_APC_DELTA_CHECK=1: the record assembled from blocks must equal the whole snapshot taken at the same moment, bit for bit"""
        rec = self.store.snaps.get(key)
        if rec is None:
            self.log(f"prefix cache: delta check at {B}: record not stored (refused or evicted)")
            return
        full = self.store.assemble(rec)
        bad = []
        if set(full.carried) != set(ref.carried):
            bad.append(f"names {sorted(set(full.carried) ^ set(ref.carried))}")
        for name, per_col in ref.carried.items():
            got = full.carried.get(name)
            if got is None or any(not torch.equal(got[min(c, len(got) - 1)], x) for c, x in enumerate(per_col)):
                bad.append(name)
        for src, lat in ref.pool.items():
            if not torch.equal(full.pool[src], lat):
                bad.append(f"pool{src}")
        self.stats["delta_checks"] = self.stats.get("delta_checks", 0) + 1
        if bad:
            self.stats["delta_check_failures"] = self.stats.get("delta_check_failures", 0) + 1
        self.log(
            f"prefix cache: delta check at {B} tokens: {'FAIL ' + ', '.join(bad[:8]) if bad else 'OK (bit-identical to the whole snapshot)'}"
        )

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

    def maybe_snapshot(self, b, B, tokens, full=True, vs=None):
        if not self.active() or not self.wants(B):
            return
        kt = self._keyed(tokens, B, vs, B)
        if kt is None:
            self.stats["skipped_vision"] += 1
            return
        t_h0 = time.perf_counter()
        if self.store.has(kt, B, full):
            return
        t_h1 = time.perf_counter()
        try:
            snap = snapshot_user(self.m, b, B, tokens, self._delta_start(kt, B))
            snap.full = bool(full)
            if kt is not tokens:
                snap.ktoks = torch.as_tensor(kt).long().reshape(-1)[:B].clone()
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

    def lookup(self, b, tokens, end, zone=None, vs=None, min_s=0):
        """-> the deepest usable snapshot for user b's prompt (end = this call's end position), else None. ``zone``: first replay chunk start of the prompt (bounded replay) or None.
        ``vs``: the user's ``VisionSeq`` (image prompts, DSV41_APC_IMAGES=1); ``min_s``: only a snapshot deeper than this is of use (a continuation piece that already holds state up to it).
        """
        if not self.active():
            return None
        kt = self._keyed(tokens, end, vs)
        if kt is None:
            self.stats["skipped_vision"] += 1
            return None
        snap = self.store.lookup(kt, end, end, margin=self.margin, zone=zone)
        return snap if snap is not None and snap.s > min_s else None

    # ---- decode output (DSV41_APC_DECODE=1): the next turn's prompt contains this request's answer ----------------------------------------------------------
    def decode_ok(self):
        """decode-output snapshots are on, the delta store is in use and the build can convert its decode state (checked once, logged)"""
        if not self.decode or not self.delta or not isinstance(self.store, DeltaStore) or not self.active():
            return False
        if "_dec_ok" not in self.__dict__:
            why = decode_unsupported(self.m)
            self._dec_ok = not why
            if why:
                self.log(f"prefix cache: DSV41_APC_DECODE ignored ({why})")
            else:
                # the ring keeps the last ``ring_rows`` positions (a halo needs 128 of them, speculative writes run up to ~16 ahead): the halo of s can be gathered while n - s <= ring_rows - 128 - 16
                # (a 128-row ring: only at n == s, which a plain decode step by step reaches exactly)
                rr = int(getattr(getattr(self.m, "pool", None), "ring_rows", 288))
                self.dec_lag = min(self.dec_lag, max(rr - 128 - 16, 0) if rr > 128 else 0)
                self.log(
                    f"prefix cache: decode-output snapshots on (ring {rr} rows, halo capture lag <= {self.dec_lag} tokens)"
                )
        return self._dec_ok

    def decode_poll(self, b, n):
        """after a decode / verify step: user ``b`` holds the KV of ``n`` tokens. When n crossed a new multiple s of the chunk that lies beyond the first one seen for this request (the prompt's own tip is
        already a prefill snapshot), capture the halo of s from the rings now (they only keep the last ``ring_rows`` positions). Keeps the latest capture.
        """
        if not self.decode_ok():
            return
        C = self.chunk
        s = n // C * C
        st = self.dec.get(b)
        if st is not None and n < st["last_n"]:  # (a new request on this user that was never finished here)
            st = None
        if st is None:
            self.dec[b] = st = {"s0": s, "cap": None, "last_n": n}
        st["last_n"] = n
        if s <= st["s0"] or (st["cap"] is not None and s <= st["cap"][0]) or n - s > self.dec_lag:
            return
        try:
            st["cap"] = decode_halo_begin(self.m, b, s)
            self.stats["dec_captures"] = self.stats.get("dec_captures", 0) + 1
        except Exception as e:  # (never let the cache break a decode step)
            self.log(f"prefix cache: decode halo capture failed for user {b} at {s}: {e!r}")
            self.decode = False

    def decode_drop(self, b):
        self.dec.pop(b, None)

    def decode_finish(self, b, tokens):
        """the request of user ``b`` ended (before its pages are released): store the decode-state snapshot at the latest captured s (``tokens``: every token whose KV the model holds)."""
        st = self.dec.pop(b, None)
        if st is None or st["cap"] is None or not self.decode_ok():
            return
        s = st["cap"][0]
        tokens = torch.as_tensor(tokens).long().reshape(-1)
        if tokens.numel() < s or self._has_image(tokens, s):
            return
        t0 = time.perf_counter()
        self.finish_pending()
        kt = tokens[:s]
        if self.store.has(kt, s, True):
            return
        try:
            s_lo = self._delta_start(kt, s)
            snap = decode_snapshot(self.m, b, s, tokens, s_lo, st["cap"])
        except Exception as e:
            self.log(f"prefix cache: decode snapshot of user {b} at {s} failed: {e!r}")
            self.decode = False
            return
        snap.full, snap.kind, snap.serial = True, "dec", self.serial
        self._store(snap, b, s, time.perf_counter() - t0, "decode")
        self.stats["dec_snapshots"] = self.stats.get("dec_snapshots", 0) + 1

    def _delta_start(self, kt, B):
        """delta snapshots: the first token of the rows still to read at ``B`` = the end of the longest run of cached (or already pending) blocks of this chain; 0 = a whole snapshot. A model whose carried
        tensors do not split into whole chunks (``delta_supported``) falls back to the whole-prefix store (log line, once).
        """
        if not self.delta:
            return 0
        if not isinstance(self.store, DeltaStore):
            return 0
        if not self.__dict__.get("_delta_checked"):
            self._delta_checked = True
            why = delta_unsupported(self.m, self.chunk)
            if why:
                self.log(f"prefix cache: DSV41_APC_DELTA ignored ({why}); whole-prefix snapshots")
                st = PrefixStore(self.chunk, self.store.budget, self.store.signature)
                st.tag = self.store.tag
                self.store, self.delta = st, False
                return 0
        keys = self.store.block_keys(kt, B)
        return self.store.have_prefix(keys, self.__dict__.get("pending_bkeys", ())) * self.chunk

    def restore_group(self, snap, bs):
        t_a = time.perf_counter()
        snap = self.store.assemble(snap)
        t_a = time.perf_counter() - t_a
        if self.delta:
            self.stats["assemble_s"] = self.stats.get("assemble_s", 0.0) + t_a
            self.log(f"prefix cache: delta record of {snap.s} tokens assembled from blocks in {t_a:.3f} s")
        t = restore_users(self.m, snap, bs)
        self.stats["restores"] += len(bs)
        self.stats["restored_tokens"] += snap.s * len(bs)
        self.stats["restore_s"] += t
        self.log(
            f"prefix cache: users {bs} restored at {snap.s} tokens in {t:.2f} s ({self.m.__dict__.get('_restore_timing')})"
        )
        return t
