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

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P

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

    def end_row(self, N):
        return self.end or N - self.pad_rows

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
        if sp.kvt is not None:
            add(CarriedSpec(f"L{lid}.kvt", sp.kvt.t, "tail", ratio=pa.ratio, end=sp.kvt.LAT))
        ix = sp.indexer
        if ix is not None and sp.owner and ix.key_owner is None:
            sh = getattr(ix, "sh", None)
            if sh is not None:  # column-sharded FIFO
                add(CarriedSpec(f"L{lid}.keys_rm", ix.keys_rm, "tail", ratio=pa.ratio, div=sh["cols"]))
                add(CarriedSpec(f"L{lid}.keys", ix.keys, "tail", ratio=pa.ratio, div=sh["cols"], pad_rows=32))
            elif ix.keys is not None:
                add(CarriedSpec(f"L{lid}.keys", ix.keys, "full"))
    return out


# ---- per-device shard access ---------------------------------------------------------------------------------------------------------------------------
def _dev(t, k):
    return ttnn.get_device_tensors(t)[k]


def read_shard(t, k):
    """device shard k (row-major over (mesh row, column)) of mesh tensor t -> torch."""
    return ttnn.to_torch(_dev(t, k))


PIECE = 2048  # rows per ``paged_scatter_rows`` call: the row count is a COMPILE-time argument (one program per distinct count, ~0.4 s at first use), so every write is split into pieces of the same size (the last one padded with SKIP ids)


def scatter_rows_mesh(model, t, r, rows_, data, base=0):
    """In place: t[rows_[j] + base] = data[j] on mesh row ``r`` only (other mesh rows get SKIP ids), t a bf16 row-major tensor whose rows are ``data.shape[1]`` wide; fixed-size pieces."""
    md, rows, cols = model.md, model.rows, model.cols
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    n, D = data.shape
    for a in range(0, n, PIECE):
        m = min(PIECE, n - a)
        ids = torch.full((rows, 1, PIECE), SKIP, dtype=torch.int64)
        ids[r, 0, :m] = rows_[a : a + m].long() + base
        src = torch.zeros(rows, 1, PIECE, D, dtype=torch.bfloat16)
        src[r, 0, :m] = data[a : a + m]
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
        ttnn.synchronize_device(
            md
        )  # (the host tensors of the next piece reuse nothing, but the device buffers are freed here)
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


def slot_mask(model, r, i):
    """per-(mesh row, slot) 0/1 mask [rows * Up, 1, 1, 1] sharded over the mesh rows (the ``DynCtx.umask`` layout), one-hot at (r, i)"""
    rows, cols, Up, md = model.rows, model.cols, model.Up, model.md
    mask = torch.zeros(rows * Up, 1, 1, 1)
    mask[r * Up + i] = 1.0
    return ttnn.from_torch(
        mask,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )


def write_slot(model, t, r, i, per_col, row0=0, use_scatter=True, mask_d=None):
    """Write ``per_col`` (list [cols] of torch [1, 1, n, D]: the content of slot ``i`` on each column device of mesh row ``r``) into rows [row0, row0 + n) of slot i of persistent tensor t [Up, 1, N, D]
    on mesh row r only, in place, outside any trace; everything else (other rows / slots / rows of the slot) keeps its content, except on the masked path where the REST of the target slot is zeroed
    (don't-care rows of a FIFO / table). No device synchronisation here (the caller syncs once)."""
    rows, cols, Up = model.rows, model.cols, model.Up
    N, D = t.shape[2], t.shape[3]
    n = per_col[0].shape[2]
    assert t.dtype == ttnn.bfloat16, "kv table in fp8 (DSV41_PFA_SP_FP8) is not supported by the prototype"
    md = model.md
    if (
        use_scatter
        and t.layout == ttnn.ROW_MAJOR_LAYOUT
        and D == 512
        and all(torch.equal(x, per_col[0]) for x in per_col[1:])
    ):
        # replicated over the columns, rows of 1 KiB: scatter only the valid rows (ids are SKIP on the other mesh rows)
        scatter_rows_mesh(model, t, r, torch.arange(n) + (i * N + row0), per_col[0][0, 0])
        return
    mesh = ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(rows, cols))
    host = torch.zeros(rows * Up, cols, N, D, dtype=torch.bfloat16)
    for c in range(cols):
        host[r * Up + i, c, row0 : row0 + n] = per_col[c][0, 0]
    d_new = ttnn.from_torch(
        host, device=md, dtype=ttnn.bfloat16, layout=t.layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mesh
    )
    own = mask_d is None
    d_mask = slot_mask(model, r, i) if own else mask_d
    upd = _where(d_mask, d_new, t)
    ttnn.copy(upd, t)
    for x in (upd, d_new) + ((d_mask,) if own else ()):
        ttnn.deallocate(x)


@dataclass
class PrefixSnapshot:
    s: int
    toks: torch.Tensor  # int64 [s]
    carried: dict = field(default_factory=dict)  # name -> [cols] torch (slot dim kept, trimmed rows)
    pool: dict = field(default_factory=dict)  # src -> torch bf16 [n_entries, 512] latents of the kv-source layer
    seconds: float = 0.0

    @property
    def nbytes(self):
        n = sum(x.numel() * x.element_size() for v in self.carried.values() for x in v)
        return n + sum(x.numel() * x.element_size() for x in self.pool.values())


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


def snapshot_user(model, b, s, toks):
    """State of user ``b`` right after a step that ended at position ``s`` (``model.pf_resume[b] == s``) -> PrefixSnapshot. ``toks``: the user's tokens [>= s]."""
    t0 = time.perf_counter()
    assert model.__dict__.get("pf_resume", {}).get(b) == s, f"user {b} is not resumable at {s}"
    r, i = _slot_of(model, b)
    cols = model.cols
    snap = PrefixSnapshot(s=s, toks=torch.as_tensor(toks).long().reshape(-1)[:s].clone())
    for name, sp in carried_specs(model).items():
        n = sp.count(s)
        vals = []
        for c in range(cols):
            x = read_shard(sp.tensor, r * cols + c)[i : i + 1]
            if sp.kind == "tail":
                end = sp.end_row(x.shape[2])
                x = x[:, :, end - n : end]
            vals.append(x.clone())
        snap.carried[name] = vals
    # pool latents of the kv-source layers (replicated over the columns: column 0 of the mesh row)
    pool = model.pool
    if _sources(model):
        full = read_shard(pool.pool, r * cols)[0, 0]  # [total_rows, 512]
        for src in _sources(model):
            ent = torch.arange(s // P.SRC_RATIO[src])
            snap.pool[src] = full[pool.phys_rows(b, src, ent).long()].clone()
    snap.seconds = time.perf_counter() - t0
    return snap


def restore_user(model, snap, b, tokens=None):
    """Put ``snap`` into user ``b`` (any mesh row / prefill slot): pages, pool latents, carried state of its slot, Engram history, ``pf_resume``. Afterwards
    ``model.prefill_interleaved([(b, tokens, snap.s, e)], ...)`` continues the prompt. -> seconds."""
    t0 = time.perf_counter()
    s, cols = snap.s, model.cols
    resume = model.__dict__.setdefault("pf_resume", {})
    # 1. pages + slot
    model.release_user(b)
    model.pool.release(b)
    model.pool.admit(b, s + 1)
    model.pool.sync_page_table()
    if model.Up != model.U:
        slot = model._slot_alloc([(b, None, 0, s)], resume)[b]
        r, i = slot // model.Up, slot % model.Up
    else:
        r, i = b // model.U, b % model.U
    # 2. carried state of the target (mesh row, slot): masked write, outside the trace
    t_c = time.perf_counter()
    per = {}
    mask_d = slot_mask(model, r, i)
    for name, sp in carried_specs(model).items():
        n = sp.count(s)
        per_col = snap.carried[name]
        t_n = time.perf_counter()
        if sp.kind == "tail":
            write_slot(model, sp.tensor, r, i, per_col, row0=sp.end_row(sp.tensor.shape[2]) - n, mask_d=mask_d)
        else:
            write_slot(model, sp.tensor, r, i, per_col, mask_d=mask_d)
        per[name] = round(time.perf_counter() - t_n, 4)
    ttnn.synchronize_device(model.md)
    ttnn.deallocate(mask_d)
    t_c = time.perf_counter() - t_c
    # 3. pool latents of the kv-source layers (page table of the target user; replicated over the columns)
    t_p = time.perf_counter()
    pool = model.pool
    for src, lat in snap.pool.items():
        rows_ = pool.phys_rows(b, src, torch.arange(lat.shape[0])).long()
        scatter_rows_mesh(model, pool.pool, b // model.U, rows_, lat)
    t_p = time.perf_counter() - t_p
    model.__dict__["_restore_timing"] = {"carried_s": round(t_c, 3), "pool_s": round(t_p, 3), "per_tensor_s": per}
    # 4. Engram history (host, pure function of the tokens)
    if model.host_rows is not None:
        st = model.hasher.st
        st.cache[b, :s] = st.token_map[snap.toks]
    resume[b] = s
    return time.perf_counter() - t0


# ---- host store -------------------------------------------------------------------------------------------------------------------------------------------
class PrefixStore:
    """LRU store of snapshots under a host byte budget. Key = (signature, sha256 of the first s tokens): ``signature`` ties a snapshot to the model / chunk / vision configuration
    (the generator hashes the tokens itself, not vLLM's block hashes). ``lookup(tokens, start, end)`` -> the snapshot with the LARGEST s such that s % chunk == 0, s <= start and
    s <= end - 128 (the last 128 positions of a prompt are always replayed: ring rows, ``prev_cs`` and the drafter taps come from those chunks), or None. A miss means recompute
    from 0 (always correct)."""

    def __init__(self, chunk, budget_bytes, signature=""):
        self.chunk, self.budget, self.signature = int(chunk), int(budget_bytes), signature
        self.snaps = OrderedDict()  # key -> PrefixSnapshot (oldest first)
        self.bytes = 0
        self.hits = self.misses = self.evictions = 0

    def _key(self, tokens, s):
        import hashlib

        t = torch.as_tensor(tokens).long().reshape(-1)[:s].contiguous()
        return (self.signature, s, hashlib.sha256(t.numpy().tobytes()).hexdigest())

    def put(self, snap):
        key = self._key(snap.toks, snap.s)
        if key in self.snaps:
            self.snaps.move_to_end(key)
            return False
        if snap.nbytes > self.budget:
            return False
        self.snaps[key] = snap
        self.bytes += snap.nbytes
        while self.bytes > self.budget:
            _, old = self.snaps.popitem(last=False)
            self.bytes -= old.nbytes
            self.evictions += 1
        return True

    def lookup(self, tokens, start, end):
        tokens = torch.as_tensor(tokens).long().reshape(-1)
        top = min(int(start), int(end) - 128, tokens.numel())
        for s in range(top // self.chunk * self.chunk, 0, -self.chunk):
            key = self._key(tokens, s)
            snap = self.snaps.get(key)
            if snap is not None:
                self.snaps.move_to_end(key)
                self.hits += 1
                return snap
        self.misses += 1
        return None

    def clear(self):
        """``reset_prefix_cache`` / a weight reload."""
        self.snaps.clear()
        self.bytes = 0
