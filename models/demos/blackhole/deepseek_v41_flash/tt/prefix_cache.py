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
    replicated: bool = True  # identical on the 8 columns of a mesh row (everything but the column-sharded key FIFO): one host copy, one upload per mesh row

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
        row0 = sp.end_row(sp.tensor.shape[2]) - n if tail else 0
        _fill(host, flats, j, rn, per_col, tail, N, row0, replicated)
    mesh = ttnn.ShardTensor2dMesh(md, dims=(0, None if replicated else 1), mesh_shape=(rows, cols))
    big = ttnn.from_torch(
        host, device=md, dtype=ttnn.bfloat16, layout=t0_.layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mesh
    )
    for j, (sp, per_col) in enumerate(batch):
        t = sp.tensor
        new_j = ttnn.slice(big, [0, 0, j * rn, 0], [Up, 1, (j + 1) * rn, D]) if len(batch) > 1 else big
        if tail:
            row0 = sp.end_row(t.shape[2]) - n
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


def snapshot_user(model, b, s, toks):
    """State of user ``b`` right after a step that ended at position ``s`` (``model.pf_resume[b] == s``) -> PrefixSnapshot. ``toks``: the user's tokens [>= s]. The pool latents of the kv-source layers
    are NOT read from the pool (an fp8 pool, and reading it would mean the whole per-device pool): the kv table of a source layer holds the same bf16 latents (entry j = FIFO row LAT - n + j).
    """
    t0 = time.perf_counter()
    assert model.__dict__.get("pf_resume", {}).get(b) == s, f"user {b} is not resumable at {s}"
    r, i = _slot_of(model, b)
    cols = model.cols
    snap = PrefixSnapshot(s=s, toks=torch.as_tensor(toks).long().reshape(-1)[:s].clone())
    specs = carried_specs(model)
    for src in _sources(model):
        if f"L{src}.kvt" not in specs:
            raise ApcUnsupported(f"no kv table for source layer {src} (dense prefill path)")
    for name, sp in specs.items():
        if sp.tensor.dtype != ttnn.bfloat16:
            raise ApcUnsupported(f"{name} is {sp.tensor.dtype}")
        n = sp.count(s)
        vals = []
        for c in range(1 if sp.replicated else cols):
            x = read_shard(sp.tensor, r * cols + c)[i : i + 1]
            if sp.kind == "tail":
                end = sp.end_row(x.shape[2])
                x = x[:, :, end - n : end]
            vals.append(x.clone())
        snap.carried[name] = vals * cols if sp.replicated else vals
    for src in _sources(model):
        snap.pool[src] = snap.carried[f"L{src}.kvt"][0][0, 0]  # [s / ratio, 512], shared with the carried copy
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
            out[s] = (self.signature, s, h.copy().hexdigest())
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

    * restore: a new prompt (no carried state) takes the deepest stored snapshot s <= end - margin (``margin`` = the ring rows of the pool: the continuation must rewrite every ring row), the
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

    def maybe_snapshot(self, b, B, tokens):
        if not self.active() or not self.wants(B):
            return
        if self._has_image(tokens, B):
            self.stats["skipped_vision"] += 1
            return
        if self.store.has(tokens, B):
            return
        try:
            snap = snapshot_user(self.m, b, B, tokens)
        except ApcUnsupported as e:
            self.enabled = False
            self.log(f"prefix caching disabled: {e}")
            return
        self.store.put(snap)
        self.stats["snapshots"] += 1
        self.stats["snap_s"] += snap.seconds
        self.log(
            f"prefix cache: snapshot of user {b} at {B} tokens ({snap.nbytes / 2**20:.1f} MiB, {snap.seconds:.2f} s; store {self.store.bytes / 2**20:.0f} MiB, {len(self.store.snaps)} snapshots)"
        )

    def lookup(self, b, tokens, end):
        """-> the deepest usable snapshot for user b's prompt (end = this call's end position), else None"""
        if not self.active():
            return None
        if self._has_image(tokens, end):
            self.stats["skipped_vision"] += 1
            return None
        return self.store.lookup(tokens, end, end, margin=self.m.pool.ring_rows)

    def restore_group(self, snap, bs):
        t = restore_users(self.m, snap, bs)
        self.stats["restores"] += len(bs)
        self.stats["restored_tokens"] += snap.s * len(bs)
        self.stats["restore_s"] += t
        self.log(
            f"prefix cache: users {bs} restored at {snap.s} tokens in {t:.2f} s ({self.m.__dict__.get('_restore_timing')})"
        )
        return t
