# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The store end to end with a prefill stack that knows nothing of it (KV_STORE.md 4, variant 3b; 7, step 3).

The prefill stack keeps ordinary contiguous user slots; ONE of them is the WORKBENCH. Every conversation lives in
the paged pool (``manager.KvStore``); around a request the engine

    1. reserves the conversation's frames (LRU eviction by tokens, in the manager),
    2. copies its cached prefix, pool frames -> workbench slot (``copy.KvCopy``, bank-local, every chip),
    3. runs the prefill passes for the new tokens in the workbench, restarting on the 4096 grid below the shared
       prefix (the stack's passes, traced or not: the engine only calls ``prefill(slot, start, end, ids)``),
    4. copies the frames the passes wrote, workbench -> pool,
    5. rewrites the frame tables when the decode-row binding changed.

The stack is not imported here except by ``StackPrefill`` (the adapter of ``TtPrefillRuntime.prefill_chunk``), so
the engine also runs against a stub. Everything the engine allocates on the device (pool, plans, tables) and every
program it launches is created in ``__init__`` and ``warm()``: call both BEFORE the stack captures its trace.

What decode needs from this later is in README.md ("Decode on the store").
"""

from __future__ import annotations

import time

import numpy as np

import ttnn
from models.experimental.hy4_preview.fused.kv_store import frames as fr
from models.experimental.hy4_preview.fused.kv_store.copy import KvCompare, KvCopy
from models.experimental.hy4_preview.fused.kv_store.device import FrameTables, allocate, check_layout
from models.experimental.hy4_preview.fused.kv_store.manager import KvStore

CHUNK = fr.PERIOD  # a prefill pass: 4096 token rows, the stripe period


def dram(mesh_device) -> dict:
    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return {
        "allocated_gb": v.total_bytes_allocated_per_bank * v.num_banks / 1e9,
        "free_gb": v.total_bytes_free_per_bank * v.num_banks / 1e9,
    }


class StackPrefill:
    """``prefill(slot, start, end, ids)`` on the prefill stack's runtime: one chunk pass of tokens [start, end)
    (at most 4096, start on the 32-row grid) into user slot ``slot``. Trace mode: the pass is a replay and
    (slot, start, end) ride a device record, as in the stack's serving loop."""

    def __init__(self, runtime, kv_caches, mesh_device, mesh_shape=(8, 4), trace: bool = True):
        self.runtime, self.kv, self.mesh, self.mesh_shape, self.trace = (
            runtime,
            kv_caches,
            mesh_device,
            mesh_shape,
            trace,
        )
        self.sp = runtime.config.sp_factor
        self.passes = 0

    def __call__(self, slot: int, start: int, end: int, ids) -> dict:
        from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor

        assert start % 32 == 0 and 0 < end - start <= CHUNK
        padded = [int(v) for v in ids[start:end]] + [0] * (CHUNK - (end - start))
        t_prep = time.perf_counter()
        tt_in = prepare_prefill_input_tensor(padded, self.mesh, self.sp, False, self.mesh_shape, 0, chunk_start=start)
        ttnn.synchronize_device(self.mesh)
        t0 = time.perf_counter()
        md = self.runtime._meta_words_dev((slot, start, end)) if self.trace else None
        self.runtime.prefill_chunk(tt_in, self.kv, slot_id=slot, actual_start=start, actual_end=end, metadata_msg=md)
        ttnn.synchronize_device(self.mesh)
        t1 = time.perf_counter()
        if md is not None:
            ttnn.deallocate(md)
        self.passes += 1
        return {"prep_s": t0 - t_prep, "chunk_s": t1 - t0}


def common_prefix(a, b) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    differ = np.nonzero(np.asarray(a[:n]) != np.asarray(b[:n]))[0]
    return int(differ[0]) if differ.size else n


class Engine:
    def __init__(
        self,
        mesh_device,
        store: KvStore,
        bench_latent,
        bench_index,
        prefill,
        *,
        bench_users: int,
        bench_user: int = 0,
        zero_pool: bool = True,
        latent_table_cores=None,
        index_table_cores=None,
    ):
        """``bench_latent`` / ``bench_index``: the prefill stack's two cache tensors (``kv_caches.kvpe.storage``,
        ``kv_caches.index``), ``bench_users`` user slots each; user ``bench_user`` is the workbench, the others stay
        ordinary slots. ``store``: the manager of a pool this constructor allocates (``store.num_frames`` frames).
        The table cores are where decode's readers will want the frame tables (defaults: the sparse-MLA reader
        columns and the scan grid)."""
        self.mesh, self.store, self.prefill = mesh_device, store, prefill
        self.bench_users, self.bench_user = bench_users, bench_user
        layers, index_layers = bench_latent.shape[0] // bench_users, bench_index.shape[0] // bench_users
        assert bench_latent.shape[0] == layers * bench_users and bench_index.shape[0] == index_layers * bench_users
        self.capacity_frames = bench_latent.shape[2] // fr.FRAME_ROWS
        assert bench_latent.shape[2] == self.capacity_frames * fr.FRAME_ROWS == bench_index.shape[2]
        self.latent = fr.Kind("latent", layers, fr.LATENT.width, fr.LATENT.shard_bytes, fr.LATENT.row_stride)
        self.index = fr.Kind("index", index_layers, fr.INDEX.width, fr.INDEX.shard_bytes, 0)
        self.kinds = (self.latent, self.index)
        self.bench = {self.latent: bench_latent, self.index: bench_index}
        self.bench_g = {k: fr.Geometry.contiguous(k, self.capacity_frames) for k in self.kinds}
        self.pool_g = {k: fr.Geometry.pool(k) for k in self.kinds}
        for k in self.kinds:
            check_layout(self.bench[k], self.bench_g[k], bench_users)
        self.pool = {k: allocate(mesh_device, self.pool_g[k], store.num_frames) for k in self.kinds}
        if zero_pool:  # a frame is only ever read after a copy-out wrote all of it; zeros make that checkable
            from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

            for k in self.kinds:
                DRAMZeroFill.op(self.pool[k])
        to_bench = [(self.pool[k], self.pool_g[k], self.bench[k], self.bench_g[k]) for k in self.kinds]
        to_pool = [(self.bench[k], self.bench_g[k], self.pool[k], self.pool_g[k]) for k in self.kinds]
        self.copy_in, self.copy_out = KvCopy(mesh_device, to_bench), KvCopy(mesh_device, to_pool)
        # one compare per kind: a slot-major frame number of a user other than 0 depends on the kind's layer count
        self._cmps = {
            k: KvCompare(mesh_device, [(self.pool[k], self.pool_g[k], self.bench[k], self.bench_g[k])])
            for k in self.kinds
        }
        lat_cores = latent_table_cores or [(x, y) for x in (0, 7) for y in range(8)]
        idx_cores = index_table_cores or [(x, y) for x in range(8) for y in range(8)]
        self.latent_tables = FrameTables(mesh_device, lat_cores, rows=store.rows)
        self.index_tables = FrameTables(mesh_device, idx_cores, rows=store.rows)
        self._tables_written = -1
        self.prompts = {}  # key -> the prompt whose rows are valid in the pool (np.int32)
        self.history = {}  # key -> [(start, end, prompt)]: the passes since the conversation was (re)created
        self._lost = {}  # key -> the prompt an eviction dropped (to count what is prefilled again)
        store.on_evict = self._evicted

    # ---- the device side ---------------------------------------------------------------------------------------
    def warm(self) -> None:
        """Compile and launch every program of the engine once with nothing to do, and write the tables."""
        for copy in (self.copy_in, self.copy_out):
            copy.set_pairs([])
            copy()
        for cmp in self._cmps.values():
            cmp.set_pairs([], 0)
            cmp()
        self.sync_tables(force=True)
        ttnn.synchronize_device(self.mesh)

    def _bench_pairs(self, pairs, to_bench: bool):
        """Manager pairs -> copy pairs. The workbench is user slot 0, so its frame number is the frame's position
        k for both kinds (one plan serves both)."""
        assert self.bench_user == 0, "the workbench is user slot 0"
        return [(f, k) for f, k in pairs] if to_bench else [(k, f) for k, f in pairs]

    def _launch(self, program) -> float:
        t0 = time.perf_counter()
        program()
        ttnn.synchronize_device(self.mesh)
        return time.perf_counter() - t0

    def sync_tables(self, force: bool = False) -> float:
        """Rewrite the frame tables if the manager's tables changed; seconds on the host."""
        if not force and self._tables_written == self.store.table_version:
            return 0.0
        t0 = time.perf_counter()
        self.latent_tables.write(self.store.table(self.pool_g[self.latent]))
        self.index_tables.write(self.store.table(self.pool_g[self.index]))
        ttnn.synchronize_device(self.mesh)
        self._tables_written = self.store.table_version
        return time.perf_counter() - t0

    # ---- requests ----------------------------------------------------------------------------------------------
    def _evicted(self, conv) -> None:
        if conv.key in self.prompts:
            self._lost[conv.key] = self.prompts.pop(conv.key)
        self.history.pop(conv.key, None)

    def begin(self, key, tokens, max_new: int = 0):
        """A request: the conversation's whole prompt. Returns the record of what was done (None: no room now)."""
        tokens = np.asarray(tokens, dtype=np.int64)
        n = len(tokens)
        assert n + max_new <= self.capacity_frames * fr.FRAME_TOKENS, "the prompt does not fit the workbench slot"
        known = self.prompts.get(key, self._lost.get(key))
        shared = common_prefix(known, tokens) if known is not None else 0
        t_begin = time.perf_counter()
        grant = self.store.begin(key, n + max_new, shared_prefix=shared)
        if grant is None:
            return None
        rec = {
            "key": key,
            "prompt": n,
            "shared": shared,
            "restart": grant.restart,
            "new_tokens": n - grant.restart,
            "frames": len(grant.frames),
            "new_frames": len(grant.new_frames),
            "reprefill": grant.reprefill,
            "copy_in_frames": 0,
            "copy_in_s": 0.0,
            "copy_in_plan_s": 0.0,
            "copy_out_frames": 0,
            "copy_out_s": 0.0,
            "copy_out_plan_s": 0.0,
            "passes": [],
        }
        restart = grant.restart
        if n > restart:
            pairs = self._bench_pairs(self.store.copy_in(key, restart), to_bench=True)
            if pairs:
                t0 = time.perf_counter()
                self.copy_in.set_pairs(pairs)
                t1 = time.perf_counter()
                self.copy_in()
                ttnn.synchronize_device(self.mesh)
                rec.update(copy_in_frames=len(pairs), copy_in_s=time.perf_counter() - t0, copy_in_plan_s=t1 - t0)
            for start in range(restart, n, CHUNK):
                rec["passes"].append(self.prefill(self.bench_user, start, min(start + CHUNK, n), tokens))
            pairs = self._bench_pairs(self.store.copy_out(key, restart, n), to_bench=False)
            t0 = time.perf_counter()
            self.copy_out.set_pairs(pairs)
            t1 = time.perf_counter()
            self.copy_out()
            ttnn.synchronize_device(self.mesh)
            rec.update(copy_out_frames=len(pairs), copy_out_s=time.perf_counter() - t0, copy_out_plan_s=t1 - t0)
            if grant.keep == 0 or key not in self.history:
                self.history[key] = []
            self.history[key].append((restart, n, tokens))
        self.prompts[key] = tokens
        self._lost.pop(key, None)
        rec["prefill_s"] = sum(p["chunk_s"] for p in rec["passes"])
        rec["prep_s"] = sum(p["prep_s"] for p in rec["passes"])
        rec["total_s"] = time.perf_counter() - t_begin
        rec["store_s"] = rec["total_s"] - rec["prefill_s"] - rec["prep_s"]  # everything the store adds
        return rec

    def end(self, key, trim: bool = False) -> None:
        self.store.end(key, cached=len(self.prompts[key]), trim=trim)

    def evict(self, key) -> None:
        self.store.evict(key)

    # ---- ordinary slots (references, fixed-slot baseline) ---------------------------------------------------------
    def prefill_slot(self, slot: int, start: int, end: int, tokens) -> dict:
        """Passes for tokens [start, end) into an ordinary user slot (not the workbench)."""
        assert slot != self.bench_user
        passes = [self.prefill(slot, s, min(s + CHUNK, end), tokens) for s in range(start, end, CHUNK)]
        return {
            "passes": passes,
            "prefill_s": sum(p["chunk_s"] for p in passes),
            "prep_s": sum(p["prep_s"] for p in passes),
        }

    def reference(self, key, slot: int) -> int:
        """Prefill the conversation ALONE into ordinary slot ``slot``: the same passes, in order, that its pool
        rows came from since it was (re)created. Returns the number of passes."""
        n = 0
        for start, end, tokens in self.history[key]:
            n += len(self.prefill_slot(slot, start, end, tokens)["passes"])
        return n

    def compare(self, key, slot: int) -> dict:
        """The conversation's pool frames against user slot ``slot`` of the bench tensors, on every chip, every
        layer of both kinds, positions [0, ceil32(prompt)) (the passes write whole 32-row tiles)."""
        conv = self.store.conversation(key)
        positions = -(-len(self.prompts[key]) // 32) * 32
        frames = conv.frames[: self.store.frames_for(positions)]
        out = {"positions": positions, "frames": len(frames)}
        for kind in self.kinds:
            cmp = self._cmps[kind]
            g = self.bench_g[kind]
            cmp.set_pairs([(f, g.contiguous_frame(slot, k)) for k, f in enumerate(frames)], positions)
            cmp()
            ttnn.synchronize_device(self.mesh)
            out[kind.name] = cmp.read()
        return out
