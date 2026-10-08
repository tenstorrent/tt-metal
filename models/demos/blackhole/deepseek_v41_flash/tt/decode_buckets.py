# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Right-sized decode steps (VLLM_BUCKETS_NOTES.md): a decode graph for ``Ub`` < U users per mesh row that runs on the SAME persistent state as the full (U) model.

A ``DecodeBucket`` is a second ``DSV41Decoder`` whose layers are shallow copies of the model's layers with
  * the batch dependent configuration of the new size (``T``, core grids, MoE decode scratch buffers / config, step-state ``T``),
  * every weight, the routed-expert weights, the router, the shared expert, mHC, Engram device weights, the head, the rope tables and the KV POOL shared with the full model (nothing
    batch independent is uploaded again),
  * per-user state used through prefix views of the full model's tensors: the pool / page table / ring regions are addressed by user index and ring stride (``PagedKVPool.ring_base`` keeps
    the stride of the full U), the compressor 'previous token' buffer ``prev_cs`` ([1,1,U,1024] tile, padded to 32 rows) is viewed as [1,1,Ub,1024], the decode index-key slab keeps its [U,..]
    shape (a prefix slice for the score matmul, a padded update for the key append).
The users of a step must be the users u < Ub of every mesh row (model user r * U + u): a user with a larger in-row index keeps its state (the step only writes the rows it computes, except
the padding rows of the tile based buffers, which hold no live state by construction) but is not computed.

All persistent buffers of a bucket are allocated by ``__init__`` / ``prepare`` and every program is compiled by ``compile`` before any trace is captured (``capture``).
"""

import copy
import time

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.decoder import DSV41Decoder


def check_trace_allocations(md, trace_id, name, seen=set()):
    """TT_METAL_TRACE_ALLOC_TRACKING=1 (+ TT_METAL_TRACE_ALLOC_TRACEBACKS=1): log (once per trace and buffer set) the live buffers that were allocated after ``trace_id`` was captured and can
    be clobbered by its replay: the evidence of a device allocation under a captured trace. Cheap no-op without the env.
    """
    if not (hasattr(ttnn, "trace_allocation_tracking_enabled") and ttnn.trace_allocation_tracking_enabled()):
        return
    try:
        if hasattr(
            trace_id, "prog"
        ):  # tt/moe_overlap.SegTrace: the segment traces (a sub-device manager must be active for its own ids: best effort on the first)
            ids = {}
            for kind, tid in trace_id.prog:
                if kind == "trace":
                    ids.update(ttnn.get_unsafe_tracked_ids(md, tid))
        else:
            ids = ttnn.get_unsafe_tracked_ids(md, trace_id)
    except Exception as e:  # noqa: BLE001
        if (name, "err") not in seen:
            seen.add((name, "err"))
            print(f"TRACE-ALLOC check not possible for {name}: {e!r}", flush=True)
        return
    key = (name, tuple(sorted(ids)))
    if ids and key not in seen:
        seen.add(key)
        first = next(iter(ids.values()))
        print(
            f"TRACE-ALLOC {name}: {len(ids)} live buffer(s) allocated after the capture can be clobbered by the replay; first traceback:\n{first}",
            flush=True,
        )


def view_rows(t, n):
    """[1,1,U,W] tile tensor viewed as [1,1,n,W] over the SAME buffer (U, n <= 32: both are padded to one 32-row tile)."""
    w = int(t.shape[-1])
    return ttnn.reshape(t, [1, 1, n, w], [1, 1, 32, w])


class DecodeBucket:
    def __init__(self, model, Ub, log=print):
        m = model
        self.m, self.U_full, self.U, self.rows, self.cols = m, m.U, int(Ub), m.rows, m.cols
        self.B = self.rows * self.U
        self.log = log
        assert 1 <= self.U < self.U_full, (self.U, self.U_full)
        t0 = time.time()
        self.trace_id = None
        self.last_logits = None
        self.rows_cat = None
        self._warm = False
        md = m.md
        attn_b, idx_b, built_b, moe_buffers = {}, {}, [], None
        for L, layer, key in m.built:
            a = m.attns[L]
            ab = copy.copy(a)
            ab.T = self.U
            ab._ucfg = self._ucfg(a)
            if getattr(a, "prev_cs", None) is not None:
                ab.prev_cs = view_rows(a.prev_cs, self.U)
            if getattr(a, "source", None) is not None:
                ab.source = attn_b[id(a.source)]
            if getattr(a, "indexer", None) is not None:
                if id(a.indexer) not in idx_b:
                    idx_b[id(a.indexer)] = self._clone_indexer(a.indexer)
                ab.indexer = idx_b[id(a.indexer)]
            attn_b[id(a)] = ab
            lb = copy.copy(layer)
            lb.T = self.U
            lb.attention = ab
            lb.moe = layer.moe.for_batch(self.U, buffers=moe_buffers)
            if moe_buffers is None:
                moe_buffers = lb.moe.decode.buffers
                lb.moe.warmup()  # steers the moe_compute global semaphore of this size to the top of L1 (see DSV41MoEBlock.warmup)
            built_b.append((L, lb, key))
        self.built = built_b
        self.step_groups = {}
        for k, ss in m.step_groups.items():
            sb = copy.copy(ss)
            sb.T = self.U
            self.step_groups[k] = sb
        keep = m._keep
        self.embedding = keep["embedding"].rebatch(self.U)
        self.dev_engram = keep["dev_engram"]
        self.dec = DSV41Decoder(md, built_b, self.embedding, m.head, self.dev_engram, step_states=self.step_groups)
        self.dec.mesh_config, self.dec.ccl = m.mc, m.ccl
        self.engram_ids = m.engram_ids
        self.engram_kin = m.engram_kin
        log(f"decode bucket U'={self.U} (batch {self.B}) objects built in {time.time() - t0:.1f} s")

    # ---- construction helpers --------------------------------------------------------------------------------------------------------
    def _ucfg(self, a):
        from models.demos.blackhole.deepseek_v41_flash.tt.attention import HEAD_DIM, PAD_HEADS

        return ttnn.create_sharded_memory_config(
            shape=(PAD_HEADS, HEAD_DIM),
            core_grid=ttnn.num_cores_to_corerangeset(self.U, ttnn.CoreCoord(8, 8), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _clone_indexer(self, ix):
        nb = copy.copy(ix)
        nb.T = self.U
        nb.B = self.rows * self.U
        nb.U_slab = int(ix.k_cache.shape[0])  # the shared key slab keeps its full user dimension
        # (``_kcfg``, the key-append shard layout, stays that of the full slab: the update takes one row per slab user)
        # persistent zero rows that pad the key append / update indices of the users outside the bucket (allocated here, before any trace)
        nb._pad_idx = ttnn.from_torch(
            torch.zeros(ix.k_cache.shape[0] - self.U, dtype=torch.int32),
            device=self.m.md,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.m.md),
        )
        return nb

    # ---- inputs -----------------------------------------------------------------------------------------------------------------------
    def _mp(self):
        return self.m._mp()

    def set_loop_state(self, tokens, pos):
        """tokens / pos [B'] (bucket row order) -> the persistent device token / position buffers (created on first use)."""
        tok = ttnn.from_torch(
            tokens.reshape(-1, 1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._mp(),
        )
        ps = ttnn.from_torch(
            pos.reshape(-1).to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._mp()
        )
        d = self.dec
        if getattr(d, "tok_dev", None) is None:
            d.enable_device_loop(self.m.mc, self.m.ccl, tokens.reshape(-1), pos.reshape(-1))
            d.engram_rows_fn = self._engram_rows_fn if self.engram_ids else None
        else:
            ttnn.copy_host_to_device_tensor(tok, d.tok_dev)
            ttnn.copy_host_to_device_tensor(ps, d.pos_dev)

    def _engram_rows_fn(self, tokens, pos):
        Tn = self.U
        rc = ttnn.reshape(self.rows_cat, [1, 1, Tn, self.rows_cat.shape[-1]])
        out, off = {}, 0
        for l in self.engram_ids:
            k = self.engram_kin[l]
            out[l] = ttnn.to_layout(ttnn.slice(rc, [0, 0, 0, off], [1, 1, Tn, off + k]), ttnn.TILE_LAYOUT)
            off += k
        return out

    def engram_host_rows(self, tokens, pos, phys):
        """Engram rows of the step's tokens at their positions (model user ``phys[i]`` owns the token history of bucket row i) -> host tensor for ``rows_cat`` (None without Engram)."""
        m = self.m
        if not self.engram_ids:
            return None
        hashes = m.hasher(tokens.reshape(self.B, 1).long(), pos.long(), rows=torch.as_tensor(phys).long())
        rows = m._rows_threads(hashes)
        cat = torch.cat([rows[l].reshape(self.B, 1, 1, -1) for l in self.engram_ids], dim=-1).to(torch.bfloat16)
        return ttnn.from_torch(cat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._mp())

    def upload_rows(self, host_rows):
        if host_rows is None:
            return
        if self.rows_cat is None:
            self.rows_cat = ttnn.to_device(host_rows, self.m.md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn.copy_host_to_device_tensor(host_rows, self.rows_cat)

    # ---- allocate / compile / capture (in this order, for every bucket, before ANY trace is captured) ---------------------------------------
    def prepare(self, tokens, pos, phys):
        """Allocate the persistent step buffers (tokens / positions, packed Engram rows) with the given (filler) inputs."""
        self.set_loop_state(tokens, pos)
        if self.engram_ids:
            self.upload_rows(self.engram_host_rows(tokens, pos, phys))

    def compile(self, tokens, pos, phys):
        """One eager step (compiles every program of the bucket); the step-carried compressor state is restored."""
        snaps = self.dec.snapshot_states()
        self.last_logits = self.dec.forward()
        ttnn.synchronize_device(self.m.md)
        self.dec.restore_states(snaps)
        self.set_loop_state(tokens, pos)
        self._warm = True

    def capture(self, tokens, pos):
        md = self.m.md
        snaps = self.dec.snapshot_states()
        self.trace_id = ttnn.begin_trace_capture(md, cq_id=0)
        self.last_logits = self.dec.forward()
        ttnn.end_trace_capture(md, self.trace_id, cq_id=0)
        ttnn.synchronize_device(md)
        self.dec.restore_states(snaps)
        self.set_loop_state(tokens, pos)

    def release_trace(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.m.md, self.trace_id)
            self.trace_id = None

    # ---- one step -----------------------------------------------------------------------------------------------------------------------
    def step(self, tokens, pos, phys, reload_inputs=True, enable_trace=True):
        """One decode step of the bucket's users. tokens / pos [B'] in bucket row order, ``phys`` [B'] the model user of every row. -> next greedy tokens [B'] (long)."""
        m = self.m
        t0 = time.perf_counter()
        full_pos = torch.zeros(m.B, dtype=torch.long)
        full_pos[torch.as_tensor(phys).long()] = pos.long()
        m.pool.ensure(
            full_pos, lookahead=16
        )  # pages for the next replays (no-op, no upload, unless a page boundary is near)
        if reload_inputs:
            self.set_loop_state(tokens, pos)
        host_rows = self.engram_host_rows(tokens, pos, phys)
        t1 = time.perf_counter()
        self.upload_rows(host_rows)
        if enable_trace and self.trace_id is None:
            # normally captured by Model.warm_serving; this fallback (after a trace release, e.g. a re-captured prefill trace) captures under the live prefill trace
            self.log(f"WARNING: decode bucket {self.B}: capturing the trace while serving (not captured at load time)")
            self.capture(tokens, pos)
            self.upload_rows(host_rows)
        t2 = time.perf_counter()
        if enable_trace:
            check_trace_allocations(m.md, self.trace_id, f"decode bucket B'={self.B}")
            ttnn.execute_trace(m.md, self.trace_id, cq_id=0, blocking=False)
        else:
            self.last_logits = self.dec.forward()
        devs = ttnn.get_device_tensors(ttnn.from_device(self.dec.tok_dev))
        out = torch.cat([ttnn.to_torch(devs[r * self.cols]).reshape(-1) for r in range(self.rows)]).long()
        t3 = time.perf_counter()
        m.timing["decode_host_prep"] = t1 - t0
        m.timing["decode_upload"] = t2 - t1
        m.timing["decode_device_read"] = t3 - t2
        return out

    def read_logits(self):
        ttnn.synchronize_device(self.m.md)
        return self.m.head.gather_logits(self.last_logits)[: self.B]
