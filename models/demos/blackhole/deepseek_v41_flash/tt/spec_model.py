# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speculative decoding (DSpark drafter + greedy verification of k drafts) on top of a built ``dsv41_model.Model``: SAME weights, SAME paged pool.

``SpecRunner(model, k)`` builds, without duplicating any weight tensor:
  * "views" of the model's decode layers for n = 1 + k token rows per user (shallow copies of the attention / MoE objects with their T-dependent members replaced:
    ``SpecPaged*Attention`` class swap on the model's own weights, pool, ring slots, ``prev_cs`` and indexer key slabs; the MoE block reuses the expert weights with a
    batch-per-device = U * n config and its own scratch buffers),
  * the drafter (``mtp.DSparkDrafter``, the checkpoint's 3 DSpark stages) and a ``SpecDecoder`` (verify + accept + draft as ONE traced step).
The model must be built with ``DSV41_RING_ROWS = 288`` (>= 255 + k for the tail-replay seeding; 160 with DSV41_SPEC_FULL_REPLAY=1): a rejected speculative write must not overwrite a window row the next round still needs.

Flow after the model's real PREFILL (traced, paged hand-off incl. index keys):
  ``seed(prompt_tokens, lens, first)``  seeds the drafter's rings and the first 5 drafts from the TAPS of the prefill (tt/prefill_taps.py: the stream means at the input of layers 37-39 of the
                                       last 128 prompt positions, written inside the traced prefill chunk; ``seed_from_prefill``: eager, a few tens of ms for any number of users). Without taps for a user (a
                                       context that is not what the prefill left: the model was built without spec / the user decoded since) it replays the last 128 prompt tokens through the verify step
                                       (even start position, accept count forced), which gives the same state (the compressor state / pool / keys of the hand-off are used as they are);
  ``run(first, lens, max_new, ...)``   the speculative loop: one traced round per iteration (verify n rows per user, accept, state commit, draft), host = Engram rows.
"""

import copy
import os
import time

import torch
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attention import HEAD_DIM, PAD_HEADS
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import CONFIG_PATH
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.mtp import (
    BLOCK,
    L_D,
    RING,
    ChunkedDrafter,
    DSparkDrafter,
    load_mtp_stage,
)
from models.demos.blackhole.deepseek_v41_flash.tt.paged_attention import DSV41PagedStepState
from models.demos.blackhole.deepseek_v41_flash.tt.spec_decoder import SpecDecoder
from models.demos.blackhole.deepseek_v41_flash.tt.spec_paged import (
    SpecIndexer,
    SpecPagedCompressedAttention,
    SpecPagedWindowAttention,
)


def _ucfg(T):
    return ttnn.create_sharded_memory_config(
        shape=(PAD_HEADS, HEAD_DIM),
        core_grid=ttnn.num_cores_to_corerangeset(min(T, 64), ttnn.CoreCoord(8, 8), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class _MoEBig:
    """DSV41_SPEC_ROWS=1, Tn > 32 rows per mesh row: the grouped moe_compute front-end of the prefill (G = Tn/32 chunks of 32 rows per call, router per 32-row slice)
    over the SAME expert weights, with the DSV41MoEBlock.forward signature."""

    def __init__(self, moe, Tn, buffers):
        from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import DSV41PrefillMoE

        assert Tn % 32 == 0
        self.pm = DSV41PrefillMoE(moe, T=32, buffers=buffers, g=Tn // 32)
        self.decode = self.pm.decode
        self.mesh_device = moe.mesh_device

    def warmup(self):
        self.pm.warmup()

    def forward(self, h, h_tok, forced_routing=None):
        assert forced_routing is None
        return self.pm.forward(h, h_tok)


def _moe_view(moe, Tn, buffers):
    """DSV41MoEBlock over the SAME expert weights / gate with batch_per_device = Tn (own config + scratch buffers)."""
    if Tn > 32 and os.environ.get("DSV41_SPEC_ROWS") == "1":
        mb = _MoEBig(moe, Tn, buffers)
        return mb, mb.decode.buffers
    from models.common.modules.moe.tt_moe_decode import _TTMoEDecodeBuffers
    from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig

    md = moe.mesh_device
    text = CONFIG_PATH.read_text()
    text = text.replace("batch_per_device: 4 ", f"batch_per_device: {Tn} ", 1)
    text = text.replace("num_shared_experts: 1", "num_shared_experts: 0").replace(
        "  shared_expert_ids_to_devices: fully_replicated\n", ""
    )
    cfg = TTMoEDecodeConfig.from_yaml(text, topology=ttnn.Topology.Linear)
    if cfg.mesh_shape != tuple(md.shape):
        cfg = cfg.with_mesh_shape(tuple(md.shape))
    if cfg.batch_per_device != Tn:
        cfg = cfg.model_copy(update={"batch_per_device": Tn})
    if cfg.num_fast_reduce_outputs == 1:
        cfg = cfg.model_copy(
            update={"reduce": cfg.reduce.model_copy(update={"output_memory_config": ttnn.DRAM_MEMORY_CONFIG})}
        )
    if buffers is None:
        bd = cfg.buffers.model_dump()
        bd["compute_tilize_drain_core"] = ttnn.experimental.get_moe_tilize_drain_core(
            md,
            cfg.compute.output_height_shard_dim,
            auto_output_width_shard_dim(cfg.hidden_size, matmul_ring_size=effective_matmul_ring_size(md)),
            cfg.hidden_size,
            mux_core_range_set=cfg.compute.mux_core_range_set,
        )
        buffers = _TTMoEDecodeBuffers(md, **bd)
    dec = copy.copy(moe.decode)  # shares expert_state, mesh handles, padding flags
    dec.config, dec.buffers = cfg, buffers
    mv = copy.copy(moe)
    mv.decode, mv.decode_config = dec, cfg
    return mv, buffers


class SpecRunner:
    def __init__(self, model, k, max_pos=None, drafter=None, draft=True, Ub=None):
        """``drafter``: an existing (root) drafter of a sibling runner (adaptive verification length): shared weights + rings, viewed for this block size."""
        assert (
            (0 if drafter is not None else 1) <= k <= BLOCK
        )  # k = 0: plain-like round (adaptive scheduler's no-spec mode), only as a sibling of a real runner
        self.m, self.k, self.n = model, k, k + 1
        self.draft = draft
        self.md, self.U, self.rows, self.cols, self.B = model.md, model.U, model.rows, model.cols, model.B
        # decode bucket (adapter, Ub < U users per mesh row): the runner steps the users u < Ub of every mesh row (model user r * U + u), over the SAME state (prefix views of prev_cs and of
        # the index-key slab, the model's pool / rings by user index, the first chunks of the drafter); ``phys`` = the model user of every runner row
        self.Ufull, self.phys = model.U, None
        if Ub is not None and Ub < model.U:
            assert drafter is not None, "a bucket runner needs the drafter (first chunks) of the full runner"
            self.U, self.B = int(Ub), self.rows * int(Ub)
            self.phys = torch.tensor([(i // self.U) * model.U + i % self.U for i in range(self.B)], dtype=torch.long)
        n, U = self.n, self.U
        # tail-replay seeding: rows q in [S-128, S) read the window [q-127, q] => the prefill-written ring must hold 255 rows (+k slack) below S
        assert model.pool.ring_rows >= 255 + k or (
            os.environ.get("DSV41_SPEC_FULL_REPLAY") == "1" and model.pool.ring_rows >= 128 + k
        ), f"pool ring_rows {model.pool.ring_rows}: build the model with DSV41_RING_ROWS=288 (160 is enough only with DSV41_SPEC_FULL_REPLAY=1)"
        self.T = U * n
        self.max_pos = max_pos or (model.max_ctx + 64)
        t0 = time.time()
        model.log_dram("spec: before runner build")
        views, by_id, idx_views, layers, groups = {}, {}, {}, [], {}
        idx_by_id = {}
        buffers, first = None, True
        for L, layer, key in model.built:
            a = model.attns[L]
            v = copy.copy(a)
            v.T, v.U, v.n, v.nq, v.cs_block, v._ucfg = self.T, U, n, n, None, _ucfg(self.T)
            if hasattr(a, "ratio"):
                v.__class__ = SpecPagedCompressedAttention
                v.source = by_id[id(a.source)] if a.source is not None else None
                v.indexer = None
                if self.phys is not None and getattr(a, "prev_cs", None) is not None:
                    from models.demos.blackhole.deepseek_v41_flash.tt.decode_buckets import view_rows

                    v.prev_cs = view_rows(a.prev_cs, U)  # [1,1,Ub,1024] over the SAME buffer
                if a.indexer is not None:
                    assert a.indexer.backend == "matmul", "spec verify needs the matmul indexer backend"
                    iv = copy.copy(a.indexer)
                    iv.__class__ = SpecIndexer
                    iv.n, iv.R = n, U * n
                    if self.phys is not None:
                        iv.T = U
                        iv.U_slab = int(a.indexer.k_cache.shape[0])
                        iv._kc_view = None
                        own = getattr(a.indexer, "_slab_owner", None)
                        iv._slab_owner = idx_by_id[id(own)] if own is not None else None
                        iv._pad_idx = ttnn.from_torch(
                            torch.full((iv.U_slab - U,), -1, dtype=torch.int32),
                            device=self.md,
                            dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
                        )
                    idx_by_id[id(a.indexer)] = iv
                    v.indexer = iv
            else:
                v.__class__ = SpecPagedWindowAttention
            by_id[id(a)] = v
            lv = copy.copy(layer)
            lv.T, lv.attention = self.T, v
            lv.moe, buffers = _moe_view(layer.moe, self.T, buffers)
            if first:
                lv.moe.warmup()  # compile pass with the L1 hole (semaphore placement), before any trace exists
                first = False
            if key not in groups:
                groups[key] = DSV41PagedStepState(
                    v, max_pos=self.max_pos, with_indexer=model.use_indexer, per_user_valid=model.use_indexer
                )
            layers.append((L, lv, key))
        emb = copy.copy(model.dec.embedding)
        emb.pre = ttnn.from_torch(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.T, 1, 1, 1),
            device=self.md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
        )
        if drafter is not None:
            self.drafter_root = drafter
            self.drafter = drafter.view_n(n)
        else:
            sh = _Shards()
            stage_w = [load_mtp_stage(i, sh) for i in range(3)]
            Uc = (
                U if BLOCK * U <= 32 else 4
            )  # the drafter's token rows (5 per user) must stay <= 32 per mesh row: chunks of 4 users share weights
            assert U % Uc == 0, f"users per mesh row {U}: the chunked drafter needs a multiple of {Uc}"
            base = DSparkDrafter(
                self.md,
                model.mc,
                model.ccl,
                stage_w,
                emb.weight,
                model.head,
                users_per_row=Uc,
                n=n,
                max_pos=self.max_pos,
            )
            self.drafter = self.drafter_root = base if Uc == U else ChunkedDrafter(base, U // Uc)
            del stage_w
        self.siblings, self.round_ms, self.conf = [], None, None
        self.dec = SpecDecoder(
            self.md, layers, emb, model.head, self.drafter, model.dec.engram, step_states=groups, n=n
        )
        self.dec.enable_sampling(model.mc, model.ccl)
        self.dec.draft_on = draft
        # in-trace sampling of every verify row (DSV41_INTRACE_SAMPLE=1): one sampler over the T = U * n block rows of a mesh row; params uploaded per round (``sample_cfg``), allocated before any trace
        self.dec.samp = self.samp = model._make_sampler(self.T)  # (replayed by trace S of a sampled round only)
        self.sample_cfg = None  # None = greedy rows; else dict(temperature=[B], top_k=[B], top_p=[B], gens=[B generators] | one generator)
        if self.samp is not None and os.environ.get("DSV41_DEMO_SAMPLE"):
            t_, k_, p_ = os.environ["DSV41_DEMO_SAMPLE"].split(":")
            self.sample_cfg = dict(
                temperature=float(t_), top_k=int(k_), top_p=float(p_), gens=torch.Generator().manual_seed(1234)
            )
        self.tid = None
        # sampled verify (adapter): a round with sampled rows replays three traces, A (verify block -> logits) | S (the in-trace sampler draws every block row) | B (accept / commit / draft over the
        # drawn ids), instead of the single trace; the persistent buffers are allocated here, before any trace
        self.pack_mono = self.pack_b = None
        self.sampled, self.tid_a, self.tid_s, self.tid_b = self.samp is not None, None, None, None
        self.sample_stats = {"rows": 0}
        if self.sampled:
            self.dec.alloc_sampled()
        self.log = model.log
        model.log_dram("spec: after runner build (views + drafter)")
        self.log(f"spec runner built: k={k} n={n} T={self.T} rows/mesh-row ({time.time() - t0:.0f}s)")

    # ---- host feed / readback ---------------------------------------------------------------------------------------------------
    def _feed(self, X, base):
        """X [B, n] block tokens, base [B] position of column 0."""
        m, B, n = self.m, self.B, self.n
        rows_d = {}
        if m.host_rows is not None:
            hs = m.hasher(X, base, rows=self.phys)
            rows_d = {l: r.reshape(B * n, 1, -1) for l, r in m.host_rows.rows_all(hs, m.engram_ids).items()}
        pos = (base.reshape(B, 1) + torch.arange(n).reshape(1, n)).reshape(-1)
        self.dec.set_packed_inputs(X.reshape(-1), rows_d, pos)

    def _readback(self, pack=None):
        U, n, rows, cols = self.U, self.n, self.rows, self.cols
        # ``dec.pack`` is re-bound by every traced forward (a capture binds the NEW output tensor): a sampled runner keeps the output buffer of each of its traces
        pk = pack if pack is not None else (self.pack_mono if self.pack_mono is not None else self.dec.pack)
        v = torch.cat([ttnn.to_torch(ttnn.get_device_tensors(pk)[r * cols]).reshape(1, -1) for r in range(rows)]).long()
        T_loc = U * n
        a = v[:, :T_loc].reshape(self.B, n)
        mm = v[:, T_loc : T_loc + U].reshape(self.B)
        if not self.draft:  # no-draft round: stale drafts are the caller's business
            self.conf = None
            return a, mm, torch.zeros(self.B, BLOCK, dtype=torch.long)
        d = v[:, T_loc + U : T_loc + U + BLOCK * U].reshape(rows, BLOCK, U).permute(0, 2, 1).reshape(self.B, BLOCK)
        rest = v[:, T_loc + U + BLOCK * U :]
        self.conf = (
            rest.reshape(rows, BLOCK, U).permute(0, 2, 1).reshape(self.B, BLOCK).float() / 65535.0
            if rest.shape[1] == BLOCK * U
            else None
        )  # confidence head: P(draft j accepted | drafts < j accepted), j = 1..5 (see tools/spec_conf)
        return a, mm, d

    def _ensure(self, base):
        """pages of the block rows (no-op / in-place upload); a bucket runner maps its rows to the model users (the others stay at position 0)."""
        if self.phys is None:
            self.m.pool.ensure(base + self.n, lookahead=self.n + 16)
        else:
            full = torch.zeros(self.m.B, dtype=torch.long)
            full[self.phys] = base.long()
            self.m.pool.ensure(full + self.n, lookahead=self.n + 16)

    def _round(self, X, base, force=None):
        self._ensure(base)
        self.dec.set_force(-1 if force is None else force)
        self._feed(X, base)
        if self.sample_cfg is not None:
            if not self.sampled:
                raise ValueError("sampled rows need the in-trace sampler (DSV41_INTRACE_SAMPLE=1, the default)")
            # sampled rows: verify | sample | accept-commit-draft, three replays and one synchronization (a round without sampled rows replays the single trace: the greedy fast path)
            self._set_sample_params()
            ttnn.execute_trace(self.md, self.tid_a, cq_id=0, blocking=False)
            ttnn.execute_trace(self.md, self.tid_s, cq_id=0, blocking=False)
            ttnn.execute_trace(self.md, self.tid_b, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.md)
            self.sample_stats["rows"] += 1
            return self._readback(self.pack_b)
        ttnn.execute_trace(self.md, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.md)
        return self._readback()

    def _set_sample_params(self):
        """Per block row (user-major, ``n`` rows per user) sampling params of the in-trace sampler for the next sampled round: every row draws from its own request's distribution with its own uniform
        (``sample_cfg``: temperature / top_k / top_p [B] and ``gens`` (a generator per user, or one shared by all users)).
        """
        from models.demos.blackhole.deepseek_v41_flash.tt.device_sampler import sampling_rows

        n, B = self.n, self.B
        c = self.sample_cfg
        rep = lambda x: [v for v in (list(x) if isinstance(x, (list, tuple)) else [x] * B) for _ in range(n)]
        g = c["gens"]
        gens = [x for x in g for _ in range(n)] if isinstance(g, (list, tuple)) else g
        self.samp.set_params(*sampling_rows(rep(c["temperature"]), rep(c["top_k"]), rep(c["top_p"]), gens, B * n))

    def free_other_traces(self):
        """Release the plain-decode trace and the prefill chunk trace + per-chunk buffers (they are re-captured by the next prefill / decode call)."""
        self.m.release_trace()
        self.m.prefill_model.teardown_dyn()  # NOTE: frees only ~11 MiB/bank; at ISL > 512 the next spec round read garbage keys after it (off by default)
        self.m.log_dram("spec: after freeing decode + prefill traces")

    def prepare(self, X, base):
        """Eager compile pass (creates every lazily built constant / output) + state snapshot. With several runners ALL of them must be prepared before the FIRST trace is captured:
        a tensor allocated after a trace was captured can land on that trace's (freed) scratch memory and is then overwritten by every replay of that trace (hang).
        """
        dec = self.dec
        self.snaps = dec.snapshot_states()
        self._ensure(base)
        dec.set_force(self.n - 1)
        self._feed(X, base)
        dec.forward()
        ttnn.synchronize_device(self.md)
        dec.restore_states(self.snaps)
        root = self.drafter_root
        if (
            self.phys is None
            and getattr(getattr(self.m, "sink", None), "taps", None) is not None
            and not getattr(root, "_taps_compiled", False)
        ):  # compile pass of the seeding from the prefill taps (nothing selected: no ring row written), before any trace exists
            root._taps_compiled = True
            self._taps_seed(
                torch.zeros(self.B, dtype=torch.bool),
                torch.ones(self.B, dtype=torch.long),
                torch.zeros(self.B, dtype=torch.long),
                dry=True,
            )
        if (
            self.sampled
        ):  # compile pass of the three parts of the sampled round (creates their lazily built tensors before any trace exists)
            dec.set_force(self.n - 1)
            self._feed(X, base)
            dec.forward_verify()
            dec.forward_sampler()
            dec.forward_tail()
            ttnn.synchronize_device(self.md)
            dec.restore_states(self.snaps)

    def capture_trace(self):
        dec = self.dec
        self.tid = ttnn.begin_trace_capture(self.md, cq_id=0)
        dec.forward()
        ttnn.end_trace_capture(self.md, self.tid, cq_id=0)
        ttnn.synchronize_device(self.md)
        dec.restore_states(self.snaps)
        if self.sampled:
            self.pack_mono = dec.pack
            self.tid_a = ttnn.begin_trace_capture(self.md, cq_id=0)
            dec.forward_verify()
            ttnn.end_trace_capture(self.md, self.tid_a, cq_id=0)
            self.tid_s = ttnn.begin_trace_capture(self.md, cq_id=0)
            dec.forward_sampler()
            ttnn.end_trace_capture(self.md, self.tid_s, cq_id=0)
            self.tid_b = ttnn.begin_trace_capture(self.md, cq_id=0)
            dec.forward_tail()
            ttnn.end_trace_capture(self.md, self.tid_b, cq_id=0)
            self.pack_b = dec.pack
            ttnn.synchronize_device(self.md)
            dec.restore_states(self.snaps)
        self.m.log_dram("spec: after trace capture")

    def calibrate(self, X, base, reps):
        """Round-time calibration: replay the captured block with all drafts forced accepted (state restored afterwards)."""
        ws = []
        for _ in range(reps + 1):
            t = time.perf_counter()
            self._round(X, base, torch.full((self.B,), self.n - 1))
            ws.append((time.perf_counter() - t) * 1e3)
        self.round_ms = sorted(ws[1:])[len(ws[1:]) // 2]
        self.dec.restore_states(self.snaps)
        self.log(
            f"spec calibration: k={self.k} n={self.n} T={self.T} round {self.round_ms:.1f} ms (reps {[round(w, 1) for w in ws]})"
        )

    def capture(self, X, base):
        """Compile pass + trace capture (state restored). Call AFTER the prefill, before ``seed``; X [B,n] / base [B] any valid block."""
        self.prepare(X, base)
        self.capture_trace()
        reps = int(os.environ.get("DSV41_SPEC_CALIB", "0"))
        if reps > 0:
            self.calibrate(X, base, reps)

    def release(self):
        for name in ("tid", "tid_a", "tid_s", "tid_b"):
            if getattr(self, name, None) is not None:
                ttnn.release_trace(self.md, getattr(self, name))
                setattr(self, name, None)

    # ---- hand-off: drafter seeding ----------------------------------------------------------------------------------------------
    def _capture_all(self, block_of, b0):
        """First use after a release: compile pass + trace capture of this runner and of every sibling (adaptive set) on the block ``block_of(n)`` ([B, n] tokens) at positions ``b0``."""
        B = self.B
        rs = [self] + list(
            self.siblings
        )  # adaptive: every runner of the set captures its trace on the same first block
        blocks = {id(r_): block_of(r_.n) for r_ in rs}
        for r_ in rs:  # phase 1: all compile passes, before ANY trace exists
            r_.prepare(blocks[id(r_)], b0)
        for r_ in rs:  # phase 2: the traces
            r_.capture_trace()
        reps = int(os.environ.get("DSV41_SPEC_CALIB", "0"))
        if os.environ.get("DSV41_SPEC_SWITCHTEST") == "1":  # diagnostic: alternate the traces of the runners
            for it in range(3):
                for r_ in rs:
                    self.log(f"switchtest it {it} -> k={r_.k}")
                    r_._round(blocks[id(r_)], b0, torch.full((B,), r_.n - 1))
            self.log("switchtest done")
        if reps > 0:
            for r_ in rs:
                r_.calibrate(blocks[id(r_)], b0, reps)

    def taps_ready(self, lens, users=None):
        """True when the prefill's drafter taps (tt/prefill_taps.py) hold the last 128 positions of every user of ``users`` (default: every user with a prompt longer than the dummy
        length 1 of an idle row) at the prompt length ``lens``."""
        sink = getattr(self.m, "sink", None)
        if getattr(sink, "taps", None) is None or self.phys is not None:
            return False
        lens = torch.as_tensor(lens).long()
        users = [b for b in range(self.B) if int(lens[b]) > 1] if users is None else list(users)
        return bool(users) and all(self.m.tap_n.get(b) == int(lens[b]) for b in users)

    def seed_from_prefill(self, lens, first, users=None, draft=True):
        """Seed the drafter of ``users`` (default: every user with taps) from the PREFILL's taps: no replay of the prompt tail through the verify trace. ``lens`` [B] prompt lengths (the position
        of the first generated token), ``first`` [B] the first generated token. Writes the drafter's ring rows of the last 128 positions of every selected user (eager, tt/mtp.py
        ``DSparkDrafter.seed_from_taps``), then drafts the first 5 tokens exactly as the last replay round does (token ``first`` at the frontier ``lens - 1``). Rows of the other users are
        untouched; ``dev_first`` / ``d_final`` / ``conf_final`` are updated for the selected users only. Returns (X [B,n] the first block, base [B]).
        """
        B, n, rows, cols = self.B, self.n, self.rows, self.cols
        lens = torch.as_tensor(lens).long()
        first = torch.as_tensor(first).long()
        taps, Ud = self.m.sink.taps, self.m.U
        sel = torch.zeros(B, dtype=torch.bool)
        for b in range(B) if users is None else users:
            if self.m.tap_n.get(int(b)) == int(lens[b]) and int(lens[b]) > 0:
                sel[int(b)] = True
        if (
            self.tid is None
        ):  # (after a release) compile pass + trace capture on a block that writes only positions >= the prompt length
            X = torch.zeros(B, self.n, dtype=torch.long)
            X[:, 0] = first
            b0 = torch.clamp(lens, max=self.m.max_ctx - self.n - 1)
            self._capture_all(lambda nn: torch.cat([X[:, :1], torch.zeros(B, nn - 1, dtype=torch.long)], dim=1), b0)
        if getattr(self, "d_final", None) is None:
            self.d_final = torch.zeros(B, BLOCK, dtype=torch.long)
            self.conf_final = torch.zeros(B, BLOCK)
            self.dev_first = torch.zeros(B, dtype=torch.long)
        self._taps_seed(sel, lens, first, draft=draft)
        X0 = torch.zeros(B, n, dtype=torch.long)
        X0[:, 0] = first
        X0[:, 1:] = self.d_final[:, : self.k]
        return X0, lens.clone()

    def _taps_seed(self, sel, lens, first, dry=False, draft=True):
        """Body of ``seed_from_prefill``: per drafter chunk with a selected user, ring rows from the stash + the first drafts. ``dry``: run every chunk with nothing selected (compile pass:
        writes no ring row, keeps no draft). ``draft`` False: only the ring rows (the caller's next verify round drafts).
        """
        from models.demos.blackhole.deepseek_v41_flash.tt.prefill_taps import TAP_ROWS

        B, rows, cols = self.B, self.rows, self.cols
        taps, Ud = self.m.sink.taps, self.m.U
        root = self.drafter_root
        subs = getattr(root, "subs", [root])
        Uc = taps.Uc
        shard = taps.shard
        up = lambda t, dt, lay=ttnn.ROW_MAJOR_LAYOUT: ttnn.from_torch(
            t.contiguous(),
            device=self.md,
            dtype=dt,
            layout=lay,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        j = torch.arange(TAP_ROWS)
        for c, sub in enumerate(subs):
            assert sub.U == Uc, f"drafter chunk of {sub.U} users != tap chunk {Uc}"
            gb = torch.tensor(
                [[r * Ud + c * Uc + u for u in range(Uc)] for r in range(rows)]
            )  # [rows, Uc] global users of this chunk
            ok = sel[gb]
            if not bool(ok.any()) and not dry:
                continue
            S = lens[gb]  # [rows, Uc]
            # position held by stash slot j of user (r, u): the latest position < S with position % 128 == j (valid: >= 0 and the user is selected)
            p = (S.unsqueeze(-1) - 1) - ((S.unsqueeze(-1) - 1 - j.reshape(1, 1, -1)) % TAP_ROWS)  # [rows, Uc, 128]
            valid = ok.unsqueeze(-1) & (p >= 0)
            pos = torch.where(valid, p, torch.zeros_like(p)).reshape(rows * Uc * TAP_ROWS)
            # selection matrix (ring slot <- stash slot) and keep mask of the ring writes
            perm = torch.zeros(rows, Uc, L_D, TAP_ROWS)
            keep = torch.ones(rows, Uc, L_D, HEAD_DIM)
            ri, ui, ji = valid.nonzero(as_tuple=True)
            slot = p[ri, ui, ji] % RING
            perm[ri, ui, slot, ji] = 1.0
            keep[ri, ui, slot, :] = 0.0
            hidden = taps.chunk_hidden(c)
            sub.seed_from_taps(
                hidden,
                up(pos.to(torch.int32), ttnn.int32),
                up(perm.reshape(rows * Uc, 1, L_D, TAP_ROWS).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
                up(keep.reshape(rows * Uc, 1, L_D, HEAD_DIM).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            )
            ttnn.deallocate(hidden)
            if not draft:
                continue
            # first drafts: token ``first`` at the frontier lens - 1 (what the last replay round drafts from)
            t_dev = up(first[gb].reshape(rows * Uc, 1).to(torch.int32), ttnn.uint32)
            f_dev = up((lens[gb] - 1).clamp(min=0).reshape(rows * Uc, 1).to(torch.int32), ttnn.int32)
            d, drafts = sub.draft_full(t_dev, f_dev)
            cq = sub.conf_q
            toks = torch.stack(
                [
                    torch.stack(
                        [
                            ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).reshape(-1)[:Uc].long()
                            for r in range(rows)
                        ]
                    )
                    for t in d["tokens"]
                ],
                dim=-1,
            )  # [rows, Uc, 5]
            conf = torch.stack(
                [ttnn.to_torch(ttnn.get_device_tensors(cq)[r * cols]).reshape(BLOCK, Uc).float() for r in range(rows)]
            ).permute(
                0, 2, 1
            )  # [rows, Uc, 5]
            for r in range(rows):
                for u in range(Uc):
                    b = int(gb[r, u])
                    if sel[b]:
                        self.d_final[b] = toks[r, u]
                        self.conf_final[b] = conf[r, u] / 65535.0
                        self.dev_first[b] = int(first[b])
        ttnn.synchronize_device(self.md)

    def seed(self, tokens, lens, first):
        """tokens [B, L] prompts, lens [B], first [B] (the prefill's first generated token). Seeds the drafter from the prefill's taps when it has them for every user (``seed_from_prefill``), else
        replays [p0_u, S_u) per user (p0 even, >= S-128-1) with forced accepts.
        Returns (X [B,n] the first block (first token + drafts), base [B])."""
        lens = torch.as_tensor(lens).long()
        if self.taps_ready(lens):
            X0, base = self.seed_from_prefill(lens, first)
            if (
                os.environ.get("DSV41_SEED_COMPARE") == "1"
            ):  # oracle: the same state after the replay of the prompt tail, compared in this process; the run continues from the taps' state
                self._compare_with_replay(tokens, lens, first)
                X0, base = self.seed_from_prefill(lens, first)
            return X0, base
        return self._seed_replay(tokens, lens, first)

    def _ring_rows(self, lens):
        """host {user: [3 stages][positions p in [max(0, S - 128), S)] -> ring row [512] fp32} of the drafter's attention rings (diagnostics)."""
        rows, cols, Ud = self.rows, self.cols, self.m.U
        subs = getattr(self.drafter_root, "subs", [self.drafter_root])
        out = {}
        for c, sub in enumerate(subs):
            for s_, a in enumerate(sub.attn):
                devs = ttnn.get_device_tensors(a.cache)
                for r in range(rows):
                    cache = ttnn.to_torch(devs[r * cols]).float().reshape(sub.U, -1, 512)
                    for u in range(sub.U):
                        b = r * Ud + c * sub.U + u
                        S = int(lens[b])
                        ps = list(range(max(0, S - 128), S))
                        out.setdefault(b, [None, None, None])[s_] = cache[u, [p % 160 for p in ps]].clone()
        return out

    def _compare_with_replay(self, tokens, lens, first):
        """DSV41_SEED_COMPARE=1: after the seeding from the taps, run the replay seeding of the same prompts and compare the drafter ring rows of the last 128 positions, the first drafts and
        the confidences (logged)."""
        B = self.B
        r_t = self._ring_rows(lens)
        d_t, c_t = self.d_final.clone(), self.conf_final.clone()
        self._seed_replay(tokens, lens, first)
        r_r = self._ring_rows(lens)
        cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))
        allc = [[], [], []]  # per stage: (user, position, cosine)
        for st in range(3):
            for b in range(B):
                S = int(lens[b])
                if S <= 1:
                    continue
                a, b_ = r_t[b][st], r_r[b][st]
                for i in range(a.shape[0]):
                    allc[st].append((b, max(0, S - 128) + i, cos(a[i], b_[i])))
        worst = []
        for st in range(3):
            cc = torch.tensor([c for _, _, c in allc[st]])
            pp = torch.tensor([p for _, p, _ in allc[st]])
            q = cc.quantile(torch.tensor([0.01, 0.05, 0.5]))
            worst.append(
                (
                    float(cc.min()),
                    float(q[0]),
                    float(q[1]),
                    float(q[2]),
                    float((cc < 0.95).float().mean()),
                    int(pp[cc.argmin()]),
                )
            )
        try:
            torch.save(
                {
                    "lens": lens,
                    "ring_taps": r_t,
                    "ring_replay": r_r,
                    "d_taps": d_t,
                    "d_replay": self.d_final,
                    "conf_taps": c_t,
                    "conf_replay": self.conf_final,
                    "cos": allc,
                },
                os.environ.get("DSV41_SEED_COMPARE_DUMP", "/tmp/seed_compare_dump.pt"),
            )
        except Exception as e:  # diagnostics only
            self.log(f"SEED_COMPARE dump failed: {e}")
        same = (d_t[:, : self.k] == self.d_final[:, : self.k]).float().mean(0).tolist()
        self.log(
            "SEED_COMPARE ring rows (replay vs taps) cosine over all (user, position) rows, per stage: "
            + "; ".join(
                f"stage {i}: min {a:.4f} (at position {pm}) p1 {b:.4f} p5 {c:.4f} median {d:.4f}, share < 0.95 {e:.3f}"
                for i, (a, b, c, d, e, pm) in enumerate(worst)
            )
            + f"; first drafts equal per draft index {[round(x, 3) for x in same]}; max |conf diff| {float((c_t - self.conf_final).abs().max()):.4f}; mean conf taps {float(c_t[:, : self.k].mean()):.3f} replay {float(self.conf_final[:, : self.k].mean()):.3f}"
        )

    def _seed_replay(self, tokens, lens, first):
        B, n = self.B, self.n
        # replay length: the last 128 tokens = the full attention window of the drafter (one verify round per 4 replayed tokens, ~115 ms each at B=32, whatever the number of rows to seed)
        tail = 128
        p0 = torch.clamp((lens - tail) // 2 * 2, min=0)
        if (
            os.environ.get("DSV41_SPEC_FULL_REPLAY") == "1"
        ):  # replay the whole prompt (no reliance on the prefill state; ISL <= ~2k) instead of the last 128 tokens
            p0 = torch.zeros_like(p0)
        nbl = -(-(lens - p0) // n)  # blocks per user
        seqs = []
        for b in range(B):
            s = torch.zeros(int(lens[b]) + 1 + n, dtype=torch.long)
            s[: int(lens[b])] = tokens[b, : int(lens[b])].long()
            s[int(lens[b])] = int(first[b])
            seqs.append(s)
        d_final = torch.zeros(B, BLOCK, dtype=torch.long)
        conf_final = torch.zeros(B, BLOCK)
        dev_first = torch.zeros(B, dtype=torch.long)
        for r in range(int(nbl.max())):
            if (
                self.tid is None
            ):  # first use: compile pass + trace capture on the first replay block (writes the same state the replay writes)
                self._capture_all(lambda nn: torch.stack([seqs[b][int(p0[b]) : int(p0[b]) + nn] for b in range(B)]), p0)
            rr = torch.minimum(torch.full((B,), r), nbl - 1)  # finished users repeat their last block (idempotent)
            base = p0 + rr * n
            X = torch.stack([seqs[b][int(base[b]) : int(base[b]) + n] for b in range(B)])
            force = torch.where(rr == nbl - 1, lens - 1 - base, torch.full((B,), n - 1))
            a, mm, d = self._round(X, base, force)
            if r % 8 == 0:
                self.log(f"spec seed: replay block {r}/{int(nbl.max())} (k={self.k})")
            for b in range(B):
                if r == int(nbl[b]) - 1:
                    d_final[b] = d[b]
                    if self.conf is not None:
                        conf_final[b] = self.conf[b]
                    dev_first[b] = a[b, int(lens[b]) - 1 - int(base[b])]
        self.dev_first, self.d_final, self.conf_final = dev_first, d_final, conf_final
        X0 = torch.zeros(B, n, dtype=torch.long)
        X0[:, 0] = torch.as_tensor(first).long()
        X0[:, 1:] = d_final[:, : self.k]
        return X0, lens.clone()

    # ---- the speculative loop -----------------------------------------------------------------------------------------------------
    def run(self, X, base, max_new, eos=None, active=None):
        """X [B,n] (first token + drafts), base [B] = position of the first token. Generates up to ``max_new`` tokens per user (counting the first token).
        -> (gen: list of token lists per user (the first token included), stats dict). ``active``: [B] bool, users that count (padding users are ignored).
        """
        B, n, k = self.B, self.n, self.k
        base = base.clone()
        X = X.clone()
        gen = [[int(X[b, 0])] for b in range(B)]
        self.m_hist = [[] for _ in range(B)]  # accepted drafts per round per user
        self.gaps = [
            [] for _ in range(B)
        ]  # gaps[b][i-1] = top1-top2 logit gap of the argmax that produced generated token i (near-tie evidence)
        done = torch.zeros(B, dtype=torch.bool) if active is None else ~torch.as_tensor(active)
        walls, emitted, ms = [], [], []
        top = self.m.max_ctx - n - 1
        while not bool(done.all()):
            t = time.perf_counter()
            a, mm, d = self._round(X, base)
            walls.append((time.perf_counter() - t) * 1e3)
            if len(walls) <= int(os.environ.get("DSV41_SPEC_DBG", "0")):
                for b_ in range(min(B, 2)):
                    self.log(
                        f"SPEC_DBG round {len(walls)} user {b_} base {int(base[b_])} X {X[b_].tolist()} a {a[b_].tolist()} m {int(mm[b_])} newd {d[b_].tolist()} conf {None if self.conf is None else [round(float(c_), 2) for c_ in self.conf[b_]]}"
                    )
            v2 = torch.cat(
                [
                    ttnn.to_torch(ttnn.get_device_tensors(self.dec.top2)[r * self.cols]).reshape(self.U * n, -1)
                    for r in range(self.rows)
                ]
            ).float()
            t2 = v2.topk(2, dim=-1).values.reshape(B, n, 2)
            gp = t2[..., 0] - t2[..., 1]
            for b in range(B):
                if done[b]:
                    continue
                mb = int(mm[b])
                toks = [int(x) for x in a[b, : mb + 1]]
                if eos is not None and eos in toks:
                    toks = toks[: toks.index(eos) + 1]
                    done[b] = True
                gen[b] += toks
                self.gaps[b] += [float(x) for x in gp[b, : len(toks)]]
                emitted.append(len(toks))
                ms.append(mb)
                self.m_hist[b].append(mb)
                base[b] += mb + 1
                X[b, 0] = a[b, mb]
                X[b, 1 : 1 + k] = d[b, :k]
                if len(gen[b]) >= max_new or int(base[b]) > top:
                    done[b] = True
            # finished users keep running inside the batch: keep their positions fixed (their rows rewrite the same cache slots)
        mt = torch.tensor(ms)
        stats = {
            "rounds": len(walls),
            "round_ms": sum(walls) / max(len(walls), 1),
            "tok_per_round": sum(emitted) / max(len(emitted), 1),
            "accepted_per_round": float(mt.float().mean()) if len(ms) else 0.0,
            "p_ge": [float((mt >= j).float().mean()) for j in range(1, k + 1)] if len(ms) else [],
        }
        stats["tok_s_user"] = 1e3 * stats["tok_per_round"] / stats["round_ms"] if walls else float("nan")
        return gen, stats


def parse_ks(spec, default):
    """'1,3,5' -> [1, 3, 5]"""
    return sorted({int(x) for x in str(spec or default).split(",") if x.strip()})


class AdaptiveSpec:
    """Adaptive speculative verification length (DSV41_SPEC_ADAPT=1). One resident, traced ``SpecRunner`` per candidate k (own verify views / MoE scratch / trace, ONE shared drafter + rings
    + pool); the drafter always proposes 5 drafts per user, a round of runner k verifies the first k of them (``n = 1 + k`` rows per user). Per round the scheduler (``choose``) picks the
    runner that maximises  E[tokens/round](k) / round_ms(k)  where  E = 1 + sum_{j<=k} S_j,  S_j = mean over active users of prod_{i<=j} p_i  (p = confidence-head acceptance probabilities of the
    drafts about to be verified) and round_ms(k) = the profiled round time (startup calibration of every trace, then an EMA of the measured rounds; ``DSV41_SPEC_TIMES='1:70,3:83,5:104'`` overrides).
    The accept rule is the plain greedy one, so the output stream equals plain greedy decoding for every policy (up to bf16 near-ties). Policies: 'adapt' | 'k<j>' (fixed j, same runner set).
    Per-user masking is NOT used: rows are shared across the batch, a masked user would only lose accepted tokens without saving any compute.
    """

    def __init__(self, model, ks, max_pos=None, seed_k=None):
        ks = sorted(set(ks))
        self.plain_ok = (
            0 in ks
        )  # k = 0 candidate: no-spec rounds (verify 1 row/user + write_main, no drafting) with a probe round (with drafting) every DSV41_SPEC_PROBE rounds
        ks = [k for k in ks if k > 0]
        os.environ.setdefault(
            "DSV41_SPEC_CALIB", "3"
        )  # every runner measures its round time when its trace is captured
        self.m, self.ks, self.B = model, ks, model.B
        seed_k = seed_k or (3 if 3 in ks else ks[-1])
        self.runners = {}
        first = SpecRunner(model, seed_k, max_pos=max_pos)
        self.runners[seed_k] = first
        for k in ks:
            if k != seed_k:
                self.runners[k] = SpecRunner(model, k, max_pos=max_pos, drafter=first.drafter_root)
        self.seedr = first
        first.siblings = [self.runners[k] for k in ks if k != seed_k]
        self.probe_every = int(os.environ.get("DSV41_SPEC_PROBE", "16"))
        self.runner0 = self.probe0 = None
        if self.plain_ok:
            self.runner0 = SpecRunner(
                model, 0, max_pos=max_pos, drafter=first.drafter_root, draft=False
            )  # the fast plain round
            self.probe0 = SpecRunner(
                model, 0, max_pos=max_pos, drafter=first.drafter_root, draft=True
            )  # a round with drafting at the same n = 1: refreshes drafts + confidence
            first.siblings += [self.runner0, self.probe0]
        self.k, self.n = max(ks), max(ks) + 1
        self.log = model.log
        self.policy = os.environ.get("DSV41_SPEC_POLICY", "adapt")
        self.times = {}
        self.cal_a, self.cal_b = [float(x) for x in os.environ.get("DSV41_SPEC_CONF_AB", "1,0").split(",")]
        self.hyst = float(os.environ.get("DSV41_SPEC_HYST", "0.0"))
        self.cal_records = []

    def release(self):
        for r in list(self.runners.values()) + [r_ for r_ in (self.runner0, self.probe0) if r_ is not None]:
            r.release()

    def seed(self, tokens, lens, first):
        X0, base = self.seedr.seed(tokens, lens, first)
        for k, r in self.runners.items():
            if k not in self.times and r.round_ms is not None:
                self.times[k] = r.round_ms
        if self.runner0 is not None and self.runner0.round_ms is not None:
            self.times[0] = self.runner0.round_ms
            self.times_probe = self.probe0.round_ms
        for kv in os.environ.get("DSV41_SPEC_TIMES", "").split(","):
            if kv:
                k, v = kv.split(":")
                self.times[int(k)] = float(v)
        self.log(f"spec adapt: round-time table (ms) {dict(sorted(self.times.items()))}")
        return X0, base

    # ---- scheduler ----------------------------------------------------------------------------------------------------------------------
    def _p(self, conf):
        c = conf.clamp(1e-4, 1 - 1e-4)
        if self.cal_a != 1.0 or self.cal_b != 0.0:
            c = torch.sigmoid(self.cal_a * torch.logit(c) + self.cal_b)
        return c

    def expected_tokens(self, conf, active):
        """E[tokens/round] for every candidate k from the confidence probabilities conf [B,5] (mean over the active users)."""
        S = torch.cumprod(self._p(conf[active]), dim=1).mean(0)  # [5] survival P(first j drafts accepted)
        return {k: 1.0 + float(S[:k].sum()) for k in self.ks}

    def choose(self, conf, active):
        if self.policy.startswith("k"):
            return int(self.policy[1:])
        if not bool(active.any()):
            return self.ks[0]
        E = self.expected_tokens(conf, active)
        score = {k: E[k] / self.times[k] for k in self.ks}
        if self.plain_ok and 0 in self.times:
            score[0] = 1.0 / self.times[0]
        best = max(score, key=score.get)
        cur = getattr(self, "_cur", None)
        if cur is not None and cur in score and score[best] < score[cur] * (1 + self.hyst):
            best = cur
        self._cur = best
        return best

    # ---- the loop -----------------------------------------------------------------------------------------------------------------------
    def run(self, X, base, max_new, eos=None, active=None, policy=None):
        B = self.B
        if policy is not None:
            self.policy = policy
        base = base.clone()
        X5 = torch.zeros(B, 1 + BLOCK, dtype=torch.long)
        X5[:, 0] = X[:, 0]
        X5[:, 1:] = self.seedr.d_final
        conf = self.seedr.conf_final.clone()
        gen = [[int(X5[b, 0])] for b in range(B)]
        self.m_hist = [[] for _ in range(B)]
        self.gaps = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool) if active is None else ~torch.as_tensor(active)
        walls, emitted, ms, ks_used, exp_tok = [], [], [], [], []
        cyc = None  # 'cycle:adapt+k1+k3+k5:4' = interleave the policies in windows of 4 rounds on the SAME stream (no re-prefill: A/B within one pass)
        if self.policy.startswith("cycle:"):
            _, names, win = self.policy.split(":")
            cyc = ([x for x in names.split("+")], int(win))
        base_policy = self.policy
        by = {}  # policy -> dict(rounds, wall, emitted, pairs, acc, ks)
        cal = []  # (user, k, conf[5], m): drafts of this round's block vs outcome
        top = self.m.max_ctx - self.n - 1
        last_pol = self.policy
        stale, since_probe, nprobe = (
            False,
            0,
            0,
        )  # stale: the drafts / confidence in X5 / conf belong to an earlier position (after no-draft k = 0 rounds)
        while not bool(done.all()):
            if cyc is not None:
                self.policy = cyc[0][(len(walls) // cyc[1]) % len(cyc[0])]
            probing = False
            probe_due = since_probe >= self.probe_every or self.policy not in ("adapt", "k0") or self.policy != last_pol
            last_pol = self.policy
            if stale and self.policy != "k0" and probe_due:
                k, probing = (
                    0,
                    True,
                )  # refresh drafts + confidence with a drafting k = 0 round, then let the scheduler decide again
            elif stale:
                k = 0
            else:
                k = self.choose(conf, ~done)
            r = (self.probe0 if probing else self.runner0) if k == 0 else self.runners[k]
            n = k + 1
            cur_pol = self.policy
            exp_tok.append(1.0 if (k == 0 or stale) else self.expected_tokens(conf, ~done)[k])
            if len(walls) < 3:
                self.log(
                    f"spec adapt: round {len(walls)} -> k={k} (policy {self.policy}, times {({q: round(v, 1) for q, v in self.times.items()})})"
                )
            t = time.perf_counter()
            a, mm, d = r._round(X5[:, :n].contiguous(), base)
            if len(walls) % 10 == 0:
                self.log(f"spec adapt: round {len(walls)} k={k} done {int(done.sum())}/{B}")
            w = (time.perf_counter() - t) * 1e3
            walls.append(w)
            ks_used.append(k)
            e0 = len(emitted)
            tk = "probe" if probing else k
            self.times[tk] = 0.8 * self.times[tk] + 0.2 * w if tk in self.times else w
            v2 = torch.cat(
                [
                    ttnn.to_torch(ttnn.get_device_tensors(r.dec.top2)[row * r.cols]).reshape(r.U * n, -1)
                    for row in range(r.rows)
                ]
            ).float()
            t2 = v2.topk(2, dim=-1).values.reshape(B, n, 2)
            gp = t2[..., 0] - t2[..., 1]
            newconf = r.conf
            for b in range(B):
                if done[b]:
                    continue
                mb = int(mm[b])
                if k > 0 and not stale:
                    cal.append((b, k, [float(x) for x in conf[b]], mb))
                toks = [int(x) for x in a[b, : mb + 1]]
                if eos is not None and eos in toks:
                    toks = toks[: toks.index(eos) + 1]
                    done[b] = True
                gen[b] += toks
                self.gaps[b] += [float(x) for x in gp[b, : len(toks)]]
                emitted.append(len(toks))
                ms.append(mb)
                self.m_hist[b].append(mb)
                base[b] += mb + 1
                X5[b, 0] = a[b, mb]
                if r.draft:
                    X5[b, 1:] = d[b]
                if newconf is not None:
                    conf[b] = newconf[b]
                if len(gen[b]) >= max_new or int(base[b]) > top:
                    done[b] = True
            if k == 0 and not probing:
                stale, since_probe = True, since_probe + 1
            elif r.draft:
                stale, since_probe = False, 0
                nprobe += int(probing)
            st_ = by.setdefault(cur_pol, {"rounds": 0, "wall": 0.0, "emitted": 0, "pairs": 0, "acc": 0, "ks": {}})
            st_["rounds"] += 1
            st_["wall"] += w
            st_["emitted"] += sum(emitted[e0:])
            st_["acc"] += sum(ms[e0:])
            st_["pairs"] += len(emitted) - e0
            st_["ks"][k] = st_["ks"].get(k, 0) + 1
        self.policy = base_policy
        self.cal_records = cal
        mt = torch.tensor(ms)
        tot_ms = sum(walls)
        nk = {k: ks_used.count(k) for k in ([0] if self.plain_ok else []) + list(self.ks)}
        nk["probes"] = nprobe
        stats = {
            "rounds": len(walls),
            "round_ms": tot_ms / max(len(walls), 1),
            "tok_per_round": sum(emitted) / max(len(emitted), 1),
            "accepted_per_round": float(mt.float().mean()) if len(ms) else 0.0,
            "p_ge": [float((mt >= j).float().mean()) for j in range(1, BLOCK + 1)] if len(ms) else [],
            "k_hist": nk,
            "policy": self.policy,
            "times": dict(self.times),
            "mean_expected_tokens": sum(exp_tok) / max(len(exp_tok), 1),
        }
        stats["tok_s_user"] = 1e3 * stats["tok_per_round"] / stats["round_ms"] if walls else float("nan")
        stats["by_policy"] = {
            q: {
                "rounds": v["rounds"],
                "round_ms": v["wall"] / max(v["rounds"], 1),
                "tok_per_round": v["emitted"] / max(v["pairs"], 1),
                "accepted_per_round": v["acc"] / max(v["pairs"], 1),
                "k_hist": v["ks"],
            }
            for q, v in by.items()
        }
        for v in stats["by_policy"].values():
            v["tok_s_user"] = 1e3 * v["tok_per_round"] / v["round_ms"]
        return gen, stats

    def conf_report(self):
        """Reliability of the confidence head: per draft position j, conditional (given the prefix accepted) predicted vs observed acceptance in 5 probability bins,
        plus the prefix survival (product) vs observed P(m >= j). Only positions inside the verified block (j <= k of the round) are observed.
        """
        rec = self.cal_records
        out = {"n_records": len(rec), "cond": {}, "surv": {}}
        for j in range(1, BLOCK + 1):
            c_p, c_o, s_p, s_o = [], [], [], []
            for _, k, cf, m in rec:
                if j > k:
                    continue
                s_p.append(float(torch.tensor(cf[:j]).prod()))
                s_o.append(float(m >= j))
                if m >= j - 1:  # position j was actually tested
                    c_p.append(cf[j - 1])
                    c_o.append(float(m >= j))
            for name, P, O in (("cond", c_p, c_o), ("surv", s_p, s_o)):
                if not P:
                    continue
                P, O = torch.tensor(P), torch.tensor(O)
                bins = []
                for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
                    sel = (P >= lo) & (P < hi)
                    if int(sel.sum()):
                        bins.append(
                            (
                                f"{lo:.1f}-{min(hi, 1):.1f}",
                                int(sel.sum()),
                                round(float(P[sel].mean()), 3),
                                round(float(O[sel].mean()), 3),
                            )
                        )
                out[name][j] = {
                    "n": len(P),
                    "pred": round(float(P.mean()), 3),
                    "obs": round(float(O.mean()), 3),
                    "bins": bins,
                }
        return out


def default_ks(U):
    """Candidate verification lengths per users-per-mesh-row (table and rationale in tt/spec_policy.py; the chunked verify rows follow DSV41_SPEC_ROWS)."""
    from models.demos.blackhole.deepseek_v41_flash.tt.spec_policy import default_ks as _dk

    return _dk(U, os.environ.get("DSV41_SPEC_ROWS") == "1")
