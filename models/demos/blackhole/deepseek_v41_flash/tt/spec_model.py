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
  ``seed(prompt_tokens, lens, first)``  replays the last 128 prompt tokens of every user through the verify step (even start position, accept count forced) which
                                       seeds the drafter's rings and the first 5 drafts (the compressor state / pool / keys of the hand-off are used as they are);
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
from models.demos.blackhole.deepseek_v41_flash.tt.mtp import BLOCK, ChunkedDrafter, DSparkDrafter, load_mtp_stage
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
        core_grid=ttnn.num_cores_to_corerangeset(T, ttnn.CoreCoord(8, 8), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _moe_view(moe, Tn, buffers):
    """DSV41MoEBlock over the SAME expert weights / gate with batch_per_device = Tn (own config + scratch buffers)."""
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
    def __init__(self, model, k, max_pos=None, drafter=None):
        """``drafter``: an existing (root) drafter of a sibling runner (adaptive verification length): shared weights + rings, viewed for this block size."""
        assert 1 <= k <= BLOCK
        self.m, self.k, self.n = model, k, k + 1
        self.md, self.U, self.rows, self.cols, self.B = model.md, model.U, model.rows, model.cols, model.B
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
        buffers, first = None, True
        for L, layer, key in model.built:
            a = model.attns[L]
            v = copy.copy(a)
            v.T, v.U, v.n, v.nq, v.cs_block, v._ucfg = self.T, U, n, n, None, _ucfg(self.T)
            if hasattr(a, "ratio"):
                v.__class__ = SpecPagedCompressedAttention
                v.source = by_id[id(a.source)] if a.source is not None else None
                v.indexer = None
                if a.indexer is not None:
                    assert a.indexer.backend == "matmul", "spec verify needs the matmul indexer backend"
                    iv = copy.copy(a.indexer)
                    iv.__class__ = SpecIndexer
                    iv.n, iv.R = n, U * n
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
        self.tid = None
        self.log = model.log
        model.log_dram("spec: after runner build (views + drafter)")
        self.log(f"spec runner built: k={k} n={n} T={self.T} rows/mesh-row ({time.time() - t0:.0f}s)")

    # ---- host feed / readback ---------------------------------------------------------------------------------------------------
    def _feed(self, X, base):
        """X [B, n] block tokens, base [B] position of column 0."""
        m, B, n = self.m, self.B, self.n
        rows_d = {}
        if m.host_rows is not None:
            hs = m.hasher(X, base)
            rows_d = {l: r.reshape(B * n, 1, -1) for l, r in m.host_rows.rows_all(hs, m.engram_ids).items()}
        pos = (base.reshape(B, 1) + torch.arange(n).reshape(1, n)).reshape(-1)
        self.dec.set_packed_inputs(X.reshape(-1), rows_d, pos)

    def _readback(self):
        U, n, rows, cols = self.U, self.n, self.rows, self.cols
        v = torch.cat(
            [ttnn.to_torch(ttnn.get_device_tensors(self.dec.pack)[r * cols]).reshape(1, -1) for r in range(rows)]
        ).long()
        T_loc = U * n
        a = v[:, :T_loc].reshape(self.B, n)
        mm = v[:, T_loc : T_loc + U].reshape(self.B)
        d = v[:, T_loc + U : T_loc + U + BLOCK * U].reshape(rows, BLOCK, U).permute(0, 2, 1).reshape(self.B, BLOCK)
        rest = v[:, T_loc + U + BLOCK * U :]
        self.conf = (
            rest.reshape(rows, BLOCK, U).permute(0, 2, 1).reshape(self.B, BLOCK).float() / 65535.0
            if rest.shape[1] == BLOCK * U
            else None
        )  # confidence head: P(draft j accepted | drafts < j accepted), j = 1..5 (see tools/spec_conf)
        return a, mm, d

    def _round(self, X, base, force=None):
        self.m.pool.ensure(base + self.n, lookahead=self.n + 16)  # pages of the block rows (no-op / in-place upload)
        self.dec.set_force(-1 if force is None else force)
        self._feed(X, base)
        ttnn.execute_trace(self.md, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.md)
        return self._readback()

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
        self.m.pool.ensure(base + self.n, lookahead=self.n + 16)
        dec.set_force(self.n - 1)
        self._feed(X, base)
        dec.forward()
        ttnn.synchronize_device(self.md)
        dec.restore_states(self.snaps)

    def capture_trace(self):
        dec = self.dec
        self.tid = ttnn.begin_trace_capture(self.md, cq_id=0)
        dec.forward()
        ttnn.end_trace_capture(self.md, self.tid, cq_id=0)
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
        if self.tid is not None:
            ttnn.release_trace(self.md, self.tid)
            self.tid = None

    # ---- hand-off: drafter seeding ----------------------------------------------------------------------------------------------
    def seed(self, tokens, lens, first):
        """tokens [B, L] prompts, lens [B], first [B] (the prefill's first generated token). Replays [p0_u, S_u) per user (p0 even, >= S-128-1) with forced accepts.
        Returns (X [B,n] the first block (first token + drafts), base [B])."""
        B, n = self.B, self.n
        lens = torch.as_tensor(lens).long()
        p0 = torch.clamp((lens - 128) // 2 * 2, min=0)
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
                b0 = p0.clone()
                rs = [self] + list(
                    self.siblings
                )  # adaptive: every runner of the set captures its trace on the same first replay block
                blocks = {id(r_): torch.stack([seqs[b][int(b0[b]) : int(b0[b]) + r_.n] for b in range(B)]) for r_ in rs}
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
        self.k, self.n = max(ks), max(ks) + 1
        self.log = model.log
        self.policy = os.environ.get("DSV41_SPEC_POLICY", "adapt")
        self.times = {}
        self.cal_a, self.cal_b = [float(x) for x in os.environ.get("DSV41_SPEC_CONF_AB", "1,0").split(",")]
        self.hyst = float(os.environ.get("DSV41_SPEC_HYST", "0.0"))
        self.cal_records = []

    def release(self):
        for r in self.runners.values():
            r.release()

    def seed(self, tokens, lens, first):
        X0, base = self.seedr.seed(tokens, lens, first)
        for k, r in self.runners.items():
            if k not in self.times and r.round_ms is not None:
                self.times[k] = r.round_ms
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
        cal = []  # (user, k, conf[5], m): drafts of this round's block vs outcome
        top = self.m.max_ctx - self.n - 1
        while not bool(done.all()):
            k = self.choose(conf, ~done)
            r, n = self.runners[k], k + 1
            exp_tok.append(self.expected_tokens(conf, ~done)[k])
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
            self.times[k] = 0.8 * self.times[k] + 0.2 * w if k in self.times else w
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
                X5[b, 1:] = d[b]
                if newconf is not None:
                    conf[b] = newconf[b]
                if len(gen[b]) >= max_new or int(base[b]) > top:
                    done[b] = True
        self.cal_records = cal
        mt = torch.tensor(ms)
        tot_ms = sum(walls)
        nk = {k: ks_used.count(k) for k in self.ks}
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
    """Candidate verification lengths per users-per-mesh-row with fast / supported row counts T = U * (1 + k) (mHC fast paths 4 / 8 / 16 / 24 / 32; T <= 32)."""
    table = {1: [1, 3], 2: [1, 3, 5], 4: [1, 3, 5], 8: [1, 3]}
    return table.get(U, [k for k in (1, 3) if U * (1 + k) <= 32] or [1])
