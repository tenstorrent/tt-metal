# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speculative decoding (DSpark drafter + greedy verification of k drafts) on top of a built ``dsv41_model.Model``: SAME weights, SAME paged pool.

``SpecRunner(model, k)`` builds, without duplicating any weight tensor:
  * "views" of the model's decode layers for n = 1 + k token rows per user (shallow copies of the attention / MoE objects with their T-dependent members replaced:
    ``SpecPaged*Attention`` class swap on the model's own weights, pool, ring slots, ``prev_cs`` and indexer key slabs; the MoE block reuses the expert weights with a
    batch-per-device = U * n config and its own scratch buffers),
  * the drafter (``mtp.DSparkDrafter``, the checkpoint's 3 DSpark stages) and a ``SpecDecoder`` (verify + accept + draft as ONE traced step).
The model must be built with ``DSV41_RING_ROWS >= 128 + k`` (160): a rejected speculative write must not overwrite a window row the next round still needs.

Flow after the model's real PREFILL (traced, paged hand-off incl. index keys):
  ``seed(prompt_tokens, lens, first)``  replays the last 128 prompt tokens of every user through the verify step (even start position, accept count forced) which
                                       seeds the drafter's rings and the first 5 drafts (the compressor state / pool / keys of the hand-off are used as they are);
  ``run(first, lens, max_new, ...)``   the speculative loop: one traced round per iteration (verify n rows per user, accept, state commit, draft), host = Engram rows.
"""

import copy
import time

import torch
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attention import HEAD_DIM, PAD_HEADS
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import CONFIG_PATH
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.mtp import BLOCK, DSparkDrafter, load_mtp_stage
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
    def __init__(self, model, k, max_pos=None):
        assert 1 <= k <= BLOCK
        self.m, self.k, self.n = model, k, k + 1
        self.md, self.U, self.rows, self.cols, self.B = model.md, model.U, model.rows, model.cols, model.B
        n, U = self.n, self.U
        assert (
            model.pool.ring_rows >= 128 + k
        ), f"pool ring_rows {model.pool.ring_rows}: build the model with DSV41_RING_ROWS=160"
        self.T = U * n
        self.max_pos = max_pos or (model.max_ctx + 64)
        t0 = time.time()
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
        sh = _Shards()
        stage_w = [load_mtp_stage(i, sh) for i in range(3)]
        self.drafter = DSparkDrafter(
            self.md, model.mc, model.ccl, stage_w, emb.weight, model.head, users_per_row=U, n=n, max_pos=self.max_pos
        )
        del stage_w
        self.dec = SpecDecoder(
            self.md, layers, emb, model.head, self.drafter, model.dec.engram, step_states=groups, n=n
        )
        self.dec.enable_sampling(model.mc, model.ccl)
        self.tid = None
        self.log = model.log
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
        d = v[:, T_loc + U :].reshape(rows, BLOCK, U).permute(0, 2, 1).reshape(self.B, BLOCK)
        return a, mm, d

    def _round(self, X, base, force=None):
        self.m.pool.ensure(base + self.n, lookahead=self.n + 16)  # pages of the block rows (no-op / in-place upload)
        self.dec.set_force(-1 if force is None else force)
        self._feed(X, base)
        ttnn.execute_trace(self.md, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.md)
        return self._readback()

    def capture(self, X, base):
        """Compile pass + trace capture (state restored). Call AFTER the prefill, before ``seed``; X [B,n] / base [B] any valid block."""
        dec = self.dec
        snaps = dec.snapshot_states()
        self.m.pool.ensure(base + self.n, lookahead=self.n + 16)
        dec.set_force(self.n - 1)
        self._feed(X, base)
        dec.forward()
        ttnn.synchronize_device(self.md)
        dec.restore_states(snaps)
        self.tid = ttnn.begin_trace_capture(self.md, cq_id=0)
        dec.forward()
        ttnn.end_trace_capture(self.md, self.tid, cq_id=0)
        ttnn.synchronize_device(self.md)
        dec.restore_states(snaps)
        self.snaps = snaps

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
        nbl = -(-(lens - p0) // n)  # blocks per user
        seqs = []
        for b in range(B):
            s = torch.zeros(int(lens[b]) + 1 + n, dtype=torch.long)
            s[: int(lens[b])] = tokens[b, : int(lens[b])].long()
            s[int(lens[b])] = int(first[b])
            seqs.append(s)
        d_final = torch.zeros(B, BLOCK, dtype=torch.long)
        dev_first = torch.zeros(B, dtype=torch.long)
        for r in range(int(nbl.max())):
            if (
                self.tid is None
            ):  # first use: compile pass + trace capture on the first replay block (writes the same state the replay writes)
                b0 = p0.clone()
                self.capture(torch.stack([seqs[b][int(b0[b]) : int(b0[b]) + n] for b in range(B)]), b0)
            rr = torch.minimum(torch.full((B,), r), nbl - 1)  # finished users repeat their last block (idempotent)
            base = p0 + rr * n
            X = torch.stack([seqs[b][int(base[b]) : int(base[b]) + n] for b in range(B)])
            force = torch.where(rr == nbl - 1, lens - 1 - base, torch.full((B,), n - 1))
            a, mm, d = self._round(X, base, force)
            for b in range(B):
                if r == int(nbl[b]) - 1:
                    d_final[b] = d[b]
                    dev_first[b] = a[b, int(lens[b]) - 1 - int(base[b])]
        self.dev_first = dev_first
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
