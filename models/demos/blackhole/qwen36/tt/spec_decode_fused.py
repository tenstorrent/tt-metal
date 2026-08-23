# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused decode-width speculative verify (contract v1 in docs/mtp_design.md).

Verify = ONE production-shaped decode step at width B*(K+1) <= 32: full-attention
layers run the standard batched paged decode over pseudo-user rows (user u's
page-table row replicated, per-row positions c_u..c_u+K); GDN layers run the
candidates SEQUENTIALLY in-kernel (fused_decode.op.recurrence_seq_rows) from the
committed anchor state with per-row state stashes and NO writeback. Committing
selects a stash row per user (commit-by-select): the anchor is always the
committed head — no block anchors, no snapshot/restore, no commit chunks, and
with them the whole alloc-under-trace hazard class is gone.

EAGER v1: correctness first. The perf steps (documented in the contract): read
the per-user anchor row index inside the kernel from a device tensor so
commit-by-select is pure data, then ride the production decode-width trace.

Requires the gdn-decode-fused branch (the fused_decode package) in the
workspace; TT_SPEC_FUSED=1 selects this loop in the demo.
"""
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.spec_decode import greedy_accept
from models.demos.blackhole.qwen36.tt.spec_decode_batched import Qwen36BatchedSpeculativeDecoder
from models.tt_transformers.tt.common import Mode


class Qwen36FusedSpeculativeDecoder(Qwen36BatchedSpeculativeDecoder):
    """Per-user drafts + ONE decode-width fused verify + commit-by-select.

    Inherits prefill (per-user scratch segments + batched anchor assembly),
    drafter legs, slots, and stats from the batched chunk-verify loop; replaces
    the verify and the entire commit machinery.
    """

    def __init__(self, model, mtp_head, page_table, draft_len=3, stop_tokens=None, seed_window=2048):
        super().__init__(
            model, mtp_head, page_table, draft_len=draft_len, stop_tokens=stop_tokens, seed_window=seed_window
        )
        # TT_SPEC_TRACE gates the traced batched DRAFTER only (0 = eager
        # per-user legs); the verify itself stays eager until it rides the
        # production decode-width trace.
        self._use_trace = os.environ.get("TT_SPEC_TRACE", "1") == "1"
        self.W = self.draft_len + 1
        assert self.B * self.W <= 32, f"width {self.B}x{self.W} must fit one tile row (K<= {32 // self.B - 1})"
        try:
            from models.demos.blackhole.qwen36.tt.gdn.fused_decode import op as fused_op
        except ImportError as e:  # pragma: no cover - workspace wiring
            raise RuntimeError(
                "the fused verify needs the gdn-decode-fused branch (fused_decode package) in the workspace"
            ) from e
        self._fop = fused_op
        args = model.args
        self._nv, self._dk, self._dv = args.gdn_nv_tp, args.linear_key_head_dim, args.linear_value_head_dim
        rep = ttnn.ReplicateTensorToMesh(model.mesh_device)
        # Per-GDN-layer persistent state stashes [B*W, Nv, Dk, Dv] fp32 (rank-4:
        # ttnn caps at 4 dims; row u*W+t = user u's state after candidate t) +
        # this iteration's conv-window stash handles (composite conv leg).
        self._stash = [
            ttnn.from_torch(
                torch.zeros(self.B * self.W, self._nv, self._dk, self._dv, dtype=torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=model.mesh_device,
                mesh_mapper=rep,
            )
            for _ in self._dns
        ]
        self._conv_win = [None] * len(self._dns)
        self._vio = None  # persistent verify I/O (built at the first fused verify)
        self._win_buf = [None] * len(self._dns)  # traced mode: persistent conv windows
        self._last_accepts = [0] * self.B  # traced mode: the commit, uploaded as data
        # Constant window-select one-hot: rows t*(K-1)+j pick x_padded row t+1+j —
        # the conv shift register ending at candidate t (x_padded = [carry | qkv rows]).
        conv_k = self._dns[0].K
        sel = torch.zeros(self.B, self.W * (conv_k - 1), (conv_k - 1) + self.W, dtype=torch.float32)
        for t in range(self.W):
            for j in range(conv_k - 1):
                sel[:, t * (conv_k - 1) + j, t + 1 + j] = 1.0
        self._win_sel = ttnn.from_torch(
            sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=model.mesh_device, mesh_mapper=rep
        )
        self._conv_k = conv_k

    # ── the fused decode-width verify ────────────────────────────────────────
    def _gdn_verify_seq(self, li, dn, attn_in, wins):
        """GDN leg: raw qkvzab projection + per-user causal FIR + seq_rows kernel.

        attn_in [1,1,R,full] replicated (post decode norm). Returns the fractured
        attention-output equivalent [1,1,R,dim/tp] (post out-proj all-reduce).
        Capture-safe: every reshape crossing padded-TILE rows goes through
        ROW_MAJOR, the FIR is inlined over the conv-window tensor (plain device
        ops), and the recurrence output rides the persistent _vio buffer. The
        fresh conv-window handle lands in wins[li] — the CALLER owns swapping
        self._conv_win (a captured graph must not free pre-existing buffers).
        """
        from models.demos.blackhole.qwen36.tt import tp_common as tpc
        from models.tt_transformers.tt.ccl import tt_all_reduce

        model = self.model
        R = self.B * self.W
        assert dn._fuse_ab, "the fused verify needs the folded qkvzab weight"
        x3 = ttnn.reshape(attn_in, (1, R, attn_in.shape[-1]))
        if getattr(model.args, "proj_1d_decode", False):
            qkvzab = tpc.matmul_1d_decode(
                x3,
                dn.tw["qkvz"],
                model.args.gdn_qkvz_decode_1d_progcfg,
                dn.cfg,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            qkvzab = dn._col_proj(x3, dn.tw["qkvz"], model.args.gdn_qkvzab_progcfg, ttnn.DRAM_MEMORY_CONFIG)

        D = dn.qkv_dim_tp
        conv_k = dn.K
        qkv = ttnn.slice(qkvzab, (0, 0, 0), (1, R, D), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # [1,R,D] -> [B,W,D] splits tile rows: go through ROW_MAJOR (a padded-
        # TILE reshape is a host-fallback read — illegal under capture).
        qkv_rm = ttnn.untilize(qkv, use_multicore=True)
        ttnn.deallocate(qkv)
        qkv_rm = ttnn.reshape(qkv_rm, (self.B, self.W, D))
        if self._use_trace:
            # Commit-as-pure-data: the carry is the accepted window of the
            # PREVIOUS verify, selected by the uploaded one-hot (read strictly
            # before this layer's window write below).
            carry = ttnn.matmul(self._vio["sel"], self._win_buf[li])  # [B, K-1, D]
            carry_rm = ttnn.untilize(carry, use_multicore=True)
            ttnn.deallocate(carry)
        else:
            carry_rm = ttnn.untilize(dn._batched_conv_carry, use_multicore=True)
        xp_rm = ttnn.concat([carry_rm, qkv_rm], dim=1)  # [B, K-1+W, D]
        ttnn.deallocate(carry_rm)
        ttnn.deallocate(qkv_rm)
        xp = ttnn.to_layout(xp_rm, ttnn.TILE_LAYOUT)
        # Conv-window stash: shift-register contents after every candidate row
        # (next iteration's carry source in traced mode; the eager commit-by-
        # select source otherwise).
        win = ttnn.matmul(self._win_sel, xp)  # [B, W*(K-1), D]
        ttnn.deallocate(xp)
        if self._use_trace:
            ttnn.copy(win, self._win_buf[li])
            ttnn.deallocate(win)
        else:
            wins[li] = win
        # Depthwise causal FIR + SiLU inlined over the SAME window tensor:
        # out = silu(sum_k xp[:, k:k+W] * tap_k). RM slices keep every step a
        # plain device op.
        out = None
        for k in range(conv_k):
            sl = ttnn.to_layout(ttnn.slice(xp_rm, (0, k, 0), (self.B, k + self.W, D)), ttnn.TILE_LAYOUT)
            if out is None:
                out = ttnn.multiply(sl, dn.tw["conv_taps"][k], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                out = ttnn.addcmul(out, sl, dn.tw["conv_taps"][k], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(sl)
        ttnn.deallocate(xp_rm)
        conv_u = ttnn.silu(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B,W,D]

        conv_rm = ttnn.untilize(conv_u, use_multicore=True)
        ttnn.deallocate(conv_u)
        conv_flat = ttnn.to_layout(ttnn.reshape(conv_rm, (1, R, D)), ttnn.TILE_LAYOUT)
        gated = self._vio["gated"]  # persistent [1,R,Nv*Dv] fp32 (kernel writes every row)
        self._fop.recurrence_seq_rows(
            conv_flat,
            qkvzab,
            dn.rec_state,
            self._stash[li],
            dn.tw["dt_bias"],
            dn.tw["neg_exp_A"],
            dn.tw["norm_w"],
            gated,
            nk=dn.Nk,
            nv=dn.Nv,
            dk=self._dk,
            dv=self._dv,
            users=self.B,
            w=self.W,
            # Traced mode anchors on stash row u*W + acc[u] (the previous
            # verify's accepted row) — rec_state drops out of the loop.
            accepts=self._vio["acc"] if self._use_trace else None,
        )
        ttnn.deallocate(conv_flat)
        ttnn.deallocate(conv_rm)
        ttnn.deallocate(qkvzab)
        partial = dn._row_proj(gated, dn.tw["out"])
        partial = ttnn.reshape(partial, (1, 1, R, partial.shape[-1]))
        return tt_all_reduce(
            partial,
            model.mesh_device,
            dn.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=model.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _ensure_verify_io(self):
        """Persistent device buffers for the verify's per-iteration inputs
        (refreshed via copy_host_to_device between replays), the constant
        replicated page-table rows, and the shared recurrence output."""
        if self._vio is not None:
            return
        model = self.model
        R = self.B * self.W
        rd = model.args.rope_head_dim
        rep = ttnn.ReplicateTensorToMesh(model.mesh_device)
        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

        def dev(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=model.device, mesh_mapper=rep)

        pt_rows = [u for u in range(self.B) for _ in range(self.W)]
        self._vio = {
            "tok": dev(torch.zeros(R, 1, dtype=torch.int64), ttnn.uint32, rm),
            "pos": dev(torch.zeros(R, dtype=torch.int32), ttnn.int32, rm),
            "cos": dev(torch.zeros(1, R, 1, rd, dtype=torch.bfloat16), ttnn.bfloat16, tile),
            "sin": dev(torch.zeros(1, R, 1, rd, dtype=torch.bfloat16), ttnn.bfloat16, tile),
            # Sequential masked KV updates: the batch-alias rows of one user
            # target the SAME cache tile, and paged_update_cache read-modify-
            # writes whole tiles — a single batched call races same-tile
            # writers. One masked call per candidate index writes at most one
            # row per tile; -1 skips a row.
            "upd": [dev(torch.full((R,), -1, dtype=torch.int32), ttnn.int32, rm) for _ in range(self.W)],
            "pt": dev(self.page_table[pt_rows].to(torch.int32).contiguous(), ttnn.int32, rm),
            "gated": dev(torch.zeros(1, R, self._nv * self._dv, dtype=torch.float32), ttnn.float32, tile),
            "trace": None,  # {"id", "xn", "idx", "val"} once captured
        }
        if self._use_trace:
            # Commit-as-pure-data inputs: the accepted-row index per user (the
            # recurrence reader anchors on stash row u*W + acc[u]) and the
            # matching conv-window select one-hot. Uploaded, never recreated.
            conv_k = self._conv_k
            dw = self._dns[0].qkv_dim_tp
            self._vio["acc"] = dev(torch.zeros(1, max(self.B, 8), dtype=torch.int32), ttnn.int32, rm)
            self._vio["sel"] = dev(
                torch.zeros(self.B, conv_k - 1, self.W * (conv_k - 1), dtype=torch.float32), ttnn.bfloat16, tile
            )
            self._win_buf = [
                dev(torch.zeros(self.B, self.W * (conv_k - 1), dw, dtype=torch.float32), ttnn.bfloat16, tile)
                for _ in self._dns
            ]

    def _verify_upload(self, drafts):
        """Stage this iteration's verify inputs into the persistent buffers
        (tokens, positions, per-row rope, masked KV-update indices)."""
        model = self.model
        R = self.B * self.W
        mk = {"mesh_mapper": ttnn.ReplicateTensorToMesh(model.mesh_device)}
        toks, poss = [], []
        for u, slot in enumerate(self.slots):
            c = len(slot.committed) - 1
            cands = [slot.committed[-1]] + list(drafts[u]) + [slot.committed[-1]] * (self.W - 1 - len(drafts[u]))
            toks.extend(cands[: self.W])
            poss.extend(range(c, c + self.W))
        rd = model.args.rope_head_dim
        inv_freq = 1.0 / (model.args.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
        freqs = torch.outer(torch.tensor(poss, dtype=torch.float32), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        stage = [
            (torch.tensor(toks, dtype=torch.int64).reshape(R, 1), ttnn.uint32, rm, self._vio["tok"]),
            (torch.tensor(poss, dtype=torch.int32), ttnn.int32, rm, self._vio["pos"]),
            (emb.cos().reshape(1, R, 1, rd).to(torch.bfloat16), ttnn.bfloat16, tile, self._vio["cos"]),
            (emb.sin().reshape(1, R, 1, rd).to(torch.bfloat16), ttnn.bfloat16, tile, self._vio["sin"]),
        ]
        for t in range(self.W):
            m = [p if (i % self.W) == t else -1 for i, p in enumerate(poss)]
            stage.append((torch.tensor(m, dtype=torch.int32), ttnn.int32, rm, self._vio["upd"][t]))
        if self._use_trace:
            # The commit, as data: last iteration's accepted row per user.
            conv_k = self._conv_k
            acc = torch.zeros(1, self._vio["acc"].shape[-1], dtype=torch.int32)
            sel = torch.zeros(self.B, conv_k - 1, self.W * (conv_k - 1), dtype=torch.float32)
            for u, mm in enumerate(self._last_accepts):
                acc[0, u] = mm
                for j in range(conv_k - 1):
                    sel[u, j, mm * (conv_k - 1) + j] = 1.0
            stage.append((acc, ttnn.int32, rm, self._vio["acc"]))
            stage.append((sel.to(torch.bfloat16), ttnn.bfloat16, tile, self._vio["sel"]))
        for host_t, dtype, layout, dst in stage:
            src = ttnn.from_torch(host_t, dtype=dtype, layout=layout, **mk)
            ttnn.copy_host_to_device_tensor(src, dst)

    def _verify_graph(self, debug=False):
        """Device-only fused verify over the persistent input buffers. Returns
        (xn, idx, val, wins) handles; the caller reads them back and owns the
        _conv_win swap. Capture-safe when debug=False (debug adds mid-graph
        readbacks for the one-run layer-type bisect)."""
        model = self.model
        vio = self._vio
        R = self.B * self.W

        def _row_uniformity(tag, t):
            rows = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(R, -1)
            base = rows[: self.W]
            bad = [u for u in range(1, self.B) if not torch.equal(rows[u * self.W : (u + 1) * self.W], base)]
            maxd = max(
                ((rows[u * self.W : (u + 1) * self.W] - base).abs().max().item() for u in range(1, self.B)), default=0.0
            )
            logger.info(f"spec-fused bisect {tag}: nonuniform_users={bad} max_abs_diff={maxd:.3e}")

        x = model.embd(vio["tok"])
        x = ttnn.reshape(x, (1, 1, R, x.shape[-1]))
        wins = [None] * len(self._dns)
        li = 0
        first_attn_logged = False
        for layer in model.layers:
            if layer.is_full_attention:
                x = layer.forward(
                    x,
                    vio["cos"],
                    vio["sin"],
                    position_tensor=vio["pos"],
                    page_table=vio["pt"],
                    mode="decode",
                    kv_update_positions=vio["upd"],
                )
                if debug and not first_attn_logged:
                    _row_uniformity("post-first-attention", x)
                    first_attn_logged = True
            else:
                nc = model.args.get_norm_config("attn", Mode.DECODE)
                attn_in = layer.attention_norm(x, mode=Mode.DECODE, norm_config=nc)
                attn_out = self._gdn_verify_seq(li, layer.attention, attn_in, wins)
                ttnn.deallocate(attn_in)
                h = ttnn.add(x, attn_out)
                ttnn.deallocate(x)
                ttnn.deallocate(attn_out)
                ff_in = layer.ffn_norm(h, mode=Mode.DECODE, norm_config=nc)
                ff_out = layer.feed_forward.forward(ff_in)
                ttnn.deallocate(ff_in)
                x = ttnn.add(h, ff_out)
                ttnn.deallocate(h)
                ttnn.deallocate(ff_out)
                if debug and li == 0:
                    _row_uniformity("post-first-gdn", x)
                li += 1
        xn = model._final_norm_decode(x)  # replicated [1,1,R,full]
        ttnn.deallocate(x)
        shard = ttnn.linear(xn, model.lm_head_weight)  # [1,1,R,Vs] per device
        rm = ttnn.untilize(shard, use_multicore=True)
        idx = ttnn.argmax(rm, dim=-1, keepdim=False)  # [1,1,32] uint32 (R == 32)
        ttnn.deallocate(rm)
        val = ttnn.max(shard, dim=-1)
        ttnn.deallocate(shard)
        return xn, idx, val, wins

    def _swap_conv_wins(self, wins):
        for li, win in enumerate(wins):
            if win is None:  # traced mode: the window lives in _win_buf instead
                continue
            if self._conv_win[li] is not None and self._conv_win[li] is not win:
                ttnn.deallocate(self._conv_win[li])
            self._conv_win[li] = win

    def _verify_read(self, xn, idx, val):
        """Assemble per-user (target_ids [W], hid [W, dim]) from the graph
        outputs (post-replay readbacks)."""
        model = self.model
        R = self.B * self.W
        hid_all = ttnn.to_torch(ttnn.get_device_tensors(xn)[0]).float().reshape(R, -1)
        comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
        idxs = ttnn.to_torch(idx, mesh_composer=comp).reshape(-1, 32)
        vals = ttnn.to_torch(val, mesh_composer=comp).float().reshape(-1, 32)
        per_shard = model.args.vocab_size // model.num_devices
        results = []
        for u in range(self.B):
            ids = []
            for t in range(self.W):
                r = u * self.W + t
                if getattr(model, "_lmhead_vocab_sharded", False):
                    d = int(vals[:, r].argmax())
                    ids.append(d * per_shard + int(idxs[d, r]))
                else:
                    ids.append(int(idxs[0, r]))
            results.append((ids, hid_all[u * self.W : (u + 1) * self.W]))
        return results

    def _capture_verify_trace(self):
        """Capture the verify graph over the persistent buffers (iteration 1's
        inputs are still staged; runs AFTER the drafter captures, and after
        iteration 1 compiled every program in this graph)."""
        tid = ttnn.begin_trace_capture(self.model.device, cq_id=0)
        xn, idx, val, wins = self._verify_graph()
        ttnn.end_trace_capture(self.model.device, tid, cq_id=0)
        self._swap_conv_wins(wins)
        self._vio["trace"] = {"id": tid, "xn": xn, "idx": idx, "val": val}

    def _fused_verify(self, drafts):
        """One decode-width forward over every user's [t_c, d_1..d_K] rows.

        Returns per user (target_ids [W], hid [W, dim]) — row u*W+t predicts the
        token after candidate t. Traced once captured; eager before that (and
        always under TT_SPEC_TRACE=0).
        """
        self._ensure_verify_io()
        self._verify_upload(drafts)
        tr = self._vio["trace"]
        if tr is not None:
            ttnn.execute_trace(self.model.device, tr["id"], cq_id=0, blocking=False)
            return self._verify_read(tr["xn"], tr["idx"], tr["val"])
        # TT_SPEC_FUSED_DEBUG=1: one-run bisect — after the first attention and
        # first GDN layer, log whether the per-user row blocks are bitwise
        # uniform (uniform prompts => any split names the diverging layer type).
        debug = os.environ.get("TT_SPEC_FUSED_DEBUG", "0") == "1" and not getattr(self, "_debug_done", False)
        xn, idx, val, wins = self._verify_graph(debug=debug)
        if debug:
            self._debug_done = True
        self._swap_conv_wins(wins)
        results = self._verify_read(xn, idx, val)
        for t in (xn, idx, val):
            ttnn.deallocate(t)
        return results

    # ── commit-by-select ─────────────────────────────────────────────────────
    def _commit_select(self, accepts):
        """Advance every user's anchor to its accepted row: rec_state[u] <-
        stash[u, m_u]; conv carry <- the window ending at candidate m_u."""
        conv_k = self._conv_k
        for li, dn in enumerate(self._dns):
            for u, m in enumerate(accepts):
                if m is None:
                    continue
                r = u * self.W + m
                row = ttnn.slice(self._stash[li], (r, 0, 0, 0), (r + 1, self._nv, self._dk, self._dv))
                dn._write_index(dn.rec_state, row, u, dim=0)
                w0 = m * (conv_k - 1)
                cw = ttnn.slice(self._conv_win[li], (u, w0, 0), (u + 1, w0 + conv_k - 1, dn.qkv_dim_tp))
                dn._write_index(dn._batched_conv_carry, cw, u, dim=0)

    # ── traced setup ─────────────────────────────────────────────────────────
    def _seed_select_state(self):
        """One-time pre-loop seeding for commit-as-pure-data (traced mode):
        place every user's post-prefill anchor where accepts=0 selects it —
        stash row u*W <- rec_state[u], win_buf leading rows <- the conv carry.
        Eager and BEFORE any capture, so its allocations are safe."""
        self._ensure_verify_io()
        conv_k = self._conv_k
        for li, dn in enumerate(self._dns):
            for u in range(self.B):
                row = ttnn.slice(dn.rec_state, (u, 0, 0, 0), (u + 1, self._nv, self._dk, self._dv))
                dn._write_index(self._stash[li], row, u * self.W, dim=0)
            tail = ttnn.slice(self._win_buf[li], (0, conv_k - 1, 0), (self.B, self.W * (conv_k - 1), dn.qkv_dim_tp))
            seeded = ttnn.concat([dn._batched_conv_carry, tail], dim=1)
            ttnn.deallocate(tail)
            ttnn.copy(seeded, self._win_buf[li])
            ttnn.deallocate(seeded)
        self._last_accepts = [0] * self.B

    def _setup_drafter_traces(self):
        """Compile + capture the batched drafter windows, then the verify trace,
        AFTER one full eager iteration: every fused-verify program must already
        exist when the first trace parks (all compiles strictly before all
        captures). Capture records without executing (dispatch runs in bypass
        mode), so the capture passes leave KV and select-state untouched. From
        here the loop is allocation-free: uploads into persistent buffers,
        replays, and readbacks only — 55287 showed ANY fresh L1 allocation
        while traces are parked can land under a parked trace's CB region."""
        K = self.draft_len
        w_max = (K + 2) + K - 1  # steady-state legs: pendings (<= K+1, +1 slack) + chain
        self.mtp.ensure_batched_window(w_max, self.B)
        setup_pos = torch.stack(
            [torch.full((w_max,), int(self._sched_pending(slot)[-1][2]), dtype=torch.int32) for slot in self.slots]
        )
        self.mtp.stage_batched_window(setup_pos)
        self.mtp.compile_batched_window()
        self.mtp.capture_batched_window()
        self._capture_verify_trace()

    # ── the loop ─────────────────────────────────────────────────────────────
    def _align_anchor_to_head(self):
        """One-time post-prefill catch-up: advance every user's state through all
        committed tokens but the head (the fused anchor invariant)."""
        model = self.model
        self._snapshot_from_live()
        for u, slot in enumerate(self.slots):
            n = len(slot.committed) - 1 - slot.a
            if n <= 0:
                slot.a = len(slot.committed) - 1
                continue
            prev = model._bind_gdn_prefill_scratch()
            try:
                for dn, _B, rec_b, _c, carry_b, _z, _s in prev:
                    dn._spec_batched_rec = rec_b
                    dn._spec_batched_carry = getattr(dn, "_batched_conv_carry", None)
                self._seed_scratch_from_snapshot(u)
                self._scratch_chunk(u, slot.committed[slot.a : slot.a + n], slot.a, n)
                self._writeback_scratch_row(u)
            finally:
                model._unbind_gdn_prefill_scratch(prev)
                for dn in self._dns:
                    dn._spec_batched_rec = None
                    dn._spec_batched_carry = None
            slot.a = len(slot.committed) - 1
        self._snapshot_from_live()

    def _draft_eager(self, max_new_tokens):
        """Per-user eager drafter legs (iteration 1, and the TT_SPEC_TRACE=0
        fallback)."""
        K = self.draft_len
        drafts = [[] for _ in range(self.B)]
        for u, slot in enumerate(self.slots):
            if slot.done or len(slot.out) >= max_new_tokens:
                continue
            slot.k_used.append(K)
            last_pos = slot.pending[-1][2]
            for tok, hid_row, pos in slot.pending[:-1]:
                self.mtp.step(tok, hid_row, pos, user=u)
            d_logits, g = self.mtp.step(*slot.pending[-1], user=u)
            slot.pending = []
            drafts[u] = [int(d_logits.argmax())]
            for j in range(1, K):
                d_logits, g = self.mtp.step(drafts[u][-1], g, last_pos + j, user=u)
                drafts[u].append(int(d_logits.argmax()))
        return drafts

    def generate(self, max_new_tokens, adaptive_k=False):
        assert not adaptive_k, "the fused verify is static-K (width B*(K+1) is fixed)"
        K = self.draft_len
        # Arm pendings + first tokens via the parent's (chunk-shaped) first verify,
        # then align every anchor to its committed head for the fused invariant.
        self._first_verify()
        self._align_anchor_to_head()
        if self._use_trace:
            self._seed_select_state()
        traced_armed = False
        first_iter_done = False

        while any(not s.done and len(s.out) < max_new_tokens for s in self.slots):
            if self._use_trace and first_iter_done and not traced_armed:
                # Iteration 1 ran fully eager and compiled every verify/commit
                # program; parking traces is safe from here on.
                self._setup_drafter_traces()
                traced_armed = True
            t0 = time.perf_counter()
            if traced_armed:
                width = max(len(self._sched_pending(s2)) for s2 in self.slots) + K - 1
                assert width <= self.mtp._bwin["w_max"], f"steady-state window {width} overflows"
                drafts = self._draft_traced(width, max_new_tokens)
            else:
                drafts = self._draft_eager(max_new_tokens)
            self._timing["draft"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            results = self._fused_verify(drafts)
            self._timing["verify"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            accepts = [None] * self.B
            first_committed = [None] * self.B
            for u, slot in enumerate(self.slots):
                if slot.done or not drafts[u] or len(slot.out) >= max_new_tokens:
                    continue
                target_ids, hid = results[u]
                c = len(slot.committed) - 1
                m, new_tokens = greedy_accept(drafts[u], target_ids[: K + 1])
                slot.accepts.append(m)
                accepts[u] = m
                first_committed[u] = new_tokens[0]
                for j, tok in enumerate(new_tokens):
                    slot.committed.append(tok)
                    slot.out.append(tok)
                    slot.pending.append((tok, hid[j], c + j))
                    if tok in self.stop_tokens:
                        slot.done = True
                        break
                    if len(slot.out) >= max_new_tokens:
                        break
                slot.a = len(slot.committed) - 1  # fused invariant: anchor == head
            if self._use_trace:
                # The commit IS data: the next verify anchors on stash row
                # u*W + m and carry-selects the matching window (uploaded in
                # _verify_upload). A finished user's selection is dead state.
                self._last_accepts = [m if m is not None else 0 for m in accepts]
            else:
                self._commit_select(accepts)
            self._timing["commit"] += time.perf_counter() - t0
            first_iter_done = True
            if os.environ.get("TT_SPEC_TIMING", "0") == "1":
                it = max(len(s2.accepts) for s2 in self.slots)
                logger.info(
                    f"spec-fused iter {it}: m={accepts} first_committed={first_committed} "
                    f"drafts={[list(d) for d in drafts]}"
                )
        return [list(s.out) for s in self.slots], self._stats()
