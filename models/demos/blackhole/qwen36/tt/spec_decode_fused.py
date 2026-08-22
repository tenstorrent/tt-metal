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
import time

import torch

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
        self._use_trace = False  # eager v1 (the traced form rides the production decode trace)
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
    def _gdn_verify_seq(self, li, dn, attn_in):
        """GDN leg: raw qkvzab projection + per-user causal FIR + seq_rows kernel.

        attn_in [1,1,R,full] replicated (post decode norm). Returns the fractured
        attention-output equivalent [1,1,R,dim/tp] (post out-proj all-reduce).
        """
        from models.demos.blackhole.qwen36.tt import tp_common as tpc
        from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
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
        qkv = ttnn.slice(qkvzab, (0, 0, 0), (1, R, D), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv_u = ttnn.reshape(qkv, (self.B, self.W, D))
        conv_u, conv_tail = _causal_conv1d_fir(
            qkv_u,
            None,
            None,
            dn.K,
            model.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            conv_state=dn._batched_conv_carry,
            weight_taps=dn.tw["conv_taps"],
            bias_dev=None,
        )
        ttnn.deallocate(conv_tail)
        # Conv-window stash: shift-register contents after every candidate row
        # (the commit-by-select source for _batched_conv_carry).
        xp = ttnn.concat([dn._batched_conv_carry, qkv_u], dim=1)  # [B, K-1+W, D]
        win = ttnn.matmul(self._win_sel, xp)  # [B, W*(K-1), D]
        ttnn.deallocate(xp)
        if self._conv_win[li] is not None:
            ttnn.deallocate(self._conv_win[li])
        self._conv_win[li] = win
        ttnn.deallocate(qkv_u)

        conv_flat = ttnn.reshape(conv_u, (1, R, D))
        gated = ttnn.from_torch(
            torch.zeros(1, R, self._nv * self._dv, dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=model.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
        )
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
        )
        ttnn.deallocate(conv_flat)
        ttnn.deallocate(qkvzab)
        partial = dn._row_proj(gated, dn.tw["out"])
        ttnn.deallocate(gated)
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

    def _fused_verify(self, drafts):
        """One decode-width forward over every user's [t_c, d_1..d_K] rows.

        Returns per user (target_ids [W], hid [W, dim]) — row u*W+t predicts the
        token after candidate t.
        """
        model = self.model
        R = self.B * self.W
        rep = ttnn.ReplicateTensorToMesh(model.mesh_device)

        toks, poss, pt_rows = [], [], []
        for u, slot in enumerate(self.slots):
            c = len(slot.committed) - 1
            cands = [slot.committed[-1]] + list(drafts[u]) + [slot.committed[-1]] * (self.W - 1 - len(drafts[u]))
            toks.extend(cands[: self.W])
            poss.extend(range(c, c + self.W))
            pt_rows.extend([u] * self.W)

        tok = ttnn.from_torch(
            torch.tensor(toks, dtype=torch.int64).reshape(R, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=model.device,
            mesh_mapper=rep,
        )
        x = model.embd(tok)
        ttnn.deallocate(tok)
        x = ttnn.reshape(x, (1, 1, R, x.shape[-1]))
        cur_pos = ttnn.from_torch(
            torch.tensor(poss, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=model.device,
            mesh_mapper=rep,
        )
        pt = ttnn.from_torch(
            self.page_table[pt_rows].to(torch.int32).contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=model.device,
            mesh_mapper=rep,
        )
        # TP decode rope: per-row rotations [1, R, 1, rope_dim].
        rd = model.args.rope_head_dim
        inv_freq = 1.0 / (model.args.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
        freqs = torch.outer(torch.tensor(poss, dtype=torch.float32), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = ttnn.from_torch(
            emb.cos().reshape(1, R, 1, rd).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=model.device,
            mesh_mapper=rep,
        )
        sin = ttnn.from_torch(
            emb.sin().reshape(1, R, 1, rd).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=model.device,
            mesh_mapper=rep,
        )

        li = 0
        for layer in model.layers:
            if layer.is_full_attention:
                x = layer.forward(x, cos, sin, position_tensor=cur_pos, page_table=pt, mode="decode")
            else:
                nc = model.args.get_norm_config("attn", Mode.DECODE)
                attn_in = layer.attention_norm(x, mode=Mode.DECODE, norm_config=nc)
                attn_out = self._gdn_verify_seq(li, layer.attention, attn_in)
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
                li += 1
        for t in (cur_pos, pt, cos, sin):
            ttnn.deallocate(t)

        xn = model._final_norm_decode(x)  # replicated [1,1,R,full]
        ttnn.deallocate(x)
        hid_all = ttnn.to_torch(ttnn.get_device_tensors(xn)[0]).float().reshape(R, -1)
        shard = ttnn.linear(xn, model.lm_head_weight)  # [1,1,R,Vs] per device
        ttnn.deallocate(xn)
        rm = ttnn.untilize(shard, use_multicore=True)
        idx = ttnn.argmax(rm, dim=-1, keepdim=False)  # [1,1,32] uint32 (R == 32)
        ttnn.deallocate(rm)
        val = ttnn.max(shard, dim=-1)
        ttnn.deallocate(shard)
        comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
        idxs = ttnn.to_torch(idx, mesh_composer=comp).reshape(-1, 32)
        vals = ttnn.to_torch(val, mesh_composer=comp).float().reshape(-1, 32)
        ttnn.deallocate(idx)
        ttnn.deallocate(val)
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

    def generate(self, max_new_tokens, adaptive_k=False):
        assert not adaptive_k, "the fused verify is static-K (width B*(K+1) is fixed)"
        K = self.draft_len
        # Arm pendings + first tokens via the parent's (chunk-shaped) first verify,
        # then align every anchor to its committed head for the fused invariant.
        self._first_verify()
        self._align_anchor_to_head()

        while any(not s.done and len(s.out) < max_new_tokens for s in self.slots):
            t0 = time.perf_counter()
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
            self._timing["draft"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            results = self._fused_verify(drafts)
            self._timing["verify"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            accepts = [None] * self.B
            for u, slot in enumerate(self.slots):
                if slot.done or not drafts[u] or len(slot.out) >= max_new_tokens:
                    continue
                target_ids, hid = results[u]
                c = len(slot.committed) - 1
                m, new_tokens = greedy_accept(drafts[u], target_ids[: K + 1])
                slot.accepts.append(m)
                accepts[u] = m
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
            self._commit_select(accepts)
            self._timing["commit"] += time.perf_counter() - t0
        return [list(s.out) for s in self.slots], self._stats()
