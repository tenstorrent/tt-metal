# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched (c8) speculative decoding: B users drafted per-user, verified in ONE
grouped chunk forward per iteration (TT_SPEC_BATCHED=1).

The batched verify amortizes the chunk's per-layer fixed costs (CCLs, launch
floors, the GDN scan) across all B users — the c8 arithmetic that clears the
campaign target even with the chunk-shaped verify. It rides the
silicon-validated grouped machinery (prefill_paged_grouped /
test_gdn_fused_batch, Bg<=8 at bucket 128): GDN layers run BATCHED with
per-row valid_len masks and carry=True over the batched per-user anchor state
(rec_state rows + _batched_conv_carry rows); full-attention layers run
per-user with each user's OWN page-table row, cos/sin window, and device
chunk_start — the desync between users (different anchors a_u, committed heads
c_u, draft counts k_u) is pure DATA.

Per-user desync bookkeeping (the hard part) mirrors the B=1 loop per slot:

- anchor a_u block-aligned per user; verify chunk_u = committed_u[a_u:] +
  drafts_u with valid_len_u; accept row_start_u = c_u - a_u — all per-row.
- verify NEVER commits: one batched snapshot (rec_state [B,...] +
  _batched_conv_carry [B,...] clones per GDN layer — the same tensor count as
  B=1) is restored after every verify.
- commits desync per user: when commit_advance fires for user u, a B=1 masked
  chunk runs on the persistent prefill scratch seeded from snapshot row u, and
  the result is row-written (TPGatedDeltaNet._write_index) into both the live
  batched state and the snapshot.

KV rollback stays implicit PER USER (each user's chunk rewrites only its own
blocks via its page-table row). The drafter keeps per-user KV block ranges
(Qwen36MTPHead users=B) and runs eager per-user legs in v1; batching the
drafter legs across users and tracing the batched verify follow the u=1
traced-loop isolation (see docs/mtp_design.md).
"""
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.spec_decode import (
    BLOCK,
    _PREFILL_SEG,
    adaptive_draft_len,
    block_aligned_prefill_len,
    commit_advance,
    greedy_accept,
)
from models.tt_transformers.tt.common import Mode

_VERIFY_BUCKET = 128
_VERIFY_ROWS = 32


@dataclass
class SpecSlot:
    """Per-user speculative state (pure bookkeeping — host-emulation testable)."""

    committed: list  # token ids 0..c_u (prompt + generated)
    a: int  # block-aligned anchor: this user's GDN state row reflects tokens 0..a-1
    pending: list = field(default_factory=list)  # (input_token, hidden_row, drafter_pos)
    out: list = field(default_factory=list)  # generated tokens
    accepts: list = field(default_factory=list)
    k_used: list = field(default_factory=list)
    k_ema: float = 0.0
    done: bool = False


class Qwen36BatchedSpeculativeDecoder:
    """Per-user draft -> ONE batched verify -> per-user accept/commit (eager v1).

    Args:
        model: Qwen36Model (TP) with batched caches allocated
            (allocate_kv_caches(batch_size=B) — the GDN modules hold [B,...] state).
        mtp_head: Qwen36MTPHead with allocate_kv_cache(..., users=B).
        page_table: torch [B, blocks_per_user] int32 — row u = user u's blocks.
        draft_len / stop_tokens / seed_window: as the B=1 decoder.
    """

    def __init__(self, model, mtp_head, page_table, draft_len=4, stop_tokens=None, seed_window=2048):
        assert model.num_devices > 1, "batched spec decode is the TP path"
        self.model = model
        self.mtp = mtp_head
        self.page_table = page_table
        self.B = int(page_table.shape[0])
        assert self.B <= 8, "grouped GDN chunk verify is validated to Bg=8 at bucket 128"
        self.draft_len = int(draft_len)
        assert 1 <= self.draft_len <= 31
        self.stop_tokens = set(stop_tokens or [])
        self.seed_window = int(seed_window)
        self._dns = [layer.attention for layer in model.layers if not layer.is_full_attention]
        for dn in self._dns:
            assert dn.B == self.B, f"GDN batched state is B={dn.B}, expected {self.B} (allocate batch_size={self.B})"
            assert getattr(dn, "_stable_state", False), "batched spec needs in-place GDN state"
        self.slots = []
        self._snap = None  # per-GDN-layer (rec [B,...], conv_carry [B,K-1,D]) anchor clones
        # Traced loop (TT_SPEC_TRACE=0 for eager). Same trace-safety rules as the
        # B=1 loop: all compiles strictly before the first capture, variation as
        # data in persistent buffers, self-restoring verify graph.
        self._use_trace = os.environ.get("TT_SPEC_TRACE", "1") == "1"
        self._vt = None  # traced-verify persistent I/O
        self._timing = defaultdict(float)

    # ── batched GDN anchor snapshot ──────────────────────────────────────────
    def _snapshot_from_live(self):
        """Clone/refresh the anchor snapshot from the LIVE batched state."""
        if self._snap is None:
            self._snap = [
                (
                    ttnn.clone(dn.rec_state, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                    ttnn.clone(dn._batched_conv_carry, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                )
                for dn in self._dns
            ]
            return
        for dn, (rec, carry) in zip(self._dns, self._snap):
            ttnn.copy(dn.rec_state, rec)
            ttnn.copy(dn._batched_conv_carry, carry)

    def _restore_from_snapshot(self):
        """Copy the anchor snapshot back into the live batched state (post-verify)."""
        for dn, (rec, carry) in zip(self._dns, self._snap):
            ttnn.copy(rec, dn.rec_state)
            # forward_prefill_batched(carry=True) replaces the carry HANDLE each
            # call; copy into whatever it currently points at.
            ttnn.copy(carry, dn._batched_conv_carry)

    # ── per-user B=1 chunk on the prefill scratch ────────────────────────────
    def _scratch_chunk(self, user, tokens, chunk_start, valid_len, want_hidden=False):
        """Run one B=1 masked chunk for `user` on the bound prefill scratch.

        The caller binds the scratch (model._bind_gdn_prefill_scratch) — the
        scratch carries GDN state in place across segments. Returns the
        pre-final-norm hidden [1,1,bucket,dim/tp] when want_hidden else None.
        """
        model = self.model
        n = len(tokens)
        assert chunk_start % BLOCK == 0
        bucket = model._mask_bucket_for(n)
        buf = torch.zeros(1, bucket, dtype=torch.int32)
        buf[0, :n] = torch.tensor(tokens, dtype=torch.int32)
        row = self.page_table[user : user + 1].contiguous()
        hidden = model._forward_prefill_chunk_masked(buf, valid_len, chunk_start, row, bucket)
        if want_hidden:
            return hidden
        ttnn.deallocate(hidden)
        return None

    def _zero_scratch(self):
        """Zero the bound B=1 scratch explicitly (rec, conv carry, decode window)."""
        for dn in self._dns:
            zero_rec = ttnn.from_torch(
                torch.zeros(list(dn.rec_state.shape), dtype=torch.float32),
                dtype=dn.rec_state.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.model.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.device),
            )
            ttnn.copy(zero_rec, dn.rec_state)
            ttnn.deallocate(zero_rec)
            if dn.conv_carry is not None:
                zero_c = ttnn.from_torch(
                    torch.zeros(list(dn.conv_carry.shape), dtype=torch.float32),
                    dtype=dn.conv_carry.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.model.device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.device),
                )
                ttnn.copy(zero_c, dn.conv_carry)
                ttnn.deallocate(zero_c)

    def _seed_scratch_from_snapshot(self, user):
        """Load snapshot row `user` into the bound B=1 scratch (device copies)."""
        for dn, (rec, carry) in zip(self._dns, self._snap):
            nv, dk, dv = rec.shape[1], rec.shape[2], rec.shape[3]
            r = ttnn.slice(rec, (user, 0, 0, 0), (user + 1, nv, dk, dv))
            ttnn.copy(r, dn.rec_state)
            ttnn.deallocate(r)
            km1, d = carry.shape[1], carry.shape[2]
            c = ttnn.slice(carry, (user, 0, 0), (user + 1, km1, d))
            c = ttnn.to_layout(c, ttnn.TILE_LAYOUT)
            ttnn.copy(c, dn.conv_carry)
            ttnn.deallocate(c)

    def _writeback_scratch_row(self, user):
        """Row-write the bound scratch state into the live batched state AND the
        snapshot at row `user` (the committed anchor moved for this user)."""
        for dn, (rec, carry) in zip(self._dns, self._snap):
            src_rec_live = ttnn.clone(dn.rec_state)
            src_rec_snap = ttnn.clone(dn.rec_state)
            src_carry_live = ttnn.clone(dn.conv_carry)
            src_carry_snap = ttnn.clone(dn.conv_carry)
            batched_rec, batched_carry = self._batched_state_handles(dn)
            dn._write_index(batched_rec, src_rec_live, user, dim=0)
            dn._write_index(rec, src_rec_snap, user, dim=0)
            dn._write_index(batched_carry, src_carry_live, user, dim=0)
            dn._write_index(carry, src_carry_snap, user, dim=0)

    @staticmethod
    def _batched_state_handles(dn):
        """The LIVE batched buffers while the B=1 scratch is bound: the bind
        stashed them via model._bind_gdn_prefill_scratch's `prev` — the decoder
        stores them per-layer at bind time (see _commit_user)."""
        return dn._spec_batched_rec, dn._spec_batched_carry

    # ── prefill: per-user segments on the scratch, then batched-anchor assembly ─
    def prefill(self, token_ids_list):
        """Per-user segmented masked prefill into the B=1 scratch; stitches every
        user's anchor state into the batched buffers and takes the snapshot.

        token_ids_list: list of B torch [1, T_u] prompts (lengths may differ).
        """
        model = self.model
        assert len(token_ids_list) == self.B
        self.slots = []
        model._build_request_rope(token_ids_list[0][:, :1], None)  # text-only: clears any M-RoPE table

        rec_host = []  # per user: per-layer host [num_dev, Nv, Dk, Dv]
        carry_host = []  # per user: per-layer host [num_dev, K-1, D]
        comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
        for u, token_ids in enumerate(token_ids_list):
            T = token_ids.shape[1]
            assert T >= 2, "prompt too short for MTP"
            slot = SpecSlot(committed=[int(t) for t in token_ids[0]], a=block_aligned_prefill_len(T))
            slot.k_ema = float(self.draft_len)
            self.slots.append(slot)

            prev = model._bind_gdn_prefill_scratch()
            try:
                self._zero_scratch()
                seed_from = max(0, slot.a - self.seed_window)
                prompt_hidden = {}
                for start in range(0, slot.a, _PREFILL_SEG):
                    length = min(_PREFILL_SEG, slot.a - start)
                    hidden = self._scratch_chunk(
                        u, slot.committed[start : start + length], start, length, want_hidden=True
                    )
                    if start + length > seed_from:
                        normed = model.norm(hidden, mode=Mode.PREFILL)
                        rows = ttnn.to_torch(ttnn.get_device_tensors(normed)[0]).float()
                        rows = rows.reshape(-1, rows.shape[-1])[:length]
                        if normed is not hidden:
                            ttnn.deallocate(normed)
                        for i in range(max(seed_from, start), start + length):
                            prompt_hidden[i] = rows[i - start]
                    ttnn.deallocate(hidden)
                # Stash this user's anchor state from the scratch (host).
                rec_host.append([ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in self._dns])
                carry_host.append([ttnn.to_torch(dn.conv_carry, mesh_composer=comp) for dn in self._dns])
            finally:
                model._unbind_gdn_prefill_scratch(prev)

            # Seed this user's drafter KV (eager; per-user block range).
            for i in range(seed_from, slot.a):
                self.mtp.step(slot.committed[i + 1], prompt_hidden[i], i, user=u)
            logger.info(f"spec[c{self.B}] prefill user {u}: T={T} anchor a0={slot.a}")

        self._assemble_anchor_state(rec_host, carry_host)
        self._snapshot_from_live()

    def _assemble_anchor_state(self, rec_host, carry_host):
        """Stitch per-user host anchor states into the batched buffers.

        Host layout for ShardTensorToMesh(dim=0): device d's block holds rows
        [d*B, (d+1)*B) = users 0..B-1 on device d.
        """
        model = self.model
        num_dev = model.num_devices
        mapper = ttnn.ShardTensorToMesh(model.mesh_device, dim=0)
        for li, dn in enumerate(self._dns):
            rec_rows = torch.stack(
                [rec_host[u][li][d] for d in range(num_dev) for u in range(self.B)], dim=0
            )  # [num_dev*B, Nv, Dk, Dv]
            rec = ttnn.from_torch(
                rec_rows,
                dtype=dn.rec_state.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=model.mesh_device,
                mesh_mapper=mapper,
            )
            ttnn.copy(rec, dn.rec_state)
            ttnn.deallocate(rec)
            carry_rows = torch.stack([carry_host[u][li][d] for d in range(num_dev) for u in range(self.B)], dim=0)
            carry = ttnn.from_torch(
                carry_rows, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=model.mesh_device, mesh_mapper=mapper
            )
            if getattr(dn, "_batched_conv_carry", None) is not None:
                ttnn.copy(carry, dn._batched_conv_carry)
                ttnn.deallocate(carry)
            else:
                dn._batched_conv_carry = carry

    # ── batched verify ───────────────────────────────────────────────────────
    def _batched_verify(self, chunks):
        """ONE grouped chunk forward over all users' [tail + drafts] chunks.

        chunks: per user (tokens, chunk_start a_u, valid_len n_u, hid_start,
        score_start). Returns per user (target_ids from score_start — at most 32,
        the multicore-argmax row contract — and hid rows from hid_start to the
        chunk end, up to 64 for the first verify's prompt tail).

        Adapted from prefill_paged_grouped: batched embedding + batched GDN
        (per-row valid_len, carry=True over the anchor rows) + per-user
        attention with per-user page tables / cos-sin / device chunk_start.
        """
        model = self.model
        B, bucket = self.B, _VERIFY_BUCKET
        rep = ttnn.ReplicateTensorToMesh(model.mesh_device)
        block_size = BLOCK

        tok_bg = torch.zeros(B, bucket, dtype=torch.int32)
        vlens = []
        for u, (tokens, a_u, n_u, _hid_start, _score_start) in enumerate(chunks):
            assert a_u % BLOCK == 0 and n_u <= bucket
            tok_bg[u, :n_u] = torch.tensor(tokens, dtype=torch.int32)
            vlens.append(n_u)
        tok = ttnn.from_torch(tok_bg, dtype=ttnn.uint32, device=model.device, mesh_mapper=rep)
        x = model.embd(tok)
        d = x.shape[-1]
        x = ttnn.reshape(x, (1, 1, B * bucket, d))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)

        # Per-user attention inputs: page tables, rope windows, device chunk starts.
        full_pts, chunk_pts, coss, sins, csis = [], [], [], [], []
        from models.tt_transformers.tt.common import num_blocks_in_seq

        for u, (_tokens, a_u, n_u, _hid_start, _score_start) in enumerate(chunks):
            row = self.page_table[u : u + 1].contiguous()
            full_pts.append(ttnn.from_torch(row, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=model.device))
            blk0 = a_u // block_size
            blkN = num_blocks_in_seq(a_u + n_u, block_size)
            chunk_pts.append(
                ttnn.from_torch(
                    row[:, blk0:blkN].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=model.device
                )
            )
            cos_t, sin_t = model._rope_tp_cos_sin_torch(a_u, bucket)
            coss.append(
                ttnn.from_torch(
                    cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=model.device, mesh_mapper=rep
                )
            )
            sins.append(
                ttnn.from_torch(
                    sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=model.device, mesh_mapper=rep
                )
            )
            csis.append(
                ttnn.from_torch(
                    torch.tensor([a_u], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=model.device,
                )
            )

        for layer in model.layers:
            attn_in = layer.attention_norm(x, mode=Mode.PREFILL)  # [1,1,B*bucket, full]
            full = attn_in.shape[-1]
            if layer.is_full_attention:
                attn_in_b = ttnn.reshape(attn_in, (1, B, bucket, full))
                outs = []
                for u in range(B):
                    xi = ttnn.reshape(attn_in_b[:, u : u + 1, :, :], (1, 1, bucket, full))
                    oi = layer.attention.forward_prefill_paged(
                        xi,
                        coss[u],
                        sins[u],
                        full_pts[u],
                        chunk_page_table=chunk_pts[u],
                        chunk_start_idx=0,
                        chunk_start_idx_tensor=csis[u],
                        user_id=0,
                    )
                    ttnn.deallocate(xi)
                    outs.append(ttnn.reshape(oi, (1, 1, bucket, oi.shape[-1])))
                attn_out = ttnn.concat(outs, dim=2) if B > 1 else outs[0]
                for o in outs:
                    if o is not attn_out:
                        ttnn.deallocate(o)
            else:
                gdn_in = ttnn.reshape(attn_in, (B, bucket, full))
                attn_out = layer.attention.forward_prefill_batched(
                    gdn_in, chunk_size=model.args.gdn_chunk_size, valid_lens=vlens, carry=True, carry_inplace=True
                )
                attn_out = ttnn.reshape(attn_out, (1, 1, B * bucket, attn_out.shape[-1]))
            ttnn.deallocate(attn_in)
            h = ttnn.add(x, attn_out)
            ttnn.deallocate(x)
            ttnn.deallocate(attn_out)
            ff_in = layer.ffn_norm(h, mode=Mode.PREFILL)
            ff_out = layer.feed_forward.forward(ff_in)
            ttnn.deallocate(ff_in)
            x = ttnn.add(h, ff_out)
            ttnn.deallocate(h)
            ttnn.deallocate(ff_out)

        xn = model.norm(x, mode=Mode.PREFILL)  # [1,1,B*bucket, full] replicated
        ttnn.deallocate(x)
        xn_b = ttnn.reshape(xn, (1, B, bucket, xn.shape[-1]))

        results = []
        comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
        per_shard = model.args.vocab_size // model.num_devices
        for u, (_tokens, _a_u, n_u, hid_start, score_start) in enumerate(chunks):
            xu = ttnn.reshape(xn_b[:, u : u + 1, :, :], (1, 1, bucket, xn.shape[-1]))
            # Hidden rows [hid_start, n_u): up to 64 for the first verify's tail.
            n_hid = n_u - hid_start
            hid_padded = ((n_hid + 31) // 32) * 32
            sel_h = torch.zeros(1, 1, hid_padded, bucket, dtype=torch.float32)
            for j in range(n_hid):
                sel_h[0, 0, j, hid_start + j] = 1.0
            sel_h_tt = ttnn.from_torch(
                sel_h, dtype=xn.dtype, layout=ttnn.TILE_LAYOUT, device=model.device, mesh_mapper=rep
            )
            hrows = ttnn.matmul(sel_h_tt, xu)
            ttnn.deallocate(sel_h_tt)
            hid = ttnn.to_torch(ttnn.get_device_tensors(hrows)[0]).float().reshape(hid_padded, -1)[:n_hid]
            ttnn.deallocate(hrows)
            # Score rows [score_start, ...): exactly 32 padded (multicore argmax contract).
            n_score = min(_VERIFY_ROWS, n_u - score_start)
            sel_s = torch.zeros(1, 1, _VERIFY_ROWS, bucket, dtype=torch.float32)
            for j in range(n_score):
                sel_s[0, 0, j, score_start + j] = 1.0
            sel_s_tt = ttnn.from_torch(
                sel_s, dtype=xn.dtype, layout=ttnn.TILE_LAYOUT, device=model.device, mesh_mapper=rep
            )
            rows = ttnn.matmul(sel_s_tt, xu)  # [1,1,32, full] replicated
            ttnn.deallocate(sel_s_tt)
            ttnn.deallocate(xu)
            shard = ttnn.linear(rows, model.lm_head_weight)  # [1,1,32,Vs] per device
            ttnn.deallocate(rows)
            rm = ttnn.untilize(shard, use_multicore=True)
            idx = ttnn.argmax(rm, dim=-1, keepdim=False)  # [1,1,32] uint32
            ttnn.deallocate(rm)
            val = ttnn.max(shard, dim=-1)  # [1,1,32]
            ttnn.deallocate(shard)
            idxs = ttnn.to_torch(idx, mesh_composer=comp).reshape(-1, _VERIFY_ROWS)
            vals = ttnn.to_torch(val, mesh_composer=comp).float().reshape(-1, _VERIFY_ROWS)
            ttnn.deallocate(idx)
            ttnn.deallocate(val)
            if getattr(model, "_lmhead_vocab_sharded", False):
                dwin = vals.argmax(dim=0)
                target_ids = [int(idxs[int(dwin[r]), r]) + int(dwin[r]) * per_shard for r in range(n_score)]
            else:
                target_ids = [int(idxs[0, r]) for r in range(n_score)]
            results.append((target_ids, hid))
        ttnn.deallocate(xn)
        for t in full_pts + chunk_pts + coss + sins + csis:
            ttnn.deallocate(t)
        return results

    # ── traced batched verify ────────────────────────────────────────────────
    def _init_verify_trace(self):
        """Persistent device I/O for the traced grouped verify (alloc once, pre-capture).

        Per-user variation (positions, rope windows, chunk page tables, device
        chunk starts, GDN per-row valid masks, row-select one-hots) is DATA in
        stacked [B, ...] buffers the graph slices per user statically.
        """
        model = self.model
        mesh = model.mesh_device
        B, bucket = self.B, _VERIFY_BUCKET
        rd = model.args.rope_head_dim
        conv_k = self._dns[0].K
        rep = dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

        def dev(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, **rep)

        self._vt = {
            "tok": dev(torch.zeros(B, bucket, dtype=torch.int64), ttnn.uint32, rm),
            "cos": dev(torch.zeros(B, bucket, rd, dtype=torch.bfloat16), ttnn.bfloat16, rm),
            "sin": dev(torch.zeros(B, bucket, rd, dtype=torch.bfloat16), ttnn.bfloat16, rm),
            "csi": dev(torch.zeros(B, dtype=torch.int32), ttnn.int32, rm),
            "chunk_pt": dev(torch.zeros(B, 2, dtype=torch.int32), ttnn.int32, rm),
            "full_pts": [
                dev(self.page_table[u : u + 1].contiguous().to(torch.int32), ttnn.int32, rm) for u in range(B)
            ],
            "conv_sel": dev(
                torch.zeros(B, conv_k - 1, (conv_k - 1) + bucket, dtype=torch.float32), ttnn.bfloat16, tile
            ),
            "mask_f32": dev(torch.zeros(B, bucket, 1, dtype=torch.float32), ttnn.float32, tile),
            "mask_bf16": dev(torch.zeros(B, bucket, 1, dtype=torch.float32), ttnn.bfloat16, tile),
            "sel_h": dev(torch.zeros(B, 2 * _VERIFY_ROWS, bucket, dtype=torch.float32), ttnn.bfloat16, tile),
            "sel_s": dev(torch.zeros(B, _VERIFY_ROWS, bucket, dtype=torch.float32), ttnn.bfloat16, tile),
        }

    def _stage_verify_trace(self, chunks):
        """Refresh the persistent traced-verify inputs from this iteration's chunks."""
        model, vt = self.model, self._vt
        B, bucket = self.B, _VERIFY_BUCKET
        rd = model.args.rope_head_dim
        conv_k = self._dns[0].K
        rep = dict(mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device))
        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

        tok = torch.zeros(B, bucket, dtype=torch.int64)
        cos = torch.zeros(B, bucket, rd, dtype=torch.bfloat16)
        sin = torch.zeros(B, bucket, rd, dtype=torch.bfloat16)
        csi = torch.zeros(B, dtype=torch.int32)
        cpt = torch.zeros(B, 2, dtype=torch.int32)
        conv_sel = torch.zeros(B, conv_k - 1, (conv_k - 1) + bucket, dtype=torch.float32)
        mask = torch.zeros(B, bucket, 1, dtype=torch.float32)
        sel_h = torch.zeros(B, 2 * _VERIFY_ROWS, bucket, dtype=torch.float32)
        sel_s = torch.zeros(B, _VERIFY_ROWS, bucket, dtype=torch.float32)
        for u, (tokens, a_u, n_u, hid_start, score_start) in enumerate(chunks):
            assert a_u % BLOCK == 0 and n_u <= bucket
            blk0 = a_u // BLOCK
            assert blk0 + 2 <= self.page_table.shape[1], "verify window exceeds the page-table block budget"
            tok[u, :n_u] = torch.tensor(tokens, dtype=torch.int64)
            cos_t, sin_t = model._rope_tp_cos_sin_torch(a_u, bucket)
            cos[u], sin[u] = cos_t.reshape(bucket, rd), sin_t.reshape(bucket, rd)
            csi[u] = a_u
            cpt[u] = self.page_table[u, blk0 : blk0 + 2].to(torch.int32)
            for j in range(conv_k - 1):
                conv_sel[u, j, n_u + j] = 1.0
            mask[u, :n_u, 0] = 1.0
            for j in range(min(2 * _VERIFY_ROWS, n_u - hid_start)):
                sel_h[u, j, hid_start + j] = 1.0
            for j in range(min(_VERIFY_ROWS, n_u - score_start)):
                sel_s[u, j, score_start + j] = 1.0
        for host_t, dtype, layout, dst in (
            (tok, ttnn.uint32, rm, "tok"),
            (cos, ttnn.bfloat16, rm, "cos"),
            (sin, ttnn.bfloat16, rm, "sin"),
            (csi, ttnn.int32, rm, "csi"),
            (cpt, ttnn.int32, rm, "chunk_pt"),
            (conv_sel, ttnn.bfloat16, tile, "conv_sel"),
            (mask, ttnn.float32, tile, "mask_f32"),
            (mask, ttnn.bfloat16, tile, "mask_bf16"),
            (sel_h, ttnn.bfloat16, tile, "sel_h"),
            (sel_s, ttnn.bfloat16, tile, "sel_s"),
        ):
            src = ttnn.from_torch(host_t, dtype=dtype, layout=layout, **rep)
            ttnn.copy_host_to_device_tensor(src, vt[dst])

    def _verify_trace_body(self):
        """The grouped verify graph over the persistent buffers.

        SELF-RESTORING (anchor snapshot copied back in-graph) and score-on-device
        per user. Per-batch-row TILE padding never crosses a reshape: batch-row
        merges/splits go through ROW_MAJOR or stay per-user slices.
        Returns per-user (hid_rows, idx, val) persistent output handles.
        """
        model, vt = self.model, self._vt
        B, bucket = self.B, _VERIFY_BUCKET
        rd = model.args.rope_head_dim
        for dn, (rec, carry) in zip(self._dns, self._snap):
            ttnn.copy(rec, dn.rec_state)
            ttnn.copy(carry, dn._batched_conv_carry)
        vm = {"conv_sel": vt["conv_sel"], "f32": vt["mask_f32"], "bf16": vt["mask_bf16"]}
        x = model.embd(vt["tok"])  # [B, bucket, d]
        d = x.shape[-1]
        x = ttnn.reshape(x, (1, 1, B * bucket, d))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        for layer in model.layers:
            attn_in = layer.attention_norm(x, mode=Mode.PREFILL)
            full = attn_in.shape[-1]
            if layer.is_full_attention:
                attn_in_b = ttnn.reshape(attn_in, (1, B, bucket, full))
                outs = []
                for u in range(B):
                    xi = ttnn.reshape(attn_in_b[:, u : u + 1, :, :], (1, 1, bucket, full))
                    cos_u = ttnn.to_layout(
                        ttnn.reshape(ttnn.slice(vt["cos"], [u, 0, 0], [u + 1, bucket, rd]), (1, 1, bucket, rd)),
                        ttnn.TILE_LAYOUT,
                    )
                    sin_u = ttnn.to_layout(
                        ttnn.reshape(ttnn.slice(vt["sin"], [u, 0, 0], [u + 1, bucket, rd]), (1, 1, bucket, rd)),
                        ttnn.TILE_LAYOUT,
                    )
                    csi_u = ttnn.slice(vt["csi"], [u], [u + 1])
                    cpt_u = ttnn.slice(vt["chunk_pt"], [u, 0], [u + 1, 2])
                    oi = layer.attention.forward_prefill_paged(
                        xi,
                        cos_u,
                        sin_u,
                        vt["full_pts"][u],
                        chunk_page_table=cpt_u,
                        chunk_start_idx=0,
                        chunk_start_idx_tensor=csi_u,
                        user_id=0,
                    )
                    for t in (xi, cos_u, sin_u, csi_u, cpt_u):
                        ttnn.deallocate(t)
                    outs.append(ttnn.reshape(oi, (1, 1, bucket, oi.shape[-1])))
                attn_out = ttnn.concat(outs, dim=2) if B > 1 else outs[0]
                for o in outs:
                    if o is not attn_out:
                        ttnn.deallocate(o)
            else:
                gdn_in = ttnn.reshape(attn_in, (B, bucket, full))
                attn_out = layer.attention.forward_prefill_batched(
                    gdn_in,
                    chunk_size=model.args.gdn_chunk_size,
                    valid_lens=None,
                    carry=True,
                    valid_masks=vm,
                    carry_inplace=True,
                )
                attn_out = ttnn.reshape(attn_out, (1, 1, B * bucket, attn_out.shape[-1]))
            ttnn.deallocate(attn_in)
            h = ttnn.add(x, attn_out)
            ttnn.deallocate(x)
            ttnn.deallocate(attn_out)
            ff_in = layer.ffn_norm(h, mode=Mode.PREFILL)
            ff_out = layer.feed_forward.forward(ff_in)
            ttnn.deallocate(ff_in)
            x = ttnn.add(h, ff_out)
            ttnn.deallocate(h)
            ttnn.deallocate(ff_out)

        xn = model.norm(x, mode=Mode.PREFILL)
        ttnn.deallocate(x)
        xn_b = ttnn.reshape(xn, (1, B, bucket, xn.shape[-1]))
        outputs = []
        for u in range(B):
            xu = ttnn.reshape(xn_b[:, u : u + 1, :, :], (1, 1, bucket, xn.shape[-1]))
            hsel = ttnn.reshape(
                ttnn.slice(vt["sel_h"], [u, 0, 0], [u + 1, 2 * _VERIFY_ROWS, bucket]),
                (1, 1, 2 * _VERIFY_ROWS, bucket),
            )
            hrows = ttnn.matmul(hsel, xu)  # [1,1,64, full] replicated
            ttnn.deallocate(hsel)
            ssel = ttnn.reshape(
                ttnn.slice(vt["sel_s"], [u, 0, 0], [u + 1, _VERIFY_ROWS, bucket]), (1, 1, _VERIFY_ROWS, bucket)
            )
            rows = ttnn.matmul(ssel, xu)
            ttnn.deallocate(ssel)
            ttnn.deallocate(xu)
            shard = ttnn.linear(rows, model.lm_head_weight)  # [1,1,32,Vs] per device
            ttnn.deallocate(rows)
            rm_l = ttnn.untilize(shard, use_multicore=True)
            idx = ttnn.argmax(rm_l, dim=-1, keepdim=False)  # [1,1,32] uint32
            ttnn.deallocate(rm_l)
            val = ttnn.max(shard, dim=-1)  # [1,1,32]
            ttnn.deallocate(shard)
            outputs.append((hrows, idx, val))
        ttnn.deallocate(xn)
        return outputs

    def _verify_trace_compile(self, chunks):
        """Eager compile pass of the traced-verify graph (BEFORE any capture)."""
        self._init_verify_trace()
        self._stage_verify_trace(chunks)
        outputs = self._verify_trace_body()
        ttnn.synchronize_device(self.model.mesh_device)
        for hrows, idx, val in outputs:
            for t in (hrows, idx, val):
                ttnn.deallocate(t)

    def _verify_trace_capture(self):
        """Capture the traced verify (the compile pass ran; inputs still staged)."""
        mesh = self.model.mesh_device
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        outputs = self._verify_trace_body()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        self._vt["id"] = tid
        self._vt["outputs"] = outputs
        logger.info(f"spec[c{self.B}] verify trace captured (grouped bucket 128, self-restoring)")

    def _batched_verify_traced(self, chunks):
        """Replay the grouped verify; read per-user accept ids + hidden rows."""
        self._stage_verify_trace(chunks)
        ttnn.execute_trace(self.model.mesh_device, self._vt["id"], cq_id=0, blocking=False)
        model = self.model
        comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
        per_shard = model.args.vocab_size // model.num_devices
        results = []
        for u, (_tokens, _a_u, n_u, hid_start, score_start) in enumerate(chunks):
            hrows, idx, val = self._vt["outputs"][u]
            n_hid = min(2 * _VERIFY_ROWS, n_u - hid_start)
            hid = ttnn.to_torch(ttnn.get_device_tensors(hrows)[0]).float().reshape(2 * _VERIFY_ROWS, -1)[:n_hid]
            n_score = min(_VERIFY_ROWS, n_u - score_start)
            idxs = ttnn.to_torch(idx, mesh_composer=comp).reshape(-1, _VERIFY_ROWS)
            vals = ttnn.to_torch(val, mesh_composer=comp).float().reshape(-1, _VERIFY_ROWS)
            if getattr(model, "_lmhead_vocab_sharded", False):
                dwin = vals.argmax(dim=0)
                target_ids = [int(idxs[int(dwin[r]), r]) + int(dwin[r]) * per_shard for r in range(n_score)]
            else:
                target_ids = [int(idxs[0, r]) for r in range(n_score)]
            results.append((target_ids, hid))
        return results

    def _setup_traces(self):
        """One-time trace setup, honoring the compile-before-any-capture rule:
        (1) drafter window compile passes, (2) verify-trace compile pass, then
        (3) drafter captures, (4) verify capture. Runs after the eager first
        verify (which warmed the shared eager programs)."""
        K = self.draft_len
        w_max = (K + 2) + K - 1  # steady-state legs: pendings (<= K+1, +1 slack) + chain
        self.mtp.ensure_batched_window(w_max, self.B)
        # Capture-time staging: any REAL positions work (iteration 1's oversized
        # prompt-tail window runs eager legs, so no real schedule exists yet);
        # the compile/capture KV writes land at these slots and are rewritten in
        # order by later legs before anything attends to them.
        setup_pos = torch.stack(
            [torch.full((w_max,), int(slot.pending[-1][2]), dtype=torch.int32) for slot in self.slots]
        )
        self.mtp.stage_batched_window(setup_pos)
        self.mtp.compile_batched_window()
        chunks = []
        for slot in self.slots:
            tail = slot.committed[slot.a :]
            chunks.append((tail, slot.a, len(tail), 0, max(0, len(tail) - 1)))
        self._verify_trace_compile(chunks)
        self.mtp.capture_batched_window()
        self._verify_trace_capture()

    def _sched_pending(self, slot):
        """The pending list a drafter schedule sees for `slot`: a finished (or
        drained) slot contributes ONE idempotent replay leg, so a dead user's
        long tail never widens (or overflows) the shared window."""
        if slot.pending and not slot.done:
            return slot.pending
        if slot.pending:
            return slot.pending[-1:]
        return [(slot.committed[-1], None, max(0, len(slot.committed) - 2))]

    def _schedule_positions(self, width):
        """End-aligned per-user drafter schedule for `width` legs.

        Returns (pos_table [B, width], pads [B]): user u's real legs occupy the
        LAST n_legs_u slots; earlier slots replay its first pending (an
        idempotent KV rewrite whose output is ignored).
        """
        K = self.draft_len
        pos = torch.zeros(self.B, width, dtype=torch.int32)
        pads = []
        for u, slot in enumerate(self.slots):
            pending = self._sched_pending(slot)
            n_legs = len(pending) + K - 1
            pad = width - n_legs
            assert pad >= 0, f"user {u}: {n_legs} legs exceed window {width}"
            first_pos = pending[0][2]
            for j in range(width):
                if j < pad:
                    pos[u, j] = first_pos
                elif j < pad + len(pending):
                    pos[u, j] = pending[j - pad][2]
                else:
                    pos[u, j] = pending[-1][2] + (j - pad - len(pending) + 1)
            pads.append(pad)
        return pos, pads

    def _draft_traced(self, width, max_new_tokens):
        """One end-aligned traced drafter window advancing ALL users K legs.

        Per-iteration pos/rope upload; catch-up legs read nothing back; each
        draft leg reads two tiny score tensors for all users at once. Returns
        per-user drafts ([] for inactive users, whose legs are idempotent
        replays)."""
        K = self.draft_len
        pos_table, pads = self._schedule_positions(width)
        self.mtp.stage_batched_window(pos_table)
        active = [not (s.done or len(s.out) >= max_new_tokens) for s in self.slots]
        for u, slot in enumerate(self.slots):
            if active[u]:
                slot.k_used.append(K)
        # Catch-up legs 0..width-K (leg width-K is every user's LAST pending).
        for j in range(width - K + 1):
            toks, hids = [], []
            for u, slot in enumerate(self.slots):
                pending = self._sched_pending(slot)
                i = min(max(0, j - pads[u]), len(pending) - 1)
                tok_u, hid_u, _pos = pending[i]
                toks.append(tok_u)
                hids.append(hid_u if hid_u is not None else torch.zeros(self.model.args.dim))
            picks = self.mtp.step_batched(j, toks, torch.stack([h.float() for h in hids]), want_tokens=(j == width - K))
        drafts = [[picks[u]] for u in range(self.B)]
        # Chained draft legs (uniform across users at static K).
        for j in range(width - K + 1, width):
            toks = [drafts[u][-1] for u in range(self.B)]
            picks = self.mtp.step_batched(j, toks, chain_hidden=True, want_tokens=True)
            for u in range(self.B):
                drafts[u].append(picks[u])
        for u, slot in enumerate(self.slots):
            if active[u]:
                slot.pending = []
            else:
                drafts[u] = []
        return drafts

    # ── per-user commit ──────────────────────────────────────────────────────
    def _commit_user(self, u):
        """Advance user u's anchor by whole blocks when due (B=1 scratch chunk)."""
        slot = self.slots[u]
        k = commit_advance(len(slot.committed) - slot.a)
        if k == 0:
            return
        t0 = time.perf_counter()
        model = self.model
        if self._use_trace and self._snap is not None:
            self._restore_from_snapshot()
        prev = model._bind_gdn_prefill_scratch()
        try:
            # Stash the live batched handles so the row write-back can target them
            # while the scratch is bound.
            for dn, _B, rec_b, _conv_b, carry_b, _z, _s in prev:
                dn._spec_batched_rec = rec_b
                dn._spec_batched_carry = dn_batched_carry = getattr(dn, "_batched_conv_carry", None)
                assert dn_batched_carry is not None
            self._seed_scratch_from_snapshot(u)
            self._scratch_chunk(u, slot.committed[slot.a : slot.a + k], slot.a, k)
            self._writeback_scratch_row(u)
        finally:
            model._unbind_gdn_prefill_scratch(prev)
            for dn in self._dns:
                dn._spec_batched_rec = None
                dn._spec_batched_carry = None
        slot.a += k
        self._timing["commit"] += time.perf_counter() - t0

    # ── generation ───────────────────────────────────────────────────────────
    def _first_verify(self):
        """Draft-less batched verify over every user's prompt tail: samples each
        user's first token and arms the catch-up pairs."""
        chunks = []
        for slot in self.slots:
            tail = slot.committed[slot.a :]
            chunks.append((tail, slot.a, len(tail), 0, len(tail) - 1))
        results = self._batched_verify(chunks)
        self._restore_from_snapshot()
        for u, slot in enumerate(self.slots):
            target_ids, hid = results[u]
            first_token = target_ids[0]  # score rows anchored at the tail's last row
            c = len(slot.committed) - 1
            for i in range(slot.a, c):
                slot.pending.append((slot.committed[i + 1], hid[i - slot.a], i))
            slot.pending.append((first_token, hid[c - slot.a], c))
            slot.committed.append(first_token)
            slot.out.append(first_token)
            if first_token in self.stop_tokens:
                slot.done = True
            self._commit_user(u)

    def generate(self, max_new_tokens, adaptive_k=False):
        """Greedy batched speculative generation.

        Returns (per-user generated lists, stats dict).
        """
        K = self.draft_len
        use_trace = self._use_trace
        assert not (use_trace and adaptive_k), "the traced batched drafter is static-K (TT_SPEC_TRACE=0 for adaptive)"
        self._first_verify()
        if use_trace:
            self._setup_traces()
        w_max = self.mtp._bwin["w_max"] if use_trace else 0

        while any(not s.done and len(s.out) < max_new_tokens for s in self.slots):
            # 1. Per-user drafting. Traced: one end-aligned window of B-row legs
            # (per-iteration pos/rope upload; catch-up legs read nothing back;
            # each draft leg reads two tiny score tensors for ALL users). Eager
            # (or an oversized iteration-1 window): per-user B=1 legs.
            t0 = time.perf_counter()
            drafts = [[] for _ in range(self.B)]
            width = max(len(self._sched_pending(s2)) for s2 in self.slots) + K - 1
            if use_trace and width <= w_max:
                drafts = self._draft_traced(width, max_new_tokens)
            else:
                for u, slot in enumerate(self.slots):
                    if slot.done or len(slot.out) >= max_new_tokens:
                        continue
                    k_t = adaptive_draft_len(slot.k_ema, K) if adaptive_k else K
                    slot.k_used.append(k_t)
                    last_pos = slot.pending[-1][2]
                    for tok, hid_row, pos in slot.pending[:-1]:
                        self.mtp.step(tok, hid_row, pos, user=u)
                    d_logits, g = self.mtp.step(*slot.pending[-1], user=u)
                    slot.pending = []
                    drafts[u] = [int(d_logits.argmax())]
                    for j in range(1, k_t):
                        d_logits, g = self.mtp.step(drafts[u][-1], g, last_pos + j, user=u)
                        drafts[u].append(int(d_logits.argmax()))
            self._timing["draft"] += time.perf_counter() - t0

            # 2. ONE batched verify over every user's [tail + drafts] chunk.
            # Finished users ride along with their (never-committed) tail.
            t0 = time.perf_counter()
            chunks = []
            for u, slot in enumerate(self.slots):
                c = len(slot.committed) - 1
                tokens = slot.committed[slot.a :] + drafts[u]
                chunks.append((tokens, slot.a, len(tokens), c - slot.a, c - slot.a))
            if use_trace:
                # The trace self-restores to the anchor snapshot in-graph.
                results = self._batched_verify_traced(chunks)
            else:
                results = self._batched_verify(chunks)
                self._restore_from_snapshot()
            self._timing["verify"] += time.perf_counter() - t0

            # 3. Per-user accept/commit (fully desynced).
            for u, slot in enumerate(self.slots):
                if slot.done or not drafts[u] or len(slot.out) >= max_new_tokens:
                    continue
                target_ids, hid = results[u]
                c = len(slot.committed) - 1
                k_t = len(drafts[u])
                m, new_tokens = greedy_accept(drafts[u], target_ids[: k_t + 1])
                slot.accepts.append(m)
                slot.k_ema = 0.7 * slot.k_ema + 0.3 * m
                for j, tok in enumerate(new_tokens):
                    slot.committed.append(tok)
                    slot.out.append(tok)
                    slot.pending.append((tok, hid[j], c + j))
                    if tok in self.stop_tokens:
                        slot.done = True
                        break
                    if len(slot.out) >= max_new_tokens:
                        break
                self._commit_user(u)
        return [list(s.out) for s in self.slots], self._stats()

    def _stats(self):
        iters = max((len(s.accepts) for s in self.slots), default=0)
        total = sum(sum(s.accepts) for s in self.slots)
        proposed = sum(sum(s.k_used) for s in self.slots)
        # Active-normalized throughput: a user that stops early idles its slot
        # for the remaining iterations — the plain per-user metric charges those
        # dead rows, understating what a refilled slot would sustain.
        active_iters = sum(len(s.accepts) for s in self.slots)
        # Conditional accept per draft position: p_j = P(draft j lands | drafts
        # 1..j-1 landed). Flat p's mean K is the lever's limit; a die-off at
        # some j says the drafter chain degrades there.
        ms = [m for s in self.slots for m in s.accepts]
        per_pos = []
        for j in range(1, self.draft_len + 1):
            prev = sum(1 for m in ms if m >= j - 1)
            per_pos.append(round(sum(1 for m in ms if m >= j) / prev, 3) if prev else 0.0)
        return {
            "users": self.B,
            "iterations": iters,
            "accepted_drafts": total,
            "accept_rate": (total / proposed) if proposed else 0.0,
            "tokens_per_user_iteration": (sum(len(s.out) for s in self.slots) / (self.B * iters) if iters else 0.0),
            "tokens_per_active_iteration": (
                sum(len(s.accepts) + sum(s.accepts) for s in self.slots) / active_iters if active_iters else 0.0
            ),
            "per_position_accept": per_pos,
            "per_user_accepts": [list(s.accepts) for s in self.slots],
            "timing_s": dict(self._timing),
        }
