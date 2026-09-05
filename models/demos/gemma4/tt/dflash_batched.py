# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched (B>1) dFlash fused decoder — one trace, B users, ragged acceptance.

Design constraints inherited from the B=1 work (see dflash_drafter.py and the
gemma4_galaxy_effort ledger):
  * ONE metal trace total — B per-user decoders would replay B alternating
    CCL-bearing traces, the documented deadlock.
  * The verify's B*(V+1) rows obey the measured 32-row cliff, so B<=8 at V=3
    and B<=4 at V=7 (code). The value case is CODE at B=2..4.
  * All trace outputs land in persistent slots (mid-graph tensors get
    reclaimed — root cause #6).
  * Drafter context K/V live in per-layer STACKED roped caches
    [1, local_kv, B*cap, hd]; commits are embed-gather merges through one
    concatenated per-user row map. Blocks attend through BLOCK-DIAGONAL masks
    (stationary in steady state, per user, exactly like B=1).
  * Target verify = ttnn_packed_verify_forward with B in the batch dim,
    per-position fallback KV writes, width-matched per-layer tables.

v1 scope: greedy, code-targeted B<=4 (V>=7) or B<=8 (V=3); per-user positions
diverge (ragged); prefill is per-user and never traced.
"""

import os as _os

import torch

import ttnn


class DFlashBatchedDecoder:
    def __init__(self, target_model, drafter, kv_layers, page_table_torch, B, ctx_cap=2048):
        self.target = target_model
        self.drafter = drafter
        self.kv_layers = kv_layers
        self.page_table_torch = page_table_torch
        self.B = B
        self.cap = ctx_cap
        self.mesh_device = drafter.mesh_device
        self._mapper = drafter._replicate
        self._tp = drafter.tp
        K = drafter.block_size - 1
        self.K = K
        H = drafter.hidden
        self.V = min(int(_os.environ.get("GEMMA4_DFLASH_VERIFY", str(K))), K)
        self.P_v = self.V + 1
        # Verify modes:
        #   fold (default): ONE B=1-shaped packed verify -- users folded into the
        #     packed-rows dim (packed_p = B*P_v), single-row VIRTUAL page table
        #     (user page rows concatenated), block-offset masks, virtual write
        #     positions b*S + pos. Reuses the exact kernel config the B=1 V=15
        #     runs validated (PNH=64, one table row).
        #   split (GEMMA4_DFLASH_BVSPLIT=1): B separate B=1 verifies -- correctness
        #     reference, ~2x target weight reads.
        #   bbatch (GEMMA4_DFLASH_BBATCH=1): B in the SDPA batch dim. BROKEN as of
        #     2026-09-05: user 0 clean, user 1+ corrupt even with identical
        #     prompts -- the packed SDPA (q [1,B,H*P,hd], mask [B,1,H*P,S]) has
        #     never been exercised at batch>1; kept for kernel-side debugging.
        self.verify_split = _os.environ.get("GEMMA4_DFLASH_BVSPLIT", "0") == "1"
        self.verify_bbatch = _os.environ.get("GEMMA4_DFLASH_BBATCH", "0") == "1"
        if B * self.P_v > 32:
            raise ValueError(
                f"B*(V+1) = {B * self.P_v} > 32: the verify row cliff (measured on MTP) — "
                f"lower GEMMA4_DFLASH_VERIFY or B"
            )

        z = torch.zeros
        mkT = dict(device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._mapper)
        mkU = dict(device=self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._mapper)

        # stacked per-layer roped ctx K/V
        kvshape = (1, drafter.local_kv, B * ctx_cap, drafter.head_dim)
        self.ctx_k = [ttnn.from_torch(z(*kvshape, dtype=torch.bfloat16), **mkT) for _ in drafter.layers]
        self.ctx_v = [ttnn.from_torch(z(*kvshape, dtype=torch.bfloat16), **mkT) for _ in drafter.layers]
        # commit machinery: previous verify's fc rows (user-major) + row map
        self.fc_prev = ttnn.from_torch(z(1, 1, B * self.P_v, H, dtype=torch.bfloat16), **mkT)
        self.merge_idx = ttnn.from_torch(torch.arange(B * ctx_cap, dtype=torch.int64).reshape(1, B * ctx_cap), **mkU)
        self.commit_pos = ttnn.from_torch(z(1, B * self.P_v, dtype=torch.int64), **mkU)
        # drafter block inputs (user-major folded rows)
        K1 = K + 1
        self.K1 = K1
        self.noise_rows = ttnn.from_torch(z(1, 1, B * K1, H, dtype=torch.bfloat16), **mkT)
        self.blk_pos = ttnn.from_torch(z(1, B * K1, dtype=torch.int64), **mkU)
        S_d = B * ctx_cap + B * K1
        self.S_d = S_d
        self.dmask_full = ttnn.from_torch(z(1, 1, B * K1, S_d, dtype=torch.bfloat16), **mkT)
        self.dmask_slide = ttnn.from_torch(z(1, 1, B * K1, S_d, dtype=torch.bfloat16), **mkT)
        # verify inputs
        self.vx = ttnn.from_torch(z(1, B * self.P_v, dtype=torch.int64), **mkU)
        self.pv_pos = ttnn.from_torch(z(1, B * self.P_v, dtype=torch.int64), **mkU)
        self.pv_widx = [
            ttnn.from_torch(
                z(B, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self._mapper,
            )
            for _ in range(self.P_v)
        ]
        self.pv_widx_fold = [
            ttnn.from_torch(
                z(1, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self._mapper,
            )
            for _ in range(B * self.P_v)
        ]
        self.pv_widx_u = None
        if self.verify_split:
            self.pv_widx_u = [
                [
                    ttnn.from_torch(
                        z(1, dtype=torch.int32),
                        device=self.mesh_device,
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=self._mapper,
                    )
                    for _ in range(self.P_v)
                ]
                for _ in range(B)
            ]
        self.v_pt = None  # width-matched batched table (built in capture)
        self.pv_ptl = None
        self.pv_sk = None
        self.pv_mask_full = None
        self.pv_mask_slide = None
        self._sampler = getattr(target_model, "sampling", None)
        drafter._sampler = self._sampler
        if drafter._mask_rows is None or drafter._mask_rows.shape[2] != K:
            _mrows = drafter._embed_w_host[[drafter.mask_token_id] * K].to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
            drafter._mask_rows = ttnn.from_torch(_mrows, **mkT)
        self.tap_bufs = [None for _ in drafter.target_layer_ids]
        self.out_draft = None
        self.out_vidx = None
        self.trace = None
        # per-user host state
        self.start = [0] * B
        self.anchor = [0] * B
        self.win_first = [0] * B
        self.ctx_len = [0] * B
        self._dmask_key = None

    # ── host input builders ────────────────────────────────────────────────
    def _upload(self, host, dev, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT):
        h = ttnn.from_torch(host, layout=layout, dtype=dtype, mesh_mapper=self._mapper)
        ttnn.copy_host_to_device_tensor(h, dev)
        h.deallocate(True)

    def _upload_iter_inputs(self):
        d = self.drafter
        B, K1, P_v = self.B, self.K1, self.P_v
        # noise rows: [anchor_b, mask x K] per user, user-major
        rows = []
        for b in range(B):
            rows.append(d._embed_w_host[[self.anchor[b]]].to(torch.bfloat16))
            rows.append(d._embed_w_host[[d.mask_token_id] * self.K].to(torch.bfloat16))
        noise = torch.cat(rows, dim=0).reshape(1, 1, B * K1, -1)
        self._upload(noise, self.noise_rows, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        bp = torch.zeros(1, B * K1, dtype=torch.int64)
        for b in range(B):
            bp[0, b * K1 : (b + 1) * K1] = torch.arange(self.start[b], self.start[b] + K1)
        self._upload(bp, self.blk_pos)
        # verify tokens are bound in-body from draft output; positions + write idxs here
        pp = torch.zeros(1, B * P_v, dtype=torch.int64)
        for b in range(B):
            pp[0, b * P_v : (b + 1) * P_v] = torch.arange(self.start[b], self.start[b] + P_v)
        self._upload(pp, self.pv_pos)
        for p in range(P_v):
            w = torch.tensor([self.start[b] + p for b in range(B)], dtype=torch.int32)
            self._upload(w, self.pv_widx[p], dtype=ttnn.int32)
        if self.pv_widx_u is not None:
            for b in range(B):
                for p in range(P_v):
                    self._upload(
                        torch.tensor([self.start[b] + p], dtype=torch.int32),
                        self.pv_widx_u[b][p],
                        dtype=ttnn.int32,
                    )
        # verify masks: per-user causal/window rows over the capped S_k
        NEG = -1e9
        j = torch.arange(self.pv_sk)
        W = self.target.hf_config.sliding_window
        ring = self.pv_ring
        if not (self.verify_split or self.verify_bbatch):
            # fold mode: rows user-major over B*P_v, columns block-diagonal per
            # user at stride pv_sk; write idxs are VIRTUAL positions b*pv_sk+pos
            S = self.pv_sk
            mf = torch.full((1, 1, B * P_v, B * S), NEG)
            ms = torch.full((1, 1, B * P_v, B * S), NEG)
            for b in range(B):
                for p in range(P_v):
                    up = self.start[b] + p
                    r = b * P_v + p
                    mf[0, 0, r, b * S : (b + 1) * S] = torch.where(j <= up, 0.0, NEG)
                    ms[0, 0, r, b * S : (b + 1) * S] = torch.where((j <= up) & (j > up - W), 0.0, NEG)
            self._upload(mf.to(torch.bfloat16), self.pv_mask_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            self._upload(ms.to(torch.bfloat16), self.pv_mask_slide, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            for b in range(B):
                for p in range(P_v):
                    self._upload(
                        torch.tensor([b * S + self.start[b] + p], dtype=torch.int32),
                        self.pv_widx_fold[b * P_v + p],
                        dtype=ttnn.int32,
                    )
        else:
            mf = torch.empty(B, 1, P_v, self.pv_sk)
            ms = torch.empty(B, 1, P_v, self.pv_sk)
            for b in range(B):
                for p in range(P_v):
                    up = self.start[b] + p
                    mf[b, 0, p] = torch.where(j <= up, 0.0, NEG)
                if ring:
                    top = self.start[b] + P_v - 1
                    jr = torch.arange(ring)
                    dd = torch.remainder(top - jr, ring)
                    for p in range(P_v):
                        lo = P_v - 1 - p
                        ok = (dd >= lo) & (dd < lo + W) & (dd <= top)
                        ms[b, 0, p, :ring] = torch.where(ok, 0.0, NEG)
                    ms[b, 0, :, ring:] = NEG
                else:
                    for p in range(P_v):
                        up = self.start[b] + p
                        ms[b, 0, p] = torch.where((j <= up) & (j > up - W), 0.0, NEG)
            self._upload(mf.to(torch.bfloat16), self.pv_mask_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            self._upload(ms.to(torch.bfloat16), self.pv_mask_slide, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        # drafter block-diagonal masks: stationary once every user's window is full
        key = tuple((self.ctx_len[b] - self.win_first[b], self.start[b] - self.win_first[b]) for b in range(B))
        if self._dmask_key != key:
            S_d = self.S_d
            dmf = torch.full((B * K1, S_d), NEG)
            dms = torch.full((B * K1, S_d), NEG)
            w = d.sliding_window
            for b in range(B):
                win_len = self.ctx_len[b] - self.win_first[b]
                qpos = torch.arange(self.start[b], self.start[b] + K1)[:, None]
                cpos = torch.arange(self.win_first[b], self.win_first[b] + self.cap)[None, :]
                cvalid = (torch.arange(self.cap) < win_len)[None, :]
                blk = slice(b * K1, (b + 1) * K1)
                ctxc = slice(b * self.cap, (b + 1) * self.cap)
                blkc = slice(B * self.cap + b * K1, B * self.cap + (b + 1) * K1)
                dmf[blk, ctxc] = torch.where(cvalid.expand(K1, -1), 0.0, NEG)
                dmf[blk, blkc] = 0.0
                if w:
                    vis = cvalid & (qpos - cpos < w) & (cpos - qpos < w)
                    dms[blk, ctxc] = torch.where(vis, 0.0, NEG)
                    dms[blk, blkc] = 0.0
                else:
                    dms[blk, ctxc] = dmf[blk, ctxc]
                    dms[blk, blkc] = 0.0
            self._upload(
                dmf.reshape(1, 1, B * K1, S_d).to(torch.bfloat16),
                self.dmask_full,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            self._upload(
                dms.reshape(1, 1, B * K1, S_d).to(torch.bfloat16),
                self.dmask_slide,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            self._dmask_key = key

    # ── graph ──────────────────────────────────────────────────────────────
    def _body(self):
        d = self.drafter
        B, K1, P_v, K = self.B, self.K1, self.P_v, self.K
        # commit previous fc rows into the stacked ctx K/V caches
        cos_c = ttnn.unsqueeze_to_4D(ttnn.embedding(self.commit_pos, d._cos_2d, layout=ttnn.TILE_LAYOUT))
        sin_c = ttnn.unsqueeze_to_4D(ttnn.embedding(self.commit_pos, d._sin_2d, layout=ttnn.TILE_LAYOUT))
        kv_new = d.project_ctx_kv(self.fc_prev, cos_c, sin_c)
        cos_c.deallocate(True)
        sin_c.deallocate(True)
        hd = d.head_dim
        for li, (k_new, v_new) in enumerate(kv_new):
            for cache, new in ((self.ctx_k[li], k_new), (self.ctx_v[li], v_new)):
                src = ttnn.concat([cache, new], dim=2)
                src2d = ttnn.reshape(src, (B * self.cap + B * P_v, hd))
                m = ttnn.embedding(self.merge_idx, src2d, layout=ttnn.TILE_LAYOUT)
                m4 = ttnn.reshape(m, (1, 1, B * self.cap, hd))
                ttnn.assign(m4, cache)
                for t in (m4, m, src2d, src):
                    try:
                        t.deallocate(True)
                    except Exception:
                        pass
            k_new.deallocate(True)
            v_new.deallocate(True)
        # drafter block (folded rows)
        cos_blk = ttnn.unsqueeze_to_4D(ttnn.embedding(self.blk_pos, d._cos_2d, layout=ttnn.TILE_LAYOUT))
        sin_blk = ttnn.unsqueeze_to_4D(ttnn.embedding(self.blk_pos, d._sin_2d, layout=ttnn.TILE_LAYOUT))
        draft_ids = d.block_forward_cached_batched(
            self.noise_rows,
            self.ctx_k,
            self.ctx_v,
            cos_blk,
            sin_blk,
            self.dmask_full,
            self.dmask_slide,
            B,
            self.cap,
        )  # user-major [.., B*K] uint32
        cos_blk.deallocate(True)
        sin_blk.deallocate(True)
        # verify tokens: per user [anchor | drafts[:V]] — assembled on device
        # via explicit ttnn.slice (the proven MTP fused idiom; bracket slicing
        # RM uint32 tensors segfaulted here).
        flat = ttnn.reshape(draft_ids, (1, B * K))
        H_l = self.target.layers[0].self_attn.config.num_attention_heads // self._tp
        self.target._dflash_sharded_logits = self._sampler is not None
        if self.verify_split:
            # Isolation probe: B separate B=1-shaped verifies (own 1-row table,
            # [1] write idxs, own mask) + per-user sampler. Slower (B x target
            # weight reads) -- correctness reference only.
            assert self._sampler is not None
            S_f = int(self.pv_mask_full.shape[3])
            S_s = int(self.pv_mask_slide.shape[3])
            vidx_parts, fc_parts = [], []
            vx = None
            for b in range(B):
                a_b = ttnn.slice(self.noise_anchor_tok, [0, b], [1, b + 1])
                dr_b = ttnn.slice(flat, [0, b * K], [1, b * K + self.V])
                vx_b = ttnn.concat([a_b, dr_b], dim=1)
                a_b.deallocate(True)
                dr_b.deallocate(True)
                pos_b = ttnn.slice(self.pv_pos, [0, b * P_v], [1, (b + 1) * P_v])
                mfb = ttnn.slice(self.pv_mask_full, [b, 0, 0, 0], [b + 1, 1, P_v, S_f])
                msb = ttnn.slice(self.pv_mask_slide, [b, 0, 0, 0], [b + 1, 1, P_v, S_s])
                m_full = ttnn.repeat(mfb, ttnn.Shape([1, 1, H_l, 1]))
                m_slide = ttnn.repeat(msb, ttnn.Shape([1, 1, H_l, 1]))
                mfb.deallocate(True)
                msb.deallocate(True)
                lg_b, _vh_b = self.target.ttnn_packed_verify_forward(
                    x=vx_b,
                    position_idx=pos_b,
                    attn_mask_full=m_full,
                    attn_mask_sliding=m_slide,
                    packed_p=P_v,
                    page_table=self.v_pt_u[b],
                    kv_cache=self.kv_layers,
                    kv_write_idxs=self.pv_widx_u[b],
                    page_tables_per_layer=None,
                )
                for t in (m_full, m_slide, pos_b, vx_b):
                    t.deallocate(True)
                v_b, _lp = self._sampler.sample(lg_b, enable_trace=False)
                vidx_parts.append(v_b)
                cat_b = ttnn.concat(self.tap_bufs, dim=3)
                fc_parts.append(ttnn.linear(cat_b, d.fc, compute_kernel_config=d._ckc))
                cat_b.deallocate(True)
            vidx = ttnn.concat(vidx_parts, dim=len(vidx_parts[0].shape) - 1)
            fc_out = ttnn.concat(fc_parts, dim=2)
            for t in vidx_parts + fc_parts:
                t.deallocate(True)
            ttnn.assign(fc_out, self.fc_prev)
            fc_out.deallocate(True)
        else:
            parts = []
            for b in range(B):
                parts.append(ttnn.slice(self.noise_anchor_tok, [0, b], [1, b + 1]))
                parts.append(ttnn.slice(flat, [0, b * K], [1, b * K + self.V]))
            vx = ttnn.concat(parts, dim=1)  # [1, B*P_v]
            for t in parts:
                try:
                    t.deallocate(True)
                except Exception:
                    pass
            m_full = ttnn.repeat(self.pv_mask_full, ttnn.Shape([1, 1, H_l, 1]))
            m_slide = ttnn.repeat(self.pv_mask_slide, ttnn.Shape([1, 1, H_l, 1]))
            if self.verify_bbatch:
                logits, vhidden = self.target.ttnn_packed_verify_forward(
                    x=vx,
                    position_idx=self.pv_pos,
                    attn_mask_full=m_full,
                    attn_mask_sliding=m_slide,
                    packed_p=P_v,
                    page_table=self.v_pt,
                    kv_cache=self.kv_layers,
                    kv_write_idxs=self.pv_widx,
                    page_tables_per_layer=self.pv_ptl,
                )
            else:
                # fold: users in the packed dim -- ONE B=1-shaped verify with the
                # virtual single-row table and block-offset masks (see __init__)
                logits, vhidden = self.target.ttnn_packed_verify_forward(
                    x=vx,
                    position_idx=self.pv_pos,
                    attn_mask_full=m_full,
                    attn_mask_sliding=m_slide,
                    packed_p=B * P_v,
                    page_table=self.v_pt_virt,
                    kv_cache=self.kv_layers,
                    kv_write_idxs=self.pv_widx_fold,
                    page_tables_per_layer=None,
                )
            m_full.deallocate(True)
            m_slide.deallocate(True)
            if self._sampler is not None:
                vidx, _lp = self._sampler.sample(logits, enable_trace=False)
            else:
                vidx = ttnn.argmax(logits[:, :, :, : d.vocab], dim=-1)
            # NOTE: match the validated B=1 verify tail EXACTLY -- neither logits
            # nor vhidden is deallocated here. Deallocating vhidden mid-graph
            # segfaulted (it can alias model-internal state); both are trace-region
            # tensors with a fixed footprint, not per-replay leaks.
            # tap fc -> persistent for next commit
            cat = ttnn.concat(self.tap_bufs, dim=3)
            fc_out = ttnn.linear(cat, d.fc, compute_kernel_config=d._ckc)
            ttnn.assign(fc_out, self.fc_prev)
            fc_out.deallocate(True)
        if self.out_draft is None:
            self.out_draft = ttnn.clone(draft_ids)
            self.out_vidx = ttnn.clone(vidx)
        ttnn.copy(draft_ids, self.out_draft)
        ttnn.copy(vidx, self.out_vidx)
        draft_ids.deallocate(True)
        vidx.deallocate(True)
        if vx is not None:
            vx.deallocate(True)
        return self.out_draft, self.out_vidx

    def capture(self, anchors, starts, max_new=256):
        B, P_v = self.B, self.P_v
        self.anchor = list(anchors)
        self.start = list(starts)
        # verify horizon + width-matched tables (mirrors the B=1 decoder)
        horizon = max(starts) + max_new + P_v + 64
        if not (self.verify_split or self.verify_bbatch):
            # fold mode: block-granular S_k -- the virtual table is B*(pv_sk/64)
            # wide and paged_update_cache requires table width <= pool blocks,
            # so every wasted block costs B entries. 64-aligned satisfies the
            # mask/k_chunk contract.
            self.pv_sk = ((horizon + 63) // 64) * 64
        else:
            self.pv_sk = ((horizon + 1023) // 1024) * 1024
        t = self.target
        self.pv_ring = None
        for layer in t.layers:
            mod = getattr(layer.self_attn.config, "cache_position_modulo", None)
            if mod is not None:
                self.pv_ring = int(mod)
                break
        z = torch.zeros
        mkT = dict(device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._mapper)
        fold = not (self.verify_split or self.verify_bbatch)
        if fold and self.pv_ring:
            raise NotImplementedError(
                "batched dFlash fold-verify + bounded ring: virtual ring tables not wired yet "
                "(use GEMMA4_DFLASH_BVSPLIT=1 for bounded B>1, or unbounded KV)"
            )
        if fold:
            # users folded into packed rows; columns are per-user S_k blocks
            self.pv_mask_full = ttnn.from_torch(z(1, 1, B * P_v, B * self.pv_sk, dtype=torch.bfloat16), **mkT)
            self.pv_mask_slide = ttnn.from_torch(z(1, 1, B * P_v, B * self.pv_sk, dtype=torch.bfloat16), **mkT)
        else:
            self.pv_mask_full = ttnn.from_torch(z(B, 1, P_v, self.pv_sk, dtype=torch.bfloat16), **mkT)
            s_slide = self.pv_ring if self.pv_ring else self.pv_sk
            self.pv_mask_slide = ttnn.from_torch(z(B, 1, P_v, s_slide, dtype=torch.bfloat16), **mkT)
        # per-user distinct table rows, width-matched to the mask
        bs = 64
        width = min(self.pv_sk // bs, int(self.page_table_torch.shape[1]))
        pt = self.page_table_torch[:B, :width].to(torch.int32)
        self.v_pt = ttnn.from_torch(
            pt, device=self.mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._mapper
        )
        self.v_pt_u = [
            ttnn.from_torch(
                pt[b : b + 1],
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self._mapper,
            )
            for b in range(B)
        ]
        # fold mode: ONE virtual row = users' page rows concatenated at a FIXED
        # per-user stride of pv_sk//64 blocks, so virtual position b*pv_sk + pos
        # lands on user b's page for pos (pv_sk % 64 == 0). Stride can exceed a
        # user's physical width; the pad entries (page 0) sit under NEG-masked
        # columns only -- table width == mask width, the root-cause-#4 rule.
        self.v_pt_virt = None
        if fold:
            vw = self.pv_sk // bs
            pool_blocks = int(self.kv_layers[0][0].shape[0]) if self.kv_layers else B * vw
            if B * vw > pool_blocks:
                raise ValueError(
                    f"fold-verify virtual table needs B*(pv_sk/64) = {B * vw} blocks but the KV pool has "
                    f"{pool_blocks}; allocate more blocks (max_num_blocks) or lower max_new"
                )
            vrow = torch.zeros(1, B * vw, dtype=torch.int32)
            for b in range(B):
                w_b = min(vw, int(pt.shape[1]))
                vrow[0, b * vw : b * vw + w_b] = pt[b, :w_b]
            self.v_pt_virt = ttnn.from_torch(
                vrow,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self._mapper,
            )
        installed = getattr(t, "_active_page_tables_per_layer", None)
        self.pv_ptl = None
        if installed:
            self.pv_ptl = []
            per_type = {}
            for i, layer in enumerate(t.layers):
                lt = t.hf_config.layer_types[i]
                if lt not in per_type:
                    lpt = installed[i]
                    rows = lpt if lpt.dim() > 1 else lpt.unsqueeze(0)
                    mod = getattr(layer.self_attn.config, "cache_position_modulo", None)
                    w = (int(mod) // bs) if mod else width
                    w = min(w, int(rows.shape[1]))
                    per_type[lt] = ttnn.from_torch(
                        rows[:B, :w].to(torch.int32),
                        device=self.mesh_device,
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=self._mapper,
                    )
                self.pv_ptl.append(per_type[lt])
        # anchor tokens (persistent [1, B])
        self.noise_anchor_tok = ttnn.from_torch(
            torch.tensor([list(anchors)], dtype=torch.int64),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._mapper,
        )
        self.target.dflash_capture_taps(self.drafter.target_layer_ids, buffers=self.tap_bufs)
        self._upload_iter_inputs()
        a, b = self._body()  # compile
        ttnn.synchronize_device(self.mesh_device)
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._out = self._body()
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        self.trace = tid

    def seed_users(self, per_user):
        """One-time seed of the stacked ctx K/V from per-user projected rows.

        per_user: list of B (mirror [cap, H] torch bf16, win_first, ctx_len) --
        the same projected-tap rows the B=1 prefill_ingest builds. Each user's
        rows are projected/roped at their absolute positions and the B results
        are concatenated straight into the stacked caches (one assign each).
        """
        d = self.drafter
        ks = [[] for _ in d.layers]
        vs = [[] for _ in d.layers]
        tmp = ttnn.from_torch(
            torch.zeros(1, 1, self.cap, d.hidden, dtype=torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._mapper,
        )
        pos_dev = ttnn.from_torch(
            torch.zeros(1, self.cap, dtype=torch.int64),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._mapper,
        )
        for b, (mirror, wf, cl) in enumerate(per_user):
            self.win_first[b] = wf
            self.ctx_len[b] = cl
            self._upload(mirror.reshape(1, 1, self.cap, -1), tmp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            cp = torch.arange(wf, wf + self.cap, dtype=torch.int64).clamp(min=0).reshape(1, self.cap)
            self._upload(cp, pos_dev)
            cos = ttnn.reshape(
                ttnn.embedding(pos_dev, d._cos_2d, layout=ttnn.TILE_LAYOUT), (1, 1, self.cap, d.head_dim)
            )
            sin = ttnn.reshape(
                ttnn.embedding(pos_dev, d._sin_2d, layout=ttnn.TILE_LAYOUT), (1, 1, self.cap, d.head_dim)
            )
            kv = d.project_ctx_kv(tmp, cos, sin)
            cos.deallocate(True)
            sin.deallocate(True)
            for li, (k_new, v_new) in enumerate(kv):
                ks[li].append(k_new)
                vs[li].append(v_new)
        for li in range(len(d.layers)):
            k_all = ttnn.concat(ks[li], dim=2) if self.B > 1 else ks[li][0]
            v_all = ttnn.concat(vs[li], dim=2) if self.B > 1 else vs[li][0]
            ttnn.assign(k_all, self.ctx_k[li])
            ttnn.assign(v_all, self.ctx_v[li])
            for t in ks[li] + vs[li] + [k_all, v_all]:
                try:
                    t.deallocate(True)
                except Exception:
                    pass
        tmp.deallocate(True)
        pos_dev.deallocate(True)

    def _read_ids(self, t):
        src = ttnn.get_device_tensors(t)[0] if self._tp > 1 else t
        return ttnn.to_torch(src).reshape(-1).to(torch.int64).tolist()

    def step(self, first=False, frozen=()):
        """One batched iteration. Returns per-user (committed, produced).

        ``frozen``: user ids past their token budget -- their state stops
        advancing (they re-verify the same block each replay, which is
        deterministic and idempotent) so a straggler can't push a finished
        user's positions past the mask/table horizon."""
        B, K, P_v = self.B, self.K, self.P_v
        if not first:
            self._upload_iter_inputs()
            self._upload(torch.tensor([self.anchor], dtype=torch.int64), self.noise_anchor_tok)
            ttnn.execute_trace(self.mesh_device, self.trace, cq_id=0, blocking=False)
        draft_t, vidx_t = self._out
        drafts_all = self._read_ids(draft_t)
        post_all = self._read_ids(vidx_t)
        merge = torch.arange(B * self.cap, dtype=torch.int64)
        cpos = torch.zeros(1, B * P_v, dtype=torch.int64)
        results = []
        for b in range(B):
            if b in frozen:
                cpos[0, b * P_v : (b + 1) * P_v] = torch.arange(self.start[b], self.start[b] + P_v)
                results.append(([], 0))
                continue
            drafts = drafts_all[b * K : b * K + self.V]
            post = post_all[b * P_v : (b + 1) * P_v]
            acc = 0
            for dtok, ptok in zip(drafts, post[:-1]):
                if dtok == ptok:
                    acc += 1
                else:
                    break
            bonus = post[acc]
            produced = acc + 1
            committed = drafts[:acc] + [bonus]
            # per-user commit map (offset into the stacked cache/new rows)
            win_len = self.ctx_len[b] - self.win_first[b]
            base = b * self.cap
            if win_len + produced <= self.cap:
                merge[base + win_len : base + win_len + produced] = B * self.cap + b * P_v + torch.arange(produced)
            else:
                shift = win_len + produced - self.cap
                keep_n = self.cap - produced
                merge[base : base + keep_n] = base + shift + torch.arange(keep_n)
                merge[base + keep_n : base + self.cap] = B * self.cap + b * P_v + torch.arange(produced)
                self.win_first[b] += shift
            cpos[0, b * P_v : (b + 1) * P_v] = torch.arange(self.start[b], self.start[b] + P_v)
            self.ctx_len[b] = self.start[b] + produced
            self.anchor[b] = bonus
            self.start[b] += produced
            results.append((committed, produced))
        self._upload(merge.reshape(1, B * self.cap), self.merge_idx)
        self._upload(cpos, self.commit_pos)
        return results
