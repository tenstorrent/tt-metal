# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-side bookkeeping of one speculative round (task mtp-device; DESIGN in $PROGRAM/spec-mtp-dev/RESULT.md s4): nothing on the host inside the loop.

Per mesh row (= one user) the round state is resident on device: ``pos`` (fp32, position P of the pending token x_P = committed KV length), ``tok_pending``, the k drafts, ``cap`` (max
drafts that may be accepted: token budget), ``act``. ``SpecDerive.derive(p)`` turns the position of row 0 into EVERY per-step input of an S-row pool decode (rope tables, write rows, per-row key counts,
col info, dense ids; same values as ``PoolCtx.set_step`` + host rope, proven by tests/mtp_spec_dev.py). ``accept`` computes m (leading run of d_{j+1} == a_j, capped), the pending token, the new position
P + m + 1 and the one-hot of row m (used to pick the MTP s0 output / hidden of row m) entirely with elementwise ops on [1,1,1,16] vectors.
"""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_device_loop import B2, CH, NC, SL, K, _rope_tables_two_level

W16 = 16


class SpecDerive:
    def __init__(self, md, ctx, rope, idx_rope_first, S):
        """ctx: PoolCtx(S=S) (its persistent tensors are overwritten by ``derive``); rope: RopeDev of the layer (dim, theta, interleave)."""
        self.md, self.ctx, self.S = md, ctx, S
        rep = lambda t, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT: upload(md, t, dt, lay)
        row = lambda v: rep(torch.tensor(v, dtype=torch.float32).reshape(1, 1, 1, W16))
        ar = torch.arange(W16, dtype=torch.float32)
        self.ar16 = row(ar.tolist())
        self.maskS = row((ar < S).float().tolist())
        self.oh0 = row((ar == 0).float().tolist())
        self.not_oh0 = row((ar != 0).float().tolist())
        self.oh1 = row((ar == 1).float().tolist())
        mc = ((ar >= 2) & (ar < 2 + S) & (S > 1)).float()
        self.maskC = row(mc.tolist())
        self.offC = row((mc * (ar - 2)).tolist())
        self.own_col = upload(
            md, torch.arange(NC, dtype=torch.float32).reshape(1, NC, 1, 1), ttnn.float32, ttnn.TILE_LAYOUT, shard_dim=1
        )
        self.arK = rep(torch.arange(K, dtype=torch.float32).reshape(1, 1, 1, K))
        self.arS = rep(torch.arange(S, dtype=torch.float32).reshape(S, 1, 1, 1))
        self.first = idx_rope_first
        t = _rope_tables_two_level(rope.dim, rope.theta, rope.interleave)
        ch_, sh_, cl_, sl_ = t
        split = lambda x: torch.cat([x.bfloat16().float(), (x - x.bfloat16().float()).bfloat16().float()], dim=-1)
        pack = lambda c, s_: torch.cat([split(c), split(s_)], dim=-1).reshape(B2, 4 * rope.dim)
        self.rt = [
            rep(pack(ch_, sh_), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            rep(pack(cl_, sl_), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ]
        self.D = rope.dim
        self.ones = rep(torch.ones(1, 1, S, 64), ttnn.bfloat16)
        self.zeros = rep(torch.zeros(1, 1, S, 64), ttnn.bfloat16)
        R = int(
            md.shape[0]
        )  # mesh rows = users: every per-user tensor is row-sharded, the derive ops run per device on its own row's position
        z = lambda *s: upload(md, torch.zeros(R, *s), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
        self.cos, self.sin = z(1, S, 64), z(
            1, S, 64
        )  # persistent outputs: the layer's rope tables (cos, sin) and the 128-wide indexer tables
        self.c128, self.s128 = z(1, S, 128), z(1, S, 128)
        from models.demos.blackhole.hy4_preview.tt import hy4_derive_fused as _b40

        self._b40 = _b40 if _b40.enabled() else None
        if self._b40 is not None:  # B40: persistent rope hi / lo index rows (uint32 [1, S] per mesh row)
            zi = lambda: upload(md, torch.zeros(R, S, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, row_dim=0)
            self.ih_b40, self.il_b40 = zi(), zi()

    def tabs(self):
        return (self.cos, self.sin, self.cos, self.sin, self.c128, self.s128)

    @staticmethod
    def _i(x, dt):
        return ttnn.to_layout(ttnn.typecast(x, dt), ttnn.ROW_MAJOR_LAYOUT)

    def derive(self, p):
        """p fp32 [1,1,1,1] (position of row 0 of the S verify rows) -> overwrite the ctx tensors + rope tabs (all in place, traceable)."""
        ctx, S, cp = self.ctx, self.S, ttnn.copy
        pv = None if self._b40 is not None else ttnn.add(p, self.ar16)  # [1,1,1,16]: position of row j in element j
        # --- rope (two-level tables)
        if self._b40 is not None:
            self._b40.derive_geom(
                p, self.own_col, ctx.wr, ctx.kvcnt, ctx.colinfo, self.ih_b40, self.il_b40, ctx.upd_rows, S, CH, SL, B2
            )
            ih, il = self.ih_b40, self.il_b40
        else:
            pr = ttnn.slice(pv, [0, 0, 0, 0], [1, 1, 1, S])
            hi = ttnn.floor(ttnn.multiply(pr, 1.0 / B2))
            lo = ttnn.subtract(pr, ttnn.multiply(hi, float(B2)))
            ih = ttnn.reshape(self._i(hi, ttnn.uint32), [1, S])
            il = ttnn.reshape(self._i(lo, ttnn.uint32), [1, S])
        D = self.D

        def g(tbl, idx):
            e = ttnn.reshape(ttnn.embedding(idx, tbl, layout=ttnn.TILE_LAYOUT), [1, 1, S, 4 * D])
            f = ttnn.typecast(e, ttnn.float32)
            sl = lambda i: ttnn.slice(f, [0, 0, 0, i * D], [1, 1, S, (i + 1) * D])
            return ttnn.add(sl(0), sl(1)), ttnn.add(sl(2), sl(3))

        (ch_, sh_), (cl_, sl_) = g(self.rt[0], ih), g(self.rt[1], il)
        cos = ttnn.subtract(ttnn.multiply(ch_, cl_), ttnn.multiply(sh_, sl_))
        sin = ttnn.add(ttnn.multiply(sh_, cl_), ttnn.multiply(ch_, sl_))
        cosb, sinb = ttnn.typecast(cos, ttnn.bfloat16), ttnn.typecast(sin, ttnn.bfloat16)
        cp(cosb, self.cos)
        cp(sinb, self.sin)
        cp(
            ttnn.concat([cosb, self.ones] if self.first else [self.ones, cosb], dim=3), self.c128
        )  # product9 fix: order as hy4_indexer (was flipped)
        cp(ttnn.concat([sinb, self.zeros] if self.first else [self.zeros, sinb], dim=3), self.s128)
        if self._b40 is None:
            # --- KV geometry of the S appended tokens (vector over j)
            h5 = ttnn.floor(ttnn.multiply(pv, 1.0 / CH))
            l5 = ttnn.subtract(pv, ttnn.multiply(h5, float(CH)))
            col = ttnn.floor(ttnn.multiply(l5, 1.0 / SL))
            loc = ttnn.add(ttnn.multiply(h5, float(SL)), ttnn.subtract(l5, ttnn.multiply(col, float(SL))))
            mine = ttnn.eq(col, self.own_col)
            e = ttnn.subtract(
                ttnn.multiply(mine, ttnn.add(loc, 1.0)), 1.0
            )  # local row on the owner column, -1 elsewhere
            wr = ttnn.subtract(ttnn.multiply(ttnn.add(e, 1.0), self.maskS), 1.0)  # -1 for j >= S
            cp(self._i(wr, ttnn.int32), ctx.wr)
            for j in range(S):
                cp(
                    ttnn.reshape(self._i(ttnn.slice(e, [0, 0, 0, j], [1, 1, 1, j + 1]), ttnn.int32), [1, 1]),
                    ctx.upd_rows[j],
                )
            n = ttnn.add(pv, 1.0)
            fu = ttnn.floor(ttnn.multiply(n, 1.0 / CH))
            rem = ttnn.subtract(n, ttnn.multiply(fu, float(CH)))
            kv = ttnn.add(
                ttnn.multiply(fu, float(SL)),
                ttnn.minimum(ttnn.maximum(ttnn.subtract(rem, ttnn.multiply(self.own_col, float(SL))), 0.0), float(SL)),
            )
            cp(self._i(ttnn.multiply(kv, self.maskS), ttnn.uint32), ctx.kvcnt)
            n0 = ttnn.add(p, 1.0)
            ci = ttnn.add(ttnn.multiply(self.own_col, self.oh0), ttnn.multiply(n0, self.oh1))
            ci = ttnn.add(ci, ttnn.add(ttnn.multiply(n0, self.maskC), self.offC))
            cp(self._i(ci, ttnn.uint32), ctx.colinfo)
        # --- dense ids [S,1,1,K]: row j sees tokens 0..p+j
        if self._b40 is not None and ctx.mode != "dense":
            return  # B40: the dense ids are only read in mode "dense"
        pj = ttnn.add(p, self.arS)  # [S,1,1,1]
        valid = ttnn.le(self.arK, pj)
        ids = ttnn.subtract(ttnn.multiply(valid, ttnn.add(self.arK, 1.0)), 1.0)
        cp(ttnn.to_layout(ttnn.typecast(ttnn.typecast(ids, ttnn.int32), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), ctx.ids_g)


def accept(a_row, d_row, cap, k, ar16, oh0, maskS):
    """greedy accept of one user. a_row fp32 [1,1,1,16]: target tokens a_0..a_k (element j; rest 0), d_row fp32 [1,1,1,16]: drafts, d_{j+1} at element j for j < k and -1 at j >= k (never matches);
    cap fp32 [1,1,1,1] (max accepted drafts). Returns (m [1,1,1,1], onehot_m [1,1,1,16], next_tok [1,1,1,1]) fp32.
    m = min(cap, k, leading run of eq): the run is the cumulative product of the match vector (k <= 3 so a short unrolled prefix product).
    """
    eq = ttnn.eq(a_row, d_row)  # 1.0 where a_j == d_{j+1}
    run = None
    runs = []
    for j in range(k):
        e = ttnn.slice(eq, [0, 0, 0, j], [1, 1, 1, j + 1])
        run = e if run is None else ttnn.multiply(run, e)
        runs.append(run)
    m = runs[0]
    for r in runs[1:]:
        m = ttnn.add(m, r)
    m = ttnn.minimum(m, cap)
    onehot = ttnn.eq(ar16, m)  # [1,1,1,16]
    nxt = ttnn.sum(ttnn.multiply(onehot, a_row), dim=-1, keepdim=True)
    return m, onehot, nxt


# ---------------------------------------------------------------------------------------------------- the draft side of a round (in one trace with the verify)
class MTPRoundTail:
    """Everything after the main-model verify rows of ONE round, as traceable device ops (design s4 of spec-mtp-dev/RESULT.md):
    accept -> MTP step 0 over the S = k+1 verify rows (tokens = the verify output tokens a_0..a_k, hidden = the verify hidden rows) -> one-hot gather of row m -> k-1 recursive MTP steps whose
    token / hidden / position feedback never leaves the device -> next-round inputs (verify tokens [nxt, d_1..d_k], position P + m + 1, drafts).
    All MTP passes run at S rows per user (the recursive steps carry 1 real row + S-1 filler rows after it; filler rows write KV beyond the committed length, which is never read and overwritten next round):
    ONE XMerge / PoolCtx S per process. Persistent tensors are allocated here (before any trace capture); ``run`` is the traced body.
    """

    def __init__(self, ch, mtp, derive, ctx, k, greedy, sampling=None):
        self.sampling = sampling  # SpecSampling (lossless draft-q rejection sampling) or None (greedy)
        self.ch, self.mtp, self.sd, self.ctx, self.k, self.S = ch, mtp, derive, ctx, k, k + 1
        self.greedy = greedy  # fn(logits [1,1,T,V/8]) -> uint32 [T,1] ids (hy4_mtp.greedy_tokens)
        # HY4_DRAFT_HEAD=bfp8 (verify-opt track, DRAFT-ONLY numerics: the verify rows decide every token): the draft lm_head matmuls (k per round, DRAM-bound at 455 us each with the bf16 weights)
        # read a bfp8 copy of the head (half the bytes). Built here (outside any trace); the main-model head stays bf16.
        self.lm_draft, self.ckc_draft = ch.lm_head, ch.ckc_head
        if os.environ.get("HY4_DRAFT_HEAD", "bf16") == "bfp8":
            from models.demos.blackhole.hy4_preview.tt.hy4_common import compute_cfg

            self.lm_draft = ttnn.typecast(ch.lm_head, ttnn.bfloat8_b)
            self.ckc_draft = compute_cfg(ch.md, os.environ.get("HY4_DRAFT_HEAD_FID", "hifi2"), True)
        md, S = ch.md, self.S
        R = int(md.shape[0])
        rep = lambda t, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT: upload(md, t, dt, lay)
        rows = lambda t, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT: upload(
            md, t, dt, lay, row_dim=0
        )  # one slice per user (mesh row)
        self.ar16 = rep(torch.arange(W16, dtype=torch.float32).reshape(1, 1, 1, W16))
        self.arS_col = rep(torch.arange(S, dtype=torch.float32).reshape(1, 1, S, 1))
        self.d_row = rows(torch.full((R, 1, 1, W16), -1.0))  # drafts of this round (d_{j+1} at j < k, -1 beyond)
        self.pos = rows(torch.zeros(R, 1, 1, 1))  # P: position of the pending token (resident)
        self.cap = rows(torch.full((R, 1, 1, 1), float(k)))
        self.tok = rows(
            torch.zeros(R, S).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )  # verify tokens [x_P, d_1..d_k] (input of the next verify)
        self.park = rows(
            torch.zeros(R, 1, 1, 1)
        )  # 1.0 = idle (parked) row: its next-round state is forced to (PAD token, position 0, no drafts) ON DEVICE, exactly what a host reload would upload
        self.padvec = rep(torch.tensor([120002.0] + [0.0] * (S - 1)).reshape(1, 1, 1, S))
        # HY4_PFX=1 (prefix cache): a parked row keeps (PAD, parkvec[row]) instead of position 0. The engine sets parkvec = the length of the row's retained prefix, so the idle rounds write their junk KV rows just past
        # the retained blocks (never over page 0 of the prefix) and only attend to initialised keys.
        self.parkvec = rows(torch.zeros(R, 1, 1, 1)) if os.environ.get("HY4_PFX", "0") == "1" else None
        self.m_out = rows(torch.zeros(R, 1, 1, 1))
        self.a_out = rows(torch.zeros(R, 1, 1, W16))  # emitted tokens a_0..a_k of the round (host reads a_out + m_out)
        self.zeros_pad = rep(torch.zeros(1, 1, 1, W16 - S))
        from models.demos.blackhole.hy4_preview.tt import hy4_derive_fused as _b40

        self._b40 = _b40 if _b40.tail_enabled() else None
        if self._b40 is not None:  # B40: persistent outputs of the one-core accept program (per mesh row)
            self.p_new_t, self.nxt_t = rows(torch.zeros(R, 1, 1, 1)), rows(torch.zeros(R, 1, 1, 1))
            self.oh_col_t = rows(torch.zeros(R, 1, S, 1), ttnn.bfloat16)
        self.neg_pad = rep(torch.full((1, 1, 1, W16 - k), -1.0))

    def _u32_row(self, f):
        """f32 [1,1,1,n] TILE -> uint32 [1, n] ROW_MAJOR."""
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(f, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, int(f.shape[3])])

    def _dense8(self):
        return self.ctx.dense_ids8() if self.ctx.mode == "dense" else None

    def _step(self, tok_u32_1S, h, dense8):
        return self.mtp.forward(tok_u32_1S, h, self.sd.tabs(), dense8)

    def run(self, a_tok, h_rows, derived=False):
        """a_tok: uint32 [S, 1] verify output tokens (device greedy / sampler), h_rows bf16 [1,1,S,D] verify hidden rows (post model.norm). Updates self.pos / d_row / tok / a_out / m_out in place."""
        S, k, mtp, ch = self.S, self.k, self.mtp, self.ch
        if self._b40 is not None:
            self._b40.spec_accept(
                a_tok,
                self.d_row,
                self.cap,
                self.pos,
                self.a_out,
                self.m_out,
                self.oh_col_t,
                self.p_new_t,
                self.nxt_t,
                S,
                k,
            )
        else:
            a_f = ttnn.typecast(ttnn.to_layout(ttnn.reshape(a_tok, [1, 1, 1, S]), ttnn.TILE_LAYOUT), ttnn.float32)
            a_row = ttnn.concat([a_f, self.zeros_pad], dim=3) if W16 > S else a_f
            m, onehot, nxt = accept(a_row, self.d_row, self.cap, k, self.ar16, None, None)
            ttnn.copy(a_row, self.a_out)
            ttnn.copy(m, self.m_out)
        # MTP step 0 on all S rows: pair (h_j, a_j) at position P + j  (no dependence on the accept result)
        if not derived:  # the verify of the same trace already derived the position-P inputs
            self.sd.derive(self.pos)
        out0 = self._step(ttnn.reshape(a_tok, [1, S]), h_rows, self._dense8())
        oh_col = (
            self.oh_col_t if self._b40 is not None else ttnn.typecast(ttnn.eq(self.arS_col, m), ttnn.bfloat16)
        )  # [1,1,S,1]
        hm = ttnn.sum(ttnn.multiply(out0, oh_col), dim=2, keepdim=True)  # [1,1,1,D] the MTP hidden of row m
        ttnn.deallocate(out0)
        p_new = self.p_new_t if self._b40 is not None else ttnn.add(ttnn.add(self.pos, m), 1.0)  # L' = P + m + 1
        drafts = []
        h_cur = hm
        for i in range(k):
            lg = ttnn.linear(h_cur, self.lm_draft, compute_kernel_config=self.ckc_draft, dtype=ttnn.float32)
            if self.sampling is None:
                t = self.greedy(lg)  # [1,1] u32
                drafts.append(
                    ttnn.typecast(ttnn.to_layout(ttnn.reshape(t, [1, 1, 1, 1]), ttnn.TILE_LAYOUT), ttnn.float32)
                )
            else:
                tok_f = self.sampling.draft(
                    lg, i, ttnn.add(p_new, float(i + 1))
                )  # the drafted token sits at position L' + i + 1
                t = ttnn.reshape(ttnn.to_layout(ttnn.typecast(tok_f, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, 1])
                drafts.append(tok_f)
            ttnn.deallocate(lg)
            if i == k - 1:
                break
            # recursive step i+1: row 0 = (token d_{i+1}, hidden h_cur) at position L' + i; filler rows repeat it
            self.sd.derive(ttnn.add(p_new, float(i)))
            tok_rows = ttnn.concat([ttnn.reshape(t, [1, 1])] * S, dim=1)
            h_rows_ = ttnn.concat([h_cur] * S, dim=2)
            o = self._step(tok_rows, h_rows_, self._dense8())
            ttnn.deallocate(h_rows_)
            h_cur = ttnn.slice(o, [0, 0, 0, 0], [1, 1, 1, int(o.shape[3])])
            ttnn.deallocate(o)
        if self.sampling is not None:
            self.sampling.finish()
        if self._b40 is not None:
            self._b40.spec_state(
                drafts, self.nxt_t, p_new, self.park, self.parkvec, self.d_row, self.tok, self.pos, S, k
            )
            return
        # next-round state (a parked row keeps (PAD, 0, no drafts): the same values ``load`` uploads for an idle row)
        keep = ttnn.subtract(ttnn.multiply(self.park, -1.0), -1.0)  # 1 - park
        dcat = ttnn.concat(drafts, dim=3)  # [1,1,1,k]
        d_pad = ttnn.concat([dcat, self.neg_pad], dim=3)
        d_eff = ttnn.subtract(ttnn.multiply(d_pad, keep), self.park)  # parked: -1 everywhere
        ttnn.copy(d_eff, self.d_row)
        tok_f = ttnn.concat([nxt, dcat], dim=3)  # [1,1,1,S]: [x_{P'}, d_1..d_k]
        tok_eff = ttnn.add(ttnn.multiply(tok_f, keep), ttnn.multiply(self.padvec, self.park))
        ttnn.copy(self._u32_row(tok_eff), self.tok)
        p_eff = ttnn.multiply(p_new, keep)
        if self.parkvec is not None:
            p_eff = ttnn.add(p_eff, ttnn.multiply(self.parkvec, self.park))
        ttnn.copy(p_eff, self.pos)


class Hy4SpecRunner:
    """ONE traced speculative round per k (D-design s4): derive(P) -> verify S = k+1 rows through the 78 main layers -> greedy a_j -> accept -> MTP s0 over the S rows -> row m -> k-1 recursive MTP steps ->
    next-round inputs, all resident on device. The host only reloads on a transition (``load``) and reads (a_0..a_k, m, P) per user per round (``read``, async).
    Build order: chain + head + MTP layer, Hy4PoolVerifier (pools, XMerge), ``Hy4SpecRunner`` (MTP pool + persistent tensors), then ``capture`` (eager warm round first).
    """

    def __init__(self, dec, mtp, k, greedy, pages, key_dtype=ttnn.bfloat8_b, sampling=None):
        self.dec, self.mtp, self.k, self.S = dec, mtp, k, k + 1
        ch, md, ctx = dec.ch, dec.md, dec.ctx
        ly = mtp.ly
        self.ly = ly
        rope = ch.layers[0].attn.rope
        self.sd = SpecDerive(md, ctx, rope, ch.layers[0].attn.indexer.idx_rope_first, self.S)
        dec.attach_derive(self.sd)
        dec.prepare_blocks()
        ly.blk = (
            ly.blk.for_batch(self.S) if getattr(ly, "blk", None) is not None else None
        )  # the MTP MoE at batch_per_device = S too
        if getattr(ly, "blk_tp", None) is not None and ly.blk is not None:
            # B42 (HY4_B42_MTP_TP24=1): the MTP MoE on the tp24 kernel (TP32) instead of the EP ring block: block at T = S tokens per mesh row, router of the batch-S block
            from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32_block import Hy4MoeTp32Block

            assert self.S * int(md.shape[0]) <= 16
            ly.blk_tp = Hy4MoeTp32Block(md, ch.ccl, ly.blk.gate, ly.blk_tp.tpe, self.S)
        self.mtp_kv = ttnn.empty(
            [1, 1, pages * 64, 656],
            dtype=ttnn.fp8_e4m3,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.mtp_keys = ttnn.empty(
            [pages, 1, 64, 128],
            dtype=key_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ly.attn.init_pool(ctx, self.mtp_kv, self.mtp_keys)
        self.sampling = sampling
        self.samp_obj = (
            sampling  # kept for the variant switch (product serving: a greedy AND a sampled trace share all state)
        )
        self.default_variant = "S" if sampling is not None else "G"
        self.traces = {}
        self.tail = MTPRoundTail(ch, mtp, self.sd, ctx, k, greedy, sampling=sampling)
        dec.tok = self.tail.tok  # the verify input IS the tail's next-round token tensor
        self.greedy = greedy
        self.trace = None

    def _set_variant(self, v):
        """'G' = greedy round, 'S' = lossless sampled round (needs ``samp_obj``): the persistent state (pos / tok / drafts / cap / a_out) is shared by both traces."""
        if v is None:
            v = self.default_variant
        assert (
            v == "G" or self.samp_obj is not None
        ), "sampled round requested but the runner was built without SpecSampling"
        smp = self.samp_obj if v == "S" else None
        self.sampling = smp
        self.tail.sampling = smp
        return v

    def body(self, variant=None):
        dec, tail = self.dec, self.tail
        self._set_variant(variant)
        self.sd.derive(tail.pos)
        hn, logits = dec.step_verify()
        a_tok = (
            self.greedy(logits) if self.sampling is None else self.sampling.verify(logits, tail.pos)
        )  # [S, 1] uint32 per device row
        ttnn.deallocate(logits)
        tail.run(a_tok, hn, derived=True)
        ttnn.deallocate(hn)
        ttnn.deallocate(a_tok)

    def capture(self, variant=None):
        """eager warm round (compiles every program; it is a REAL round: its result is returned, the state has advanced), then the trace capture (records, executes nothing). One trace per variant ('G' / 'S')."""
        v = self._set_variant(variant)
        self.body(v)
        ttnn.synchronize_device(self.dec.md)
        warm = self.read()
        tr = ttnn.begin_trace_capture(self.dec.md, cq_id=0)
        self.body(v)
        ttnn.end_trace_capture(self.dec.md, tr, cq_id=0)
        ttnn.synchronize_device(self.dec.md)
        self.traces[v] = tr
        if self.trace is None or v == self.default_variant:
            self.trace = tr
        return warm

    def load(self, pending, pos, drafts, cap=None):
        """transition (admission / reload): pending token x_P, P and the k drafts of every user row (host -> persistent device tensors)."""
        md, k, S = self.dec.md, self.k, self.S
        R = int(md.shape[0])
        mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
        cp = lambda t, dt, lay, tgt: ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=mp), tgt
        )
        tok = torch.zeros(R, S, dtype=torch.int32)
        tok[:, 0] = torch.tensor(pending, dtype=torch.int32)
        for r in range(R):
            tok[r, 1:] = torch.tensor(
                [max(int(x), 0) for x in drafts[r][:k]], dtype=torch.int32
            )  # -1 = no draft: a valid token id for the embedding, the draft row d_row keeps -1 (never accepted)
        cp(tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.tail.tok)
        cp(torch.tensor([float(p) for p in pos]).reshape(R, 1, 1, 1), ttnn.float32, ttnn.TILE_LAYOUT, self.tail.pos)
        d = torch.full((R, 1, 1, W16), -1.0)
        for r in range(R):
            d[r, 0, 0, :k] = torch.tensor([float(x) for x in drafts[r][:k]])
        cp(d, ttnn.float32, ttnn.TILE_LAYOUT, self.tail.d_row)
        c = (
            torch.full((R, 1, 1, 1), float(k))
            if cap is None
            else torch.tensor([float(x) for x in cap]).reshape(R, 1, 1, 1)
        )
        cp(c, ttnn.float32, ttnn.TILE_LAYOUT, self.tail.cap)

    def submit(self, variant=None):
        tr = self.trace if variant is None else self.traces[variant]
        ttnn.execute_trace(self.dec.md, tr, cq_id=0, blocking=False)

    def set_park(self, park, park_pos=None):
        """per user flag (1 = idle row): the device parks it every round (see MTPRoundTail.park). park_pos (prefix cache): the position a parked row keeps per user."""
        md = self.dec.md
        R = int(md.shape[0])
        mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
        if park_pos is not None and self.tail.parkvec is not None:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor([float(x) for x in park_pos]).reshape(R, 1, 1, 1),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=mp,
                ),
                self.tail.parkvec,
            )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([float(x) for x in park]).reshape(R, 1, 1, 1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mp,
            ),
            self.tail.park,
        )

    def set_cap(self, cap):
        """per user max accepted drafts of the next round (host -> persistent device tensor, 4 floats)."""
        md = self.dec.md
        R = int(md.shape[0])
        mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([float(x) for x in cap]).reshape(R, 1, 1, 1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mp,
            ),
            self.tail.cap,
        )

    def read_enqueue(self):
        """enqueue the 5 small round outputs behind the last submitted trace (in CQ order, before any later trace overwrites them) and record an event: returns a handle for ``read_finish``."""
        t = self.tail
        hs = [ttnn.from_device(x, blocking=False) for x in (t.a_out, t.m_out, t.pos, t.d_row, t.tok)]
        return hs, ttnn.record_event(self.dec.md, 0)

    def read_finish(self, handle):
        """wait for the event of a ``read_enqueue`` (NOT for traces enqueued after it) and parse."""
        md, S, k = self.dec.md, self.S, self.k
        hs, ev = handle
        ttnn.event_synchronize(ev)
        R, c = int(md.shape[0]), int(md.shape[1])
        g = [ttnn.get_device_tensors(h) for h in hs]
        out = []
        for r in range(R):
            val = lambda i: ttnn.to_torch(g[i][r * c]).reshape(-1)
            a, m, p, d, tk = val(0), val(1), val(2), val(3), val(4)
            mv = int(round(float(m[0])))
            out.append(
                dict(
                    a=[int(round(float(x))) for x in a[:S].tolist()],
                    m=mv,
                    pos=int(round(float(p[0]))),
                    drafts=[int(round(float(x))) for x in d[:k].tolist()],
                    tok=[int(x) for x in tk[:S].tolist()],
                )
            )
        return out

    def read_state(self):
        """-> per user row dict(a, m, pos, tok, drafts) of the round just run (blocking: waits for the event of the read, i.e. for everything enqueued before it)."""
        return self.read_finish(self.read_enqueue())

    def read(self):
        """-> per user row (emitted tokens a_0..a_m, m, new P): blocking read of the small per-round outputs."""
        md, S = self.dec.md, self.S
        R, c = int(md.shape[0]), int(md.shape[1])
        ttnn.synchronize_device(md)
        g = lambda t: ttnn.get_device_tensors(t)
        a, m, p = g(self.tail.a_out), g(self.tail.m_out), g(self.tail.pos)
        out = []
        for r in range(R):
            av = ttnn.to_torch(a[r * c]).reshape(-1)[:S].tolist()
            mv = int(round(float(ttnn.to_torch(m[r * c]).reshape(-1)[0])))
            pv = int(round(float(ttnn.to_torch(p[r * c]).reshape(-1)[0])))
            out.append(([int(x) for x in av[: mv + 1]], mv, pv))
        return out
