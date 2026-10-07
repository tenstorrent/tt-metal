# SPDX-License-Identifier: Apache-2.0
"""DFlash drafter (MiMo-V2.6-Flash, 5 SWA layers, block 8 = anchor + 7 mask tokens) on the 4x8 mesh, incremental context + traceable steps.

Spec: reference/DFLASH_NOTES.md, oracle reference/dflash_ref.py. Mapping: mesh columns (8) = tensor parallel (64 q heads -> 8 per column, 8 kv heads -> 1 per column,
MLP 16384 -> 2048 per column), mesh rows = independent user groups (U users per row, own context each).
State per layer: K/V ring [U,1,1024,128] per column (slot = position % 1024; K already normed + roped, V already x0.612), 20 KB per position over 5 layers and 8 columns.
* append(feats, positions, valid): hn = hidden_norm(fc(concat of the 5 target hidden states)) for R new rows per user (ragged: `valid` masks rows), per layer K,V of the rows,
  written into the ring with a one-hot scatter matmul  ring = ring*keep + S @ new  (S [U,1,1024,R] one-hot per valid row; no custom op, exact in bf16, trace friendly, ragged).
* draft(tokens, anchor_pos): x = [embed(anchor), mask_embedding x7], 5 layers of (norm, q/k/v, q/k norm, partial rotary 64/128 at absolute positions, attention over
  [ring (slots visible iff their stored position is in [anchor-1024, anchor-1]), 8 block keys bidirectional, sink] with matmul + additive mask (the sink is an extra all-zero key whose
  mask value is the per-head sink bias, V row zero), o_proj + all-reduce, SwiGLU), final norm, lm_head argmax over the 7 draft rows.
All per-round data (tokens, masks, cos/sin, scatter matrices) are persistent device tensors written with copy_host_to_device_tensor, so both steps can be captured in a trace.
"""
import torch

import ttnn
from models.demos.blackhole.mimo_v26.reference import dflash_ref as D

HIDDEN, NQ, NKV, HD, ROT, BLK, WIN, TP = 4096, 64, 8, 128, 64, 8, 1024, 8
NQC = NQ // TP
KPAD = 1056  # 1024 ring + 8 block + 1 sink key + 23 pad (tile multiple)
VSCALE = 0.612
EPS = 1e-6
NEG = -30000.0


def _ck(fp32=True, hifi4=False):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4 if hifi4 else ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
    )


def rot_matrix():
    R = torch.zeros(HD, HD)
    for j in range(ROT // 2):
        R[j + ROT // 2, j] = -1.0
        R[j, j + ROT // 2] = 1.0
    return R


def rope_cs(pos, theta=1e4):
    inv = theta ** (-torch.arange(0, ROT, 2).float() / ROT)
    f = pos.float()[..., None] * inv
    e = torch.cat([f, f], -1)
    cos = torch.cat([e.cos(), torch.ones(*e.shape[:-1], HD - ROT)], -1)
    sin = torch.cat([e.sin(), torch.zeros(*e.shape[:-1], HD - ROT)], -1)
    return cos, sin


class DFlashDrafter:
    def __init__(self, mesh, embed, lm_head, U, wdtype=ttnn.bfloat16, sd=None, R=BLK):
        """embed: target embedding table (replicated, row-major); lm_head [1,1,4096,V/8] per column; U users per mesh row; R = rows appended per call."""
        self.mesh, self.embed, self.lm_head, self.wdtype, self.U, self.R = mesh, embed, lm_head, wdtype, U, R
        self.rows = mesh.shape[0]
        sd = sd or D.load_dflash()
        self.sd = sd
        rep = ttnn.ReplicateTensorToMesh(mesh)
        col = lambda d: ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(None, d))
        self.rowmap = ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, None))
        self.rowcol = ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, 1))
        self.vocab_local = lm_head.shape[-1]

        def up(t, dtype, mapper, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t, device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
            )

        self.up = up
        bf = ttnn.bfloat16
        self.fc = up(sd["fc.weight"].T.contiguous().reshape(1, 1, 20480, HIDDEN), wdtype, col(2))
        nw = lambda w: up(w.reshape(1, 1, -1, 32), bf, rep, ttnn.ROW_MAJOR_LAYOUT)
        self.hidden_norm = nw(sd["hidden_norm.weight"])
        self.norm = nw(sd["norm.weight"])
        self.rot = up(rot_matrix().reshape(1, 1, HD, HD), bf, rep)
        self.mask_emb = up(
            sd["mask_embedding"].reshape(1, 1, 1, HIDDEN).expand(self.rows, U, 7, HIDDEN).contiguous(),
            bf,
            self.rowmap,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.layers = []
        for li in range(5):
            p = f"layers.{li}."
            W = lambda n: sd[p + n]
            wq = (
                W("self_attn.q_proj.weight")
                .T.reshape(HIDDEN, TP, NQC * HD)
                .permute(1, 0, 2)
                .reshape(1, TP, HIDDEN, NQC * HD)
                .contiguous()
            )
            wk = (
                W("self_attn.k_proj.weight")
                .T.reshape(HIDDEN, TP, HD)
                .permute(1, 0, 2)
                .reshape(1, TP, HIDDEN, HD)
                .contiguous()
            )
            wv = (
                (W("self_attn.v_proj.weight") * VSCALE)
                .T.reshape(HIDDEN, TP, HD)
                .permute(1, 0, 2)
                .reshape(1, TP, HIDDEN, HD)
                .contiguous()
            )
            wo = W("self_attn.o_proj.weight").T.reshape(1, TP, NQC * HD, HIDDEN).contiguous()
            sink = W("self_attn.attention_sink_bias").float().reshape(TP, NQC)
            sk = torch.zeros(1, TP, NQC * BLK, KPAD)
            for c in range(TP):
                for h in range(NQC):
                    sk[0, c, h * BLK : (h + 1) * BLK, WIN + BLK] = sink[c, h]
            self.layers.append(
                dict(
                    wq=up(wq, wdtype, col(1)),
                    wk=up(wk, wdtype, col(1)),
                    wv=up(wv, wdtype, col(1)),
                    wo=up(wo, wdtype, col(1)),
                    qn=nw(W("self_attn.q_norm.weight")),
                    kn=nw(W("self_attn.k_norm.weight")),
                    ln1=nw(W("input_layernorm.weight")),
                    ln2=nw(W("post_attention_layernorm.weight")),
                    wg=up(W("mlp.gate_proj.weight").T.contiguous().reshape(1, 1, HIDDEN, -1), wdtype, col(3)),
                    wu=up(W("mlp.up_proj.weight").T.contiguous().reshape(1, 1, HIDDEN, -1), wdtype, col(3)),
                    wd=up(W("mlp.down_proj.weight").T.contiguous().reshape(1, 1, -1, HIDDEN), wdtype, col(2)),
                    sinkc=up(sk.to(torch.bfloat16), bf, col(1)),  # [1,1,64,KPAD] per column
                )
            )
        z = lambda *s: torch.zeros(*s)
        # rings (persistent state) + host bookkeeping of the position stored in every slot (-1 = empty)
        self.ring_k = [up(z(U, 1, WIN, HD).to(bf), bf, rep) for _ in range(5)]
        self.ring_v = [up(z(U, 1, WIN, HD).to(bf), bf, rep) for _ in range(5)]
        self.ring_pos = torch.full((self.rows, U, WIN), -1, dtype=torch.long)
        # persistent per-call inputs
        self.in_feats = up(
            z(self.rows, 1, U * R, 20480).to(bf), bf, ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, 3))
        )
        self.in_acos = up(z(self.rows, 1, U * R, HD).to(bf), bf, self.rowmap)
        self.in_asin = up(z(self.rows, 1, U * R, HD).to(bf), bf, self.rowmap)
        self.in_S = up(z(self.rows, U, WIN, R).to(bf), bf, self.rowmap)
        self.in_keep = up(z(self.rows, U, WIN, HD).to(bf), bf, self.rowmap)
        self.in_tok = up(z(self.rows, U).to(torch.int32), ttnn.uint32, self.rowmap, ttnn.ROW_MAJOR_LAYOUT)
        self.in_mask = up(z(self.rows, U, NQC * BLK, KPAD).to(bf), bf, self.rowmap)
        self.in_cq = up(z(self.rows, U, NQC * BLK, HD).to(bf), bf, self.rowmap)
        self.in_sq = up(z(self.rows, U, NQC * BLK, HD).to(bf), bf, self.rowmap)
        self.in_ck = up(z(self.rows, U, BLK, HD).to(bf), bf, self.rowmap)
        self.in_sk = up(z(self.rows, U, BLK, HD).to(bf), bf, self.rowmap)
        self._tr = {}
        self._keep = []

    # ------------------------------------------------------------------ small helpers
    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=EPS, weight=w, compute_kernel_config=_ck(hifi4=True))

    def _ar(self, x):
        return ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _rope(self, x, cos, sin):
        xr = ttnn.matmul(x, self.rot, compute_kernel_config=_ck(hifi4=True))
        a, b = ttnn.multiply(x, cos), ttnn.multiply(xr, sin)
        ttnn.deallocate(xr)
        o = ttnn.add(a, b)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        return o

    def _rm_view(self, x, shape):
        return ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), shape)

    def _h2d(self, host_t, dev_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mapper=None):
        h = ttnn.from_torch(host_t, dtype=dtype, layout=layout, mesh_mapper=mapper or self.rowmap)
        ttnn.copy_host_to_device_tensor(h, dev_t)
        self._keep.append(h)

    # ------------------------------------------------------------------ append (context update)
    def set_append(self, feats, positions, valid):
        """feats [rows,U,R,20480] bf16; positions [rows,U,R] long; valid [rows,U,R] bool. Updates the host ring bookkeeping and the persistent inputs of append()."""
        rows, U, R, _ = feats.shape
        assert (U, R) == (self.U, self.R)
        self._keep = []
        S = torch.zeros(rows, U, WIN, R)
        for r in range(rows):
            for u in range(U):
                for i in range(R):
                    if valid[r, u, i]:
                        S[r, u, int(positions[r, u, i]) % WIN, i] = 1.0
        keep = (1.0 - S.sum(-1, keepdim=True)).expand(rows, U, WIN, HD).contiguous()
        for r in range(rows):
            for u in range(U):
                for i in range(R):
                    if valid[r, u, i]:
                        self.ring_pos[r, u, int(positions[r, u, i]) % WIN] = int(positions[r, u, i])
        cos, sin = rope_cs(positions)
        self._h2d(
            feats.reshape(rows, 1, U * R, 20480).bfloat16(),
            self.in_feats,
            mapper=ttnn.ShardTensor2dMesh(self.mesh, self.mesh.shape, dims=(0, 3)),
        )
        self._h2d(cos.reshape(rows, 1, U * R, HD).bfloat16(), self.in_acos)
        self._h2d(sin.reshape(rows, 1, U * R, HD).bfloat16(), self.in_asin)
        self._h2d(S.bfloat16(), self.in_S)
        self._h2d(keep.bfloat16(), self.in_keep)
        ttnn.synchronize_device(self.mesh)

    def _append_forward(self):
        U, R = self.U, self.R
        hp = ttnn.linear(self.in_feats, self.fc, dtype=ttnn.bfloat16, compute_kernel_config=_ck())
        h = self._ar(hp)
        ttnn.deallocate(hp)
        hn = self._rms(h, self.hidden_norm)
        ttnn.deallocate(h)
        S = ttnn.reshape(self.in_S, (U, 1, WIN, R))
        keep = ttnn.reshape(self.in_keep, (U, 1, WIN, HD))
        for li, L in enumerate(self.layers):
            k = self._rms(ttnn.linear(hn, L["wk"], dtype=ttnn.bfloat16, compute_kernel_config=_ck()), L["kn"])
            k = self._rope(k, self.in_acos, self.in_asin)
            v = ttnn.linear(hn, L["wv"], dtype=ttnn.bfloat16, compute_kernel_config=_ck())
            for new, ring in ((k, self.ring_k[li]), (v, self.ring_v[li])):
                nr = ttnn.to_layout(self._rm_view(new, (U, 1, R, HD)), ttnn.TILE_LAYOUT)
                add = ttnn.matmul(S, nr, compute_kernel_config=_ck(hifi4=True))  # [U,1,1024,128]
                kept = ttnn.multiply(ring, keep)
                upd = ttnn.add(kept, add)
                ttnn.assign(upd, ring)
                for t in (nr, add, kept, upd, new):
                    ttnn.deallocate(t)
        ttnn.deallocate(hn)

    def append(self):
        self._run("append", self._append_forward, use_trace=getattr(self, "use_trace", False))

    # ------------------------------------------------------------------ draft
    def set_round(self, tokens, anchor_pos):
        """tokens [rows,U] anchor token ids, anchor_pos [rows,U]. Mask is derived from the ring bookkeeping (slot visible iff stored position in [anchor-1024, anchor-1])."""
        rows, U = tokens.shape
        self._keep = []
        vis = (
            (self.ring_pos >= 0)
            & (self.ring_pos < anchor_pos[:, :, None])
            & (self.ring_pos >= anchor_pos[:, :, None] - WIN)
        )
        mask = torch.full((rows, U, NQC * BLK, KPAD), NEG)
        mask[..., :WIN] = torch.where(vis, 0.0, NEG)[:, :, None, :]
        mask[..., WIN : WIN + BLK] = 0
        pos = anchor_pos[:, :, None] + torch.arange(BLK)
        cos, sin = rope_cs(pos)
        cq = cos[:, :, None].expand(rows, U, NQC, BLK, HD).reshape(rows, U, NQC * BLK, HD)
        sq = sin[:, :, None].expand(rows, U, NQC, BLK, HD).reshape(rows, U, NQC * BLK, HD)
        self._h2d(tokens.to(torch.int32), self.in_tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self._h2d(mask.bfloat16(), self.in_mask)
        self._h2d(cq.bfloat16(), self.in_cq)
        self._h2d(sq.bfloat16(), self.in_sq)
        self._h2d(cos.bfloat16(), self.in_ck)
        self._h2d(sin.bfloat16(), self.in_sk)
        ttnn.synchronize_device(self.mesh)

    def _draft_forward(self):
        U = self.U
        e = ttnn.embedding(self.in_tok, self.embed, layout=ttnn.ROW_MAJOR_LAYOUT)
        e = ttnn.reshape(e, (U, 1, 1, HIDDEN))
        me = ttnn.reshape(self.mask_emb, (U, 1, 7, HIDDEN))
        x = ttnn.concat([e, me], dim=2)
        ttnn.deallocate(e)
        x = ttnn.to_layout(ttnn.reshape(x, (1, 1, U * BLK, HIDDEN)), ttnn.TILE_LAYOUT)
        mask = ttnn.reshape(self.in_mask, (U, 1, NQC * BLK, KPAD))
        cq, sq = ttnn.reshape(self.in_cq, (U, 1, NQC * BLK, HD)), ttnn.reshape(self.in_sq, (U, 1, NQC * BLK, HD))
        ck, sk = ttnn.reshape(self.in_ck, (U, 1, BLK, HD)), ttnn.reshape(self.in_sk, (U, 1, BLK, HD))
        zpad = ttnn.zeros([U, 1, KPAD - WIN - BLK, HD], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)
        for li, L in enumerate(self.layers):
            xn = self._rms(x, L["ln1"])
            q = ttnn.linear(xn, L["wq"], dtype=ttnn.bfloat16, compute_kernel_config=_ck())
            k = ttnn.linear(xn, L["wk"], dtype=ttnn.bfloat16, compute_kernel_config=_ck())
            v = ttnn.linear(xn, L["wv"], dtype=ttnn.bfloat16, compute_kernel_config=_ck())
            ttnn.deallocate(xn)
            qr = ttnn.permute(self._rm_view(q, (U, BLK, NQC, HD)), (0, 2, 1, 3))
            qr = ttnn.to_layout(ttnn.reshape(qr, (U, 1, NQC * BLK, HD)), ttnn.TILE_LAYOUT)
            ttnn.deallocate(q)
            qr = self._rope(self._rms(qr, L["qn"]), cq, sq)
            kb = self._rope(
                self._rms(ttnn.to_layout(self._rm_view(k, (U, 1, BLK, HD)), ttnn.TILE_LAYOUT), L["kn"]), ck, sk
            )
            vb = ttnn.to_layout(self._rm_view(v, (U, 1, BLK, HD)), ttnn.TILE_LAYOUT)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            K = ttnn.concat([self.ring_k[li], kb, zpad], dim=2)
            V = ttnn.concat([self.ring_v[li], vb, zpad], dim=2)
            ttnn.deallocate(kb)
            ttnn.deallocate(vb)
            s = ttnn.matmul(qr, K, transpose_b=True, compute_kernel_config=_ck(hifi4=True))
            s = ttnn.multiply(s, HD**-0.5)
            s = ttnn.add(s, mask)
            s = ttnn.add(s, L["sinkc"])
            p = ttnn.softmax(s, dim=-1, compute_kernel_config=_ck(hifi4=True))
            ttnn.deallocate(s)
            o = ttnn.matmul(p, V, compute_kernel_config=_ck(hifi4=True))
            for t in (p, K, V, qr):
                ttnn.deallocate(t)
            o = ttnn.permute(self._rm_view(o, (U, NQC, BLK, HD)), (0, 2, 1, 3))
            o = ttnn.to_layout(ttnn.reshape(o, (1, 1, U * BLK, NQC * HD)), ttnn.TILE_LAYOUT)
            part = ttnn.linear(o, L["wo"], dtype=ttnn.bfloat16, compute_kernel_config=_ck())
            ttnn.deallocate(o)
            att = self._ar(part)
            ttnn.deallocate(part)
            x1 = ttnn.add(x, att)
            ttnn.deallocate(x)
            ttnn.deallocate(att)
            xm = self._rms(x1, L["ln2"])
            g = ttnn.linear(xm, L["wg"], dtype=ttnn.bfloat16, compute_kernel_config=_ck())
            u = ttnn.linear(xm, L["wu"], dtype=ttnn.bfloat16, compute_kernel_config=_ck())
            ttnn.deallocate(xm)
            h = ttnn.multiply(
                g, u, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)], dtype=ttnn.bfloat16
            )
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            d = ttnn.linear(h, L["wd"], dtype=ttnn.bfloat16, compute_kernel_config=_ck())
            ttnn.deallocate(h)
            dd = self._ar(d)
            ttnn.deallocate(d)
            x = ttnn.add(x1, dd)
            ttnn.deallocate(x1)
            ttnn.deallocate(dd)
        ttnn.deallocate(zpad)
        y = self._rms(x, self.norm)
        ttnn.deallocate(x)
        lg = ttnn.linear(y, self.lm_head, dtype=ttnn.bfloat16, compute_kernel_config=_ck(hifi4=True))
        vals = ttnn.max(lg, dim=3, keepdim=True)
        idx = ttnn.argmax(ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT), dim=3, keepdim=True)
        ttnn.deallocate(lg)
        return y, vals, idx

    # ------------------------------------------------------------------ trace plumbing
    def _run(self, name, fn, use_trace=False):
        if not use_trace:
            return fn()
        if name not in self._tr:
            fn()  # eager warm-up / compile
            ttnn.synchronize_device(self.mesh)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            out = fn()
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            ttnn.synchronize_device(self.mesh)
            self._tr[name] = (tid, out)
        tid, out = self._tr[name]
        ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=True)
        return out

    def draft(self, use_trace=False):
        """Run the drafter on the inputs of the last set_round(); returns (draft tokens [rows,U,7] long, hidden y device [1,1,U*8,4096]) for rows 1..7 of each block."""
        y, vals, idx = self._run("draft", self._draft_forward, use_trace)
        comp = ttnn.ConcatMesh2dToTensor(self.mesh, dims=(0, 3), mesh_shape=tuple(self.mesh.shape))
        v = ttnn.to_torch(vals, mesh_composer=comp).float()[:, 0]  # [rows, U*8, cols]
        ix = ttnn.to_torch(idx, mesh_composer=comp)[:, 0].long()
        cols = v.shape[-1]
        glob = ix + torch.arange(cols).reshape(1, 1, -1) * self.vocab_local
        best = v.max(-1).values
        cand = torch.where(v == best.unsqueeze(-1), glob, torch.full_like(glob, 1 << 30))
        tok = cand.min(-1).values.reshape(self.rows, self.U, BLK)[:, :, 1:]
        if not use_trace:
            for t in (vals, idx):
                ttnn.deallocate(t)
        return tok, y

    def hidden_host(self, y):
        dts = ttnn.get_device_tensors(y)
        cols = self.mesh.shape[1]
        out = torch.stack([ttnn.to_torch(dts[r * cols]).float() for r in range(self.rows)])
        return out.reshape(self.rows, -1, BLK, HIDDEN)

    def release(self):
        for tid, _ in self._tr.values():
            ttnn.release_trace(self.mesh, tid)
        self._tr = {}

    # ------------------------------------------------------------------ convenience: bulk context load (prompt / test)
    def load_context(self, feats, pos0, use_trace=False):
        """feats [rows,U,N,20480] for positions pos0..pos0+N-1 ([rows,U] long): appended R rows at a time (only the last 1024 positions can matter)."""
        rows, U, N, _ = feats.shape
        self.use_trace = use_trace
        for s in range(0, N, self.R):
            n = min(self.R, N - s)
            f = torch.zeros(rows, U, self.R, 20480, dtype=torch.bfloat16)
            f[:, :, :n] = feats[:, :, s : s + n]
            pos = pos0[:, :, None] + s + torch.arange(self.R)[None, None]
            valid = torch.zeros(rows, U, self.R, dtype=torch.bool)
            valid[:, :, :n] = True
            self.set_append(f, pos, valid)
            self.append()
