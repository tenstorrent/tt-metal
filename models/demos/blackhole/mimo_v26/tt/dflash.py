# SPDX-License-Identifier: Apache-2.0
"""DFlash drafter (MiMo-V2.6-Flash, 5 SWA layers, block 8 = anchor + 7 mask tokens) on the 4x8 mesh.

Spec: reference/DFLASH_NOTES.md, oracle reference/dflash_ref.py (draft_blocks). Mapping: mesh columns (8) = tensor parallel (64 q heads -> 8 per column, 8 kv heads -> 1 per column,
MLP 16384 -> 2048 per column), mesh rows = independent user groups (U users per row, each with its own context). One round for U users per row:
  ctx:   hn = hidden_norm(fc(concat target hidden states of layers [0,11,23,35,47]))      (fc is K-sharded over columns + all-reduce)
         per layer K = rope_partial(k_norm(hn @ Wk)), V = 0.612 * hn @ Wv  -> per-user context cache [U,1,1024,128] (left aligned, valid length per user; last 1024 positions)
  block: x = [embed(anchor token), mask_embedding x7]; per layer q/k/v of the 8 block rows (q_norm, k_norm, partial rotary at absolute positions), attention over
         [ctx cache (<=1024 visible keys), 8 block keys bidirectional, sink] done with plain matmul + additive mask (head sink = an extra all-zero key whose mask value is
         the sink bias, its V row is zero), o_proj (row-sharded, all-reduce), SwiGLU MLP (TP + all-reduce); final norm; target lm_head on rows 1..7.
Round-local state only (the 8 block keys are never cached). The context cache here is rebuilt from hn (gate/probe version; incremental append/ring is the production step).
"""
import torch

import ttnn
from models.demos.blackhole.mimo_v26.reference import dflash_ref as D

HIDDEN, NQ, NKV, HD, ROT, BLK, WIN, TP = 4096, 64, 8, 128, 64, 8, 1024, 8
NQC = NQ // TP  # 8 q heads per column
KPAD = 1056  # 1024 ctx + 8 block + 1 sink key + 23 pad (tile multiple)
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
    """pos [...] float -> cos, sin [..., 128] (partial: dims >= 64 pass through)."""
    inv = theta ** (-torch.arange(0, ROT, 2).float() / ROT)
    f = pos.float()[..., None] * inv
    e = torch.cat([f, f], -1)
    cos = torch.cat([e.cos(), torch.ones(*e.shape[:-1], HD - ROT)], -1)
    sin = torch.cat([e.sin(), torch.zeros(*e.shape[:-1], HD - ROT)], -1)
    return cos, sin


class DFlashDrafter:
    def __init__(self, mesh, embed, lm_head, wdtype=ttnn.bfloat8_b, sd=None):
        """embed: target embedding device table (replicated, row-major); lm_head: target lm_head [1,1,4096,V/8] per column."""
        self.mesh, self.embed, self.lm_head, self.wdtype = mesh, embed, lm_head, wdtype
        self.rows = mesh.shape[0]
        sd = sd or D.load_dflash()
        self.sd = sd
        rep = ttnn.ReplicateTensorToMesh(mesh)
        col = lambda d: ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(None, d))
        self.rowmap = ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, None))

        def up(t, dtype, mapper, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t, device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
            )

        self.up = up
        bf = ttnn.bfloat16
        self.fc = up(sd["fc.weight"].T.contiguous().reshape(1, 1, 20480, HIDDEN), wdtype, col(2))  # K-sharded
        nw = lambda w: up(w.reshape(1, 1, -1, 32), bf, rep, ttnn.ROW_MAJOR_LAYOUT)
        self.hidden_norm = nw(sd["hidden_norm.weight"])
        self.norm = nw(sd["norm.weight"])
        self.rot = up(rot_matrix().reshape(1, 1, HD, HD), bf, rep)
        self.mask_emb = sd["mask_embedding"].reshape(1, 1, 1, HIDDEN)
        self.layers = []
        for li in range(5):
            p = f"layers.{li}."
            W = lambda n: sd[p + n]
            wq = W("self_attn.q_proj.weight").T.reshape(HIDDEN, TP, NQC * HD)  # columns own heads [8c, 8c+8)
            wk = W("self_attn.k_proj.weight").T.reshape(HIDDEN, TP, HD)  # kv head c
            wv = (W("self_attn.v_proj.weight") * VSCALE).T.reshape(HIDDEN, TP, HD)
            wo = W("self_attn.o_proj.weight").T.reshape(TP, NQC * HD, HIDDEN)
            L = dict(
                wq=up(wq.permute(1, 0, 2).reshape(1, TP, HIDDEN, NQC * HD).contiguous(), wdtype, col(1)),
                wk=up(wk.permute(1, 0, 2).reshape(1, TP, HIDDEN, HD).contiguous(), wdtype, col(1)),
                wv=up(wv.permute(1, 0, 2).reshape(1, TP, HIDDEN, HD).contiguous(), wdtype, col(1)),
                wo=up(wo.reshape(1, TP, NQC * HD, HIDDEN).contiguous(), wdtype, col(1)),
                qn=nw(W("self_attn.q_norm.weight")),
                kn=nw(W("self_attn.k_norm.weight")),
                ln1=nw(W("input_layernorm.weight")),
                ln2=nw(W("post_attention_layernorm.weight")),
                wg=up(W("mlp.gate_proj.weight").T.contiguous().reshape(1, 1, HIDDEN, -1), wdtype, col(3)),
                wu=up(W("mlp.up_proj.weight").T.contiguous().reshape(1, 1, HIDDEN, -1), wdtype, col(3)),
                wd=up(W("mlp.down_proj.weight").T.contiguous().reshape(1, 1, -1, HIDDEN), wdtype, col(2)),
                sink=W("self_attn.attention_sink_bias").float().reshape(TP, NQC),  # host
            )
            self.layers.append(L)
        self.kctx = [None] * 5
        self.vctx = [None] * 5
        self.lc = None  # valid context length per (row, user) [rows, U]

    # ------------------------------------------------------------------ helpers
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

    def _rows_tensor(self, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        """host torch [rows, ...] -> device, row r's slice on the 8 devices of mesh row r."""
        return self.up(t, dtype, self.rowmap, layout)

    def _rm_view(self, x, shape):
        r = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        r = ttnn.reshape(r, shape)
        return r

    # ------------------------------------------------------------------ context
    def set_context(self, feats, pos0=None):
        """feats [rows, U, N, 20480] bf16 torch (concat of target hidden states of layers 0,11,23,35,47, positions pos0..pos0+N-1 per user; N <= 1024 here).
        Rebuilds the per-layer context K/V caches. pos0 [rows,U] (default 0)."""
        rows, U, N, _ = feats.shape
        assert N <= WIN
        self.U = U
        pos0 = torch.zeros(rows, U, dtype=torch.long) if pos0 is None else pos0
        x = self.up(
            feats.reshape(rows, 1, U * N, 20480),
            ttnn.bfloat16,
            ttnn.ShardTensor2dMesh(self.mesh, self.mesh.shape, dims=(0, 3)),
        )
        hp = ttnn.linear(x, self.fc, dtype=ttnn.bfloat16, compute_kernel_config=_ck())
        ttnn.deallocate(x)
        h = self._ar(hp)
        ttnn.deallocate(hp)
        hn = self._rms(h, self.hidden_norm)  # [1,1,U*N,4096]
        ttnn.deallocate(h)
        pos = pos0[:, :, None] + torch.arange(N)[None, None]
        cos, sin = rope_cs(pos)  # [rows,U,N,128]
        cos_t = self._rows_tensor(cos.reshape(rows, 1, U * N, HD))
        sin_t = self._rows_tensor(sin.reshape(rows, 1, U * N, HD))
        for li, L in enumerate(self.layers):
            k = ttnn.linear(hn, L["wk"], dtype=ttnn.bfloat16, compute_kernel_config=_ck())  # [1,1,U*N,128]
            k = self._rms(k, L["kn"])
            k = self._rope(k, cos_t, sin_t)
            v = ttnn.linear(hn, L["wv"], dtype=ttnn.bfloat16, compute_kernel_config=_ck())
            self.kctx[li] = self._to_cache(k, U, N)
            self.vctx[li] = self._to_cache(v, U, N)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        for t in (hn, cos_t, sin_t):
            ttnn.deallocate(t)
        self.lc = torch.full((rows, U), N, dtype=torch.long)

    def _to_cache(self, t, U, N):
        """[1,1,U*N,128] -> [U,1,1024,128] (left aligned, zero padded)"""
        r = self._rm_view(t, (U, 1, N, HD))
        if N < WIN:
            r = ttnn.pad(r, [(0, 0), (0, 0), (0, WIN - N), (0, 0)], 0.0)
        return ttnn.to_layout(r, ttnn.TILE_LAYOUT)

    # ------------------------------------------------------------------ one block round
    def draft(self, tokens, anchor_pos):
        """tokens [rows, U] anchor token ids (torch long), anchor_pos [rows, U]. Returns device hidden [1,1,U*8,4096] (after final norm; rows 1..7 of each user are the
        draft rows) and host logits [rows,U,7,V]."""
        rows, U = tokens.shape
        mask_np, sink, cos_q, sin_q, cos_k, sin_k = self._round_inputs(anchor_pos)
        tok = self.up(tokens.reshape(rows, U).to(torch.int32), ttnn.uint32, self.rowmap, ttnn.ROW_MAJOR_LAYOUT)
        e = ttnn.embedding(tok, self.embed, layout=ttnn.ROW_MAJOR_LAYOUT)  # [U,1,4096]? -> [1,U,4096]
        ttnn.deallocate(tok)
        e = ttnn.reshape(e, (U, 1, 1, HIDDEN))
        me = self.up(
            self.mask_emb.expand(rows, U, 7, HIDDEN).reshape(rows, U, 7, HIDDEN).contiguous(),
            ttnn.bfloat16,
            self.rowmap,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        x = ttnn.concat([e, me], dim=2)  # [U,1,8,4096] row major
        ttnn.deallocate(e)
        ttnn.deallocate(me)
        x = ttnn.to_layout(ttnn.reshape(x, (1, 1, U * BLK, HIDDEN)), ttnn.TILE_LAYOUT)
        flip = lambda t: ttnn.reshape(t, (U, 1) + tuple(t.shape)[2:])
        cq, sq = flip(self._rows_tensor(cos_q)), flip(self._rows_tensor(sin_q))
        ck, sk = flip(self._rows_tensor(cos_k)), flip(self._rows_tensor(sin_k))
        colrow = ttnn.ShardTensor2dMesh(self.mesh, self.mesh.shape, dims=(0, 1))
        for li, L in enumerate(self.layers):
            xn = self._rms(x, L["ln1"])
            q = ttnn.linear(xn, L["wq"], dtype=ttnn.bfloat16, compute_kernel_config=_ck())  # [1,1,8U,1024]
            k = ttnn.linear(xn, L["wk"], dtype=ttnn.bfloat16, compute_kernel_config=_ck())  # [1,1,8U,128]
            v = ttnn.linear(xn, L["wv"], dtype=ttnn.bfloat16, compute_kernel_config=_ck())
            ttnn.deallocate(xn)
            # q rows (user, head, tok): [8U,1024] -> [U,8tok,8head,128] -> [U,8head,8tok,128] -> [U,1,64,128]
            qr = self._rm_view(q, (U, BLK, NQC, HD))
            qr = ttnn.permute(qr, (0, 2, 1, 3))
            qr = ttnn.to_layout(ttnn.reshape(qr, (U, 1, NQC * BLK, HD)), ttnn.TILE_LAYOUT)
            ttnn.deallocate(q)
            qr = self._rms(qr, L["qn"])
            qr = self._rope(qr, cq, sq)
            kb = ttnn.to_layout(self._rm_view(k, (U, 1, BLK, HD)), ttnn.TILE_LAYOUT)
            vb = ttnn.to_layout(self._rm_view(v, (U, 1, BLK, HD)), ttnn.TILE_LAYOUT)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            kb = self._rms(kb, L["kn"])
            kb = self._rope(kb, ck, sk)
            zpad = ttnn.zeros(
                [U, 1, KPAD - WIN - BLK, HD], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh
            )
            K = ttnn.concat([self.kctx[li], kb, zpad], dim=2)
            V = ttnn.concat([self.vctx[li], vb, zpad], dim=2)
            ttnn.deallocate(kb)
            ttnn.deallocate(vb)
            ttnn.deallocate(zpad)
            s = ttnn.matmul(qr, K, transpose_b=True, compute_kernel_config=_ck(hifi4=True))  # [U,1,64,KPAD]
            s = ttnn.multiply(s, HD**-0.5)
            mk = mask_np.clone()
            for c in range(TP):
                for h in range(NQC):
                    mk[:, c, :, h * BLK : (h + 1) * BLK, WIN + BLK] = sink[li, c, h]
            mt = ttnn.reshape(self.up(mk.to(torch.bfloat16), ttnn.bfloat16, colrow), (U, 1, NQC * BLK, KPAD))
            s = ttnn.add(s, mt)
            ttnn.deallocate(mt)
            p = ttnn.softmax(s, dim=-1, compute_kernel_config=_ck(hifi4=True))
            ttnn.deallocate(s)
            o = ttnn.matmul(p, V, compute_kernel_config=_ck(hifi4=True))  # [U,1,64,128]
            for t in (p, K, V, qr):
                ttnn.deallocate(t)
            o = self._rm_view(o, (U, NQC, BLK, HD))
            o = ttnn.permute(o, (0, 2, 1, 3))
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
        for t in (cq, sq, ck, sk):
            ttnn.deallocate(t)
        y = self._rms(x, self.norm)
        ttnn.deallocate(x)
        return y

    def logits(self, y):
        """y [1,1,U*8,4096] -> host [rows,U,8,V] (row 0 of each block is unused by the caller)."""
        lg = ttnn.linear(y, self.lm_head, dtype=ttnn.bfloat16, compute_kernel_config=_ck(hifi4=True))
        out = ttnn.to_torch(
            lg, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh, dims=(0, 3), mesh_shape=tuple(self.mesh.shape))
        ).float()
        ttnn.deallocate(lg)
        rows = out.shape[0]
        return out[:, 0].reshape(rows, -1, BLK, out.shape[-1])

    def hidden_host(self, y):
        dts = ttnn.get_device_tensors(y)
        cols = self.mesh.shape[1]
        out = torch.stack([ttnn.to_torch(dts[r * cols]).float() for r in range(self.rows)])  # [rows,1,1,U*8,4096]
        return out.reshape(self.rows, -1, BLK, HIDDEN)

    def _round_inputs(self, anchor_pos):
        """host tensors of one round: mask [rows, TP, U, 64, KPAD] (additive, per column because the sink bias is per head), cos/sin for the q rows (head, tok) and the block keys."""
        rows, U = anchor_pos.shape
        mask = torch.full((rows, TP, U, NQC * BLK, KPAD), NEG)
        for r in range(rows):
            for u in range(U):
                n = int(self.lc[r, u])
                mask[r, :, u, :, :n] = 0
                mask[r, :, u, :, WIN : WIN + BLK] = 0
        sink = torch.stack([L["sink"] for L in self.layers])  # [5,TP,8]
        pos = anchor_pos[:, :, None] + torch.arange(BLK)  # [rows,U,8]
        cos_t, sin_t = rope_cs(pos)  # [rows,U,8,128]
        cq = cos_t[:, :, None].expand(rows, U, NQC, BLK, HD).reshape(rows, U, NQC * BLK, HD)
        sq = sin_t[:, :, None].expand(rows, U, NQC, BLK, HD).reshape(rows, U, NQC * BLK, HD)
        return mask, sink, cq, sq, cos_t, sin_t
