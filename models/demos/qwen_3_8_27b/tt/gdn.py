# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet (linear attention) token mixer on the SP x TP mesh.

Math: upstream ``Qwen3_5GatedDeltaNet.forward`` (see ``reference/qwen3_8_ref.py``). Op composition
learned from ``models/demos/blackhole/qwen36/tt/gdn`` (math only — that package never ran on the
galaxy): the delta-rule core is the fused ``ttnn.transformer.chunk_gated_delta_rule`` op (flat
token-major q/k/v, in-kernel q/k L2-norm, GVA 16->48 head expansion inside the op, fp32 state).

Parallel layout
  TP (columns): heads. Column c owns key heads [4c, 4c+4) and value heads [12c, 12c+12) — the GVA
    groups stay local, so the recurrence needs no TP communication. in_proj is column-parallel,
    out_proj row-parallel + a TP all-reduce.
  SP (rows): the recurrence runs over the whole sequence, so after the (local) projections the chunk's
    q/k/v + gate streams are all-gathered over SP and *every row runs the full chunk's recurrence*
    (identical results on all rows). Each row then keeps its own tokens (``ttnn.mesh_partition``) for
    the gated norm and out_proj. Redundant compute (x sp) in the GDN core, but exact and simple; a
    cross-row state hand-off is the perf follow-on. See README "GDN under SP".

Carried state (per layer, lives in ``Qwen38Caches.gdn``):
  recurrent  fp32 [Nv_local, Dk, Dv]      final state after the chunk
  conv       bf16 ROW_MAJOR [1, 1, K-1, C_local]  last K-1 *pre-conv* projection rows
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.qwen_3_8_27b.config import Qwen38Config
from models.demos.qwen_3_8_27b.tt.common import hifi4_fp32, residual_dtype, upload
from models.demos.qwen_3_8_27b.tt.gdn_core import ChunkDeltaRule, gdn_core_mode
from models.demos.qwen_3_8_27b.tt.rms_norm import rms_norm_fp32

# the fused op's internal chunk (32 = one tile; larger chunks hit L1 limits / conditioning, see qwen36)
GDN_OP_CHUNK = 32
# max tokens per fused-op call; longer sequences are split with the state carried (exact).
GDN_MAX_TOKENS_PER_CALL = 5120


def _const_tiles(mesh_device, C=GDN_OP_CHUNK):
    eye = torch.eye(C)
    tril = torch.tril(torch.ones(C, C))
    ones = torch.ones(C, C)
    ii, jj = torch.arange(32)[:, None], torch.arange(32)[None, :]
    lo_i, lo_j = ii < 16, jj < 16
    masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)

    def up(t):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return up(eye), up(tril), up(ones), up(masks)


class TtGatedDeltaNet:
    AB_PAD = 32  # b and a each padded to one tile of columns

    def __init__(
        self, mesh_config, cfg: Qwen38Config, sd: dict, *, weight_dtype, cache=None, prefix="", part_dtypes=None
    ):
        """``sd``: HF-named tensors relative to ``linear_attn.`` (in_proj_qkv.weight, conv1d.weight, ...)."""
        self.mc = mesh_config
        self.mesh = mesh_config.mesh_device
        self.cfg = cfg
        tp = mesh_config.tp
        self.nk, self.nv = cfg.linear_num_key_heads // tp, cfg.linear_num_value_heads // tp
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.K = cfg.linear_conv_kernel_dim
        self.qd, self.vd = self.nk * self.dk, self.nv * self.dv  # per-column widths: 512, 1536
        self.cd = 2 * self.qd + self.vd  # per-column conv channels: 2560
        kd, VD = cfg.linear_key_dim, cfg.linear_value_dim
        self.eps = cfg.rms_norm_eps
        self.ckc = hifi4_fp32()

        def get(k):
            return None if sd is None else sd[k]

        if sd is not None:  # sd None => every tensor comes from the tensor cache
            w_qkv = get("in_proj_qkv.weight").float()  # [conv_dim, H]
            w_z = get("in_proj_z.weight").float()  # [VD, H]
            w_b, w_a = get("in_proj_b.weight").float(), get("in_proj_a.weight").float()  # [Nv, H]
            conv = get("conv1d.weight").float()[:, 0, :]  # [conv_dim, K]

            def chan(c):  # this column's conv channels, in conv-dim order q|k|v
                return torch.cat(
                    [
                        torch.arange(c * self.qd, (c + 1) * self.qd),
                        kd + torch.arange(c * self.qd, (c + 1) * self.qd),
                        2 * kd + torch.arange(c * self.vd, (c + 1) * self.vd),
                    ]
                )

            qkvz_blocks, ab_blocks, conv_blocks, out_blocks = [], [], [], []
            w_out = get("out_proj.weight").float()  # [H, VD]
            for c in range(tp):
                vsl = slice(c * self.vd, (c + 1) * self.vd)
                hsl = slice(c * self.nv, (c + 1) * self.nv)
                qkvz_blocks.append(torch.cat([w_qkv[chan(c)], w_z[vsl]], 0).T)  # [H, cd + vd]
                ab = torch.zeros(2 * self.AB_PAD, w_b.shape[1])
                ab[: self.nv] = w_b[hsl]
                ab[self.AB_PAD : self.AB_PAD + self.nv] = w_a[hsl]
                ab_blocks.append(ab.T)  # [H, 64]
                conv_blocks.append(conv[chan(c)].T)  # [K, cd]
                out_blocks.append(w_out[:, vsl].T)  # [vd, H]
            host = {
                "w_qkvz": torch.cat(qkvz_blocks, 1)[None, None],
                "w_ab": torch.cat(ab_blocks, 1)[None, None],
                "conv": torch.cat(conv_blocks, 1)[:, None, None, :],  # [K, 1, 1, tp*cd]
                "w_out": torch.cat(out_blocks, 0)[None, None],
                "neg_a": (-get("A_log").float().exp()).reshape(1, 1, 1, -1),
                "dt_bias": get("dt_bias").float().reshape(1, 1, 1, -1),
                "norm_w": get("norm.weight").float().reshape(1, 1, 1, -1),
            }
        else:
            host = dict.fromkeys(["w_qkvz", "w_ab", "conv", "w_out", "neg_a", "dt_bias", "norm_w"])

        col = mesh_config.shard(None, 3)  # shard last dim over TP columns
        rowp = mesh_config.shard(None, 2)  # shard dim 2 over TP columns (row-parallel weight)
        rep = mesh_config.replicate()
        up = lambda n, **kw: upload(host[n], self.mesh, cache=cache, name=f"{prefix}{n}", **kw)  # noqa: E731
        pd = {"w_qkvz": weight_dtype, "w_ab": weight_dtype, "w_out": weight_dtype, **(part_dtypes or {})}
        self.w_qkvz = up("w_qkvz", dtype=pd["w_qkvz"], mapper=col)
        self.w_ab = up("w_ab", dtype=pd["w_ab"], mapper=col)
        self.w_out = up("w_out", dtype=pd["w_out"], mapper=rowp)
        conv_all = up("conv", dtype=ttnn.float32, mapper=col)  # [K, 1, 1, cd] per column
        self.conv_w = [ttnn.slice(conv_all, [j, 0, 0, 0], [j + 1, 1, 1, self.cd]) for j in range(self.K)]
        ttnn.deallocate(conv_all)
        self.neg_a = up("neg_a", dtype=ttnn.float32, mapper=col)
        self.dt_bias = up("dt_bias", dtype=ttnn.float32, mapper=col)
        self.norm_w = up("norm_w", dtype=ttnn.float32, mapper=rep)
        self.consts = _const_tiles(self.mesh)
        self.core_mode = gdn_core_mode()
        self.composed = ChunkDeltaRule(self.mesh) if self.core_mode == "composed" else None

    # ------------------------------------------------------------------------------------------
    def _causal_conv(self, qkv_full, conv_state, valid_len):
        """qkv_full [1,1,T,cd] TILE bf16 (whole chunk) -> silu(conv) [1,1,T,cd] TILE bf16, new state
        (the K-1 pre-conv rows ending at ``valid_len``)."""
        T = qkv_full.shape[2]
        rm = ttnn.to_layout(qkv_full, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if conv_state is None:
            conv_state = ttnn.from_torch(
                torch.zeros(1, 1, self.K - 1, self.cd),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=self.mc.replicate(),
            )
        xp = ttnn.concat([conv_state, rm], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,T+K-1,cd]
        ttnn.deallocate(rm)
        new_state = ttnn.slice(
            xp, [0, 0, valid_len, 0], [1, 1, valid_len + self.K - 1, self.cd], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        acc = None
        for j in range(self.K):
            s = ttnn.slice(xp, [0, 0, j, 0], [1, 1, j + T, self.cd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            s = ttnn.to_layout(s, ttnn.TILE_LAYOUT, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            term = ttnn.multiply(s, self.conv_w[j], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(s)
            if acc is None:
                acc = term
            else:
                acc2 = ttnn.add(acc, term, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(acc)
                ttnn.deallocate(term)
                acc = acc2
        ttnn.deallocate(xp)
        out = ttnn.silu(acc, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fp32
        ttnn.deallocate(acc)
        return out, new_state

    def _gates(self, ab_full, valid_len):
        """ab_full [1,1,T,64] -> beta, g as fp32 [1, T, Nv_local]. Past ``valid_len`` both are zeroed, which
        makes the padded tail an identity state update (so the carried state is the state at valid_len)."""
        T = ab_full.shape[2]
        ab = ttnn.typecast(ab_full, ttnn.float32)
        b = ttnn.slice(ab, [0, 0, 0, 0], [1, 1, T, self.nv])
        a = ttnn.slice(ab, [0, 0, 0, self.AB_PAD], [1, 1, T, self.AB_PAD + self.nv])
        ttnn.deallocate(ab)
        beta = ttnn.sigmoid(b)
        sp = ttnn.softplus(ttnn.add(a, self.dt_bias), beta=1.0, threshold=20.0)
        g = ttnn.multiply(sp, self.neg_a)
        for t in (a, b, sp):
            ttnn.deallocate(t)
        if valid_len < T:
            m = torch.zeros(1, 1, T, 1)
            m[:, :, :valid_len] = 1.0
            m = ttnn.from_torch(
                m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=self.mc.replicate()
            )
            beta, g = ttnn.multiply(beta, m), ttnn.multiply(g, m)
            ttnn.deallocate(m)
        return ttnn.reshape(beta, [1, T, self.nv]), ttnn.reshape(g, [1, T, self.nv])

    def _core_composed(self, conv_out, beta, g, rec_state):
        """fp32 composed delta rule (tt/gdn_core.py). conv_out [1,1,T,cd] fp32; beta, g [1,T,Nv] fp32."""
        T = conv_out.shape[2]
        C = self.composed.C
        assert T % C == 0, f"GDN chunk {T} tokens is not a multiple of {C}"
        NC = T // C
        rep = self.nv // self.nk

        def heads(lo, hi, nh, expand):
            x = ttnn.slice(conv_out, [0, 0, 0, lo], [1, 1, T, hi])
            x = ttnn.reshape(x, [1, T, nh, self.dk])
            x = ttnn.permute(x, (0, 2, 1, 3))  # [1, nh, T, D]
            if expand > 1:
                x = ttnn.repeat_interleave(x, expand, dim=1)
            return ttnn.reshape(x, [self.nv, NC, C, self.dk])

        q = heads(0, self.qd, self.nk, rep)
        k = heads(self.qd, 2 * self.qd, self.nk, rep)
        v = heads(2 * self.qd, self.cd, self.nv, 1)

        def per_head(t):  # [1, T, Nv] -> [Nv, NC, C, 1]
            t = ttnn.reshape(t, [1, 1, T, self.nv])
            t = ttnn.permute(t, (0, 3, 2, 1))  # [1, Nv, T, 1]
            return ttnn.reshape(t, [self.nv, NC, C, 1])

        bt, gt = per_head(beta), per_head(g)
        S0 = None if rec_state is None else ttnn.reshape(rec_state, [self.nv, 1, self.dk, self.dv])
        o, S = self.composed(q, k, v, gt, bt, S0)
        for t in (q, k, v, bt, gt):
            ttnn.deallocate(t)
        o = ttnn.reshape(o, [1, self.nv, T, self.dv])
        o = ttnn.permute(o, (0, 2, 1, 3))  # [1, T, Nv, Dv]
        o = ttnn.typecast(ttnn.reshape(o, [1, 1, T, self.vd]), ttnn.bfloat16)
        return o, S

    def _core(self, conv_out, beta, g, rec_state):
        """Delta rule over the whole chunk, split into <= GDN_MAX_TOKENS_PER_CALL pieces with the state carried."""
        T = conv_out.shape[2]
        outs = []
        step = min(T, GDN_MAX_TOKENS_PER_CALL)
        assert T % step == 0 and step % GDN_OP_CHUNK == 0
        s0 = rec_state
        for t0 in range(0, T, step):
            t1 = t0 + step
            piece = conv_out if step == T else ttnn.slice(conv_out, [0, 0, t0, 0], [1, 1, t1, self.cd])
            q = ttnn.reshape(ttnn.slice(piece, [0, 0, 0, 0], [1, 1, step, self.qd]), [1, step, self.qd])
            k = ttnn.reshape(ttnn.slice(piece, [0, 0, 0, self.qd], [1, 1, step, 2 * self.qd]), [1, step, self.qd])
            v = ttnn.reshape(ttnn.slice(piece, [0, 0, 0, 2 * self.qd], [1, 1, step, self.cd]), [1, step, self.vd])
            bt = beta if step == T else ttnn.slice(beta, [0, t0, 0], [1, t1, self.nv])
            gt = g if step == T else ttnn.slice(g, [0, t0, 0], [1, t1, self.nv])
            init = None if s0 is None else ttnn.reshape(s0, [1, self.nv, self.dk, self.dv])
            eye, tril, ones, masks = self.consts
            o, s1 = ttnn.transformer.chunk_gated_delta_rule(
                q,
                k,
                v,
                gt,
                bt,
                scale=self.dk**-0.5,
                initial_state=init,
                output_final_state=True,
                chunk_size=GDN_OP_CHUNK,
                output_head_major=False,
                eye=eye,
                tril=tril,
                ones=ones,
                masks=masks,
            )
            # token-major [1, step, Nv, Dv] ROW_MAJOR -> [1, 1, step, Nv*Dv] TILE
            o = ttnn.to_layout(ttnn.reshape(o, [1, 1, step, self.vd]), ttnn.TILE_LAYOUT)
            outs.append(o)
            for t in (q, k, v):
                ttnn.deallocate(t)
            if s0 is not None and s0 is not rec_state:
                ttnn.deallocate(s0)
            s0 = s1
        o = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)
        return o, s0

    def __call__(self, x, state: dict, valid_len: int | None = None):
        """x [1,1,S_local,H] (TP-replicated, SP-sharded). ``state`` = {"recurrent", "conv"} updated in place.
        ``valid_len``: valid tokens in this chunk (default all); the tail past it is padding."""
        qkvz = ttnn.linear(x, self.w_qkvz, compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ab = ttnn.linear(x, self.w_ab, compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        S = qkvz.shape[2]
        qkv = ttnn.slice(qkvz, [0, 0, 0, 0], [1, 1, S, self.cd])
        z = ttnn.slice(qkvz, [0, 0, 0, self.cd], [1, 1, S, self.cd + self.vd])
        ttnn.deallocate(qkvz)

        qkv_full = self.mc.all_gather_sp(qkv, dim=2)  # [1,1,T,cd] identical on every row
        ab_full = self.mc.all_gather_sp(ab, dim=2)
        ttnn.deallocate(qkv)
        ttnn.deallocate(ab)

        T = qkv_full.shape[2]
        valid_len = T if valid_len is None else valid_len
        assert 0 < valid_len <= T
        conv_out, new_conv = self._causal_conv(qkv_full, state.get("conv"), valid_len)
        ttnn.deallocate(qkv_full)
        beta, g = self._gates(ab_full, valid_len)
        ttnn.deallocate(ab_full)
        if self.core_mode == "composed":
            o_full, new_rec = self._core_composed(conv_out, beta, g, state.get("recurrent"))
        else:
            conv_bf = ttnn.typecast(conv_out, ttnn.bfloat16)  # the fused op's q/k/v contract is bf16
            ttnn.deallocate(conv_out)
            conv_out = conv_bf
            o_full, new_rec = self._core(conv_out, beta, g, state.get("recurrent"))
        ttnn.deallocate(conv_out)
        ttnn.deallocate(beta)
        ttnn.deallocate(g)

        o = ttnn.mesh_partition(o_full, dim=2, cluster_axis=self.mc.sp_axis)  # this row's tokens
        ttnn.deallocate(o_full)

        # gated RMSNorm per value head (plain weight), then * silu(z)
        o = ttnn.reshape(o, [1, 1, S * self.nv, self.dv])
        o = rms_norm_fp32(o, self.norm_w, self.eps)
        o = ttnn.reshape(o, [1, 1, S, self.vd])
        zs = ttnn.silu(z)
        ttnn.deallocate(z)
        o2 = ttnn.multiply(o, zs)
        ttnn.deallocate(o)
        ttnn.deallocate(zs)
        out = ttnn.linear(
            o2,
            self.w_out,
            dtype=residual_dtype(),
            compute_kernel_config=self.ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(o2)
        out = self.mc.all_reduce_tp(out)

        for key, new in (("recurrent", new_rec), ("conv", new_conv)):
            old = state.get(key)
            if old is not None:
                ttnn.deallocate(old)
            state[key] = new
        return out

    # ---- state read-back in the golden trace's layout ----
    def read_state(self, state: dict, sp_row: int = 0):
        """-> recurrent_state [1, Nv, Dk, Dv] fp32, conv_state [1, conv_dim, K-1] in upstream channel order.

        Every SP row holds the same state; ``sp_row`` picks which row's copy to read."""
        return read_gdn_state(state, self.mc, self.nv, self.qd, self.vd, sp_row)


def read_gdn_state(state: dict, mesh_config, nv_local, qd, vd, sp_row=0):
    tp = mesh_config.tp
    rec_dev = ttnn.get_device_tensors(state["recurrent"])
    conv_dev = ttnn.get_device_tensors(state["conv"])
    idx = [sp_row * tp + c for c in range(tp)]  # row-major mesh order
    rec = torch.cat([ttnn.to_torch(rec_dev[i]).float().reshape(nv_local, *list(rec_dev[i].shape)[-2:]) for i in idx], 0)
    conv = [ttnn.to_torch(conv_dev[i]).float().reshape(-1, 2 * qd + vd) for i in idx]  # [K-1, cd] per column
    q = torch.cat([cv[:, :qd] for cv in conv], 1)
    k = torch.cat([cv[:, qd : 2 * qd] for cv in conv], 1)
    v = torch.cat([cv[:, 2 * qd :] for cv in conv], 1)
    conv_state = torch.cat([q, k, v], 1).T[None]  # [1, conv_dim, K-1]
    return rec[None], conv_state
