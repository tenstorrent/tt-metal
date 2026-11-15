# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lossless speculative SAMPLING on device (task mtp-device, stage 6; spec-boost BOOST2.md item 1): draft-from-q rejection sampling, full vocabulary, temperature only (top-k / top-p of the production
sampler would apply to p and q alike: not implemented here).
Round with k drafts (rows j = 0..k of the verify, vocab sharded over the 8 mesh columns, logits [1,1,T=S,shard] fp32):
  draft j+1 ~ q_j = softmax(MTP logits_j / tau)   (``sample_q``: inverse-CDF draw with the draft uniform)
  verify row j: p_j = softmax(logits_j / tau); accept d_{j+1} iff u_a < p_j(d) / q_j(d); on reject emit y ~ norm(max(p_j - q_j, 0)); row k (all accepted): bonus y ~ p_k
  ``sampled_tokens`` returns a_j = d_{j+1} if accepted else y_j (row k: the bonus): the SAME vector the greedy path feeds to ``accept`` (a rejected y never equals d because the residual has no mass at d), so the
  accept / MTP / rollback machinery is unchanged; the emitted tokens of a round are a_0..a_m.
The draws use the weights -> inverse CDF machinery of ``Hy4DeviceSampler.forward`` (per-column prefix via triangular matmuls, column totals gathered), here as ``draw_from_weights``.
"""
import torch

import ttnn


class SpecSampleOps:
    def __init__(self, smp):
        """smp: a Hy4DeviceSampler built with users_per_row = T (rows of one verify = S)."""
        self.s, self.T = smp, smp.T
        md = smp.md
        cols, shard, PR = smp.cols, smp.shard, smp.PR
        rep = lambda t: ttnn.from_torch(
            t,
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        # global token ids of this column's slice as [1,1,PR,32] tiles: own_col * shard + (row * 32 + j)
        self.local_ids = rep(torch.arange(PR * 32, dtype=torch.float32).reshape(1, 1, PR, 32))

    # ---------------------------------------------------------------------------------------------- softmax pieces
    def prepare(self, logits, invT):
        """logits [1,1,T,shard] fp32, invT [1,1,T,1] -> (p4 [T,1,PR,32] probabilities of this column's slice, Z, M)."""
        s, T, PR, shard = self.s, self.T, self.s.PR, self.s.shard
        M, _mn, _tok = s._greedy(logits, with_min=False)  # global max [1,1,T,1]
        x4 = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), [T, 1, PR, 32]), ttnn.TILE_LAYOUT
        )
        e4 = ttnn.exp(ttnn.multiply(ttnn.subtract(x4, s._v4(M)), s._v4(invT)))
        loc = s._row(ttnn.sum(e4, dim=[2, 3], keepdim=True))  # [1,1,T,1] this column's mass
        Z = s._allsum(loc)  # [1,1,T,1]
        p4 = ttnn.multiply(e4, s._v4(ttnn.reciprocal(Z)))
        return p4

    def prob_of(self, p4, tok_f):
        """probability of token ids tok_f [1,1,T,1] fp32 (global ids) under p4 -> [1,1,T,1] (replicated)."""
        s = self.s
        gid = ttnn.add(
            self.local_ids, ttnn.multiply(s.own_col, float(s.shard))
        )  # [1,1,PR,32] global ids of this column
        hit = ttnn.eq(gid, s._v4(tok_f))  # [T,1,PR,32]
        loc = s._row(ttnn.sum(ttnn.multiply(p4, hit), dim=[2, 3], keepdim=True))
        return s._allsum(loc)

    def draw_from_weights(self, w4, u):
        """w4 [T,1,PR,32] non-negative weights over this column's slice, u [1,1,T,1] fp32 in [0,1) -> token ids [1,1,T,1] fp32 (identical on every device)."""
        s, T, cols, shard, PR = self.s, self.T, self.s.cols, self.s.shard, self.s.PR
        P = ttnn.matmul(w4, s.u32, compute_kernel_config=s.ckc)
        R = ttnn.slice(P, [0, 0, 0, 31], [T, 1, PR, 32])
        offs = ttnn.transpose(ttnn.matmul(ttnn.transpose(R, 2, 3), s.ustrict, compute_kernel_config=s.ckc), 2, 3)
        cs = ttnn.add(P, offs)
        tot_c = ttnn.reshape(ttnn.slice(cs, [0, 0, PR - 1, 31], [T, 1, PR, 32]), [1, 1, T, 1])
        tot_all = s.mc.allgather(tot_c, s.ccl, axis=1, dim=3)  # [1,1,T,cols]
        total = ttnn.sum(tot_all, dim=-1, keepdim=True)
        incl = ttnn.matmul(tot_all, s.tri, compute_kernel_config=s.ckc)
        tgt = ttnn.multiply(u, total)
        colsel = ttnn.minimum(ttnn.sum(ttnn.le(incl, tgt), dim=-1, keepdim=True), float(cols - 1))
        excl_sel = ttnn.sum(
            ttnn.multiply(ttnn.eq(s.col_ids, colsel), ttnn.subtract(incl, tot_all)), dim=-1, keepdim=True
        )
        local_t = ttnn.subtract(tgt, excl_sel)
        cnt = ttnn.reshape(ttnn.sum(ttnn.le(cs, s._v4(local_t)), dim=[2, 3], keepdim=True), [1, 1, T, 1])
        cnt = ttnn.minimum(cnt, float(shard - 1))
        mine = ttnn.add(ttnn.multiply(s.own_col, float(shard)), cnt)
        cand = s.mc.allgather(mine, s.ccl, axis=1, dim=3)
        return ttnn.sum(ttnn.multiply(ttnn.eq(s.col_ids, colsel), cand), dim=-1, keepdim=True)

    # ---------------------------------------------------------------------------------------------- round pieces
    def sample_q(self, q_logits, invT, u):
        """draft tokens d ~ softmax(q_logits * invT): [1,1,T,1] fp32 ids (and q4 for the later ratio)."""
        q4 = self.prepare(q_logits, invT)
        return self.draw_from_weights(q4, u), q4

    def sampled_tokens(self, p_logits, q4_rows, d_f, invT, u_acc, u_res, u_bonus, is_bonus):
        """p_logits [1,1,T,shard]; q4_rows [T,1,PR,32] = q probabilities of the draft that produced d_f [1,1,T,1] (row k: any, ignored); returns a [1,1,T,1] fp32:
        a_j = d if u_acc < p(d)/q(d) else y_res (row with is_bonus == 1: the bonus draw from p)."""
        p4 = self.prepare(p_logits, invT)
        pd_, qd = self.prob_of(p4, d_f), self.prob_of(q4_rows, d_f)
        acc = ttnn.lt(ttnn.multiply(u_acc, qd), pd_)  # u < p/q  <=>  u q < p (q(d) > 0 for a sampled draft)
        res = ttnn.relu(ttnn.subtract(p4, q4_rows))
        y_res = self.draw_from_weights(res, u_res)
        y_b = self.draw_from_weights(p4, u_bonus)
        a_draft = ttnn.add(ttnn.multiply(acc, d_f), ttnn.multiply(ttnn.subtract(1.0, acc), y_res))
        return ttnn.add(ttnn.multiply(is_bonus, y_b), ttnn.multiply(ttnn.subtract(1.0, is_bonus), a_draft)), acc


class SpecSampling:
    """Sampled-spec state + RNG for one round trace (plugs into MTPRoundTail / Hy4SpecRunner via ``sampling=``). Persistent tensors are allocated here (before any capture).
    RNG: u = lowbias32(seed ^ pos * golden ^ stream) / 2^24-style, a function of (request seed, position of the token being decided, stream id): rollback needs no RNG state (DESIGN_ROUND.md).
    """

    GOLD, STREAM = 0x9E3779B9, 0x632BE5AB

    def __init__(self, smp, k, invT=1.0, seed=0):
        self.k, self.S = k, k + 1
        self.ops = SpecSampleOps(smp)
        S, md, s = self.S, smp.md, smp
        R = int(md.shape[0])
        rep = lambda t, dt=ttnn.float32: ttnn.from_torch(
            t,
            device=md,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        rows = lambda t, dt=ttnn.float32: ttnn.from_torch(
            t,
            device=md,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape)),
        )
        self.invT = rows(torch.full((R, 1, S, 1), float(invT)))  # per row (user) temperature, set by ``set_params``
        self.seed = rows(torch.full((R, 1, 1, 1), int(seed) & 0x7FFFFFFF, dtype=torch.int32), ttnn.uint32)
        self.arS_col = rep(torch.arange(S, dtype=torch.float32).reshape(1, 1, S, 1))
        self.is_bonus = rep((torch.arange(S) == S - 1).float().reshape(1, 1, S, 1))
        self.q_persist = ttnn.from_torch(
            torch.zeros(R * S, 1, s.PR, 32),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape)),
        )
        self.d_col = rows(torch.zeros(R, 1, S, 1))
        self.zero_q = rep(torch.zeros(1, 1, s.PR, 32))
        self.s = s
        self._qrows, self._d = [], []

    def set_params(self, invT, seed):
        """host (transition only): per user invT [R], seed [R]."""
        md = self.ops.s.md
        R = int(md.shape[0])
        mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
        cp = lambda t, dt, tgt: ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, mesh_mapper=mp), tgt
        )
        cp(torch.tensor(invT, dtype=torch.float32).reshape(R, 1, 1, 1).repeat(1, 1, self.S, 1), ttnn.float32, self.invT)
        cp(
            torch.tensor([int(x) & 0x7FFFFFFF for x in seed], dtype=torch.int32).reshape(R, 1, 1, 1),
            ttnn.uint32,
            self.seed,
        )

    def draw_u(self, pos_col, stream):
        """pos_col fp32 [1,1,n,1] -> u fp32 [1,1,n,1] in [0,1) (24 bits)."""
        pu = ttnn.typecast(pos_col, ttnn.uint32)
        x = ttnn.bitwise_xor(self.seed, ttnn.multiply(pu, self.GOLD))
        x = ttnn.bitwise_xor(x, int(stream * self.STREAM) & 0x7FFFFFFF)
        x = ttnn.bitwise_xor(x, ttnn.logical_right_shift(x, 16))
        x = ttnn.multiply(x, 0x7FEB352D)
        x = ttnn.bitwise_xor(x, ttnn.logical_right_shift(x, 15))
        x = ttnn.multiply(x, 0x846CA68B)
        x = ttnn.bitwise_xor(x, ttnn.logical_right_shift(x, 16))
        return ttnn.multiply(ttnn.typecast(ttnn.logical_right_shift(x, 8), ttnn.float32), 1.0 / (1 << 24))

    # ---- verify side
    def verify(self, p_logits, pos):
        """p_logits [1,1,S,shard] fp32, pos [1,1,1,1] fp32 (P) -> a_tok uint32 [S,1] (a_j = d_{j+1} if accepted else the residual draw; row k: bonus)."""
        pcol = ttnn.add(pos, self.arS_col)  # position of verify row j = P + j
        ua, ur, ub = (self.draw_u(pcol, st) for st in (10, 11, 12))
        a, _acc = self.ops.sampled_tokens(p_logits, self.q_persist, self.d_col, self.invT, ua, ur, ub, self.is_bonus)
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(a, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [self.S, 1])

    # ---- draft side
    def draft(self, lg, i, pos_scalar):
        """lg [1,1,1,shard] fp32 MTP logits of draft i+1; pos_scalar [1,1,1,1] fp32 = position of the drafted token -> token fp32 [1,1,1,1]; keeps q for the next verify."""
        S = self.S
        lgS = ttnn.concat([lg] * S, dim=2)  # the T = S row ops with the real row 0 (rows 1.. repeat it)
        pcol = ttnn.add(pos_scalar, ttnn.multiply(self.arS_col, 0.0))
        u = self.draw_u(pcol, 1 + i)
        d, q4 = self.ops.sample_q(lgS, self.invT, u)
        PR = self.s.PR
        self._qrows.append(ttnn.slice(q4, [0, 0, 0, 0], [1, 1, PR, 32]))
        tok = ttnn.slice(d, [0, 0, 0, 0], [1, 1, 1, 1])
        self._d.append(tok)
        return tok

    def finish(self):
        """after the k draft heads: q_persist rows 0..k-1 (+ zero filler row k), d_col [1,1,S,1] = drafts (row k filler)."""
        k, S = self.k, self.S
        q = ttnn.concat(self._qrows + [self.zero_q] * (S - k), dim=0)
        ttnn.copy(q, self.q_persist)
        dcol = ttnn.concat(self._d + [ttnn.multiply(self._d[0], 0.0)] * (S - k), dim=2)
        ttnn.copy(dcol, self.d_col)
        self._qrows, self._d = [], []
