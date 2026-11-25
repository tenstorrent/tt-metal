# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention prefill, tensor parallel over heads (modeling_glm5_next.py:620-769).

Each of the ``tp`` chips owns ``64 / tp`` heads: its slice of q/k/v (+ their conv taps), of the forget gate's
``f_b`` / ``dt_bias`` / ``A_log``, of ``b_proj`` and of the output gate's ``g_b``; ``f_a`` / ``g_a`` are replicated and
``o_proj`` is row parallel (one all-reduce per sublayer).

The recurrence is a composite of chunk-parallel ttnn ops (``ttnn.experimental.kda.prepare_chunk_recurrence`` needs 33
dataflow buffers per core and cannot run on Wormhole). Per chunk of L = 32 tokens with in-chunk cumulative log decay
``G`` (per key channel), ``kb = beta * k``, ``vb = beta * v``:

    A[i, m]  = kb_i . (k_m e^{G_i - G_m})         m < i          (strictly lower)
    Tinv     = (I + A)^-1                (block doubling, 5 levels)
    W        = Tinv (kb e^G)            Uv = Tinv vb
    M        = diag(e^{G_L}) - (k e^{G_L - G})^T W            c = (k e^{G_L - G})^T Uv
    S_out    = M S_in + c                                       (chunk-to-chunk affine map)
    U        = Uv - W S_in
    Y        = (q e^G) S_in + tril(q_i . k_m e^{G_i - G_m}) U

The affine maps are composed across chunks with a Hillis-Steele scan (log2(#chunks) batched matmul rounds), so every
step is a batched matmul over (heads, chunks). ``e^{G_i - G_m}`` is factored as ``e^{G_i - r} e^{r - G_m}`` with the
anchor ``r = G_L / 2``, which bounds both factors by ``e^{-5 L / 2}`` (fp32 safe); the masked upper triangle may
overflow and is removed with ``where``. This is the same math as HF ``chunk_kimi_delta_attention`` (chunk 64).
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

import ttnn

from .common import DRAM, RM, TPContext, WeightLoader, to_rm, to_tile

CHUNK = 32


@dataclass
class KDAState:
    """Per-sequence carry of one KDA layer: conv tail (last ``K - 1`` raw q/k/v rows, RM bf16 ``[1, 1, 3, 3 W]``)
    and the recurrent state (fp32 ``[h, 128, 128]``, key x value)."""

    conv: ttnn.Tensor
    state: ttnn.Tensor


class _Consts:
    """Masks / identities of the chunked recurrence, per number of chunks."""

    def __init__(self, ctx: TPContext, heads: int):
        self.ctx = ctx
        self.heads = heads
        self._by_n = {}
        up = lambda x: ctx.upload(x, ttnn.float32)
        self.strict = up(torch.tril(torch.ones(CHUNK, CHUNK), -1).reshape(1, 1, CHUNK, CHUNK))
        self.incl = up(torch.tril(torch.ones(CHUNK, CHUNK)).reshape(1, 1, CHUNK, CHUNK))
        self.eye = up(torch.eye(CHUNK).reshape(1, 1, CHUNK, CHUNK))
        i, m = torch.arange(CHUNK)[:, None], torch.arange(CHUNK)[None, :]
        self.levels = []
        b = 1
        while b < CHUNK:  # (i, m) in the same 2b block, i in its lower b half, m in the upper
            sel = (i // (2 * b) == m // (2 * b)) & (i // b != m // b) & (i > m)
            self.levels.append(up(sel.float().reshape(1, 1, CHUNK, CHUNK)))
            b *= 2

    def eye_k(self, n: int, dk: int) -> ttnn.Tensor:
        if n not in self._by_n:
            self._by_n[n] = self.ctx.upload(torch.eye(dk).expand(self.heads, n, dk, dk), ttnn.float32)
        return self._by_n[n]


def _mm(ctx, a, b, **kw):
    return ttnn.matmul(a, b, dtype=ttnn.float32, compute_kernel_config=ctx.hifi4, memory_config=DRAM, **kw)


def kda_core(ctx: TPContext, consts: _Consts, q, k, v, g, beta, s0):
    """q, k, v, g: fp32 ``[h, N, 32, d]`` (q, k L2-normalized, q scaled by d^-0.5); beta: fp32 ``[h, N, 32, d]``
    (the per-head input gate broadcast over the head dim); s0: fp32 ``[h, d, d]``.
    Returns (o ``[h, N, 32, d]`` fp32, final state ``[h, d, d]`` fp32)."""
    h, n, c, d = q.shape
    G = ttnn.cumsum(g, dim=2)
    G_last = ttnn.sum(g, dim=2, keepdim=True)  # [h, N, 1, d]
    ref = ttnn.multiply(G_last, 0.5)
    e_pos = ttnn.exp(ttnn.subtract(G, ref))
    e_neg = ttnn.exp(ttnn.subtract(ref, G))
    e_G = ttnn.exp(G)
    k_end = ttnn.multiply(k, ttnn.exp(ttnn.subtract(G_last, G)))
    kb = ttnn.multiply(k, beta)
    vb = ttnn.multiply(v, beta)
    k_neg = ttnn.multiply(k, e_neg)

    # strictly-lower A (rows scaled by beta) and inclusive-lower intra-chunk scores
    A = _mm(ctx, ttnn.multiply(kb, e_pos), k_neg, transpose_b=True)
    A = ttnn.where(consts.strict, A, 0.0)
    intra = _mm(ctx, ttnn.multiply(q, e_pos), k_neg, transpose_b=True)
    intra = ttnn.where(consts.incl, intra, 0.0)
    for t in (e_pos, e_neg, k_neg):
        ttnn.deallocate(t)

    # Tinv = (I + A)^-1 by block doubling: X holds the inverses of the b x b diagonal blocks, and with L_b the
    # sub-diagonal b x b blocks inside each 2b x 2b block, (D + L_b)^-1 = X - X L_b X. Every X is a block of the true
    # inverse, so nothing grows (unlike the Neumann product, whose A^8 / A^16 cancel catastrophically when |A| ~ 1).
    tinv = None
    for lvl in consts.levels:
        lb = ttnn.multiply(A, lvl)
        if tinv is None:
            tinv = ttnn.subtract(consts.eye, lb)
        else:
            tinv = ttnn.subtract(tinv, _mm(ctx, tinv, _mm(ctx, lb, tinv)))
    W = _mm(ctx, tinv, ttnn.multiply(kb, e_G))
    Uv = _mm(ctx, tinv, vb)
    qd = ttnn.multiply(q, e_G)

    # chunk affine maps S -> M S + c
    decay_col = ttnn.transpose(ttnn.exp(G_last), -2, -1)  # [h, N, d, 1]
    M = ttnn.subtract(ttnn.multiply(consts.eye_k(n, d), decay_col), _mm(ctx, k_end, W, transpose_a=True))
    cvec = _mm(ctx, k_end, Uv, transpose_a=True)

    # inclusive scan over chunks: (M, c)[j] <- (M[j] M[j-s], M[j] c[j-s] + c[j])
    s = 1
    while s < n:
        m_hi = ttnn.slice(M, (0, s, 0, 0), (h, n, d, d))
        m_lo = ttnn.slice(M, (0, 0, 0, 0), (h, n - s, d, d))
        c_hi = ttnn.slice(cvec, (0, s, 0, 0), (h, n, d, d))
        c_lo = ttnn.slice(cvec, (0, 0, 0, 0), (h, n - s, d, d))
        new_m = _mm(ctx, m_hi, m_lo)
        new_c = ttnn.add(_mm(ctx, m_hi, c_lo), c_hi)
        M = ttnn.concat([ttnn.slice(M, (0, 0, 0, 0), (h, s, d, d)), new_m], dim=1)
        cvec = ttnn.concat([ttnn.slice(cvec, (0, 0, 0, 0), (h, s, d, d)), new_c], dim=1)
        s *= 2

    # chunk exit states S_j = Mcum_j s0 + ccum_j; entry states shift by one chunk
    s_exit = _mm(ctx, ttnn.reshape(M, (h, n * d, d)), s0)
    s_exit = ttnn.add(ttnn.reshape(s_exit, (h, n, d, d)), cvec)
    s_final = ttnn.reshape(ttnn.slice(s_exit, (0, n - 1, 0, 0), (h, n, d, d)), (h, d, d))
    s0_4d = ttnn.reshape(s0, (h, 1, d, d))
    s_in = s0_4d if n == 1 else ttnn.concat([s0_4d, ttnn.slice(s_exit, (0, 0, 0, 0), (h, n - 1, d, d))], dim=1)

    U = ttnn.subtract(Uv, _mm(ctx, W, s_in))
    o = ttnn.add(_mm(ctx, qd, s_in), _mm(ctx, intra, U))
    return o, s_final


class KDAPrefill:
    """One KDA sublayer on a stage (TP over heads)."""

    def __init__(self, ctx: TPContext, cfg, src, load: WeightLoader):
        self.ctx = ctx
        tp = ctx.tp
        self.H = cfg.linear_num_heads
        self.d = cfg.linear_head_dim
        assert self.H % tp == 0
        self.h = self.H // tp  # heads per chip
        self.W = self.h * self.d  # q / k / v channels per chip
        self.K = cfg.linear_conv_kernel_dim
        self.lower = float(cfg.linear_lower_bound)
        self.eps = cfg.rms_norm_eps
        D, H, d, h, W = cfg.hidden_size, self.H, self.d, self.h, self.W
        self.b_pad = 32  # b_proj columns per chip, padded to a tile
        sa = lambda n: src[f"self_attn.{n}"]

        def by_chip(x, dim):  # [.., C, ..] -> chunks of C / tp per chip
            return x.chunk(tp, dim=dim)

        def w_qkv():
            q, k, v = (by_chip(sa(f"{p}_proj.weight").t(), 1) for p in "qkv")
            return torch.cat([torch.cat([q[i], k[i], v[i]], 1) for i in range(tp)], 1)

        def taps():
            q, k, v = (by_chip(sa(f"{p}_conv1d.weight")[:, 0, :].t().float(), 1) for p in "qkv")  # [K, 8192]
            return torch.cat([torch.cat([q[i], k[i], v[i]], 1) for i in range(tp)], 1).reshape(1, 1, self.K, -1)

        def w_low():
            fa, ga = sa("f_a_proj.weight").t(), sa("g_a_proj.weight").t()
            b = by_chip(sa("b_proj.weight").t(), 1)
            pad = torch.zeros(D, self.b_pad - h, dtype=fa.dtype)
            return torch.cat([torch.cat([fa, ga, b[i], pad], 1) for i in range(tp)], 1)

        self.w_qkv = load("w_qkv", w_qkv, dtype=ttnn.bfloat8_b, shard_dim=-1)
        self.taps = load("conv_taps", taps, dtype=ttnn.float32, shard_dim=-1)
        self.w_low = load("w_low", w_low, dtype=ttnn.bfloat16, shard_dim=-1)
        self.w_fb = load("w_fb", lambda: sa("f_b_proj.weight").t(), dtype=ttnn.bfloat16, shard_dim=-1)
        self.w_gb = load("w_gb", lambda: sa("g_b_proj.weight").t(), dtype=ttnn.bfloat16, shard_dim=-1)
        self.dt_bias = load(
            "dt_bias", lambda: sa("dt_bias").float().reshape(1, 1, 1, -1), dtype=ttnn.float32, shard_dim=-1
        )
        self.decay_rate = load(
            "decay_rate",
            lambda: sa("A_log").float().exp().repeat_interleave(d).reshape(1, 1, 1, -1),
            dtype=ttnn.float32,
            shard_dim=-1,
        )
        self.o_norm = load(
            "o_norm", lambda: sa("o_norm.weight").float().repeat(h).reshape(1, 1, 1, -1), dtype=ttnn.float32
        )
        self.w_o = load("w_o", lambda: sa("o_proj.weight").t(), dtype=ttnn.bfloat8_b, shard_dim=0)
        self.consts = _Consts(ctx, h)

    def new_state(self) -> KDAState:
        ctx = self.ctx
        conv = ttnn.from_torch(
            torch.zeros(1, 1, self.K - 1, 3 * self.W),
            dtype=ttnn.bfloat16,
            layout=RM,
            device=ctx.device,
            memory_config=DRAM,
            mesh_mapper=ctx.replicate(),
        )
        state = ctx.upload(torch.zeros(self.h, self.d, self.d), ttnn.float32)
        return KDAState(conv=conv, state=state)

    def _heads(self, x, n):
        """[1, 1, T, h d] -> [h, N, 32, d]."""
        t = x.shape[-2]
        x = ttnn.reshape(x, (1, t, self.h, self.d))
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (self.h, n, CHUNK, self.d))

    def _unheads(self, x, t):
        x = ttnn.reshape(x, (1, self.h, t, self.d))
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (1, 1, t, self.W))

    def _l2(self, x):
        ss = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        return ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ss, 1e-6)))

    def forward(self, x: ttnn.Tensor, state: KDAState, n_valid: int, valid: ttnn.Tensor):
        """x: normalized sublayer input, bf16 TILE ``[1, 1, T, D]`` (T a multiple of 32, rows >= n_valid padding);
        valid: fp32 TILE ``[1, 1, T, 1]`` (1 on real rows). Returns (bf16 TILE ``[1, 1, T, D]`` replicated, new state).
        """
        ctx = self.ctx
        t = x.shape[-2]
        n = t // CHUNK
        K, W = self.K, self.W

        # q/k/v projection + depthwise causal conv (fp32, conv weight is a strict-fp32 parameter) + SiLU
        qkv = ttnn.matmul(x, self.w_qkv, dtype=ttnn.bfloat16, compute_kernel_config=ctx.hifi2, memory_config=DRAM)
        ext = ttnn.concat([state.conv, to_rm(qkv)], dim=2)  # [1, 1, T + K - 1, 3W] RM
        ttnn.deallocate(qkv)
        acc = None
        for j in range(K):
            rows = to_tile(ttnn.slice(ext, (0, 0, j, 0), (1, 1, j + t, 3 * W)), ttnn.float32)
            tap = ttnn.slice(self.taps, (0, 0, j, 0), (1, 1, j + 1, 3 * W))
            acc = ttnn.multiply(rows, tap) if acc is None else ttnn.addcmul(acc, rows, tap)
        new_conv = ttnn.slice(ext, (0, 0, n_valid, 0), (1, 1, n_valid + K - 1, 3 * W))
        ttnn.deallocate(ext)
        qkv = ttnn.silu(acc)
        q = ttnn.slice(qkv, (0, 0, 0, 0), (1, 1, t, W))
        k = ttnn.slice(qkv, (0, 0, 0, W), (1, 1, t, 2 * W))
        v = ttnn.slice(qkv, (0, 0, 0, 2 * W), (1, 1, t, 3 * W))
        ttnn.deallocate(qkv)

        # gates: [f_a | g_a | b_local] low-rank projection
        low = ttnn.matmul(x, self.w_low, dtype=ttnn.bfloat16, compute_kernel_config=ctx.hifi4, memory_config=DRAM)
        d = self.d
        fa = ttnn.slice(low, (0, 0, 0, 0), (1, 1, t, d))
        ga = ttnn.slice(low, (0, 0, 0, d), (1, 1, t, 2 * d))
        bl = ttnn.slice(low, (0, 0, 0, 2 * d), (1, 1, t, 2 * d + self.h))
        f = ttnn.matmul(fa, self.w_fb, dtype=ttnn.bfloat16, compute_kernel_config=ctx.hifi4, memory_config=DRAM)
        z = ttnn.multiply(ttnn.add(ttnn.typecast(f, ttnn.float32), self.dt_bias), self.decay_rate)
        g = ttnn.multiply(ttnn.multiply(ttnn.sigmoid(z), self.lower), valid)  # padded rows: no decay
        beta = ttnn.multiply(ttnn.sigmoid(ttnn.typecast(bl, ttnn.float32)), valid)  # padded rows: no write
        beta = ttnn.repeat_interleave(beta, d, dim=3)
        gate = ttnn.matmul(ga, self.w_gb, dtype=ttnn.bfloat16, compute_kernel_config=ctx.hifi4, memory_config=DRAM)

        qh = ttnn.multiply(self._l2(self._heads(q, n)), d**-0.5)
        kh = self._l2(self._heads(k, n))
        vh = self._heads(v, n)
        gh = self._heads(g, n)
        bh = self._heads(beta, n)
        o, s_new = kda_core(ctx, self.consts, qh, kh, vh, gh, bh, state.state)

        # gated RMSNorm over each head, then the row-parallel o_proj
        ss = ttnn.mean(ttnn.multiply(o, o), dim=-1, keepdim=True)
        o = ttnn.multiply(o, ttnn.rsqrt(ttnn.add(ss, self.eps)))
        o = self._unheads(o, t)
        o = ttnn.multiply(ttnn.multiply(o, self.o_norm), ttnn.sigmoid(ttnn.typecast(gate, ttnn.float32)))
        o = ttnn.typecast(o, ttnn.bfloat16)
        y = ttnn.matmul(o, self.w_o, dtype=ttnn.bfloat16, compute_kernel_config=ctx.hifi2, memory_config=DRAM)
        y = ctx.all_reduce(y)
        return y, KDAState(conv=new_conv, state=s_new)
