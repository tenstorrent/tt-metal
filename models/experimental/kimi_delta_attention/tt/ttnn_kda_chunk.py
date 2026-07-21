# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Chunked KDA prefill op (perf: replaces the token-by-token recurrence).
#
# Port of torch_functional/kda_ops.py::naive_chunk_kda to ttnn. Intra-chunk work (cumsum, the A matrix,
# its inverse, w/u) is fully BATCHED over chunks via matmuls; only the cross-chunk state scan is a short
# NT-iteration loop. This collapses the ~T-iteration token loop (~9000 tiny kernels/forward) to ~NT
# chunk-iterations (~100 kernels), removing the launch-overhead that dominated the recurrent path
# (see ../bringup_log.md perf section, ../ROOFLINE.md).
#
# Intra-chunk decay factoring (matmul-friendly): A[c,i] = sum_d k_c·exp(g_c)·k_i·exp(-g_i)
#   => A = (k⊙exp(g)) @ (k⊙exp(-g))ᵀ. The A-inverse (I−L)⁻¹ (L strict-lower, nilpotent) is computed by
# Neumann doubling: ∏_i (I + L^{2^i}), ⌈log2 C⌉ steps of matmuls instead of a C-length substitution loop.

from __future__ import annotations

import math

import torch
import ttnn
from einops import rearrange

# HiFi2 (2 phases) ~2x the matmul throughput of HiFi4; fp32 accumulate keeps the scan/inverse stable.
_MM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _mask(md, C, kind):
    """[1,1,1,C,C] fp32 mask: 'tril_incl' (c>=i), 'strict_lower' (c>i), 'lower_incl' (c>=i), 'eye'."""
    r = torch.arange(C)
    if kind == "eye":
        m = torch.eye(C)
    elif kind in ("tril_incl", "lower_incl"):
        m = (r[:, None] >= r[None, :]).float()
    elif kind == "strict_lower":
        m = (r[:, None] > r[None, :]).float()
    return ttnn.from_torch(m.reshape(1, 1, 1, C, C), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=md)


def _mm(a, b, transpose_b=False):
    if transpose_b:
        b = ttnn.transpose(b, -2, -1)
    return ttnn.matmul(a, b, compute_kernel_config=_MM)


def chunk_kda_ttnn(q, k, v, g, beta, scale=None, initial_state=None, device=None, chunk_size=64):
    """Chunked KDA prefill on device. Same contract as recurrent_kda_ttnn (q,k,v,g,beta already
    L2-normed / gated / sigmoided). q,k:[B,T,HV,K] v:[B,T,HV,V] g:[B,T,HV,K] beta:[B,T,HV].
    Returns (o [B,T,HV,V], S [B,HV,K,V]). Requires T % chunk_size == 0."""
    B, T, HV, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C
    assert T % C == 0, f"T={T} must be divisible by chunk_size={C}"
    if scale is None:
        scale = K ** -0.5

    tril = _mask(device, C, "tril_incl")
    slow = _mask(device, C, "strict_lower")
    linc = _mask(device, C, "lower_incl")
    eye = _mask(device, C, "eye")

    def chunks(x, D):  # [B,T,HV,D] -> [B,HV,NT,C,D]
        x = ttnn.permute(x, [0, 2, 1, 3])            # [B,HV,T,D]
        return ttnn.reshape(x, [B, HV, NT, C, D])

    q = ttnn.multiply(chunks(q, K), scale)
    k = chunks(k, K)
    v = chunks(v, V)
    g = ttnn.matmul(tril, chunks(g, K), compute_kernel_config=_MM)  # cumsum within chunk (per channel)
    beta = chunks(ttnn.reshape(beta, [B, T, HV, 1]), 1)  # [B,T,HV] -> [B,HV,NT,C,1] (same permute as q/k/v)

    eg = ttnn.exp(g)
    eng = ttnn.exp(ttnn.neg(g))

    # A = (k*eg) @ (k*eng)^T, row-scaled by beta, strict-lower, negated  -> L
    KK = _mm(ttnn.multiply(k, eg), ttnn.multiply(k, eng), transpose_b=True)  # [.,C,C]
    L = ttnn.neg(ttnn.multiply(ttnn.multiply(KK, beta), slow))               # strict-lower, negated

    # Ainv = (I - L)^-1 via Neumann doubling; Awy = Ainv * beta_col
    R = ttnn.add(eye, L)
    Lp = L
    for _ in range(1, max(1, math.ceil(math.log2(C)))):
        Lp = _mm(Lp, Lp)
        R = _mm(R, ttnn.add(eye, Lp))
    beta_col = ttnn.transpose(beta, -2, -1)  # [.,1,C]
    Awy = ttnn.multiply(R, beta_col)

    w = _mm(Awy, ttnn.multiply(eg, k))  # [.,C,K]
    u = _mm(Awy, v)                     # [.,C,V]

    # cross-chunk scan
    if initial_state is not None:
        S = ttnn.to_layout(ttnn.typecast(initial_state, ttnn.float32), ttnn.TILE_LAYOUT)
    else:
        S = ttnn.zeros([B, HV, K, V], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    outs = []
    for n in range(NT):
        qn = q[:, :, n]; kn = k[:, :, n]; un = u[:, :, n]; wn = w[:, :, n]  # [B,HV,C,*]
        egn = eg[:, :, n]; gn = g[:, :, n]
        Aqk = ttnn.multiply(_mm(ttnn.multiply(qn, egn), ttnn.multiply(kn, eng[:, :, n]), transpose_b=True),
                            ttnn.reshape(linc, [1, 1, C, C]))
        v_new = ttnn.subtract(un, _mm(wn, S))                # [.,C,V]
        o_n = ttnn.add(_mm(ttnn.multiply(qn, egn), S), _mm(Aqk, v_new))  # [.,C,V]
        outs.append(ttnn.reshape(o_n, [B, HV, 1, C, V]))
        # state update: S = S*exp(g_last) + (exp(g_last - g)*k)^T @ v_new
        g_last = ttnn.reshape(gn[:, :, C - 1], [B, HV, 1, K])             # [.,1,K]
        decayk = ttnn.multiply(ttnn.exp(ttnn.subtract(g_last, gn)), kn)   # [.,C,K]
        # per-K row scale of S[.,K,V]: transpose to [.,V,K] so exp(g_last)[.,1,K] broadcasts on dim -2
        # (ttnn supports row broadcast but not the last-dim/"subtile" column broadcast).
        St = ttnn.multiply(ttnn.transpose(S, -2, -1), ttnn.exp(g_last))   # [.,V,K]
        # (decayk^T @ v_new): transpose the FIRST arg -> [.,K,C]@[.,C,V] = [.,K,V]
        S = ttnn.add(ttnn.transpose(St, -2, -1), _mm(ttnn.transpose(decayk, -2, -1), v_new))  # [.,K,V]

    o = outs[0] if NT == 1 else ttnn.concat(outs, dim=2)   # [B,HV,NT,C,V]
    o = ttnn.reshape(o, [B, HV, T, V])
    o = ttnn.permute(o, [0, 2, 1, 3])                       # [B,T,HV,V]
    return o, S
