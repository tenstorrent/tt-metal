# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunked (WY-form) Kimi Delta Attention with PER-CHANNEL log decay, in plain torch / fp32.

Semantics per token t (matches ``tt/kda/decode_step.py`` and transformers' ``recurrent_kimi_delta_attention``):
    q,k <- l2norm; q *= K**-0.5; S <- S * exp(g_t)[:, None]; delta = beta_t * (v_t - k_t^T S); S += k_t (x) delta; o_t = q_t^T S

Chunk of C tokens with cumulative gate G_i = sum_{s<=i} g_s (per key channel), centre c = G_C / 2 (range safety):
    Kl = K * exp(G - c)          Kr = K * exp(c - G)          Q~ = Q * exp(G - c)
    A  = strict_lower(beta * Kl @ Kr^T)    T = (I + A)^-1  (exact: A is nilpotent -> Neumann doubling)
    U  = T (beta * V)            W  = T (beta * Kl)          S0c = S0 * exp(c)[:, None]
    Delta = U - W @ S0c
    O  = Q~ @ S0c + lower_incl(Q~ @ Kr^T) @ Delta
    S1 = exp(2c) * S0 + exp(c)[:, None] * (Kr^T @ Delta)
Every matmul only mixes quantities of comparable magnitude (|G - c| <= C*|g|max/2), and the state decay exp(G_C) is an
exact elementwise fp32 factor, so no error compounds across chunks. Written to be ported op-by-op to ttnn.
"""
from __future__ import annotations

import torch


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def kda_recurrent_reference(q, k, v, g, beta, S0, scale=None):
    """q,k [H,T,K]; v [H,T,V]; g [H,T,K] (log decay <= 0); beta [H,T]; S0 [H,K,V]. Returns (o [H,T,V], S [H,K,V]). fp64 math."""
    q, k, v, g, beta, S = (t.double() for t in (q, k, v, g, beta, S0))
    H, T, K = q.shape
    scale = K**-0.5 if scale is None else scale
    q = l2norm(q) * scale
    k = l2norm(k)
    outs = []
    for t in range(T):
        S = S * torch.exp(g[:, t])[:, :, None]
        kt = k[:, t]  # [H,K]
        delta = beta[:, t][:, None] * (v[:, t] - torch.einsum("hk,hkv->hv", kt, S))
        S = S + kt[:, :, None] * delta[:, None, :]
        outs.append(torch.einsum("hk,hkv->hv", q[:, t], S))
    return torch.stack(outs, 1), S


def neumann_inverse(A: torch.Tensor, C: int) -> torch.Tensor:
    """(I + A)^-1 for strictly lower-triangular A [.., C, C] via Neumann doubling (exact for nilpotent A up to rounding)."""
    eye = torch.eye(C, dtype=A.dtype, device=A.device)
    N = -A
    P = eye + N
    steps = 1
    while steps < C:
        N = N @ N
        P = P @ (eye + N)
        steps *= 2
    return P


def kda_chunked_reference(q, k, v, g, beta, S0, *, chunk: int = 32, scale=None, dtype=torch.float32):
    """Same contract as kda_recurrent_reference, computed chunk by chunk in ``dtype`` (fp32 by default)."""
    q, k, v, g, beta, S = (t.to(dtype) for t in (q, k, v, g, beta, S0))
    H, T, K = q.shape
    V = v.shape[-1]
    assert T % chunk == 0, (T, chunk)
    C = chunk
    scale = K**-0.5 if scale is None else scale
    q = l2norm(q) * scale
    k = l2norm(k)
    L_incl = torch.tril(torch.ones(C, C, dtype=dtype, device=q.device))  # j <= i
    L_strict = torch.tril(torch.ones(C, C, dtype=dtype, device=q.device), diagonal=-1)  # j < i
    outs = []
    for n in range(T // C):
        sl = slice(n * C, (n + 1) * C)
        qc, kc, vc, gc, bc = q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl][:, :, None]  # bc [H,C,1]
        G = L_incl @ gc  # [H,C,K] inclusive cumsum over the chunk
        c = 0.5 * G[:, C - 1 : C, :]  # [H,1,K] centre
        eGc = torch.exp(G - c)  # [H,C,K]
        enGc = torch.exp(c - G)
        Kl = kc * eGc
        Kr = kc * enGc
        Qt = qc * eGc
        bKl = bc * Kl
        KrT = Kr.transpose(1, 2)  # [H,K,C]
        A = L_strict * (bKl @ KrT)  # [H,C,C]
        Tinv = neumann_inverse(A, C)
        U = Tinv @ (bc * vc)  # [H,C,V]
        W = Tinv @ bKl  # [H,C,K]
        ec = torch.exp(c).transpose(1, 2)  # [H,K,1]
        S0c = S * ec  # rows scaled by exp(c)
        Delta = U - W @ S0c  # [H,C,V]
        Aqk = L_incl * (Qt @ KrT)
        O = Qt @ S0c + Aqk @ Delta
        outs.append(O)
        S = (ec * ec) * S + ec * (KrT @ Delta)
    return torch.cat(outs, 1), S


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


if __name__ == "__main__":  # CPU self-test with Kimi-like gate statistics
    import sys
    import time

    torch.manual_seed(0)
    H, K, V, T = 8, 128, 128, int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    q = torch.randn(H, T, K)
    k = torch.randn(H, T, K)
    v = torch.randn(H, T, V)
    # per-channel decays: a mixture of very slow (|g| ~ 1e-3), medium and fast (clamped at -5) channels, like layer 0
    A_log = torch.cat(
        [torch.full((K // 3,), -6.0), torch.full((K // 3,), -1.0), torch.full((K - 2 * (K // 3),), 3.0)]
    ).repeat(H, 1)
    sp = torch.nn.functional.softplus(torch.randn(H, T, K) * 0.5 + 0.2)
    g = torch.clamp(-torch.exp(A_log)[:, None, :] * sp, min=-5.0)
    beta = torch.sigmoid(torch.randn(H, T))
    S0 = torch.randn(H, K, V) * 0.1
    t0 = time.time()
    o_ref, S_ref = kda_recurrent_reference(q, k, v, g, beta, S0)
    t_ref = time.time() - t0
    for C in (32, 16):
        t0 = time.time()
        o, S = kda_chunked_reference(q, k, v, g, beta, S0, chunk=C)
        t_c = time.time() - t0
        print(
            f"C={C}: out pcc {_pcc(o_ref, o):.7f} (last 32: {_pcc(o_ref[:, -32:], o[:, -32:]):.7f}) max|err| {(o_ref - o).abs().max():.3e} "
            f"state pcc {_pcc(S_ref, S):.7f} max|err| {(S_ref - S).abs().max():.3e}  [{t_c:.1f}s vs recurrent {t_ref:.1f}s]"
        )
        # continuity across two half-length calls (state carry) must equal one call
        o1, S1 = kda_chunked_reference(
            q[:, : T // 2], k[:, : T // 2], v[:, : T // 2], g[:, : T // 2], beta[:, : T // 2], S0, chunk=C
        )
        o2, S2 = kda_chunked_reference(
            q[:, T // 2 :], k[:, T // 2 :], v[:, T // 2 :], g[:, T // 2 :], beta[:, T // 2 :], S1, chunk=C
        )
        print(f"      split carry: out pcc {_pcc(o, torch.cat([o1, o2], 1)):.7f} state pcc {_pcc(S, S2):.7f}")
