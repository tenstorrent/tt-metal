# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""float64 references for the ttml chunked gated delta rule tests.

All functions take per-head layouts q, k [B, T, H, K], v [B, T, H, V], g, beta [B, T, H] and
initial_state [B, H, K, V]; callers repeat the key heads for GVA.

`chunked_forward` mirrors the op's chunked algorithm and returns every intermediate the backward
consumes.  `recurrent_forward` is the token-by-token definition it agrees with to ~1e-15, and
`recurrent_vjp` differentiates that definition with torch autograd, so the gradient oracle does not
share the chunked algorithm the kernels implement.
"""

from __future__ import annotations

import math

import torch


def chunked_forward(q, k, v, g, beta, *, initial_state=None, chunk_size=64, scale=None):
    """Chunked forward plus intermediates; differentiable.

    Returns, with NC = T / C:
        o [B, T, H, V], final_state [B, H, K, V], h [B, NC, H, K, V] (state entering chunk i),
        v_new [B, T, H, V], g_cumsum [B, T, H] (chunk-local inclusive cumsum of g),
        A [B, T, H, C] (row r of chunk i's UT inverse (I - A_strict)^-1).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = int(chunk_size)
    assert T % C == 0, f"T={T} must be a multiple of chunk_size={C}"
    NC = T // C
    scale = K**-0.5 if scale is None else scale
    f64 = torch.float64

    q, k, v = (x.to(f64).transpose(1, 2) for x in (q, k, v))
    g, beta = (x.to(f64).transpose(1, 2) for x in (g, beta))
    q = q * scale
    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]

    chunks = lambda x: x.reshape(B, H, NC, C, x.shape[-1])
    q_c, k_c, kb_c, vb_c = (chunks(x) for x in (q, k, k_beta, v_beta))
    decay = g.reshape(B, H, NC, C).cumsum(-1)

    incl = torch.tril(torch.ones(C, C, dtype=torch.bool))
    strict = torch.tril(torch.ones(C, C, dtype=torch.bool), diagonal=-1)
    diff = decay[..., :, None] - decay[..., None, :]
    L = torch.where(incl, diff, torch.full_like(diff, -math.inf)).exp()

    A_strict = (-(kb_c @ k_c.transpose(-1, -2)) * L).masked_fill(~strict, 0.0)
    eye = torch.eye(C, dtype=f64)
    t_inv = torch.linalg.solve_triangular(eye - A_strict, eye.expand_as(A_strict), upper=False, unitriangular=True)
    v_corr = t_inv @ vb_c
    k_cumdecay = t_inv @ (kb_c * decay.exp()[..., None])

    S = torch.zeros(B, H, K, V, dtype=f64) if initial_state is None else initial_state.to(f64)
    hs, os, vns = [], [], []
    for i in range(NC):
        hs.append(S)
        q_i, k_i, d_i, L_i = q_c[:, :, i], k_c[:, :, i], decay[:, :, i], L[:, :, i]
        vn = v_corr[:, :, i] - k_cumdecay[:, :, i] @ S
        os.append((q_i * d_i.exp()[..., None]) @ S + ((q_i @ k_i.transpose(-1, -2)) * L_i) @ vn)
        vns.append(vn)
        d_last = d_i[..., -1]
        S = (
            S * d_last.exp()[..., None, None]
            + (k_i * (d_last[..., None] - d_i).exp()[..., None]).transpose(-1, -2) @ vn
        )

    tokens = lambda x: x.reshape(B, H, T, x.shape[-1]).transpose(1, 2)  # [B, H, NC, C, D] -> [B, T, H, D]
    o = tokens(torch.stack(os, dim=2))
    v_new = tokens(torch.stack(vns, dim=2))
    h = torch.stack(hs, dim=1)  # [B, NC, H, K, V]
    g_cumsum = decay.reshape(B, H, T).transpose(1, 2)
    return o, S, h, v_new, g_cumsum, tokens(t_inv)


def recurrent_forward(q, k, v, g, beta, *, initial_state=None, scale=None):
    """Token-by-token definition; returns (o [B, T, H, V], final_state [B, H, K, V]).

    S_t = exp(g_t) S_{t-1} + k_t u_t^T,  u_t = beta_t (v_t - (exp(g_t) S_{t-1})^T k_t),  o_t = S_t^T (scale q_t).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K**-0.5 if scale is None else scale
    S = torch.zeros(B, H, K, V, dtype=torch.float64) if initial_state is None else initial_state
    outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        k_t = k[:, t]
        u = beta[:, t][..., None] * (v[:, t] - torch.einsum("bhkv,bhk->bhv", S, k_t))
        S = S + k_t[..., :, None] * u[..., None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t] * scale))
    return torch.stack(outs, dim=1), S


def recurrent_vjp(q, k, v, g, beta, d_out, *, d_final_state=None, initial_state=None, scale=None):
    """Gradients of <o, d_out> + <final_state, d_final_state> through `recurrent_forward`.

    Returns (dq, dk, dv, dg, dbeta, dh0); dh0 is None when initial_state is.
    """
    leaves = [x.detach().to(torch.float64).requires_grad_(True) for x in (q, k, v, g, beta)]
    h0 = None if initial_state is None else initial_state.detach().to(torch.float64).requires_grad_(True)
    o, final_state = recurrent_forward(*leaves, initial_state=h0, scale=scale)
    loss = (o * d_out.to(torch.float64)).sum()
    if d_final_state is not None:
        loss = loss + (final_state * d_final_state.to(torch.float64)).sum()
    wrt = leaves + ([h0] if h0 is not None else [])
    grads = torch.autograd.grad(loss, wrt)
    return tuple(grads[:5]) + ((grads[5] if h0 is not None else None),)


def assert_close(name, got, exp, *, pcc_min, rms_max=None):
    """Finite, PCC >= pcc_min and (optionally) RMS error relative to the reference RMS <= rms_max."""
    got = torch.as_tensor(got).double().reshape(exp.shape)
    exp = exp.double()
    assert torch.isfinite(got).all(), f"{name}: non-finite values"
    a, e = got.flatten(), exp.flatten()
    if e.std() == 0 or a.std() == 0:
        assert torch.allclose(
            a, e, atol=1e-6
        ), f"{name}: constant tensor mismatch, max |diff| {(a - e).abs().max():.3e}"
        return
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item()
    assert pcc >= pcc_min, f"{name}: PCC {pcc:.6f} < {pcc_min}"
    if rms_max is not None:
        rms = ((a - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt().clamp_min(1e-30)).item()
        assert rms <= rms_max, f"{name}: relative RMS {rms:.4g} > {rms_max}"
