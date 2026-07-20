# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# KDA (Kimi Delta Attention) torch reference ops.
#
# Faithful port of the authoritative reference in flash-linear-attention
# (fla/ops/kda/naive.py, fla/ops/kda/gate.py; MIT, Songlin Yang et al.) — the numerical ground
# truth for the tt-metal KDA bringup. The delta versus Gated DeltaNet is the *diagonal* (per-channel)
# forget gate `g` of shape [B, T, HV, K] in log space, vs GDN's scalar per-head gate [B, T, HV].
# See ../API_SPEC.md and ../bringup_log.md.

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize over the last dim: x * rsqrt(sum(x^2) + eps)."""
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """KDA log-space decay gate.

    Args:
        g: pre-activation gate, shape ``[..., H, K]`` (output of the low-rank ``f_proj``).
        A_log: per-head parameter, shape ``[H]``.
        dt_bias: optional bias added before activation, shape ``[H*K]``.
        lower_bound: if set, use the clamped sigmoid form instead of ``-exp(A_log)*softplus``.

    Returns:
        Log-space decay ``g`` of shape ``[..., H, K]`` (values ≤ 0).
    """
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    if lower_bound is None:
        g = -A_log.view(H, 1).float().exp() * F.softplus(g)
    else:
        g = lower_bound * torch.sigmoid(A_log.view(H, 1).exp() * g)
    return g.to(output_dtype)


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """Token-by-token KDA recurrence (the ground-truth definition).

    Shapes: ``q,k [B,T,H,K]``, ``v [B,T,HV,V]``, ``g [B,T,HV,K]`` (log-space),
    ``beta [B,T,HV]``, ``initial_state [B,HV,K,V]``.

    Recurrence per step (diagonal decay + delta rule):
        S = Diag(exp(g_t)) @ S
        S = S + (beta_t * k_t) ⊗ (v_t - k_tᵀ S)
        o_t = q_tᵀ S
    Returns ``(o [B,T,HV,V], S | None)``.
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = (x.to(torch.float) for x in (q, k, v, g, beta))
    q = q.repeat_interleave(G, dim=2) * scale  # [B,T,HV,K]
    k = k.repeat_interleave(G, dim=2)          # [B,T,HV,K]

    S = k.new_zeros(B, HV, K, V)
    if initial_state is not None:
        S = S + initial_state.to(torch.float)
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()  # diagonal decay along K, broadcast over V
        S = S + torch.einsum("b h k, b h v -> b h k v", b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum("b h k, b h k v -> b h v", q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    """Chunkwise-parallel KDA (equivalent to the recurrence; used for prefill reference).

    Same shapes/semantics as :func:`naive_recurrent_kda`. Requires ``T % chunk_size == 0``.
    Port of fla ``naive_chunk_kda``.
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5
    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"

    q, k = (rearrange(x, "b (n c) h ... -> b h n c ...", c=BT).to(torch.float) for x in (q, k))
    v, g, beta = (rearrange(x, "b (n c) h ... -> b h n c ...", c=BT).to(torch.float) for x in (v, g, beta))
    q = q.repeat_interleave(G, dim=1) * scale  # [B,HV,NT,BT,K]
    k = k.repeat_interleave(G, dim=1)
    g = g.cumsum(-2)  # cumulative log-decay within chunk (per channel)

    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)

    A = torch.zeros(*g.shape[:-1], BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i + 1, :]
        A[..., i] = torch.einsum("... c d, ... d -> ... c", k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]
    A = -A.masked_fill(mask, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = k.new_zeros(B, HV, K, V)
    if initial_state is not None:
        S = S + initial_state.to(torch.float)
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(NT):
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        Aqk = torch.zeros(B, HV, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j + 1, :]
            Aqk[..., j] = torch.einsum("... c d, ... d -> ... c", q_i * (g_i - g_j).exp(), k_j)
        Aqk = Aqk.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), "b h k -> b h k 1")
        S = S + rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, "b h c k -> b h k c") @ v_i
    if not output_final_state:
        S = None
    return rearrange(o, "b h n c d -> b (n c) h d").to(dtype), S
