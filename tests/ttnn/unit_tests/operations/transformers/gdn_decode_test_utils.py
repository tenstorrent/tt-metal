# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test utilities for the fused GDN decode recurrence (ttnn.transformer.fused_recurrent_gated_delta_rule).

The reference semantics live in the op's registered golden (ttnn.get_golden_function(op), implemented in
ttnn/ttnn/operations/transformer_golden.py); this module holds what is not reference semantics: the model shapes, input
generation, the chained single-token helper, and a vendored copy of FLA's naive recurrence as the oracle the golden is
checked against."""

from dataclasses import dataclass

import torch

from ttnn.operations.transformer_golden import l2_norm


@dataclass(frozen=True)
class Shape:
    num_key_heads: int
    num_value_heads: int


# Per-device head counts of served configurations.
SHAPES = {
    "qwen27b_tp4": Shape(4, 12),  # Qwen3.6-27B TP 4
    "qwen9b_tp1": Shape(16, 32),  # Qwen3.5-9B single chip
}


def make_inputs(B, T, H, HV, K, V, *, seed, normalize_qk=True, with_state=True):
    """Recurrence inputs in the layer's post-conv convention: q, k, v ~ N(0, 1), beta ~ U(0, 1) (a sigmoid output),
    g = -2 U(0, 1) (log decay), state ~ 0.05 N(0, 1). With normalize_qk the q, k rows are L2-normalised (the golden
    module's l2_norm) as the current op expects them. Returns a dict of fp32 torch tensors, token-major [B, T, heads, D].
    """
    gen = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32)
    q, k = r(B, T, H, K), r(B, T, H, K)
    if normalize_qk:
        q, k = l2_norm(q), l2_norm(k)
    v = r(B, T, HV, V)
    beta = torch.rand(B, T, HV, generator=gen, dtype=torch.float32)
    g = -torch.rand(B, T, HV, generator=gen, dtype=torch.float32) * 2.0
    s0 = 0.05 * r(B, HV, K, V) if with_state else None
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "initial_state": s0}


def chained_decode(fn, q, k, v, g, beta, initial_state=None):
    """Run fn (op or golden, (q, k, v, g, beta, *, initial_state, output_final_state) -> (o, state)) one token at a
    time and stitch: o [B, T, HV, V], states [B, T, HV, K, V] (the state after each token)."""
    T = q.shape[1]
    state = initial_state
    outs, states = [], []
    for t in range(T):
        sl = slice(t, t + 1)
        o_t, state = fn(
            q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], initial_state=state, output_final_state=True
        )
        outs.append(o_t)
        states.append(state)
    return torch.cat(outs, dim=1), torch.stack(states, dim=1)


def fla_naive_recurrent_gated_delta_rule(q, k, v, beta, g, scale=None, initial_state=None, output_final_state=False):
    """Vendored copy of FLA's naive_recurrent_gated_delta_rule (flash-linear-attention,
    fla/ops/gated_delta_rule/naive.py, MIT license, (c) Songlin Yang et al.), FLA order (q, k, v, beta, g). The oracle
    of the golden; vendored because no tt-metal environment installs FLA."""
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h
