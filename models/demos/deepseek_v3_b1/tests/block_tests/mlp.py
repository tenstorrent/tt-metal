"""MLP (SwiGLU) — reference implementation."""

import torch.nn.functional as F


def mlp_torch(
    hidden_states,
    *,
    w_gate,
    w_up,
    w_down,
):
    """
    SwiGLU MLP applied to all tokens.

    hidden_states: [b, s, h]
    w_gate: [intermediate, h]
    w_up:   [intermediate, h]
    w_down: [h, intermediate]
    """
    gate = F.silu(F.linear(hidden_states, w_gate))
    up = F.linear(hidden_states, w_up)
    return F.linear(gate * up, w_down)


mlp_tt = mlp_torch
