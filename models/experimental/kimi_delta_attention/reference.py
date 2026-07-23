# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Independent pure-torch specification of Kimi Delta Attention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.experimental.kimi_delta_attention.config import KDAConfig


@dataclass
class KDAReferenceState:
    """Reference cache; convolution histories exclude the current token."""

    recurrent: torch.Tensor
    q_convolution: torch.Tensor
    k_convolution: torch.Tensor
    v_convolution: torch.Tensor


def _require_weight(
    weights: Mapping[str, torch.Tensor],
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    try:
        weight = weights[name]
    except KeyError as error:
        raise ValueError(f"missing KDA weight: {name}") from error
    if tuple(weight.shape) != shape:
        raise ValueError(f"{name} shape {tuple(weight.shape)} != {shape}")
    return weight.float()


def validate_reference_weights(weights: Mapping[str, torch.Tensor], config: KDAConfig) -> None:
    """Validate every canonical weight before reference execution."""
    hidden = config.hidden_size
    q_dim, k_dim, v_dim = config.q_dim, config.k_dim, config.v_dim
    kernel = config.conv_kernel_size
    key_rank = config.head_k_dim
    value_rank = config.head_v_dim
    heads = config.num_heads

    _require_weight(weights, "q_proj.weight", (q_dim, hidden))
    _require_weight(weights, "k_proj.weight", (k_dim, hidden))
    _require_weight(weights, "v_proj.weight", (v_dim, hidden))
    _require_weight(weights, "q_conv1d.weight", (q_dim, 1, kernel))
    _require_weight(weights, "k_conv1d.weight", (k_dim, 1, kernel))
    _require_weight(weights, "v_conv1d.weight", (v_dim, 1, kernel))
    _require_weight(weights, "A_log", (1, 1, heads, 1))
    _require_weight(weights, "f_a_proj.weight", (key_rank, hidden))
    _require_weight(weights, "f_b_proj.weight", (heads * key_rank, key_rank))
    _require_weight(weights, "dt_bias", (heads * key_rank,))
    _require_weight(weights, "b_proj.weight", (heads, hidden))
    _require_weight(weights, "g_a_proj.weight", (value_rank, hidden))
    _require_weight(weights, "g_b_proj.weight", (heads * value_rank, value_rank))
    _require_weight(weights, "o_norm.weight", (value_rank,))
    _require_weight(weights, "o_proj.weight", (hidden, heads * value_rank))


def causal_depthwise_conv_reference(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise conv+SiLU with `[B,W-1,D]` history."""
    batch, _, channels = inputs.shape
    if weight.ndim != 3 or tuple(weight.shape[:2]) != (channels, 1):
        raise ValueError(f"convolution weight shape {tuple(weight.shape)} incompatible with D={channels}")
    kernel = weight.shape[-1]
    state_shape = (batch, kernel - 1, channels)
    if initial_state is None:
        history = inputs.new_zeros(state_shape)
    elif tuple(initial_state.shape) != state_shape:
        raise ValueError(f"convolution state shape {tuple(initial_state.shape)} != {state_shape}")
    else:
        history = initial_state

    window = torch.cat((history.float(), inputs.float()), dim=1)
    output = F.conv1d(
        window.transpose(1, 2),
        weight.float(),
        groups=channels,
    ).transpose(1, 2)
    final_state = window[:, -(kernel - 1) :] if kernel > 1 else window[:, :0]
    return F.silu(output), final_state


def kda_gate_reference(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Convert raw gate logits to negative per-key log decay."""
    heads, key_dim = raw_gate.shape[-2:]
    if a_log.numel() != heads:
        raise ValueError(f"A_log has {a_log.numel()} values, expected {heads}")
    if dt_bias.numel() != heads * key_dim:
        raise ValueError(f"dt_bias has {dt_bias.numel()} values, expected {heads * key_dim}")
    scale = a_log.float().reshape(1, 1, heads, 1).exp()
    bias = dt_bias.float().reshape(1, 1, heads, key_dim)
    return -scale * F.softplus(raw_gate.float() + bias)


def l2_norm_reference(inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match FLA's `x / sqrt(sum(x²) + eps)` normalization."""
    inputs = inputs.float()
    return inputs * torch.rsqrt(inputs.square().sum(dim=-1, keepdim=True) + eps)


def kda_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-ordered KDA recurrence in `[B,T,H,D]` layout."""
    batch, sequence, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    expected = {
        "k": (batch, sequence, heads, key_dim),
        "v": (batch, sequence, heads, value_dim),
        "gate": (batch, sequence, heads, key_dim),
        "beta": (batch, sequence, heads),
    }
    tensors = {"k": k, "v": v, "gate": gate, "beta": beta}
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensors[name].shape)} != {shape}")

    state_shape = (batch, heads, key_dim, value_dim)
    if initial_state is None:
        state = torch.zeros(state_shape, device=q.device, dtype=torch.float32)
    elif tuple(initial_state.shape) != state_shape:
        raise ValueError(f"recurrent state shape {tuple(initial_state.shape)} != {state_shape}")
    else:
        state = initial_state.float().clone()

    q = l2_norm_reference(q) * (key_dim**-0.5)
    k = l2_norm_reference(k)
    v, gate, beta = v.float(), gate.float(), beta.float()
    output = torch.empty(batch, sequence, heads, value_dim, device=q.device, dtype=torch.float32)

    for token in range(sequence):
        q_t, k_t, v_t = q[:, token], k[:, token], v[:, token]
        state = state * gate[:, token].exp().unsqueeze(-1)
        residual = v_t - (k_t.unsqueeze(-1) * state).sum(dim=-2)
        state = state + k_t.unsqueeze(-1) * (beta[:, token].unsqueeze(-1) * residual).unsqueeze(-2)
        output[:, token] = torch.einsum("bhk,bhkv->bhv", q_t, state)

    return output, state


def sigmoid_gated_rms_norm_reference(
    inputs: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMSNorm followed by sigmoid output gating, per Kimi/FLA."""
    inputs = inputs.float()
    normalized = inputs * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + eps)
    return normalized * weight.float() * torch.sigmoid(gate.float())


def _initial_state(inputs: torch.Tensor, config: KDAConfig) -> KDAReferenceState:
    batch = inputs.shape[0]
    history = config.conv_kernel_size - 1
    return KDAReferenceState(
        recurrent=inputs.new_zeros(batch, config.num_heads, config.head_k_dim, config.head_v_dim),
        q_convolution=inputs.new_zeros(batch, history, config.q_dim),
        k_convolution=inputs.new_zeros(batch, history, config.k_dim),
        v_convolution=inputs.new_zeros(batch, history, config.v_dim),
    )


def kda_forward_reference(
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    config: KDAConfig,
    state: KDAReferenceState | None = None,
) -> tuple[torch.Tensor, KDAReferenceState]:
    """Execute the complete Kimi Delta Attention layer in pure torch."""
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != config.hidden_size:
        raise ValueError(f"hidden_states shape {tuple(hidden_states.shape)} must be [B,T,{config.hidden_size}]")
    validate_reference_weights(weights, config)
    state = _initial_state(hidden_states, config) if state is None else state
    hidden = hidden_states.float()

    q, q_state = causal_depthwise_conv_reference(
        F.linear(hidden, weights["q_proj.weight"].float()),
        weights["q_conv1d.weight"],
        state.q_convolution,
    )
    k, k_state = causal_depthwise_conv_reference(
        F.linear(hidden, weights["k_proj.weight"].float()),
        weights["k_conv1d.weight"],
        state.k_convolution,
    )
    v, v_state = causal_depthwise_conv_reference(
        F.linear(hidden, weights["v_proj.weight"].float()),
        weights["v_conv1d.weight"],
        state.v_convolution,
    )

    batch, sequence, _ = hidden.shape
    q = q.reshape(batch, sequence, config.num_heads, config.head_k_dim)
    k = k.reshape(batch, sequence, config.num_heads, config.head_k_dim)
    v = v.reshape(batch, sequence, config.num_heads, config.head_v_dim)
    raw_gate = F.linear(
        F.linear(hidden, weights["f_a_proj.weight"].float()),
        weights["f_b_proj.weight"].float(),
    ).reshape(batch, sequence, config.num_heads, config.head_k_dim)
    gate = kda_gate_reference(raw_gate, weights["A_log"], weights["dt_bias"])
    beta = torch.sigmoid(F.linear(hidden, weights["b_proj.weight"].float()))
    output, recurrent = kda_recurrent_reference(q, k, v, gate, beta, state.recurrent)

    output_gate = F.linear(
        F.linear(hidden, weights["g_a_proj.weight"].float()),
        weights["g_b_proj.weight"].float(),
    ).reshape(batch, sequence, config.num_heads, config.head_v_dim)
    output = sigmoid_gated_rms_norm_reference(
        output,
        output_gate,
        weights["o_norm.weight"],
        config.norm_eps,
    ).reshape(batch, sequence, config.v_dim)
    output = F.linear(output, weights["o_proj.weight"].float())

    return output, KDAReferenceState(recurrent, q_state, k_state, v_state)
