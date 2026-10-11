# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stateless full-layer reference transition for Kimi Delta Attention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.ops import (
    causal_depthwise_conv_reference,
    kda_gate_reference,
    kda_recurrent_reference,
    sigmoid_gated_rms_norm_reference,
)
from models.demos.deepseek_v3_d_p.reference.kda.weights import validate_kda_weights


@dataclass(frozen=True)
class KDAReferenceState:
    """Caller-owned reference state; convolution histories exclude the current token."""

    recurrent: torch.Tensor
    q_convolution: torch.Tensor
    k_convolution: torch.Tensor
    v_convolution: torch.Tensor


def _initial_state(inputs: torch.Tensor, config: KDAConfig) -> KDAReferenceState:
    batch = inputs.shape[0]
    history = config.conv_kernel_size - 1
    return KDAReferenceState(
        recurrent=inputs.new_zeros(batch, config.num_heads, config.head_k_dim, config.head_v_dim),
        q_convolution=inputs.new_zeros(batch, history, config.q_dim),
        k_convolution=inputs.new_zeros(batch, history, config.k_dim),
        v_convolution=inputs.new_zeros(batch, history, config.v_dim),
    )


@dataclass(frozen=True)
class KDAReferenceStages:
    """Every stage of one reference layer transition, for stage-by-stage comparison.

    ``q``, ``k``, ``v`` follow the causal convolution and SiLU (before the recurrence's q/k L2 normalization);
    ``recurrence`` is the delta-rule output before the gated RMS norm; ``normalized`` is the gated RMS norm
    output before ``o_proj``.
    """

    q: torch.Tensor  # [B, T, H, K]
    k: torch.Tensor  # [B, T, H, K]
    v: torch.Tensor  # [B, T, H, V]
    gate: torch.Tensor  # [B, T, H, K], negative log decay
    beta: torch.Tensor  # [B, T, H]
    recurrence: torch.Tensor  # [B, T, H, V]
    output_gate: torch.Tensor  # [B, T, H, V], sigmoid gate logits of the output norm
    normalized: torch.Tensor  # [B, T, H * V]
    output: torch.Tensor  # [B, T, hidden]
    state: KDAReferenceState


def kda_forward_reference(
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    config: KDAConfig,
    state: KDAReferenceState | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, KDAReferenceState]:
    """Execute the complete Kimi Delta Attention layer in pure Torch, computing in ``dtype``.

    ``torch.float32`` is the default reference; ``torch.float64`` is the accuracy golden.
    """
    stages = kda_forward_stages(hidden_states, weights, config, state, dtype=dtype)
    return stages.output, stages.state


def kda_forward_stages(
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    config: KDAConfig,
    state: KDAReferenceState | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> KDAReferenceStages:
    """``kda_forward_reference`` with every intermediate stage kept, all in ``dtype``."""
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != config.hidden_size:
        raise ValueError(f"hidden_states shape {tuple(hidden_states.shape)} must be [B,T,{config.hidden_size}]")
    validate_kda_weights(weights, config)
    state = _initial_state(hidden_states, config) if state is None else state
    hidden = hidden_states.to(dtype)

    q, q_state = causal_depthwise_conv_reference(
        F.linear(hidden, weights["q_proj.weight"].to(dtype)),
        weights["q_conv1d.weight"],
        state.q_convolution,
        dtype=dtype,
    )
    k, k_state = causal_depthwise_conv_reference(
        F.linear(hidden, weights["k_proj.weight"].to(dtype)),
        weights["k_conv1d.weight"],
        state.k_convolution,
        dtype=dtype,
    )
    v, v_state = causal_depthwise_conv_reference(
        F.linear(hidden, weights["v_proj.weight"].to(dtype)),
        weights["v_conv1d.weight"],
        state.v_convolution,
        dtype=dtype,
    )

    batch, sequence, _ = hidden.shape
    q = q.reshape(batch, sequence, config.num_heads, config.head_k_dim)
    k = k.reshape(batch, sequence, config.num_heads, config.head_k_dim)
    v = v.reshape(batch, sequence, config.num_heads, config.head_v_dim)
    raw_gate = F.linear(
        F.linear(hidden, weights["f_a_proj.weight"].to(dtype)), weights["f_b_proj.weight"].to(dtype)
    ).reshape(batch, sequence, config.num_heads, config.head_k_dim)
    gate = kda_gate_reference(raw_gate, weights["A_log"], weights["dt_bias"], config.gate_lower_bound, dtype=dtype)
    beta = torch.sigmoid(F.linear(hidden, weights["b_proj.weight"].to(dtype)))
    recurrence, recurrent = kda_recurrent_reference(q, k, v, gate, beta, state.recurrent, dtype=dtype)

    if config.use_full_rank_gate:
        output_gate = F.linear(hidden, weights["g_proj.weight"].to(dtype))
    else:
        output_gate = F.linear(
            F.linear(hidden, weights["g_a_proj.weight"].to(dtype)), weights["g_b_proj.weight"].to(dtype)
        )
    output_gate = output_gate.reshape(batch, sequence, config.num_heads, config.head_v_dim)
    normalized = sigmoid_gated_rms_norm_reference(
        recurrence, output_gate, weights["o_norm.weight"], config.norm_eps, dtype=dtype
    ).reshape(batch, sequence, config.v_dim)
    output = F.linear(normalized, weights["o_proj.weight"].to(dtype))

    return KDAReferenceStages(
        q=q,
        k=k,
        v=v,
        gate=gate,
        beta=beta,
        recurrence=recurrence,
        output_gate=output_gate,
        normalized=normalized,
        output=output,
        state=KDAReferenceState(recurrent, q_state, k_state, v_state),
    )
