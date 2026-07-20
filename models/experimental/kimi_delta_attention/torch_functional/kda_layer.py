# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# KDA (Kimi Delta Attention) torch reference layer — mirrors fla/layers/kda.py::KimiDeltaAttention.
# The numerical ground truth the ttnn layer is validated against. See ../API_SPEC.md.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .kda_ops import kda_gate, l2norm, naive_chunk_kda, naive_recurrent_kda


def causal_short_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Depthwise causal 1-D conv + SiLU. ``x:[B,T,D]``, ``weight:[D,kernel]``.

    Left-pads by ``kernel-1`` so output length == input length (causal).
    """
    B, T, D = x.shape
    kernel = weight.shape[-1]
    xt = rearrange(x, "b t d -> b d t")
    xt = F.pad(xt, (kernel - 1, 0))
    out = F.conv1d(xt, weight.unsqueeze(1), bias=bias, groups=D)  # depthwise
    out = rearrange(out, "b d t -> b t d")
    return F.silu(out)


def gated_rmsnorm(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Head-wise gated RMSNorm: ``weight * rmsnorm(x) * sigmoid(gate)`` (norm before gate).

    Matches the KDA paper output form ``sigmoid(W_g x) ⊙ RMSNorm(KDA(·))``. ``x,gate:[...,V]``.
    """
    x = x.float()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * normed * torch.sigmoid(gate.float())).to(x.dtype)


class KimiDeltaAttentionRef(nn.Module):
    """Torch reference for one KDA layer. Random-init by default (correctness ground truth)."""

    def __init__(
        self,
        hidden_size: int = 2304,
        head_dim: int = 128,
        num_heads: int = 32,
        num_v_heads: int | None = None,
        conv_size: int = 4,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        lower_bound: float | None = None,
        norm_eps: float = 1e-5,
        mode: str = "recurrent",  # "recurrent" | "chunk"
        chunk_size: int = 64,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_k_dim = head_dim
        self.head_v_dim = head_dim  # expand_v = 1
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.allow_neg_eigval = allow_neg_eigval
        self.lower_bound = lower_bound
        self.norm_eps = norm_eps
        self.mode = mode
        self.chunk_size = chunk_size

        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.gate_dim = self.num_v_heads * self.head_k_dim

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False, dtype=dtype)

        if use_short_conv:
            self.q_conv = nn.Parameter(torch.randn(self.key_dim, conv_size, dtype=dtype) * (conv_size ** -0.5))
            self.k_conv = nn.Parameter(torch.randn(self.key_dim, conv_size, dtype=dtype) * (conv_size ** -0.5))
            self.v_conv = nn.Parameter(torch.randn(self.value_dim, conv_size, dtype=dtype) * (conv_size ** -0.5))

        # low-rank gate f_proj: hidden -> head_v_dim -> HV*K
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False, dtype=dtype),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=False, dtype=dtype),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False, dtype=dtype)

        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16)))
        dt = torch.exp(
            torch.rand(self.gate_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        # output gate g_proj: hidden -> head_v_dim -> HV*V  (bias on last linear)
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False, dtype=dtype),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True, dtype=dtype),
        )
        self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim, dtype=dtype))
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, T, _ = hidden_states.shape

        if self.use_short_conv:
            q = causal_short_conv(self.q_proj(hidden_states), self.q_conv)
            k = causal_short_conv(self.k_proj(hidden_states), self.k_conv)
            v = causal_short_conv(self.v_proj(hidden_states), self.v_conv)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        g = self.f_proj(hidden_states)
        beta = self.b_proj(hidden_states)

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)  # [B,T,HV,K]
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)  # [B,T,HV,V]

        # explicit pre-recurrence transforms (fla does these in-kernel)
        q, k = l2norm(q), l2norm(k)
        beta = torch.sigmoid(beta)
        if self.allow_neg_eigval:
            beta = beta * 2
        g = kda_gate(g, self.A_log, self.dt_bias, self.lower_bound)

        if self.mode == "chunk":
            o, _ = naive_chunk_kda(q, k, v, g, beta, chunk_size=self.chunk_size)
        else:
            o, _ = naive_recurrent_kda(q, k, v, g, beta)

        gate = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
        o = gated_rmsnorm(o, gate, self.o_norm_weight, self.norm_eps)
        o = rearrange(o, "b t h d -> b t (h d)")
        return self.o_proj(o)
