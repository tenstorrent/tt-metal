# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 MLP and decoder-layer blocks."""

from __future__ import annotations

from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer

from .attention import Qwen3Attention
from .rmsnorm import Qwen3RMSNorm


class Qwen3MLP(AbstractModuleBase):
    """SwiGLU feed-forward block: ``down_proj(silu(gate_proj(x)) * up_proj(x))``."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = LinearLayer(hidden_size, intermediate_size, has_bias=False)
        self.up_proj = LinearLayer(hidden_size, intermediate_size, has_bias=False)
        self.down_proj = LinearLayer(intermediate_size, hidden_size, has_bias=False)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        gate = ttml.ops.unary.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(ttml.ops.binary.mul(gate, up))


class Qwen3DecoderLayer(AbstractModuleBase):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_params: ttml.ops.rope.RotaryEmbeddingParams,
        attention_bias: bool = True,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.input_layernorm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_params=rope_params,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        kv_cache: Optional[ttml.models.KvCache] = None,
        layer_idx: Optional[int] = None,
        new_tokens: Optional[int] = None,
    ) -> ttml.autograd.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, mask, kv_cache, layer_idx, new_tokens)
        h = ttml.ops.binary.add(residual, h)

        residual = h
        x = self.post_attention_layernorm(h)
        x = self.mlp(x)
        return ttml.ops.binary.add(residual, x)
