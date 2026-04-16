# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 transformer components: MLP and decoder layer."""

from __future__ import annotations

from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer

from .autograd_ops import Qwen3RMSNorm
from .attention import Qwen3Attention


class Qwen3MLP(AbstractModuleBase):
    """SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = LinearLayer(
            hidden_size,
            intermediate_size,
            has_bias=False,
            weight_init=ttml.init.normal(0.0, 0.02),
        )
        self.up_proj = LinearLayer(
            hidden_size,
            intermediate_size,
            has_bias=False,
            weight_init=ttml.init.normal(0.0, 0.02),
        )
        self.down_proj = LinearLayer(
            intermediate_size,
            hidden_size,
            has_bias=False,
            weight_init=ttml.init.normal(0.0, 0.02),
        )

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        gate = ttml.ops.unary.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(ttml.ops.binary.mul(gate, up))


class Qwen3DecoderLayer(AbstractModuleBase):
    """Single Qwen3 transformer decoder block."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: ttml.autograd.Tensor,
        attention_mask: Optional[ttml.autograd.Tensor] = None,
        past_key_values=None,
        position_offset: int = 0,
    ) -> ttml.autograd.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask,
            past_key_values,
            position_offset=position_offset,
        )
        hidden_states = ttml.ops.binary.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ttml.ops.binary.add(residual, hidden_states)
        return hidden_states
