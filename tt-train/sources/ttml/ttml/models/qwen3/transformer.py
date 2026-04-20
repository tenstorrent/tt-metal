# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 transformer building blocks.

Provides RMSNorm, SwiGLU MLP, and decoder layer for the Qwen3 architecture.
"""

from __future__ import annotations

from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, Parameter

from .autograd_ops import ConcatLastDim, RMSNormFunction
from .attention import Qwen3Attention

# Re-export for callers that import from transformer
__all__ = [
    "ConcatLastDim",
    "RMSNormFunction",
    "Qwen3RMSNorm",
    "Qwen3MLP",
    "Qwen3Block",
]


class Qwen3RMSNorm(AbstractModuleBase):
    """RMSNorm using the custom autograd function for device compatibility."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = Parameter(ttml.init.ones()((1, 1, 1, hidden_size)))

    def forward(self, hidden_states):
        return RMSNormFunction.apply(hidden_states, self.weight.tensor, self.eps)


class Qwen3MLP(AbstractModuleBase):
    """SwiGLU MLP: gate_proj + up_proj -> silu(gate) * up -> down_proj."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = LinearLayer(
            hidden_size,
            intermediate_size,
            False,
            weight_init=ttml.init.normal(0.0, 0.02),
        )
        self.up_proj = LinearLayer(
            hidden_size,
            intermediate_size,
            False,
            weight_init=ttml.init.normal(0.0, 0.02),
        )
        self.down_proj = LinearLayer(
            intermediate_size,
            hidden_size,
            False,
            weight_init=ttml.init.normal(0.0, 0.02),
        )

    def forward(self, x):
        gate = ttml.ops.unary.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(ttml.ops.binary.mul(gate, up))


class Qwen3Block(AbstractModuleBase):
    """Single Qwen3 decoder layer: pre-norm attention + pre-norm MLP."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()

        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        mask: Optional[ttml.autograd.Tensor] = None,
        past_key_values=None,
        position_offset: int = 0,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            mask,
            past_key_values,
            position_offset,
        )
        hidden_states = ttml.ops.binary.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ttml.ops.binary.add(residual, hidden_states)
        return hidden_states
