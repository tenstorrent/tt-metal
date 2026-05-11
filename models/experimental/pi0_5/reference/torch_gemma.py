# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 Gemma block with adaRMSNorm (PyTorch reference).

The pi0.5 action expert uses adaRMS in every norm — there is *no* learnable
per-channel scale. Per-batch (scale, shift, gate) come from a Dense projection
of `adarms_cond`:

    normed   = x * rsqrt(mean(x^2) + eps)               # plain RMS
    s, t, g  = chunk(linear(cond, mod_w, mod_b), 3)     # (B, 3*width)
    out      = normed * (1 + s[:, None, :]) + t[:, None, :]
    return out, g[:, None, :]                           # gate scales residual

Checkpoint key layout (per layer prefix `model.layers.{i}.`):
    input_layernorm.dense.weight             # (3*width, width)
    input_layernorm.dense.bias               # (3*width,)
    post_attention_layernorm.dense.weight    # (3*width, width)
    post_attention_layernorm.dense.bias      # (3*width,)
    self_attn.{q,k,v,o}_proj.weight
    mlp.{gate,up,down}_proj.weight
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from models.experimental.pi0.common.configs import GemmaConfig
from models.experimental.pi0.reference.torch_gemma import GemmaAttention, GemmaMLP


def plain_rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization without a learnable scale."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


def ada_rms_norm(
    x: torch.Tensor,
    cond: torch.Tensor,
    mod_weight: torch.Tensor,
    mod_bias: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adaptive RMSNorm: returns (modulated, gate)."""
    normed = plain_rms_normalize(x, eps)
    mod = F.linear(cond.to(mod_weight.dtype), mod_weight, mod_bias)
    scale, shift, gate = mod.chunk(3, dim=-1)
    scale = scale.unsqueeze(1).to(normed.dtype)
    shift = shift.unsqueeze(1).to(normed.dtype)
    gate = gate.unsqueeze(1).to(normed.dtype)
    return normed * (1.0 + scale) + shift, gate


def ada_rms_norm_no_gate(
    x: torch.Tensor,
    cond: torch.Tensor,
    mod_weight: torch.Tensor,
    mod_bias: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Adaptive RMSNorm for the final stack norm — discards the gate output."""
    out, _ = ada_rms_norm(x, cond, mod_weight, mod_bias, eps)
    return out


class AdaRMSGemmaBlock:
    """Gemma block for the PI0.5 action expert: adaRMS norms + gated residuals."""

    def __init__(self, config: GemmaConfig, weights: Dict[str, torch.Tensor], layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx

        self.pre_attn_mod_weight = weights["input_layernorm.dense.weight"]
        self.pre_attn_mod_bias = weights.get("input_layernorm.dense.bias")
        self.pre_ffw_mod_weight = weights["post_attention_layernorm.dense.weight"]
        self.pre_ffw_mod_bias = weights.get("post_attention_layernorm.dense.bias")

        self.attention = GemmaAttention(config, weights, layer_idx)
        self.mlp = GemmaMLP(config, weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adarms_cond: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        normed, attn_gate = ada_rms_norm(
            hidden_states, adarms_cond, self.pre_attn_mod_weight, self.pre_attn_mod_bias, self.config.rms_norm_eps
        )
        attn_output, new_cache = self.attention.forward(
            normed, cos, sin, attention_mask, position_ids, past_key_value, use_cache
        )
        hidden_states = hidden_states + attn_gate * attn_output

        normed, ffw_gate = ada_rms_norm(
            hidden_states, adarms_cond, self.pre_ffw_mod_weight, self.pre_ffw_mod_bias, self.config.rms_norm_eps
        )
        mlp_output = self.mlp.forward(normed)
        hidden_states = hidden_states + ffw_gate * mlp_output

        return hidden_states, new_cache
