# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 Gemma block with adaRMSNorm (TTNN).

PI0.5 expert: no learnable per-channel scale in norms; (scale, shift, gate)
come from a Dense layer over `adarms_cond`.

    normed   = ttnn.rms_norm(x, weight=ones)       # plain RMS (eps only)
    s, t, g  = chunk(linear(cond, mod_w, mod_b), 3)
    out      = normed * (1 + s) + t
    return out, g                                  # gate scales residual

Block forward:
    x' = x + g_attn * attn(adaRMS(x, cond))
    x  = x' + g_ffw  * mlp(adaRMS(x', cond))
"""

from typing import Dict, Optional, Tuple

import ttnn

from models.experimental.pi0.common.configs import GemmaConfig
from models.experimental.pi0.tt.ttnn_gemma import GemmaAttentionTTNN, GemmaMLPTTNN


def _plain_rms_norm_weight(device, width: int):
    """ones(width) tile, sized as (1, width) — used as identity scale for ttnn.rms_norm."""
    import torch as _torch  # local import keeps this file lightweight

    return ttnn.from_torch(
        _torch.ones(1, width),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def plain_rms_norm_ttnn(x: "ttnn.Tensor", ones_weight: "ttnn.Tensor", eps: float) -> "ttnn.Tensor":
    """RMSNorm without a learnable scale. `ones_weight` is a pre-built ones tile.

    NOTE: the existing pi0 rms_norm_ttnn assumes the weight has the Gemma
    `+1` offset pre-applied. By pre-building `ones_weight` we get back the
    plain `x * rsqrt(mean(x^2) + eps)` formula.
    """
    return ttnn.rms_norm(x, weight=ones_weight, epsilon=eps, memory_config=ttnn.L1_MEMORY_CONFIG)


def ada_rms_norm_ttnn(
    x: "ttnn.Tensor",
    ones_weight: "ttnn.Tensor",
    cond: "ttnn.Tensor",
    mod_weight: "ttnn.Tensor",
    mod_bias: Optional["ttnn.Tensor"],
    eps: float,
    core_grid: "ttnn.CoreGrid",
) -> Tuple["ttnn.Tensor", "ttnn.Tensor"]:
    normed = plain_rms_norm_ttnn(x, ones_weight, eps)

    mod = ttnn.linear(
        cond,
        mod_weight,
        bias=mod_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=core_grid,
    )
    width3 = mod.shape[-1]
    width = width3 // 3
    scale = ttnn.slice(mod, [0, 0], [mod.shape[0], width])
    shift = ttnn.slice(mod, [0, width], [mod.shape[0], 2 * width])
    gate = ttnn.slice(mod, [0, 2 * width], [mod.shape[0], 3 * width])
    ttnn.deallocate(mod)

    scale = ttnn.reshape(scale, (scale.shape[0], 1, width))
    shift = ttnn.reshape(shift, (shift.shape[0], 1, width))
    gate = ttnn.reshape(gate, (gate.shape[0], 1, width))

    scale_plus_one = ttnn.add(scale, 1.0)
    ttnn.deallocate(scale)
    scaled = ttnn.mul(normed, scale_plus_one)
    ttnn.deallocate(scale_plus_one)
    ttnn.deallocate(normed)
    modulated = ttnn.add(scaled, shift)
    ttnn.deallocate(scaled)
    ttnn.deallocate(shift)
    return modulated, gate


def ada_rms_norm_no_gate_ttnn(
    x: "ttnn.Tensor",
    ones_weight: "ttnn.Tensor",
    cond: "ttnn.Tensor",
    mod_weight: "ttnn.Tensor",
    mod_bias: Optional["ttnn.Tensor"],
    eps: float,
    core_grid: "ttnn.CoreGrid",
) -> "ttnn.Tensor":
    out, gate = ada_rms_norm_ttnn(x, ones_weight, cond, mod_weight, mod_bias, eps, core_grid)
    ttnn.deallocate(gate)
    return out


class AdaRMSGemmaBlockTTNN:
    """PI0.5 action-expert block (TTNN): adaRMS + gated residuals."""

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, "ttnn.Tensor"],
        layer_idx: int,
        device: "ttnn.Device",
        ones_weight: "ttnn.Tensor",
        cos_meta: Optional["ttnn.Tensor"] = None,
        sin_meta: Optional["ttnn.Tensor"] = None,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.ones_weight = ones_weight

        self.pre_attn_mod_weight = weights["input_layernorm.dense.weight"]
        self.pre_attn_mod_bias = weights.get("input_layernorm.dense.bias")
        self.pre_ffw_mod_weight = weights["post_attention_layernorm.dense.weight"]
        self.pre_ffw_mod_bias = weights.get("post_attention_layernorm.dense.bias")

        self.attention = GemmaAttentionTTNN(config, weights, layer_idx, device, cos_meta, sin_meta)
        self.mlp = GemmaMLPTTNN(config, weights, device)

        device_grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        cos: "ttnn.Tensor",
        sin: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        past_key_value: Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = None,
        use_cache: bool = False,
    ) -> Tuple["ttnn.Tensor", Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        normed, attn_gate = ada_rms_norm_ttnn(
            hidden_states,
            self.ones_weight,
            adarms_cond,
            self.pre_attn_mod_weight,
            self.pre_attn_mod_bias,
            self.config.rms_norm_eps,
            self.core_grid,
        )
        attn_output, new_cache = self.attention.forward(
            normed, cos, sin, attention_mask, position_ids, past_key_value, use_cache
        )
        ttnn.deallocate(normed)
        gated_attn = ttnn.mul(attn_output, attn_gate)
        ttnn.deallocate(attn_output)
        ttnn.deallocate(attn_gate)
        hidden_states = ttnn.add(hidden_states, gated_attn)
        ttnn.deallocate(gated_attn)

        normed, ffw_gate = ada_rms_norm_ttnn(
            hidden_states,
            self.ones_weight,
            adarms_cond,
            self.pre_ffw_mod_weight,
            self.pre_ffw_mod_bias,
            self.config.rms_norm_eps,
            self.core_grid,
        )
        mlp_output = self.mlp.forward(normed)
        ttnn.deallocate(normed)
        gated_mlp = ttnn.mul(mlp_output, ffw_gate)
        ttnn.deallocate(mlp_output)
        ttnn.deallocate(ffw_gate)
        hidden_states = ttnn.add(hidden_states, gated_mlp)
        ttnn.deallocate(gated_mlp)

        return hidden_states, new_cache
