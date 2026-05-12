# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 Gemma block with adaRMSNorm (TTNN).

PI0.5 expert: no learnable per-channel scale in norms; (scale, shift, gate)
come from a Dense layer over `adarms_cond`. The block-level modulation Dense
projects (W -> 6*W) once, producing all six tensors for the block (mirrors
tt-dit's `norm1_linear`). The (scale, shift) pair is then fused into the
RMSNorm kernel via `ttnn.rms_norm(weight=(1+scale), bias=shift, ...)`.

Per-norm:
  scale, shift, gate  = chunks of the block-level modulation
  out                 = ttnn.rms_norm(x, weight=(1+scale), bias=shift, eps)

Block forward:
  mod6   = linear(cond, mod_w, mod_b)        # (B, 6*W)  — single matmul
  six    = reshape(mod6, (B, 1, 6*W))
  sa, ta, ga, sf, tf, gf = six chunks of (B, 1, W)
  x'  = x + ga * attn(adaRMS(x, sa, ta))
  x   = x' + gf * mlp(adaRMS(x', sf, tf))
"""

from typing import Dict, List, Optional, Tuple

import ttnn

from models.experimental.pi0.common.configs import GemmaConfig
from models.experimental.pi0.tt.ttnn_gemma import GemmaAttentionTTNN, GemmaMLPTTNN


def _modulated_rms_norm(
    x: "ttnn.Tensor",
    scale: "ttnn.Tensor",
    shift: "ttnn.Tensor",
    eps: float,
) -> "ttnn.Tensor":
    """Fused: ((x · rsqrt(mean(x²)+ε)) · (1+scale)) + shift in one kernel."""
    scale_plus_one = ttnn.add(scale, 1.0)
    out = ttnn.rms_norm(
        x,
        weight=scale_plus_one,
        bias=shift,
        epsilon=eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(scale_plus_one)
    return out


def _split_modulation_6(mod: "ttnn.Tensor") -> List["ttnn.Tensor"]:
    """mod: (B, 6*W) -> 6 tensors of shape (B, 1, W)."""
    B = mod.shape[0]
    total = mod.shape[-1]
    width = total // 6
    mod3 = ttnn.reshape(mod, (B, 1, total))
    return [mod3[:, :, i * width : (i + 1) * width] for i in range(6)]


def ada_rms_norm_ttnn(
    x: "ttnn.Tensor",
    cond: "ttnn.Tensor",
    mod_weight: "ttnn.Tensor",
    mod_bias: Optional["ttnn.Tensor"],
    eps: float,
    core_grid: "ttnn.CoreGrid",
) -> Tuple["ttnn.Tensor", "ttnn.Tensor"]:
    """Standalone adaRMS (single (scale, shift, gate) from a 3*W Dense).

    Kept for the final stack norm path, which uses a separate modulation
    Dense (`model.norm.dense.*`) rather than the block-level fused Dense.
    """
    mod = ttnn.linear(
        cond,
        mod_weight,
        bias=mod_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=core_grid,
    )
    B = mod.shape[0]
    width3 = mod.shape[-1]
    width = width3 // 3
    mod3 = ttnn.reshape(mod, (B, 1, width3))
    ttnn.deallocate(mod)

    scale = mod3[:, :, 0:width]
    shift = mod3[:, :, width : 2 * width]
    gate = mod3[:, :, 2 * width : 3 * width]

    out = _modulated_rms_norm(x, scale, shift, eps)
    ttnn.deallocate(scale)
    ttnn.deallocate(shift)
    ttnn.deallocate(mod3)
    return out, gate


def ada_rms_norm_no_gate_ttnn(
    x: "ttnn.Tensor",
    cond: "ttnn.Tensor",
    mod_weight: "ttnn.Tensor",
    mod_bias: Optional["ttnn.Tensor"],
    eps: float,
    core_grid: "ttnn.CoreGrid",
) -> "ttnn.Tensor":
    """Adaptive RMSNorm for the final stack norm — gate is discarded."""
    out, gate = ada_rms_norm_ttnn(x, cond, mod_weight, mod_bias, eps, core_grid)
    ttnn.deallocate(gate)
    return out


class AdaRMSGemmaBlockTTNN:
    """PI0.5 action-expert block (TTNN): one fused 6*W modulation Dense + adaRMS + gated residuals."""

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, "ttnn.Tensor"],
        layer_idx: int,
        device: "ttnn.Device",
        cos_meta: Optional["ttnn.Tensor"] = None,
        sin_meta: Optional["ttnn.Tensor"] = None,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.device = device

        self.mod_weight = weights["adarms_mod.weight"]
        self.mod_bias = weights.get("adarms_mod.bias")

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
        # Single fused modulation Dense: (B, W) -> (B, 6*W)
        mod = ttnn.linear(
            adarms_cond,
            self.mod_weight,
            bias=self.mod_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
        )
        sa, ta, ga, sf, tf, gf = _split_modulation_6(mod)
        ttnn.deallocate(mod)

        # ---- Attention sublayer ----
        normed = _modulated_rms_norm(hidden_states, sa, ta, self.config.rms_norm_eps)
        ttnn.deallocate(sa)
        ttnn.deallocate(ta)
        attn_output, new_cache = self.attention.forward(
            normed, cos, sin, attention_mask, position_ids, past_key_value, use_cache
        )
        ttnn.deallocate(normed)
        gated_attn = ttnn.mul(attn_output, ga)
        ttnn.deallocate(attn_output)
        ttnn.deallocate(ga)
        hidden_states = ttnn.add(hidden_states, gated_attn)
        ttnn.deallocate(gated_attn)

        # ---- FFW sublayer ----
        normed = _modulated_rms_norm(hidden_states, sf, tf, self.config.rms_norm_eps)
        ttnn.deallocate(sf)
        ttnn.deallocate(tf)
        mlp_output = self.mlp.forward(normed)
        ttnn.deallocate(normed)
        gated_mlp = ttnn.mul(mlp_output, gf)
        ttnn.deallocate(mlp_output)
        ttnn.deallocate(gf)
        hidden_states = ttnn.add(hidden_states, gated_mlp)
        ttnn.deallocate(gated_mlp)

        return hidden_states, new_cache
