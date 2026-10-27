# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer inputs (PLE). EmbeddingGemma 2 uses projection-only PLE: there is no per-layer table.

PerLayerInput (HF `EmbeddingGemma2TextPLE`)
    proj  = inputs_embeds @ W_ple * hidden^-0.5          W_ple [hidden, L * P]   (scale folded into W)
    pli_i = RMSNorm_P(proj[..., i*P:(i+1)*P])            one shared norm weight, applied per layer slice

    forward(inputs_embeds [B,1,S,hidden])  -> proj [B,1,S,L*P]    bf16 TILE
    layer_input(proj, i)                   -> pli_i [B,1,S,P]     bf16 TILE

PerLayerInputBlock (HF `EmbeddingGemma2TextPLEBlock`, third residual block of every layer)
    h = h + post_norm( W_proj( gelu_tanh(W_gate h) * pli_i ) )

    forward(h [B,1,S,hidden], pli_i [B,1,S,P]) -> [B,1,S,hidden]
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import (
    EncoderLayerConfig,
    PerLayerInputConfig,
    RMSNormConfig,
    act_memory_config,
    compute_kernel_config_hifi4,
)
from models.demos.blackhole.embedding_gemma2.tt.linear import WEIGHT_DTYPE, gated_product, linear
from models.demos.blackhole.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, linear_weight, substate


class PerLayerInput(LightweightModule):
    def __init__(self, config: PerLayerInputConfig, proj_weight: ttnn.Tensor, norm: RMSNorm):
        super().__init__()
        self.config = config
        self.proj_weight = proj_weight
        self.norm = norm
        self.compute_kernel_config = compute_kernel_config_hifi4()

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], config: PerLayerInputConfig, device, *, cache_dir: Path | None = None
    ) -> "PerLayerInput":
        """`state_dict` holds the HF `ple.` sub-state (`per_layer_model_projection.weight`, ...)."""
        w = state_dict["per_layer_model_projection.weight"]
        expected = (config.num_layers * config.per_layer_dim, config.hidden_size)
        if tuple(w.shape) != expected:
            raise ValueError(f"per_layer_model_projection: expected {expected}, got {tuple(w.shape)}")
        w_scaled = (w.float() * config.projection_scale).to(torch.bfloat16)
        proj_weight = as_weight(
            linear_weight(w_scaled), device, cache_dir=cache_dir, name="ple.proj.scaled.weight", dtype=WEIGHT_DTYPE
        )
        norm = RMSNorm.from_state_dict(
            substate(state_dict, "per_layer_projection_norm"),
            RMSNormConfig(dim=config.per_layer_dim, eps=config.rms_norm_eps),
            device,
            cache_dir=cache_dir,
            name="ple.norm",
        )
        return cls(config, proj_weight, norm)

    def forward(self, inputs_embeds: ttnn.Tensor) -> ttnn.Tensor:
        return linear(inputs_embeds, self.proj_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def layer_input(self, projection: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        P = self.config.per_layer_dim
        B, _, S, _ = projection.shape
        mem = act_memory_config(projection.device(), B, S, P)
        piece = ttnn.slice(projection, (0, 0, 0, layer_idx * P), (B, 1, S, (layer_idx + 1) * P), memory_config=mem)
        out = self.norm(piece, memory_config=mem)
        ttnn.deallocate(piece)
        return out


class PerLayerInputBlock(LightweightModule):
    def __init__(self, config: EncoderLayerConfig, w_gate: ttnn.Tensor, w_proj: ttnn.Tensor, post_norm: RMSNorm):
        super().__init__()
        self.config = config
        self.w_gate = w_gate
        self.w_proj = w_proj
        self.post_norm = post_norm
        self.compute_kernel_config = compute_kernel_config_hifi4()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: EncoderLayerConfig,
        device,
        *,
        cache_dir: Path | None = None,
        name: str = "ple_block",
    ) -> "PerLayerInputBlock":
        """`state_dict` holds the HF `layers.<i>.ple_block.` sub-state."""
        H, P = config.hidden_size, config.per_layer_dim
        wg = state_dict["per_layer_input_gate.weight"]
        wp = state_dict["per_layer_projection.weight"]
        if tuple(wg.shape) != (P, H) or tuple(wp.shape) != (H, P):
            raise ValueError(f"{name}: unexpected shapes {tuple(wg.shape)} {tuple(wp.shape)}")
        w_gate = as_weight(
            linear_weight(wg), device, cache_dir=cache_dir, name=f"{name}.gate.weight", dtype=WEIGHT_DTYPE
        )
        w_proj = as_weight(
            linear_weight(wp), device, cache_dir=cache_dir, name=f"{name}.proj.weight", dtype=WEIGHT_DTYPE
        )
        post_norm = RMSNorm.from_state_dict(
            substate(state_dict, "post_per_layer_input_norm"),
            RMSNormConfig(dim=H, eps=config.rms_norm_eps),
            device,
            cache_dir=cache_dir,
            name=f"{name}.post_norm",
        )
        return cls(config, w_gate, w_proj, post_norm)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        per_layer_input: ttnn.Tensor,
        out_scale: float | None = None,
        out_memory_config=None,
    ) -> ttnn.Tensor:
        """`out_scale` (the layer's `layer_scalar`) is applied to the residual sum inside the add
        (SFPU post-activation), so the sum is rounded to bf16 once instead of twice."""
        B, S = hidden_states.shape[0], hidden_states.shape[2]
        mem = act_memory_config(hidden_states.device(), B, S, self.config.hidden_size)
        gated = gated_product(hidden_states, self.w_gate, per_layer_input, memory_config=mem)
        proj = linear(gated, self.w_proj, memory_config=mem)
        ttnn.deallocate(gated)
        normed = self.post_norm(proj, memory_config=mem)
        ttnn.deallocate(proj)
        post = [] if out_scale is None else [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, float(out_scale))]
        out = ttnn.add(
            hidden_states, normed, memory_config=out_memory_config or ttnn.DRAM_MEMORY_CONFIG, activations=post
        )
        ttnn.deallocate(normed)
        return out
