# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer inputs (PLE). EmbeddingGemma 2 uses projection-only PLE: there is no per-layer table.

PerLayerInput (HF `EmbeddingGemma2TextPLE`)
    proj  = inputs_embeds @ W_ple * hidden^-0.5          W_ple [hidden, L * P]   (scale folded into W)
    pli_i = RMSNorm_P(proj[..., i*P:(i+1)*P])            one shared norm weight, applied per layer slice

    forward(inputs_embeds [B,1,S,hidden])  -> proj [B,1,S,L*P]    bf16 TILE
    layer_input(proj, i)                   -> pli_i [B,1,S,P]     bf16 TILE

    Narrowed per-layer projection (E9.2 topology): from PER_LAYER_MIN_TOKENS tokens (B x S) up,
    `forward` returns `inputs_embeds` itself (no device op) and `layer_input` runs one
    [hidden, P] matmul per layer with that layer's weight columns, instead of slicing layer i out of
    the [B,1,S,L*P] projection. Same values per layer; 24 small matmuls replace one wide matmul and
    24 slices, which costs less at >= 8192 tokens (stage-1 profiles, doc/prefill/02-topology/), and
    the [B,1,S,L*P] tensor (1.6 GB at 65536) is never materialized. The caller must keep
    `inputs_embeds` alive until the last `layer_input` call (TextModel.encode_embeds does).

PerLayerInputBlock (HF `EmbeddingGemma2TextPLEBlock`, third residual block of every layer)
    h = h + post_norm( W_proj( gelu_tanh(W_gate h) * pli_i ) )

    forward(h [B,1,S,hidden], pli_i [B,1,S,P]) -> [B,1,S,hidden]
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.config import (
    EncoderLayerConfig,
    PerLayerInputConfig,
    RMSNormConfig,
    act_memory_config,
    compute_kernel_config_hifi4,
)
from models.demos.embedding_gemma2.tt.linear import gated_product, linear
from models.demos.embedding_gemma2.tt.precision import TextPrecision, resolve_precision
from models.demos.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.embedding_gemma2.tt.weights import as_weight, linear_weight, substate

# Tokens per forward (batch x bucket) from which each layer projects its own input (see module doc).
# Stage-1 profiles at 12 x 10 (wide matmul + 24 slices vs 24 x [S,512]x[512,512]): 4096 0.81 vs
# 1.03 ms (wide wins), 8192 2.13 vs 1.52 ms, 65536 16.2 vs 10.7 ms. Set a large value to disable.
PER_LAYER_MIN_TOKENS = int(os.environ.get("EG2_PLE_PER_LAYER_MIN_TOKENS", "8192"))


class PerLayerInput(LightweightModule):
    def __init__(
        self,
        config: PerLayerInputConfig,
        proj_weight: ttnn.Tensor,
        norm: RMSNorm,
        precision: TextPrecision | None = None,
        layer_weights: list[ttnn.Tensor] | None = None,
    ):
        super().__init__()
        self.config = config
        self.proj_weight = proj_weight
        self.layer_weights = layer_weights  # per-layer [hidden, P] columns of proj_weight
        self.norm = norm
        self.compute_kernel_config = compute_kernel_config_hifi4()
        precision = precision or resolve_precision(None).text
        self.policy = precision.matmul("ple_proj")

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: PerLayerInputConfig,
        device,
        *,
        cache_dir: Path | None = None,
        precision: TextPrecision | None = None,
    ) -> "PerLayerInput":
        """`state_dict` holds the HF `ple.` sub-state (`per_layer_model_projection.weight`, ...)."""
        precision = precision or resolve_precision(None).text
        w = state_dict["per_layer_model_projection.weight"]
        expected = (config.num_layers * config.per_layer_dim, config.hidden_size)
        if tuple(w.shape) != expected:
            raise ValueError(f"per_layer_model_projection: expected {expected}, got {tuple(w.shape)}")
        w_scaled = (w.float() * config.projection_scale).to(torch.bfloat16)
        wdt = precision.matmul("ple_proj").ttnn_dtype
        proj_weight = as_weight(
            linear_weight(w_scaled), device, cache_dir=cache_dir, name="ple.proj.scaled.weight", dtype=wdt
        )
        P = config.per_layer_dim
        layer_weights = [
            as_weight(
                linear_weight(w_scaled[i * P : (i + 1) * P]),
                device,
                cache_dir=cache_dir,
                name=f"ple.proj.scaled.layer{i}.weight",
                dtype=wdt,
            )
            for i in range(config.num_layers)
        ]
        norm = RMSNorm.from_state_dict(
            substate(state_dict, "per_layer_projection_norm"),
            RMSNormConfig(dim=config.per_layer_dim, eps=config.rms_norm_eps),
            device,
            cache_dir=cache_dir,
            name="ple.norm",
            compute_kernel_config=precision.norms.compute_kernel_config,
        )
        return cls(config, proj_weight, norm, precision, layer_weights)

    def per_layer(self, x: ttnn.Tensor) -> bool:
        """True when each layer projects its own input for this activation's token count."""
        return self.layer_weights is not None and x.shape[0] * x.shape[2] >= PER_LAYER_MIN_TOKENS

    def forward(self, inputs_embeds: ttnn.Tensor) -> ttnn.Tensor:
        """[B,1,S,L*P] projection, or `inputs_embeds` itself in the narrowed per-layer mode."""
        if self.per_layer(inputs_embeds):
            return inputs_embeds
        return linear(
            inputs_embeds,
            self.proj_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            policy=self.policy,
            dtype=self.policy.out_ttnn_dtype,
            role="ple_wide",
        )

    def layer_input(self, projection: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """`projection`: what `forward` returned (the wide projection, or the embeddings)."""
        P = self.config.per_layer_dim
        B, _, S, W = projection.shape
        mem = act_memory_config(projection.device(), B, S, P)
        if W == self.config.num_layers * P:
            piece = ttnn.slice(projection, (0, 0, 0, layer_idx * P), (B, 1, S, (layer_idx + 1) * P), memory_config=mem)
        else:
            if W != self.config.hidden_size or self.layer_weights is None:
                raise ValueError(f"layer_input: expected [.., {self.config.num_layers * P}] or embeddings, got {W}")
            piece = linear(
                projection,
                self.layer_weights[layer_idx],
                memory_config=mem,
                policy=self.policy,
                dtype=self.policy.out_ttnn_dtype,
                role="ple_layer",
            )
        out = self.norm(piece, memory_config=mem)
        ttnn.deallocate(piece)
        return out


class PerLayerInputBlock(LightweightModule):
    def __init__(
        self,
        config: EncoderLayerConfig,
        w_gate: ttnn.Tensor,
        w_proj: ttnn.Tensor,
        post_norm: RMSNorm,
        precision: TextPrecision | None = None,
    ):
        super().__init__()
        self.config = config
        self.w_gate = w_gate
        self.w_proj = w_proj
        self.post_norm = post_norm
        self.compute_kernel_config = compute_kernel_config_hifi4()
        precision = precision or resolve_precision(None).text
        self.policy = precision.matmul("ple_block", config.layer_idx)
        self.residual_dtype = precision.residual_ttnn_dtype

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: EncoderLayerConfig,
        device,
        *,
        cache_dir: Path | None = None,
        name: str = "ple_block",
        precision: TextPrecision | None = None,
    ) -> "PerLayerInputBlock":
        """`state_dict` holds the HF `layers.<i>.ple_block.` sub-state."""
        precision = precision or resolve_precision(None).text
        wdt = precision.matmul("ple_block", config.layer_idx).ttnn_dtype
        H, P = config.hidden_size, config.per_layer_dim
        wg = state_dict["per_layer_input_gate.weight"]
        wp = state_dict["per_layer_projection.weight"]
        if tuple(wg.shape) != (P, H) or tuple(wp.shape) != (H, P):
            raise ValueError(f"{name}: unexpected shapes {tuple(wg.shape)} {tuple(wp.shape)}")
        w_gate = as_weight(linear_weight(wg), device, cache_dir=cache_dir, name=f"{name}.gate.weight", dtype=wdt)
        w_proj = as_weight(linear_weight(wp), device, cache_dir=cache_dir, name=f"{name}.proj.weight", dtype=wdt)
        post_norm = RMSNorm.from_state_dict(
            substate(state_dict, "post_per_layer_input_norm"),
            RMSNormConfig(dim=H, eps=config.rms_norm_eps),
            device,
            cache_dir=cache_dir,
            name=f"{name}.post_norm",
            compute_kernel_config=precision.norms.compute_kernel_config,
        )
        return cls(config, w_gate, w_proj, post_norm, precision)

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
        p = self.policy
        dt = p.out_ttnn_dtype
        gated = gated_product(
            hidden_states, self.w_gate, per_layer_input, memory_config=mem, policy=p, dtype=dt, role="ple_gate"
        )
        proj = linear(gated, self.w_proj, memory_config=mem, policy=p, dtype=dt, role="ple_proj")
        ttnn.deallocate(gated)
        normed = self.post_norm(proj, memory_config=mem)
        ttnn.deallocate(proj)
        post = [] if out_scale is None else [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, float(out_scale))]
        out = ttnn.add(
            hidden_states,
            normed,
            dtype=self.residual_dtype,
            memory_config=out_memory_config or ttnn.DRAM_MEMORY_CONFIG,
            activations=post,
        )
        ttnn.deallocate(normed)
        return out
