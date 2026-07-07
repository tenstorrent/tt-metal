# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 full DiT model (TTTv2-pattern top-level assembly).

Reference: AceStepDiTModel.forward in modeling_acestep_v15_base.py. End-to-end generative
backbone (single denoise step):

    temb_t, proj_t = time_embed(t)
    temb_r, proj_r = time_embed_r(t - r)
    timestep_proj  = proj_t + proj_r                     # AdaLN modulation for the layers
    temb           = temb_t + temb_r                     # 1-vector modulation for norm_out
    x = concat([context_latents, hidden], dim=-1)        # -> in_channels
    x = proj_in(x)                                       # patchify
    ctx = condition_embedder(encoder_hidden_states)      # Linear inner->inner
    for layer: x = layer(x, rope, timestep_proj, ctx, mask[type])
    x = norm_out AdaLN(x, temb) ; x = proj_out(x)        # de-patchify
    x = x[:, :original_seq_len]                          # crop

Assembles the already-validated pieces: TimestepEmbedding x2, PatchEmbed, condition_embedder
(ttnn.linear), AceStepDiTStack, DiTOutput. Batch=1, no padding path (caller aligns seq), single
denoise step (no cache) — the diffusion inference contract on a single Blackhole p150.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.experimental.acestep.tt.dit_output import DiTOutput, DiTOutputConfig
from models.experimental.acestep.tt.dit_stack import AceStepDiTStack, AceStepDiTStackConfig
from models.experimental.acestep.tt.patch_embed import PatchEmbed, PatchEmbedConfig
from models.experimental.acestep.tt.timestep_embedding import TimestepEmbedding, TimestepEmbeddingConfig


@dataclass
class AceStepDiTModelConfig:
    time_embed: TimestepEmbeddingConfig
    time_embed_r: TimestepEmbeddingConfig
    patch_embed: PatchEmbedConfig
    condition_embedder_weight: LazyWeight  # [inner,inner] transposed for ttnn.linear
    condition_embedder_bias: LazyWeight  # [1, inner]
    stack: AceStepDiTStackConfig
    output: DiTOutputConfig

    dim: int = 2048
    mesh_device: ttnn.MeshDevice | None = None
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    def resolved(self) -> "AceStepDiTModelConfig":
        if self.mesh_device is None:
            self.mesh_device = self.condition_embedder_weight.device
        if self.compute_kernel_config is None:
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        return self


class AceStepDiTModel(LightweightModule):
    """forward(hidden [1,1,T,hc], context_latents [1,1,T,cc], t [B], t_r [B], cos, sin,
    encoder [1,1,E,dim], sliding_mask=None) -> [1,1,T,out_channels]."""

    def __init__(self, config: AceStepDiTModelConfig):
        self.config = config.resolved()
        cfg = self.config
        self.time_embed = TimestepEmbedding(cfg.time_embed)
        self.time_embed_r = TimestepEmbedding(cfg.time_embed_r)
        self.patch_embed = PatchEmbed(cfg.patch_embed)
        self.condition_embedder_weight = cfg.condition_embedder_weight.get_device_weight()
        self.condition_embedder_bias = cfg.condition_embedder_bias.get_device_weight()
        self.stack = AceStepDiTStack(cfg.stack)
        self.output = DiTOutput(cfg.output)

    @classmethod
    def from_config(cls, config: AceStepDiTModelConfig):
        return cls(config)

    def compute_cross_kv(self, encoder_hidden_states):
        """Precompute the cross-attention K/V for every DiT layer from the encoder context. The context
        is embedded by condition_embedder then each layer projects its K/V. All denoise-invariant, so
        this runs ONCE per (cond/uncond) context and is reused across every denoise step (matches the
        reference EncoderDecoderCache cross_attention_cache). Returns a per-layer list of (k, v)."""
        cfg = self.config
        ctx = ttnn.linear(
            encoder_hidden_states,
            self.condition_embedder_weight,
            bias=self.condition_embedder_bias,
            compute_kernel_config=cfg.compute_kernel_config,
        )
        return self.stack.compute_cross_kv(ctx)

    def forward(
        self, hidden_states, context_latents, timestep, timestep_r, cos, sin, encoder_hidden_states,
        sliding_mask=None, cross_kv=None
    ):
        cfg = self.config

        # Dual timestep embedding: timestep_proj (layers AdaLN) + temb (norm_out AdaLN).
        temb_t, proj_t = self.time_embed.forward(timestep)
        temb_r, proj_r = self.time_embed_r.forward(timestep - timestep_r)
        timestep_proj = ttnn.add(proj_t, proj_r)  # [1,6,B,dim]
        temb = ttnn.add(temb_t, temb_r)  # [1,1,B,dim]

        # Concat context_latents + hidden on channel dim -> in_channels, then patchify.
        x = ttnn.concat([context_latents, hidden_states], dim=-1)  # [1,1,T,in_channels]
        x = self.patch_embed.forward(x)  # [1,1,T/p, dim]

        # Condition embedder on encoder context. Skipped when cross_kv is cached: the embedded context
        # is only consumed to build the cross-attn K/V, which are already precomputed, so this matmul
        # (and the encoder_hidden_states input) is redundant per step.
        ctx = None
        if cross_kv is None:
            ctx = ttnn.linear(
                encoder_hidden_states,
                self.condition_embedder_weight,
                bias=self.condition_embedder_bias,
                compute_kernel_config=cfg.compute_kernel_config,
            )

        # DiT layer stack. cross_kv (when provided) supplies precomputed per-layer cross-attn K/V so
        # each denoise step only projects Q against the cached context K/V.
        x = self.stack.forward(x, cos, sin, timestep_proj, ctx, sliding_mask=sliding_mask, cross_kv=cross_kv)

        # Output head: norm_out AdaLN + de-patchify.
        x = self.output.forward(x, temb)  # [1,1,T, out_channels]
        return x
