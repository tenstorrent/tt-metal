# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-FileCopyrightText: Copyright 2024 The HuggingFace Team. All rights reserved.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from .attention import Attention
from .feed_forward import FeedForward
from .normalization import AdaLayerNormDummy


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/transformers/transformer_flux.py
class FluxSingleTransformerBlock(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim

        self.norm = AdaLayerNormDummy(dim, 3 * dim)
        self.proj_mlp = torch.nn.Linear(dim, 4 * dim)
        self.act_mlp = torch.nn.GELU(approximate="tanh")
        self.proj_out = torch.nn.Linear(5 * dim, dim)

        self.attn = Attention(
            query_dim=dim,
            dim_head=head_dim,
            heads=num_heads,
            out_dim=dim,
            qk_norm="rms_norm",
            pre_only=True,
        )

    def forward(
        self,
        combined: torch.Tensor,
        *,
        time_embed: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        emb = self.norm.linear(torch.nn.functional.silu(time_embed))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        norm_combined = self.norm.norm(combined) * (1 + scale_msa[:, None]) + shift_msa[:, None]

        mlp_combined = self.act_mlp(self.proj_mlp(norm_combined))
        attn_output, _ = self.attn(spatial=norm_combined, image_rotary_emb=image_rotary_emb)

        additional = torch.cat([attn_output, mlp_combined], dim=2)
        gate_msa = gate_msa.unsqueeze(1)
        additional = gate_msa * self.proj_out(additional)

        combined += additional
        if combined.dtype == torch.float16:
            combined = combined.clip(-65504, 65504)

        return combined


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention.py
class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        head_dim: int,
        context_pre_only: bool,
        qk_norm: str,
        use_dual_attention: bool,
    ) -> None:
        super().__init__()

        self.context_pre_only = context_pre_only
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.attn = Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=head_dim,
            heads=num_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            qk_norm=qk_norm,
        )

        self.norm2 = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, approximate="tanh")

        if context_pre_only:
            self.norm1_context = AdaLayerNormDummy(dim, 2 * dim)
            self.norm2_context = None
            self.ff_context = None
        else:
            self.norm1_context = AdaLayerNormDummy(dim, 6 * dim)
            self.norm2_context = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, approximate="tanh")

        if use_dual_attention:
            self.norm1 = AdaLayerNormDummy(dim, 9 * dim)
            self.attn2 = Attention(
                query_dim=dim,
                dim_head=head_dim,
                heads=num_heads,
                out_dim=dim,
                qk_norm=qk_norm,
            )
        else:
            self.norm1 = AdaLayerNormDummy(dim, 6 * dim)
            self.attn2 = None

    def _spatial_attn_block(
        self,
        inp: torch.Tensor,
        *,
        gate: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.attn2 is not None

        scaled = inp * (1 + scale) + shift
        attn, _ = self.attn2(spatial=scaled, image_rotary_emb=image_rotary_emb)
        return gate * attn

    def _dual_attn_block(
        self,
        *,
        spatial: torch.Tensor,
        prompt: torch.Tensor,
        spatial_gate: torch.Tensor,
        prompt_gate: torch.Tensor | None,
        prompt_scale: torch.Tensor,
        prompt_shift: torch.Tensor,
        spatial_scale: torch.Tensor,
        spatial_shift: torch.Tensor,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        spatial_scaled = spatial * (1 + spatial_scale) + spatial_shift
        prompt_scaled = prompt * (1 + prompt_scale) + prompt_shift

        spatial_attn, prompt_attn = self.attn(
            spatial=spatial_scaled,
            prompt=prompt_scaled,
            image_rotary_emb=image_rotary_emb,
        )

        spatial_attn = spatial_gate * spatial_attn
        prompt_attn = prompt_gate * prompt_attn if prompt_gate is not None else None

        return spatial_attn, prompt_attn

    def _spatial_ff_block(
        self,
        inp: torch.Tensor,
        *,
        gate: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        scaled = inp * (1 + scale) + shift
        return gate * self.ff(scaled)

    def _prompt_ff_block(
        self,
        inp: torch.Tensor,
        *,
        gate: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        assert self.ff_context is not None

        scaled = inp * (1 + scale) + shift
        return gate * self.ff_context(scaled)

    def forward(
        self,
        spatial: torch.Tensor,
        prompt: torch.Tensor,
        time_embed: torch.Tensor,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        time_embed = time_embed.unsqueeze(1)

        spatial_time = self.norm1.linear(torch.nn.functional.silu(time_embed))
        prompt_time = self.norm1_context.linear(torch.nn.functional.silu(time_embed))

        if self.attn2 is not None:
            (
                spatial_shift_dual_attn,
                spatial_scale_dual_attn,
                spatial_gate_dual_attn,
                spatial_shift_ff,
                spatial_scale_ff,
                spatial_gate_ff,
                spatial_shift_attn,
                spatial_scale_attn,
                spatial_gate_attn,
            ) = spatial_time.chunk(9, dim=-1)
        else:
            (
                spatial_shift_dual_attn,
                spatial_scale_dual_attn,
                spatial_gate_dual_attn,
                spatial_shift_ff,
                spatial_scale_ff,
                spatial_gate_ff,
            ) = spatial_time.chunk(6, dim=-1)

            spatial_gate_attn = None
            spatial_shift_attn = None
            spatial_scale_attn = None

        if self.context_pre_only:
            prompt_gate_attn = None
            prompt_shift_ff = None
            prompt_scale_ff = None
            prompt_gate_ff = None

            prompt_scale_attn, prompt_shift_attn = torch.chunk(prompt_time, 2, dim=-1)
        else:
            (
                prompt_shift_attn,
                prompt_scale_attn,
                prompt_gate_attn,
                prompt_shift_ff,
                prompt_scale_ff,
                prompt_gate_ff,
            ) = prompt_time.chunk(6, dim=-1)

        spatial_normed = self.norm1.norm(spatial)
        prompt_normed = self.norm1_context.norm(prompt)

        spatial_attn, prompt_attn = self._dual_attn_block(
            spatial=spatial_normed,
            prompt=prompt_normed,
            spatial_gate=spatial_gate_dual_attn,
            prompt_gate=prompt_gate_attn,
            prompt_scale=prompt_scale_attn,
            prompt_shift=prompt_shift_attn,
            spatial_scale=spatial_scale_dual_attn,
            spatial_shift=spatial_shift_dual_attn,
            image_rotary_emb=image_rotary_emb,
        )

        spatial += spatial_attn

        if self.attn2 is not None:
            assert spatial_gate_attn is not None
            assert spatial_scale_attn is not None
            assert spatial_shift_attn is not None

            spatial += self._spatial_attn_block(
                spatial_normed,
                gate=spatial_gate_attn,
                scale=spatial_scale_attn,
                shift=spatial_shift_attn,
                image_rotary_emb=image_rotary_emb,
            )

        spatial_normed = self.norm2(spatial)
        spatial += self._spatial_ff_block(
            spatial_normed,
            gate=spatial_gate_ff,
            scale=spatial_scale_ff,
            shift=spatial_shift_ff,
        )

        if self.context_pre_only:
            return spatial, None

        assert self.norm2_context is not None
        assert prompt_scale_ff is not None
        assert prompt_shift_ff is not None
        assert prompt_gate_ff is not None

        prompt += prompt_attn

        prompt_normed = self.norm2_context(prompt)
        prompt += self._prompt_ff_block(
            prompt_normed,
            gate=prompt_gate_ff,
            scale=prompt_scale_ff,
            shift=prompt_shift_ff,
        )

        return spatial, prompt
