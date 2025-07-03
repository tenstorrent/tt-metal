# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-FileCopyrightText: Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin

from .normalization import AdaLayerNormDummy
from .pos_embedding import FluxPosEmbed
from .timestep_embedding import CombinedTimestepTextProjEmbeddings
from .transformer_block import FluxSingleTransformerBlock, TransformerBlock

if TYPE_CHECKING:
    from collections.abc import Sequence


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/transformers/transformer_flux.py
class FluxTransformer(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(
        self,
        *,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        out_channels: int | None = None,
        guidance_embeds: bool = False,
        axes_dims_rope: Sequence[int] = (16, 56, 56),
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self._out_channels = out_channels or in_channels
        self._patch_size = patch_size
        self._in_channels = in_channels

        assert not guidance_embeds

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=list(axes_dims_rope))

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=inner_dim,
            pooled_projection_dim=pooled_projection_dim,
        )

        self.context_embedder = torch.nn.Linear(joint_attention_dim, inner_dim)
        self.x_embedder = torch.nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    dim=inner_dim,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    context_pre_only=False,
                    qk_norm="rms_norm",
                    use_dual_attention=False,
                )
                for i in range(num_layers)
            ]
        )

        self.single_transformer_blocks = torch.nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=inner_dim,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                )
                for i in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormDummy(inner_dim, 2 * inner_dim)
        self.proj_out = torch.nn.Linear(inner_dim, patch_size * patch_size * self._out_channels)

    def keep_blocks_only(self, blocks: int | None, single_blocks: int | None) -> None:
        if blocks is not None:
            del self.transformer_blocks[blocks:]
        if single_blocks is not None:
            del self.single_transformer_blocks[single_blocks:]

    def forward(
        self,
        spatial: torch.Tensor,
        prompt_embed: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        spatial = self.x_embedder(spatial)

        timestep = timestep.to(spatial.dtype)

        time_embed = self.time_text_embed(timestep, pooled_projections)
        prompt_embed = self.context_embedder(prompt_embed)

        for block in self.transformer_blocks:
            spatial, prompt_embed = block(
                spatial=spatial,
                prompt=prompt_embed,
                time_embed=time_embed,
                image_rotary_emb=image_rotary_emb,
            )

        combined = torch.cat([prompt_embed, spatial], dim=1)

        for block in self.single_transformer_blocks:
            combined = block(combined, time_embed=time_embed, image_rotary_emb=image_rotary_emb)

        spatial = combined[:, prompt_embed.shape[1] :]

        spatial = self.norm_out.norm(spatial)

        spatial_time = self.norm_out.linear(torch.nn.functional.silu(time_embed.unsqueeze(1)))
        scale, shift = torch.chunk(spatial_time, 2, dim=-1)
        spatial = spatial * (1 + scale) + shift

        return self.proj_out(spatial)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def patch_size(self) -> int:
        return self._patch_size
