# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin

from .normalization import AdaLayerNormDummy
from .patch_embedding import PatchEmbed
from .timestep_embedding import CombinedTimestepTextProjEmbeddings
from .transformer_block import TransformerBlock


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/controlnet_sd3.py
class SD3Transformer2DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(
        self,
        *,
        sample_size: int = 128,  # noqa: ARG002
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: tuple[int, ...] = (),
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self._out_channels = out_channels
        self._patch_size = patch_size
        self._in_channels = in_channels

        self.pos_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=inner_dim,
            pooled_projection_dim=pooled_projection_dim,
        )
        self.context_embedder = torch.nn.Linear(joint_attention_dim, caption_projection_dim)

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    dim=inner_dim,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                    qk_norm=qk_norm,
                    use_dual_attention=i in dual_attention_layers,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormDummy(inner_dim, 2 * inner_dim)
        self.proj_out = torch.nn.Linear(inner_dim, patch_size * patch_size * self._out_channels)

    def forward(
        self,
        spatial: torch.Tensor,
        prompt_embed: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        spatial = self.pos_embed(spatial)
        time_embed = self.time_text_embed(timestep, pooled_projections)
        prompt_embed = self.context_embedder(prompt_embed)

        for block in self.transformer_blocks:
            spatial, prompt_embed = block(
                spatial=spatial,
                prompt=prompt_embed,
                time_embed=time_embed,
            )

        spatial_time = self.norm_out.linear(torch.nn.functional.silu(time_embed.unsqueeze(1)))
        scale, shift = torch.chunk(spatial_time, 2, dim=-1)
        spatial = self.norm_out.norm(spatial) * (1 + scale) + shift

        return self.proj_out(spatial)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def patch_size(self) -> int:
        return self._patch_size
