# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from einops import rearrange

from models.experimental.audiox.reference.continuous_transformer import ContinuousTransformer
from models.experimental.audiox.reference.time_embedding import FourierFeatures


class DiffusionTransformer(nn.Module):
    """AudioX DiT outer wrapper: timestep + cond projections, prepend conditioning,
    1x1 conv residuals around a stack of continuous-transformer blocks. Mirrors
    audiox/models/dit.py:DiffusionTransformer for the inference path AudioX
    actually exercises (no adaLN/global_embed, no input_concat, no
    prepend_cond_dim, no CFG, no patching, no return_info)."""

    def __init__(
        self,
        io_channels: int = 64,
        embed_dim: int = 1536,
        depth: int = 24,
        num_heads: int = 24,
        cond_token_dim: int = 768,
        project_cond_tokens: bool = False,
    ):
        super().__init__()
        self.io_channels = io_channels
        self.embed_dim = embed_dim

        timestep_features_dim = 256
        self.timestep_features = FourierFeatures(1, timestep_features_dim)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        cond_embed_dim = embed_dim if project_cond_tokens else cond_token_dim
        self.to_cond_embed = nn.Sequential(
            nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim, bias=False),
        )

        self.transformer = ContinuousTransformer(
            dim=embed_dim,
            depth=depth,
            dim_heads=embed_dim // num_heads,
            dim_in=io_channels,
            dim_out=io_channels,
            cross_attend=True,
            cond_token_dim=cond_embed_dim,
        )

        self.preprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cross_attn_cond: torch.Tensor) -> torch.Tensor:
        cond_embed = self.to_cond_embed(cross_attn_cond)
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None]))
        prepend_inputs = timestep_embed.unsqueeze(1)

        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c t -> b t c")

        output = self.transformer(x, prepend_embeds=prepend_inputs, context=cond_embed)

        output = rearrange(output, "b t c -> b c t")[:, :, 1:]
        output = self.postprocess_conv(output) + output
        return output
