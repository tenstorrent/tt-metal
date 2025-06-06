# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/embeddings.py
class CombinedTimestepTextProjEmbeddings(torch.nn.Module):
    def __init__(self, *, embedding_dim: int, pooled_projection_dim: int) -> None:
        super().__init__()

        self.timestep_embedder = _TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = _PixArtAlphaTextProjection(in_features=pooled_projection_dim, hidden_size=embedding_dim)

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        timesteps_proj = _time_proj(num_channels=256, timesteps=timestep)

        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))

        return timesteps_emb + self.text_embedder(pooled_projection)


class _TimestepEmbedding(torch.nn.Module):
    def __init__(self, *, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_channels, time_embed_dim)
        self.act = torch.nn.SiLU()
        self.linear_2 = torch.nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


def _time_proj(num_channels: int, timesteps: torch.Tensor) -> torch.Tensor:
    assert num_channels % 2 == 0
    half_dim = num_channels // 2

    max_period = 10000

    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / half_dim

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class _PixArtAlphaTextProjection(torch.nn.Module):
    def __init__(self, *, in_features: int, hidden_size: int) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_features=in_features, out_features=hidden_size)
        self.linear_2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.act_1 = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act_1(x)
        return self.linear_2(x)
