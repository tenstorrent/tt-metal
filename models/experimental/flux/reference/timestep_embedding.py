# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-FileCopyrightText: Copyright 2024 The HuggingFace Team. All rights reserved.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/embeddings.py
class CombinedTimestepTextProjEmbeddings(torch.nn.Module):
    def __init__(self, *, embedding_dim: int, pooled_projection_dim: int) -> None:
        super().__init__()

        self.timestep_embedder = _Embedding(in_features=256, hidden_size=embedding_dim)
        self.text_embedder = _Embedding(in_features=pooled_projection_dim, hidden_size=embedding_dim)

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        time_proj_factor = self._create_time_proj_factor(num_channels=256, device=timestep.device)

        emb = timestep.unsqueeze(-1).float() * time_proj_factor
        timesteps_proj = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

        time_embed = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        text_embed = self.text_embedder(pooled_projection)

        return time_embed + text_embed

    @staticmethod
    def _create_time_proj_factor(*, num_channels: int, device: torch.device) -> torch.Tensor:
        assert num_channels % 2 == 0
        half_dim = num_channels // 2

        max_period = 10000

        exponent = -math.log(max_period) * torch.arange(
            start=0,
            end=half_dim,
            dtype=torch.float32,
            device=device,
        )
        exponent = exponent / half_dim

        return torch.exp(exponent).unsqueeze(0)


class _Embedding(torch.nn.Module):
    def __init__(self, *, in_features: int, hidden_size: int) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_features=in_features, out_features=hidden_size)
        self.linear_2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.act = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        return self.linear_2(x)
