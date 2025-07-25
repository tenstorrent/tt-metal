# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import ttnn
from models.experimental.stable_diffusion_35_large.tt.fun_linear import sd_linear, TtLinearParameters
from .parallel_config import DiTParallelConfig
from .substate import substate


@dataclass
class TtEmbeddingParameters:
    linear_1: TtLinearParameters
    linear_2: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, ttnn.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        hidden_dim_padding: int,
        parallel_config: DiTParallelConfig,
    ) -> TtEmbeddingParameters:
        return cls(
            linear_1=TtLinearParameters.from_torch(
                substate(state, "linear_1"),
                dtype=dtype,
                device=device,
                shard_dim=None,
                hidden_dim_padding=hidden_dim_padding,
                parallel_config=parallel_config,
            ),
            linear_2=TtLinearParameters.from_torch(
                substate(state, "linear_2"),
                dtype=dtype,
                device=device,
                shard_dim=None,
                hidden_dim_padding=hidden_dim_padding,
                parallel_config=parallel_config,
            ),
        )


@dataclass
class TtCombinedTimestepTextProjEmbeddingsParameters:
    timestep_embedder: TtEmbeddingParameters
    text_embedder: TtEmbeddingParameters
    time_proj_factor: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, ttnn.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        guidance_cond: int,
        hidden_dim_padding: int,
        parallel_config: DiTParallelConfig,
    ) -> TtCombinedTimestepTextProjEmbeddingsParameters:
        return cls(
            timestep_embedder=TtEmbeddingParameters.from_torch(
                substate(state, "timestep_embedder"),
                dtype=dtype,
                device=device,
                hidden_dim_padding=hidden_dim_padding,
                parallel_config=parallel_config,
            ),
            text_embedder=TtEmbeddingParameters.from_torch(
                substate(state, "text_embedder"),
                dtype=dtype,
                device=device,
                hidden_dim_padding=hidden_dim_padding,
                parallel_config=parallel_config,
            ),
            time_proj_factor=cls._create_time_proj_factor(num_channels=256, batch_size=guidance_cond, device=device),
        )

    @staticmethod
    def _create_time_proj_factor(*, num_channels: int, batch_size: int, device: ttnn.Device) -> ttnn.Tensor:
        assert num_channels % 2 == 0
        half_dim = num_channels // 2

        max_period = 10000

        exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        exponent = exponent / half_dim
        factor = torch.exp(exponent).unsqueeze(0).repeat(batch_size, 1)  # TODO: Can this broadcast be handled by ttnn?

        return ttnn.from_torch(
            factor,
            device=device,
            mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=[None, None]),
        )


def sd_combined_timestep_embed(
    timestep: ttnn.Tensor, pooled_projection: ttnn.Tensor, parameters: TtCombinedTimestepTextProjEmbeddingsParameters
) -> ttnn.Tensor:
    assert timestep.dtype == ttnn.float32

    batch_size = timestep.shape[0]

    # time_proj_factor = ttnn.repeat(self._time_proj_factor, ttnn.Shape([batch_size, 1]))
    # time_proj_factor = ttnn.to_layout(time_proj_factor, ttnn.TILE_LAYOUT)
    time_proj_factor = ttnn.to_layout(parameters.time_proj_factor, ttnn.TILE_LAYOUT)

    emb = timestep * time_proj_factor

    c = ttnn.cos(emb)
    s = ttnn.sin(emb)

    timesteps_proj = ttnn.concat([c, s], dim=-1)
    timesteps_proj = ttnn.clone(timesteps_proj, dtype=pooled_projection.dtype)

    time_embed = sd_timestep_embed(timesteps_proj, parameters.timestep_embedder)
    text_embed = sd_timestep_embed(pooled_projection, parameters.text_embedder)

    return time_embed + text_embed


def sd_timestep_embed(x: ttnn.Tensor, parameters: TtEmbeddingParameters) -> ttnn.Tensor:
    x = sd_linear(x, parameters.linear_1)
    x = ttnn.silu(x)  # Note: ttnn.silu can have poor accuracy
    return sd_linear(x, parameters.linear_2)
