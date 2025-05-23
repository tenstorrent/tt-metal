# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import ttnn
from models.experimental.stable_diffusion_35_large.tt.linear import TtLinear, TtLinearParameters

from . import utils
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
    ) -> TtEmbeddingParameters:
        return cls(
            linear_1=TtLinearParameters.from_torch(
                substate(state, "linear_1"), dtype=dtype, device=device, shard_dim=None
            ),
            linear_2=TtLinearParameters.from_torch(
                substate(state, "linear_2"), dtype=dtype, device=device, shard_dim=None
            ),
        )


@dataclass
class TtCombinedTimestepTextProjEmbeddingsParameters:
    timestep_embedder: TtEmbeddingParameters
    text_embedder: TtEmbeddingParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, ttnn.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtCombinedTimestepTextProjEmbeddingsParameters:
        return cls(
            timestep_embedder=TtEmbeddingParameters.from_torch(
                substate(state, "timestep_embedder"), dtype=dtype, device=device
            ),
            text_embedder=TtEmbeddingParameters.from_torch(
                substate(state, "text_embedder"), dtype=dtype, device=device
            ),
        )


class TtCombinedTimestepTextProjEmbeddings:
    def __init__(self, batch_size, parameters: TtCombinedTimestepTextProjEmbeddingsParameters, device) -> None:
        super().__init__()

        self._timestep_embedder = _TimestepEmbedding(parameters.timestep_embedder)
        self._text_embedder = _TimestepEmbedding(parameters.text_embedder)

        self._time_proj_factor = self._create_time_proj_factor(num_channels=256, batch_size=batch_size, device=device)

    def __call__(self, *, timestep: ttnn.Tensor, pooled_projection: ttnn.Tensor) -> ttnn.Tensor:
        assert timestep.dtype == ttnn.float32

        batch_size = timestep.shape[0]

        # time_proj_factor = ttnn.repeat(self._time_proj_factor, ttnn.Shape([batch_size, 1]))
        # time_proj_factor = ttnn.to_layout(time_proj_factor, ttnn.TILE_LAYOUT)
        time_proj_factor = ttnn.to_layout(self._time_proj_factor, ttnn.TILE_LAYOUT)

        emb = timestep * time_proj_factor

        c = ttnn.cos(emb)
        s = ttnn.sin(emb)

        timesteps_proj = ttnn.concat([c, s], dim=-1)
        timesteps_proj = ttnn.clone(timesteps_proj, dtype=pooled_projection.dtype)

        time_embed = self._timestep_embedder(timesteps_proj)
        text_embed = self._text_embedder(pooled_projection)

        return time_embed + text_embed

    @staticmethod
    def _create_time_proj_factor(*, num_channels: int, batch_size: int, device: ttnn.Device) -> ttnn.Tensor:
        assert num_channels % 2 == 0
        half_dim = num_channels // 2

        max_period = 10000

        exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        exponent = exponent / half_dim
        factor = torch.exp(exponent).unsqueeze(0).repeat(batch_size, 1)

        return ttnn.from_torch(factor, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device))


class _TimestepEmbedding:
    def __init__(self, parameters: TtEmbeddingParameters) -> None:
        super().__init__()

        self._linear_1 = TtLinear(parameters.linear_1)
        self._linear_2 = TtLinear(parameters.linear_2)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self._linear_1(x)
        x = utils.silu(x)
        return self._linear_2(x)

    @property
    def device(self) -> ttnn.Device:
        return self._linear_1.device
