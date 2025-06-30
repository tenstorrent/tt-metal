# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import ttnn

from . import utils
from .linear import Linear, LinearParameters
from .substate import substate


@dataclass
class EmbeddingParameters:
    linear_1: LinearParameters
    linear_2: LinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, ttnn.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
    ) -> EmbeddingParameters:
        return cls(
            linear_1=LinearParameters.from_torch(substate(state, "linear_1"), dtype=dtype, device=device),
            linear_2=LinearParameters.from_torch(substate(state, "linear_2"), dtype=dtype, device=device),
        )


@dataclass
class CombinedTimestepTextProjEmbeddingsParameters:
    timestep_embedder: EmbeddingParameters
    text_embedder: EmbeddingParameters
    device: ttnn.MeshDevice

    @classmethod
    def from_torch(
        cls,
        state: dict[str, ttnn.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
    ) -> CombinedTimestepTextProjEmbeddingsParameters:
        return cls(
            timestep_embedder=EmbeddingParameters.from_torch(
                substate(state, "timestep_embedder"), dtype=dtype, device=device
            ),
            text_embedder=EmbeddingParameters.from_torch(substate(state, "text_embedder"), dtype=dtype, device=device),
            device=device,
        )


class CombinedTimestepTextProjEmbeddings:
    def __init__(self, parameters: CombinedTimestepTextProjEmbeddingsParameters) -> None:
        super().__init__()

        device = parameters.device

        self._timestep_embedder = _Embedding(parameters.timestep_embedder)
        self._text_embedder = _Embedding(parameters.text_embedder)

        self._time_proj_factor = self._create_time_proj_factor(num_channels=256, device=device)

    def forward(self, *, timestep: ttnn.Tensor, pooled_projection: ttnn.Tensor) -> ttnn.Tensor:
        assert timestep.dtype == ttnn.float32
        utils.signpost("timestep embedding")

        # The result is not correct when using normal multiplication with a timestep that contains
        # more than one entry. Fortunately, the shape of the tensors involved is such that matrix
        # multiplication is equivalent. We currently  use timestep tensors with only one entry, so
        # this does not really affect us.
        emb = timestep @ self._time_proj_factor
        timesteps_proj = ttnn.concat([ttnn.cos(emb), ttnn.sin(emb)], dim=-1)
        timesteps_proj = ttnn.clone(timesteps_proj, dtype=pooled_projection.dtype)

        time_embed = self._timestep_embedder.forward(timesteps_proj)
        text_embed = self._text_embedder.forward(pooled_projection)

        return time_embed + text_embed

    @staticmethod
    def _create_time_proj_factor(*, num_channels: int, device: ttnn.MeshDevice) -> ttnn.Tensor:
        assert num_channels % 2 == 0
        half_dim = num_channels // 2

        max_period = 10000

        exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        exponent = exponent / half_dim
        factor = torch.exp(exponent).unsqueeze(0)

        return ttnn.from_torch(
            factor,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )


class _Embedding:
    def __init__(self, parameters: EmbeddingParameters) -> None:
        super().__init__()

        self._linear_1 = Linear(parameters.linear_1)
        self._linear_2 = Linear(parameters.linear_2)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self._linear_1.forward(x, activation="silu")
        return self._linear_2.forward(x)
