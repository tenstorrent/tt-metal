# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
    guidance_embedder: EmbeddingParameters | None
    device: ttnn.MeshDevice
    guidance_embeds: bool

    @classmethod
    def from_torch(
        cls,
        state: dict[str, ttnn.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        guidance_embeds: bool = False,
    ) -> CombinedTimestepTextProjEmbeddingsParameters:
        guidance_embedder = None
        if guidance_embeds:
            print("Loading guidance embedder weights...")
            print("Available keys:", list(state.keys()))
            try:
                guidance_embedder = EmbeddingParameters.from_torch(
                    substate(state, "guidance_embedder"), dtype=dtype, device=device
                )
                print("Successfully loaded guidance embedder weights")
            except Exception as e:
                print(f"Failed to load guidance embedder weights: {e}")
                print("Trying alternate key 'guidance_embedder'...")
                try:
                    guidance_embedder = EmbeddingParameters.from_torch(
                        {
                            "linear_1": state["guidance_embedder.linear_1.weight"],
                            "linear_1.bias": state["guidance_embedder.linear_1.bias"],
                            "linear_2": state["guidance_embedder.linear_2.weight"],
                            "linear_2.bias": state["guidance_embedder.linear_2.bias"],
                        },
                        dtype=dtype,
                        device=device,
                    )
                    print("Successfully loaded guidance embedder weights using alternate key")
                except Exception as e2:
                    print(f"Failed with alternate key too: {e2}")
                    raise e

        return cls(
            timestep_embedder=EmbeddingParameters.from_torch(
                substate(state, "timestep_embedder"), dtype=dtype, device=device
            ),
            text_embedder=EmbeddingParameters.from_torch(substate(state, "text_embedder"), dtype=dtype, device=device),
            guidance_embedder=guidance_embedder,
            device=device,
            guidance_embeds=guidance_embeds,
        )


class CombinedTimestepTextProjEmbeddings:
    def __init__(self, parameters: CombinedTimestepTextProjEmbeddingsParameters) -> None:
        super().__init__()

        device = parameters.device

        self._timestep_embedder = _Embedding(parameters.timestep_embedder)
        self._text_embedder = _Embedding(parameters.text_embedder)
        self._guidance_embedder = None if not parameters.guidance_embeds else _Embedding(parameters.guidance_embedder)
        self._guidance_embeds = parameters.guidance_embeds

        self._time_proj_factor = self._create_time_proj_factor(num_channels=256, device=device)

    def forward(
        self, *, timestep: ttnn.Tensor, pooled_projection: ttnn.Tensor, guidance: ttnn.Tensor | None = None
    ) -> ttnn.Tensor:
        assert timestep.dtype == ttnn.float32
        utils.signpost("timestep embedding")

        # Time projection (same as HuggingFace time_proj)
        emb = timestep @ self._time_proj_factor
        timesteps_proj = ttnn.concat([ttnn.cos(emb), ttnn.sin(emb)], dim=-1)
        timesteps_proj = ttnn.clone(timesteps_proj, dtype=pooled_projection.dtype)

        # Timestep embedding
        timesteps_emb = self._timestep_embedder.forward(timesteps_proj)

        if self._guidance_embeds and guidance is not None:
            # Guidance projection (same time_proj as timestep)
            guidance_emb = guidance @ self._time_proj_factor
            guidance_proj = ttnn.concat([ttnn.cos(guidance_emb), ttnn.sin(guidance_emb)], dim=-1)
            guidance_proj = ttnn.clone(guidance_proj, dtype=pooled_projection.dtype)

            # Guidance embedding
            guidance_emb = self._guidance_embedder.forward(guidance_proj)

            # Combine time and guidance embeddings FIRST (like HuggingFace)
            time_guidance_emb = timesteps_emb + guidance_emb

            # Text embedding
            pooled_projections = self._text_embedder.forward(pooled_projection)

            # Final conditioning
            conditioning = time_guidance_emb + pooled_projections
            return conditioning
        else:
            # Original Schnell behavior
            time_embed = timesteps_emb
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
