# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import ttnn
from models.experimental.stable_diffusion3.tt.linear import TtLinear, TtLinearParameters

from .normalization import TtLayerNorm, TtLayerNormParameters
from .patch_embedding import TtPatchEmbed, TtPatchEmbedParameters
from .substate import indexed_substates, substate
from .timestep_embedding import TtCombinedTimestepTextProjEmbeddings, TtCombinedTimestepTextProjEmbeddingsParameters
from .transformer_block import TtTransformerBlock, TtTransformerBlockParameters, chunk_device_tensors, chunk_time


@dataclass
class TtSD3Transformer2DModelParameters:
    pos_embed: TtPatchEmbedParameters
    time_text_embed: TtCombinedTimestepTextProjEmbeddingsParameters
    context_embedder: TtLinearParameters
    transformer_blocks: list[TtTransformerBlockParameters]
    time_embed_out: TtLinearParameters
    norm_out: TtLayerNormParameters
    proj_out: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        num_heads: int,
        embedding_dim: int,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtSD3Transformer2DModelParameters:
        return cls(
            pos_embed=TtPatchEmbedParameters.from_torch(
                substate(state, "pos_embed"), device=device, out_channels=embedding_dim
            ),
            time_text_embed=TtCombinedTimestepTextProjEmbeddingsParameters.from_torch(
                substate(state, "time_text_embed"), dtype=dtype, device=device
            ),
            context_embedder=TtLinearParameters.from_torch(
                substate(state, "context_embedder"), dtype=dtype, device=device, shard_dim=None
            ),
            transformer_blocks=[
                TtTransformerBlockParameters.from_torch(s, num_heads=num_heads, dtype=dtype, device=device)
                for s in indexed_substates(state, "transformer_blocks")
            ],
            time_embed_out=TtLinearParameters.from_torch(
                substate(state, "norm_out.linear"), dtype=dtype, device=device, shard_dim=None, unsqueeze_bias=True
            ),
            norm_out=TtLayerNormParameters.from_torch(
                substate(state, "norm_out.norm"), dtype=dtype, device=device, distributed=False
            ),
            proj_out=TtLinearParameters.from_torch(
                substate(state, "proj_out"), dtype=dtype, device=device, shard_dim=None
            ),
        )


class ShardingProjection:
    def __init__(self, *, dim: int, device: ttnn.MeshDevice) -> None:
        params = TtLinearParameters.from_torch(
            dict(weight=torch.eye(dim)),
            dtype=ttnn.bfloat8_b,
            device=device,
            shard_dim=-1,
        )
        self._projection = TtLinear(params)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self._projection(x)


class TtSD3Transformer2DModel:
    def __init__(
        self,
        parameters: TtSD3Transformer2DModelParameters,
        *,
        guidance_cond: int = 2,
        num_heads: int,
        device,
    ) -> None:
        super().__init__()

        self.mesh_device = device
        self._pos_embed = TtPatchEmbed(parameters.pos_embed, device)
        self._time_text_embed = TtCombinedTimestepTextProjEmbeddings(guidance_cond, parameters.time_text_embed, device)
        self._context_embedder = TtLinear(parameters.context_embedder)
        self._transformer_blocks = [
            TtTransformerBlock(block, num_heads=num_heads, device=device) for block in parameters.transformer_blocks
        ]
        self._time_embed_out = TtLinear(parameters.time_embed_out)
        self._norm_out = TtLayerNorm(parameters.norm_out, eps=1e-6)
        self._proj_out = TtLinear(parameters.proj_out)

        self._patch_size = parameters.pos_embed.patch_size

        # TODO: get dimensions from other parameters
        self._sharding = ShardingProjection(dim=2432, device=device)

    def __call__(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled_projection: ttnn.Tensor,
        timestep: ttnn.Tensor,
        N: int,
        L: int,
    ) -> ttnn.Tensor:
        spatial = self._pos_embed(spatial)
        # to avoid OOM inside the first transformer block
        spatial = ttnn.to_memory_config(spatial, ttnn.DRAM_MEMORY_CONFIG)

        time_embed = self._time_text_embed(timestep=timestep, pooled_projection=pooled_projection)
        prompt = self._context_embedder(prompt)
        time_embed = time_embed.reshape([time_embed.shape[0], 1, 1, time_embed.shape[1]])

        spatial = ttnn.unsqueeze(spatial, 1)
        assert spatial.shape[-2] % 32 == 0
        prompt = ttnn.unsqueeze(prompt, 1)
        assert prompt.shape[-2] % 32 == 0

        spatial = self._sharding(spatial)
        prompt = self._sharding(prompt)
        # num_devices = self.mesh_device.get_num_devices()
        # spatial_slices = chunk_device_tensors(spatial, num_devices)
        # prompt_slices = chunk_device_tensors(prompt, num_devices)
        # spatial = ttnn.aggregate_as_tensor(spatial_slices[::-1])
        # prompt = ttnn.aggregate_as_tensor(prompt_slices[::-1])

        for i, block in enumerate(self._transformer_blocks, start=1):
            spatial, prompt_out = block(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                N=N,  # spatial_sequence_length
                L=L,  # prompt_sequence_length
            )
            if prompt_out is not None:
                prompt = prompt_out

            if i % 6 == 0:
                ttnn.DumpDeviceProfiler(spatial.device())

        spatial_time = self._time_embed_out(ttnn.silu(time_embed))
        [scale, shift] = chunk_time(spatial_time, 2)
        spatial = ttnn.all_gather(spatial, dim=-1)
        spatial = self._norm_out(spatial) * (1 + scale) + shift
        return self._proj_out(spatial)

    def cache_and_trace(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled_projection: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> TtSD3Transformer2DModelTrace:
        device = spatial.device()

        self(spatial=spatial, prompt=prompt, pooled_projection=pooled_projection, timestep=timestep)

        tid = ttnn.begin_trace_capture(device)
        output = self(spatial=spatial, prompt=prompt, pooled_projection=pooled_projection, timestep=timestep)
        ttnn.end_trace_capture(device, tid)

        return TtSD3Transformer2DModelTrace(
            spatial_input=spatial,
            prompt_input=prompt,
            pooled_projection_input=pooled_projection,
            timestep_input=timestep,
            output=output,
            tid=tid,
        )

    @property
    def patch_size(self) -> int:
        return self._patch_size


@dataclass
class TtSD3Transformer2DModelTrace:
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    pooled_projection_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    output: ttnn.Tensor
    tid: int

    def __call__(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled_projection: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> ttnn.Tensor:
        device = self.spatial_input.device()

        ttnn.copy_host_to_device_tensor(spatial, self.spatial_input)
        ttnn.copy_host_to_device_tensor(prompt, self.prompt_input)
        ttnn.copy_host_to_device_tensor(pooled_projection, self.pooled_projection_input)
        ttnn.copy_host_to_device_tensor(timestep, self.timestep_input)

        ttnn.execute_trace(device, self.tid)

        return self.output
