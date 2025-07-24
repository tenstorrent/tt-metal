# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from . import utils
from .linear import Linear, LinearParameters
from .normalization import LayerNorm, LayerNormParameters
from .single_transformer_block import FluxSingleTransformerBlock, FluxSingleTransformerBlockParameters
from .substate import indexed_substates, substate
from .timestep_embedding import CombinedTimestepTextProjEmbeddings, CombinedTimestepTextProjEmbeddingsParameters
from .transformer_block import TransformerBlock, TransformerBlockParameters, chunk_time

if TYPE_CHECKING:
    import torch


@dataclass
class FluxTransformerParameters:
    x_embedder: LinearParameters
    time_text_embed: CombinedTimestepTextProjEmbeddingsParameters
    context_embedder: LinearParameters
    transformer_blocks: list[TransformerBlockParameters]
    single_transformer_blocks: list[FluxSingleTransformerBlockParameters]
    time_embed_out: LinearParameters
    norm_out: LayerNormParameters
    proj_out: LinearParameters
    device: ttnn.MeshDevice

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
    ) -> FluxTransformerParameters:
        _, mesh_width = device.shape
        embedding_dim = state["x_embedder.weight"].shape[0]

        return cls(
            x_embedder=LinearParameters.from_torch(
                substate(state, "x_embedder"),
                dtype=dtype,
                device=device,
                mesh_sharding_dim=1,
            ),
            time_text_embed=CombinedTimestepTextProjEmbeddingsParameters.from_torch(
                substate(state, "time_text_embed"), dtype=dtype, device=device
            ),
            context_embedder=LinearParameters.from_torch(
                substate(state, "context_embedder"),
                dtype=dtype,
                device=device,
                mesh_sharding_dim=0,
            ),
            transformer_blocks=[
                TransformerBlockParameters.from_torch(s, dtype=dtype, device=device)
                for s in indexed_substates(state, "transformer_blocks")
            ],
            single_transformer_blocks=[
                FluxSingleTransformerBlockParameters.from_torch(
                    s, dtype=dtype, device=device, linear_on_host=i > 20 and mesh_width == 1
                )
                for i, s in enumerate(indexed_substates(state, "single_transformer_blocks"))
            ],
            time_embed_out=LinearParameters.from_torch(
                substate(state, "norm_out.linear"),
                dtype=dtype,
                device=device,
                unsqueeze_bias=True,
                mesh_sharding_dim=1,
                chunks=2,
            ),
            norm_out=LayerNormParameters.from_torch(
                substate(state, "norm_out.norm"),
                dtype=dtype,
                device=device,
                weight_shape=[embedding_dim],
            ),
            proj_out=LinearParameters.from_torch(
                substate(state, "proj_out"), dtype=dtype, device=device, mesh_sharding_dim=0
            ),
            device=device,
        )


class FluxTransformer:
    def __init__(self, parameters: FluxTransformerParameters, *, num_attention_heads: int) -> None:
        super().__init__()

        self._device = parameters.device

        self._x_embedder = Linear(parameters.x_embedder)
        self._time_text_embed = CombinedTimestepTextProjEmbeddings(parameters.time_text_embed)
        self._context_embedder = Linear(parameters.context_embedder)
        self._transformer_blocks = [
            TransformerBlock(block, num_heads=num_attention_heads) for block in parameters.transformer_blocks
        ]
        self._single_transformer_blocks = [
            FluxSingleTransformerBlock(block, num_heads=num_attention_heads)
            for block in parameters.single_transformer_blocks
        ]
        self._time_embed_out = Linear(parameters.time_embed_out)
        self._norm_out = LayerNorm(parameters.norm_out, eps=1e-6)
        self._proj_out = Linear(parameters.proj_out)

    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled_projection: ttnn.Tensor,
        timestep: ttnn.Tensor,
        image_rotary_emb: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        _, mesh_width = self._device.shape

        prompt_sequence_length = prompt.shape[1]

        time_embed = self._time_text_embed.forward(timestep=timestep, pooled_projection=pooled_projection)
        ttnn.silu(time_embed, output_tensor=time_embed)

        if mesh_width > 1:
            spatial = utils.all_gather(
                spatial,
                dim=-2,
                cluster_axis=1,
                mesh_device=self._device,
                topology=ttnn.Topology.Linear,
            )

        spatial = self._x_embedder.forward(spatial)
        prompt = self._context_embedder.forward(prompt)

        time_embed = time_embed.reshape([time_embed.shape[0], 1, time_embed.shape[1]])

        for i, block in enumerate(self._transformer_blocks, start=1):
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                image_rotary_emb=image_rotary_emb,
                skip_time_embed_activation=True,
            )

            if i % 6 == 0:
                ttnn.DumpDeviceProfiler(spatial.device())

        prompt = ttnn.clone(prompt, dtype=spatial.dtype)

        combined = ttnn.concat([prompt, spatial], dim=1)
        del prompt, spatial

        for i, block in enumerate(self._single_transformer_blocks, start=1):
            combined = block.forward(
                combined=combined,
                time_embed=time_embed,
                image_rotary_emb=image_rotary_emb,
                skip_time_embed_activation=True,
            )

            if i % 6 == 0:
                ttnn.DumpDeviceProfiler(combined.device())

        spatial = combined[:, prompt_sequence_length:]
        del combined

        spatial = self._norm_out.forward(spatial)

        spatial_time = self._time_embed_out.forward(time_embed)
        [scale, shift] = chunk_time(spatial_time, 2)
        spatial = spatial * (1 + scale) + shift

        spatial = self._proj_out.forward(spatial, skip_reduce_scatter=True)
        return self._proj_out.reduce_scatter(spatial, scatter_dim=-2)
