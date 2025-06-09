# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .substate import indexed_substates, substate
from .fun_linear import sd_linear, TtLinearParameters
from .fun_normalization import sd_layer_norm, TtLayerNormParameters
from .fun_patch_embedding import sd_patch_embed, TtPatchEmbedParameters
from .fun_timestep_embedding import sd_combined_timestep_embed, TtCombinedTimestepTextProjEmbeddingsParameters
from .fun_transformer_block import sd_transformer_block, TtTransformerBlockParameters, chunk_time
from .parallel_config import DiTParallelConfig, StableDiffusionParallelManager


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
        unpadded_num_heads: int,
        embedding_dim: int,
        hidden_dim_padding: int,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig,
        guidance_cond: int,
        height: int,
        width: int,
    ) -> TtSD3Transformer2DModelParameters:
        return cls(
            pos_embed=TtPatchEmbedParameters.from_torch(
                substate(state, "pos_embed"),
                device=device,
                hidden_dim_padding=hidden_dim_padding,
                out_channels=embedding_dim,
                parallel_config=parallel_config,
                height=height,
                width=width,
            ),
            time_text_embed=TtCombinedTimestepTextProjEmbeddingsParameters.from_torch(
                substate(state, "time_text_embed"),
                dtype=dtype,
                device=device,
                guidance_cond=guidance_cond,
                hidden_dim_padding=hidden_dim_padding,
                parallel_config=parallel_config,
            ),
            context_embedder=TtLinearParameters.from_torch(
                substate(state, "context_embedder"),
                dtype=dtype,
                device=device,
                shard_dim=-1,
                hidden_dim_padding=hidden_dim_padding,
                parallel_config=parallel_config,
            ),
            transformer_blocks=[
                TtTransformerBlockParameters.from_torch(
                    s,
                    num_heads=num_heads,
                    unpadded_num_heads=unpadded_num_heads,
                    hidden_dim_padding=hidden_dim_padding,
                    dtype=dtype,
                    device=device,
                    parallel_config=parallel_config,
                )
                for s in indexed_substates(state, "transformer_blocks")
            ],
            time_embed_out=TtLinearParameters.from_torch(
                substate(state, "norm_out.linear"),
                dtype=dtype,
                device=device,
                shard_dim=None,
                unsqueeze_bias=True,
                hidden_dim_padding=hidden_dim_padding,
                parallel_config=parallel_config,
            ),
            norm_out=TtLayerNormParameters.from_torch(
                substate(state, "norm_out.norm"), dtype=dtype, device=device, distributed=False
            ),
            proj_out=TtLinearParameters.from_torch(
                substate(state, "proj_out"),
                dtype=dtype,
                device=device,
                shard_dim=None,
                hidden_dim_padding=hidden_dim_padding,
                parallel_config=parallel_config,
            ),
        )


def sd_transformer(
    *,
    spatial: ttnn.Tensor,
    prompt: ttnn.Tensor,
    pooled_projection: ttnn.Tensor,
    timestep: ttnn.Tensor,
    parameters: TtSD3Transformer2DModelParameters,
    parallel_manager: StableDiffusionParallelManager,
    num_heads: int,
    N: int,
    L: int,
    cfg_index: int,
) -> ttnn.Tensor:
    spatial = sd_patch_embed(spatial, parameters.pos_embed, parallel_manager=parallel_manager)
    time_embed = sd_combined_timestep_embed(
        timestep=timestep, pooled_projection=pooled_projection, parameters=parameters.time_text_embed
    )
    prompt = sd_linear(prompt, parameters.context_embedder)
    time_embed = time_embed.reshape([time_embed.shape[0], 1, 1, time_embed.shape[1]])
    spatial = ttnn.unsqueeze(spatial, 1)
    prompt = ttnn.unsqueeze(prompt, 1)

    local_heads = num_heads // parallel_manager.dit_parallel_config.tensor_parallel.factor
    full_seq_len = spatial.shape[2] * parallel_manager.dit_parallel_config.sequence_parallel.factor
    kv_gathered_shape = [spatial.shape[0], local_heads, full_seq_len, spatial.shape[3] // local_heads]

    parallel_manager.maybe_init_persistent_buffers(kv_gathered_shape)

    for i, block in enumerate(parameters.transformer_blocks, start=1):
        spatial, prompt_out = sd_transformer_block(
            spatial=spatial,
            prompt=prompt,
            time_embed=time_embed,
            parameters=block,
            parallel_manager=parallel_manager,
            num_heads=num_heads,
            N=N,  # spatial_sequence_length
            L=L,  # prompt_sequence_length
            cfg_index=cfg_index,
        )
        if prompt_out is not None:
            prompt = prompt_out
    spatial_time = sd_linear(ttnn.silu(time_embed), parameters.time_embed_out)
    [scale, shift] = chunk_time(spatial_time, 2)
    if parallel_manager.is_sequence_parallel:
        spatial = ttnn.experimental.all_gather_async(
            spatial,
            dim=-2,
            cluster_axis=parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis,
            mesh_device=spatial.device(),
            topology=parallel_manager.dit_parallel_config.topology,
            multi_device_global_semaphore=parallel_manager.cfg_semaphores[cfg_index]["ag"],
            num_links=1,
        )
    if parallel_manager.is_tensor_parallel:
        spatial = ttnn.experimental.all_gather_async(
            spatial,
            dim=-1,
            cluster_axis=parallel_manager.dit_parallel_config.tensor_parallel.mesh_axis,
            mesh_device=spatial.device(),
            topology=parallel_manager.dit_parallel_config.topology,
            multi_device_global_semaphore=parallel_manager.cfg_semaphores[cfg_index]["ag"],
            num_links=1,
        )

    spatial = (
        sd_layer_norm(spatial, parameters.norm_out, parallel_manager=parallel_manager, cfg_index=cfg_index)
        * (1 + scale)
        + shift
    )
    return sd_linear(spatial, parameters.proj_out)

    # def cache_and_trace(
    #     self,
    #     *,
    #     spatial: ttnn.Tensor,
    #     prompt: ttnn.Tensor,
    #     pooled_projection: ttnn.Tensor,
    #     timestep: ttnn.Tensor,
    # ) -> TtSD3Transformer2DModelTrace:
    #     device = spatial.device()

    #     self(spatial=spatial, prompt=prompt, pooled_projection=pooled_projection, timestep=timestep)

    #     tid = ttnn.begin_trace_capture(device)
    #     output = self(spatial=spatial, prompt=prompt, pooled_projection=pooled_projection, timestep=timestep)
    #     ttnn.end_trace_capture(device, tid)

    #     return TtSD3Transformer2DModelTrace(
    #         spatial_input=spatial,
    #         prompt_input=prompt,
    #         pooled_projection_input=pooled_projection,
    #         timestep_input=timestep,
    #         output=output,
    #         tid=tid,
    #     )

    # @property
    # def patch_size(self) -> int:
    #     return self._patch_size


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
