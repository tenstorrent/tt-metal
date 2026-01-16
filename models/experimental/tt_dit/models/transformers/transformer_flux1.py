# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn

from ...blocks.attention import Attention
from ...blocks.transformer_block import TransformerBlock, _chunk_time3d
from ...layers.embeddings import CombinedTimestepGuidanceTextProjEmbeddings
from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear, prepare_chunked_linear_output
from ...layers.module import Module, ModuleList
from ...layers.normalization import DistributedLayerNorm
from ...utils.substate import rename_substate
from models.common.utility_functions import is_blackhole

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager
    from ...utils.padding import PaddingConfig


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/transformers/transformer_flux.py
class Flux1SingleTransformerBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        head_dim: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
        attention_k_chunk_size: int = 512,
        attention_q_chunk_size: int = 128,
    ) -> None:
        super().__init__()

        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        mlp_hidden_dim = 4 * dim

        self.attn = Attention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            added_kv_proj_dim=0,
            pre_only=True,
            eps=1e-6,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
            use_spatial_weights_for_prompt=True,
            k_chunk_size=attention_k_chunk_size,
            q_chunk_size=attention_q_chunk_size,
        )

        self.norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        # Shard output, since size of input dimension << size of output dimension.
        self.time_embed = ColParallelLinear(
            dim,
            3 * dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        # Shard output, since size of input dimension << size of output dimension.
        self.proj_mlp = ColParallelLinear(
            dim,
            mlp_hidden_dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        # Shard input, since size of input dimension >> size of output dimension.
        self.proj_out = RowParallelLinear(
            dim + mlp_hidden_dim,
            dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm.linear", "time_embed")

        embedding_dim = state["time_embed.weight"].shape[1]

        prepare_chunked_linear_output(
            state,
            prefix="time_embed",
            device_count=self.parallel_config.tensor_parallel.factor,
            chunks=3,
        )

        if "proj_out.weight" in state:
            state["proj_out.weight"] = _re_fuse_proj_out_weight(
                state["proj_out.weight"],
                embedding_dim=embedding_dim,
                device_count=self.parallel_config.tensor_parallel.factor,
            )

    # Since we do not have operations to concatenate and slice a tensor along a sharded dimension,
    # we keep the spatial and prompt tensors separate for now.
    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        time_embed: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        skip_time_embed_activation_fn: bool = False,
        spatial_sequence_length: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the model forward.

        Args:
            combined: Tensor with shape [batch_size, sequence_length / sp_factor, query_dim / tp_factor].
            time_embed: Tensor with shape [batch_size, 1, query_dim].
            rope: Tuple of two tensors with shape [sequence_length / sp_factor, head_dim].
        """
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        if not skip_time_embed_activation_fn:
            time_embed = ttnn.silu(time_embed)
        time = self.time_embed(time_embed)

        shift_msa, scale_msa, gate_msa = _chunk_time3d(time, 3)
        norm_spatial = ttnn.squeeze(
            self.norm(ttnn.unsqueeze(spatial, 0), dynamic_weight=(1 + scale_msa), dynamic_bias=shift_msa), 0
        )
        norm_prompt = ttnn.squeeze(
            self.norm(ttnn.unsqueeze(prompt, 0), dynamic_weight=(1 + scale_msa), dynamic_bias=shift_msa), 0
        )

        norm_spatial = self.ccl_manager.all_gather_persistent_buffer(
            norm_spatial, dim=2, mesh_axis=tp_axis, use_hyperparams=True
        )
        norm_prompt = self.ccl_manager.all_gather_persistent_buffer(
            norm_prompt, dim=2, mesh_axis=tp_axis, use_hyperparams=True
        )

        # call `unsqueeze` since RowParallelLinear currently requires rank 4 tensors
        mlp_spatial = ttnn.squeeze(self.proj_mlp(ttnn.unsqueeze(norm_spatial, 0)), 0)  # OOM
        mlp_prompt = ttnn.squeeze(self.proj_mlp(ttnn.unsqueeze(norm_prompt, 0)), 0)

        # Fusing the activation function currently gives worse PCC
        ttnn.gelu(mlp_spatial, output_tensor=mlp_spatial, fast_and_approximate_mode=False)
        ttnn.gelu(mlp_prompt, output_tensor=mlp_prompt, fast_and_approximate_mode=False)

        attn_spatial, attn_prompt = self.attn.forward(
            spatial=norm_spatial,
            prompt=norm_prompt,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            spatial_sequence_length=spatial_sequence_length,
        )
        del norm_spatial, norm_prompt

        additional_spatial = ttnn.concat([attn_spatial, mlp_spatial], dim=-1)
        additional_prompt = ttnn.concat([attn_prompt, mlp_prompt], dim=-1)

        # call `unsqueeze` since RowParallelLinear currently requires rank 4 tensors
        additional_spatial = ttnn.squeeze(self.proj_out(ttnn.unsqueeze(additional_spatial, 0)), 0)
        additional_prompt = ttnn.squeeze(self.proj_out(ttnn.unsqueeze(additional_prompt, 0)), 0)
        additional_spatial = gate_msa * additional_spatial
        additional_prompt = gate_msa * additional_prompt

        spatial += additional_spatial
        prompt += additional_prompt

        # return combined
        return spatial, prompt


def _re_fuse_proj_out_weight(
    weight: torch.Tensor,
    *,
    embedding_dim: int,
    device_count: int,
) -> torch.Tensor:
    """Re-fuse out-projection parameters.

    The out-projection layer inputs are fused activations coming from the attention network and
    the MLP. In order to get the correct behavior on a mesh device, its weights must be re-fused to
    take into account mesh sharding.
    """
    if device_count == 1:
        return weight

    _, in_dim = weight.shape

    in_dim1 = embedding_dim
    in_dim2 = in_dim - in_dim1

    # unfuse
    w1, w2 = weight.split([in_dim1, in_dim2], dim=-1)

    # re-fuse
    w1 = w1.reshape([-1, device_count, in_dim1 // device_count])
    w2 = w2.reshape([-1, device_count, in_dim2 // device_count])
    return torch.cat([w1, w2], dim=-1).reshape([-1, in_dim])


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/transformers/transformer_flux.py
class Flux1Transformer(Module):
    sdpa_chunk_size_map = {
        (False, 2, 4): (128, 512),
        (False, 8, 4): (128, 256),
        (True, 2, 2): (128, 512),
        (True, 8, 4): (64, 512),
    }
    default_sdpa_chunk_size = (128, 512)

    def __init__(
        self,
        *,
        patch_size: int,
        in_channels: int,
        num_layers: int,
        num_single_layers: int,
        attention_head_dim: int,
        num_attention_heads: int,
        joint_attention_dim: int,
        pooled_projection_dim: int,
        out_channels: int,
        axes_dims_rope: Sequence[int],
        with_guidance_embeds: bool,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.patch_size = patch_size
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        q_chunk_size, k_chunk_size = self.sdpa_chunk_size_map.get(
            (
                is_blackhole(),
                self.parallel_config.sequence_parallel.factor,
                self.parallel_config.tensor_parallel.factor,
            ),
            self.default_sdpa_chunk_size,
        )

        # self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=list(axes_dims_rope))

        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
            embedding_dim=inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            with_guidance=with_guidance_embeds,
            mesh_device=mesh_device,
        )

        self.context_embedder = ColParallelLinear(
            joint_attention_dim,
            inner_dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        # Shard output, since size of input dimension << size of output dimension.
        self.x_embedder = ColParallelLinear(
            in_channels,
            inner_dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        self.transformer_blocks = ModuleList(
            TransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_pre_only=False,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                mesh_device=mesh_device,
                attention_k_chunk_size=k_chunk_size,
                attention_q_chunk_size=q_chunk_size,
            )
            for i in range(num_layers)
        )

        self.single_transformer_blocks = ModuleList(
            Flux1SingleTransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                mesh_device=mesh_device,
                attention_k_chunk_size=k_chunk_size,
                attention_q_chunk_size=q_chunk_size,
            )
            for i in range(num_single_layers)
        )

        self.time_embed_out = Linear(
            inner_dim,
            2 * inner_dim,
            mesh_device=mesh_device,
        )

        self.norm_out = DistributedLayerNorm(
            inner_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.proj_out = Linear(
            inner_dim,
            patch_size * patch_size * out_channels,
            mesh_device=mesh_device,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm_out.linear", "time_embed_out")
        rename_substate(state, "norm_out.norm", "norm_out")

    # We do not shard the last dimension of spatial, because its dimension is less than the tile
    # size for a device count of four and more. This requires padding, which is not currently
    # supported by `reduce_scatter_minimal_async`.
    def forward(
        self,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled: ttnn.Tensor,
        timestep: ttnn.Tensor,
        guidance: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
    ) -> ttnn.Tensor:
        """Run the model forward.

        Args:
            spatial: Tensor with shape [batch_size, spatial_sequence_length / sp_factor, in_channels].
            prompt: Tensor with shape [batch_size, prompt_sequence_length, joint_attention_dim].
            pooled: Tensor with shape [batch_size, pooled_projection_dim].
            timestep: Tensor with shape [batch_size, 1].
            guidance: Optional tensor with shape [batch_size, 1].
            spatial_rope: Tuple of two tensors with shape [spatial_sequence_length / sp_factor, head_dim].
            prompt_rope: Tuple of two tensors with shape [prompt_sequence_length, head_dim] (sequence is not sharded!).
        """
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        time_embed = self.time_text_embed(timestep=timestep, guidance=guidance, pooled_projection=pooled)
        ttnn.silu(time_embed, output_tensor=time_embed)
        time_embed = time_embed.reshape([time_embed.shape[-2], 1, time_embed.shape[-1]])

        spatial = self.x_embedder(spatial)
        prompt = self.context_embedder(prompt)

        for i, block in enumerate(self.transformer_blocks, start=1):
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                spatial_sequence_length=spatial_sequence_length,
                skip_time_embed_activation_fn=True,
            )

        prompt = ttnn.clone(prompt, dtype=spatial.dtype)

        for i, block in enumerate(self.single_transformer_blocks, start=1):
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                spatial_sequence_length=spatial_sequence_length,
                skip_time_embed_activation_fn=True,
            )

        spatial = ttnn.squeeze(self.norm_out(ttnn.unsqueeze(spatial, 0)), 0)

        spatial_time = self.time_embed_out(time_embed)
        [scale, shift] = _chunk_time3d(spatial_time, 2)

        spatial = self.ccl_manager.all_gather_persistent_buffer(spatial, dim=2, mesh_axis=tp_axis, use_hyperparams=True)

        spatial = spatial * (1 + scale) + shift

        return self.proj_out(spatial)
