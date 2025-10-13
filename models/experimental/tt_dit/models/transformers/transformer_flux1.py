# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn

from ...layers.embeddings import CombinedTimestepGuidanceTextProjEmbeddings
from ...layers.feedforward import ParallelFeedForward
from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import DistributedLayerNorm
from ...utils.substate import substate
from .attention_flux1 import Flux1Attention

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
    ) -> None:
        super().__init__()

        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        mlp_hidden_dim = 4 * dim

        self.attn = Flux1Attention(
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

    # TODO: migrate to _prepare_torch_state
    def load_state_dict(self, state_dict: dict[str, torch.Tensor], /) -> None:
        embedding_dim = state_dict["norm.linear.weight"].shape[1]

        self.attn.load_state_dict(substate(state_dict, "attn"))
        self.norm.load_state_dict(substate(state_dict, "norm"))
        self.time_embed.load_state_dict(self._shuffle_ada_norm_linear(substate(state_dict, "norm.linear")))
        self.proj_mlp.load_state_dict(substate(state_dict, "proj_mlp"))
        self.proj_out.load_state_dict(
            _re_fuse_proj_out_parameters(
                substate(state_dict, "proj_out"),
                embedding_dim=embedding_dim,
                device_count=self.parallel_config.tensor_parallel.factor,
            )
        )

    def _shuffle_ada_norm_linear(self, linear_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Rearrange QKV projections such column-fracturing shards the heads
        def _shuffle(x, in_dim):
            ndev = self.parallel_config.tensor_parallel.factor
            x = x.T
            cur_in_dim = x.shape[0]  # in_dim for weight, 1 for bias
            expansions = x.shape[-1] // in_dim
            x = x.reshape(-1, expansions, ndev, in_dim // ndev)
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(cur_in_dim, -1)
            assert x.shape[1] == in_dim * expansions
            x = x.T
            return x

        in_dim = linear_state["weight"].shape[1]
        weight = _shuffle(linear_state["weight"], in_dim)
        out_state = {"weight": weight}
        if "bias" in linear_state:
            bias = _shuffle(linear_state["bias"].reshape(-1, 1), in_dim)
            bias = bias.squeeze()
            out_state["bias"] = bias
        return out_state

    # Since we do not have operations to concatenate and slice a tensor along a sharded dimension,
    # we keep the spatial and prompt tensors separate for now.
    def forward(
        self,
        *,
        # combined: ttnn.Tensor,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        time_embed: ttnn.Tensor,
        # rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        skip_time_embed_activation: bool = False,
        # sequence_length: int,
        spatial_sequence_length: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the model forward.

        Args:
            combined: Tensor with shape [batch_size, sequence_length / sp_factor, query_dim / tp_factor].
            time_embed: Tensor with shape [batch_size, 1, query_dim].
            rope: Tuple of two tensors with shape [sequence_length / sp_factor, head_dim].
        """
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        if not skip_time_embed_activation:
            time_embed = ttnn.silu(time_embed)
        time = self.time_embed(time_embed)

        # combined_normed = ttnn.squeeze(self.norm(ttnn.unsqueeze(combined, 0)), 0)
        spatial_normed = ttnn.squeeze(self.norm(ttnn.unsqueeze(spatial, 0)), 0)
        prompt_normed = ttnn.squeeze(self.norm(ttnn.unsqueeze(prompt, 0)), 0)

        shift_msa, scale_msa, gate_msa = _chunk_time3d(time, 3)
        # norm_combined = combined_normed * (1 + scale_msa) + shift_msa
        norm_spatial = spatial_normed * (1 + scale_msa) + shift_msa
        norm_prompt = prompt_normed * (1 + scale_msa) + shift_msa

        # norm_combined = self.ccl_manager.all_gather(
        #     norm_combined, dim=2, mesh_axis=tp_axis
        # )
        norm_spatial = self.ccl_manager.all_gather(norm_spatial, dim=2, mesh_axis=tp_axis)
        norm_prompt = self.ccl_manager.all_gather(norm_prompt, dim=2, mesh_axis=tp_axis)

        # call `unsqueeze` since RowParallelLinear currently requires rank 4 tensors
        # mlp_combined = ttnn.squeeze(self.proj_mlp(ttnn.unsqueeze(norm_combined, 0)), 0)
        mlp_spatial = ttnn.squeeze(self.proj_mlp(ttnn.unsqueeze(norm_spatial, 0)), 0)  # OOM
        mlp_prompt = ttnn.squeeze(self.proj_mlp(ttnn.unsqueeze(norm_prompt, 0)), 0)

        # Fusing the activation function currently gives worse PCC
        # ttnn.gelu(mlp_combined, output_tensor=mlp_combined, fast_and_approximate_mode=False)
        ttnn.gelu(mlp_spatial, output_tensor=mlp_spatial, fast_and_approximate_mode=False)
        ttnn.gelu(mlp_prompt, output_tensor=mlp_prompt, fast_and_approximate_mode=False)
        # PCC of attn seems a bit low
        # attn, _ = self.attn.forward(spatial=norm_combined, spatial_rope=rope, spatial_sequence_length=sequence_length)
        # del norm_combined
        attn_spatial, attn_prompt = self.attn.forward(
            spatial=norm_spatial,
            prompt=norm_prompt,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            spatial_sequence_length=spatial_sequence_length,
        )
        del norm_spatial, norm_prompt

        # additional = ttnn.concat([attn, mlp_combined], dim=-1)
        additional_spatial = ttnn.concat([attn_spatial, mlp_spatial], dim=-1)
        additional_prompt = ttnn.concat([attn_prompt, mlp_prompt], dim=-1)
        # call `unsqueeze` since RowParallelLinear currently requires rank 4 tensors
        # additional = ttnn.squeeze(self.proj_out(ttnn.unsqueeze(additional, 0)), 0)
        additional_spatial = ttnn.squeeze(self.proj_out(ttnn.unsqueeze(additional_spatial, 0)), 0)
        additional_prompt = ttnn.squeeze(self.proj_out(ttnn.unsqueeze(additional_prompt, 0)), 0)
        # additional = gate_msa * additional
        additional_spatial = gate_msa * additional_spatial
        additional_prompt = gate_msa * additional_prompt

        # combined += additional
        spatial += additional_spatial
        prompt += additional_prompt

        # return combined
        return spatial, prompt


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
class Flux1TransformerBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        head_dim: int,
        context_pre_only: bool,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.context_pre_only = context_pre_only
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.norm1_linear = ColParallelLinear(
            dim,
            6 * dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )
        self.norm1_norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        context_norm_dim = 6 * dim if not context_pre_only else 2 * dim
        self.norm1_context_linear = ColParallelLinear(
            dim,
            context_norm_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )
        self.norm1_context_norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.attn = Flux1Attention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            added_kv_proj_dim=dim,
            context_pre_only=context_pre_only,
            eps=1e-6,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
        )

        self.norm2 = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.ff = ParallelFeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu",
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.norm2_context = None
        self.ff_context = None

        if not context_pre_only:
            self.norm2_context = DistributedLayerNorm(
                dim,
                norm_eps=1e-6,
                norm_elementwise_affine=False,
                bias=False,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )
            self.ff_context = ParallelFeedForward(
                dim=dim,
                dim_out=dim,
                activation_fn="gelu",
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
            )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    # TODO: migrate to _prepare_torch_state
    def load_state_dict(self, state_dict):
        def _shuffle_ada_norm_linear(linear_state):
            # Rearrange QKV projections such column-fracturing shards the heads
            def _shuffle(x, in_dim):
                ndev = self.parallel_config.tensor_parallel.factor
                x = x.T
                cur_in_dim = x.shape[0]  # in_dim for weight, 1 for bias
                expansions = x.shape[-1] // in_dim
                x = x.reshape(-1, expansions, ndev, in_dim // ndev)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(cur_in_dim, -1)
                assert x.shape[1] == in_dim * expansions
                x = x.T
                return x

            in_dim = linear_state["weight"].shape[1]
            weight = _shuffle(linear_state["weight"], in_dim)
            out_state = {"weight": weight}
            if "bias" in linear_state:
                bias = _shuffle(linear_state["bias"].reshape(-1, 1), in_dim)
                bias = bias.squeeze()
                out_state["bias"] = bias
            return out_state

        def rename_ff_state(state):
            out_state = {
                f"{replacement}{k[len(prefix) :]}": v
                for k, v in state.items()
                for prefix, replacement in [("net.0.proj", "ff1"), ("net.2", "ff2")]
                if prefix in k
            }
            return out_state

        self.norm1_linear.load_state_dict(_shuffle_ada_norm_linear(substate(state_dict, "norm1.linear")))
        self.norm1_norm.load_state_dict(substate(state_dict, "norm1.norm"))
        self.norm1_context_linear.load_state_dict(
            _shuffle_ada_norm_linear(substate(state_dict, "norm1_context.linear"))
        )
        self.norm1_context_norm.load_state_dict(substate(state_dict, "norm1_context.norm"))
        self.attn.load_state_dict(substate(state_dict, "attn"))
        self.norm2.load_state_dict(substate(state_dict, "norm2"))
        self.ff.load_state_dict(rename_ff_state(substate(state_dict, "ff")))
        if not self.context_pre_only:
            self.norm2_context.load_state_dict(substate(state_dict, "norm2_context"))
            self.ff_context.load_state_dict(rename_ff_state(substate(state_dict, "ff_context")))

    def forward(
        self,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        time_embed: ttnn.Tensor,
        spatial_sequence_length: int,
        *,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        skip_time_embed_activation: bool = False,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """Run the model forward.

        Args:
            spatial: Tensor with shape [batch_size, spatial_sequence_length / sp_factor, query_dim / tp_factor].
            prompt: Tensor with shape [batch_size, prompt_sequence_length, query_dim / tp_factor] (sequence is not sharded!).
            time_embed: Tensor with shape [batch_size, 1, query_dim].
            spatial_rope: Tuple of two tensors with shape [spatial_sequence_length / sp_factor, head_dim].
            prompt_rope: Tuple of two tensors with shape [prompt_sequence_length, head_dim] (sequence is not sharded!).
        """
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        if not skip_time_embed_activation:
            time_embed = ttnn.silu(time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        spatial_time = self.norm1_linear(time_embed, core_grid=self.core_grid)
        prompt_time = self.norm1_context_linear(time_embed, core_grid=self.core_grid)

        (
            spatial_shift_attn,
            spatial_scale_attn,
            spatial_gate_attn,
            spatial_shift_ff,
            spatial_scale_ff,
            spatial_gate_ff,
        ) = _chunk_time3d(spatial_time, 6)

        spatial_normed = ttnn.squeeze(self.norm1_norm(ttnn.unsqueeze(spatial, 0)), 0)
        spatial_normed = spatial_normed * (1 + spatial_scale_attn) + spatial_shift_attn

        if self.context_pre_only:
            prompt_scale_attn, prompt_shift_attn = _chunk_time3d(prompt_time, 2)
            prompt_gate_attn = None
            prompt_shift_ff = None
            prompt_scale_ff = None
            prompt_gate_ff = None
        else:
            (
                prompt_shift_attn,
                prompt_scale_attn,
                prompt_gate_attn,
                prompt_shift_ff,
                prompt_scale_ff,
                prompt_gate_ff,
            ) = _chunk_time3d(prompt_time, 6)

        prompt_normed = ttnn.squeeze(self.norm1_context_norm(ttnn.unsqueeze(prompt, 0)), 0)
        prompt_normed = prompt_normed * (1 + prompt_scale_attn) + prompt_shift_attn

        # Gather spatial, prompt before attention
        spatial_normed = self.ccl_manager.all_gather(spatial_normed, dim=2, mesh_axis=tp_axis)
        prompt_normed = self.ccl_manager.all_gather(prompt_normed, dim=2, mesh_axis=tp_axis)

        spatial_attn, prompt_attn = self.attn.forward(
            spatial=spatial_normed,
            prompt=prompt_normed,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            spatial_sequence_length=spatial_sequence_length,
        )
        spatial_attn = spatial_attn * spatial_gate_attn
        prompt_attn = prompt_attn * prompt_gate_attn if prompt_gate_attn is not None else None

        # residual
        spatial = spatial + spatial_attn

        spatial_normed = ttnn.squeeze(self.norm2(ttnn.unsqueeze(spatial, 0)), 0)
        spatial_normed = spatial_normed * (1 + spatial_scale_ff) + spatial_shift_ff

        spatial_normed = self.ccl_manager.all_gather(spatial_normed, dim=2, mesh_axis=tp_axis)

        spatial_ff = ttnn.squeeze(self.ff(ttnn.unsqueeze(spatial_normed, 0), core_grid=self.core_grid), 0)
        spatial_ff = spatial_ff * spatial_gate_ff

        spatial += spatial_ff

        if self.context_pre_only:
            return spatial, None

        prompt += prompt_attn

        prompt_normed = ttnn.squeeze(self.norm2_context(ttnn.unsqueeze(prompt, 0)), 0)
        prompt_normed = prompt_normed * (1 + prompt_scale_ff) + prompt_shift_ff

        prompt_normed = self.ccl_manager.all_gather(prompt_normed, dim=2, mesh_axis=tp_axis)

        prompt_ff = ttnn.squeeze(self.ff_context(ttnn.unsqueeze(prompt_normed, 0), core_grid=self.core_grid), 0)
        prompt_ff = prompt_ff * prompt_gate_ff

        prompt += prompt_ff

        return spatial, prompt


def _re_fuse_proj_out_parameters(
    state: dict[str, torch.Tensor],
    *,
    embedding_dim: int,
    device_count: int,
) -> dict[str, torch.Tensor]:
    """Re-fuse out-projection parameters.

    The out-projection layer inputs are fused activations coming from the attention network and
    the MLP. In order to get the correct behavior on a mesh device, its weights must be re-fused to
    take into account mesh sharding.
    """
    if device_count == 1:
        return state

    w = state["weight"]
    _, in_dim = w.shape

    in_dim1 = embedding_dim
    in_dim2 = in_dim - in_dim1

    # unfuse
    w1, w2 = w.split([in_dim1, in_dim2], dim=-1)

    # re-fuse
    w1 = w1.reshape([-1, device_count, in_dim1 // device_count])
    w2 = w2.reshape([-1, device_count, in_dim2 // device_count])
    w = torch.cat([w1, w2], dim=-1).reshape([-1, in_dim])

    return {"weight": w, "bias": state["bias"]}


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/transformers/transformer_flux.py
class Flux1Transformer(Module):
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
            Flux1TransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_pre_only=False,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                mesh_device=mesh_device,
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
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.proj_out = Linear(
            inner_dim,
            patch_size * patch_size * out_channels,
            mesh_device=mesh_device,
        )

    # TODO: migrate to _prepare_torch_state
    def load_state_dict(self, state_dict: dict[str, torch.Tensor], /) -> None:
        self.time_text_embed.load_state_dict(substate(state_dict, "time_text_embed"))
        self.context_embedder.load_state_dict(substate(state_dict, "context_embedder"))
        self.x_embedder.load_state_dict(substate(state_dict, "x_embedder"))
        for i, block in enumerate(self.transformer_blocks):
            block.load_state_dict(substate(state_dict, f"transformer_blocks.{i}"))
        for i, block in enumerate(self.single_transformer_blocks):
            block.load_state_dict(substate(state_dict, f"single_transformer_blocks.{i}"))
        self.time_embed_out.load_state_dict(substate(state_dict, "norm_out.linear"))  # chunks=2 if sharded
        self.norm_out.load_state_dict(substate(state_dict, "norm_out.norm"))
        self.proj_out.load_state_dict(substate(state_dict, "proj_out"))

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
        # combined_rope: tuple[ttnn.Tensor, ttnn.Tensor],
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
                skip_time_embed_activation=True,
            )

            if i % 6 == 0:
                ttnn.ReadDeviceProfiler(spatial.device())

        prompt = ttnn.clone(prompt, dtype=spatial.dtype)

        # combined = ttnn.concat([prompt, spatial], dim=1)
        # del prompt, spatial

        for i, block in enumerate(self.single_transformer_blocks, start=1):
            spatial, prompt = block.forward(
                # combined=combined,
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                # rope=combined_rope,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                # sequence_length=spatial_sequence_length + prompt_sequence_length,
                spatial_sequence_length=spatial_sequence_length,
                skip_time_embed_activation=True,
            )

            if i % 6 == 0:
                ttnn.ReadDeviceProfiler(self.mesh_device)

        # spatial = combined[:, prompt_sequence_length:]
        # del combined

        spatial = ttnn.squeeze(self.norm_out(ttnn.unsqueeze(spatial, 0)), 0)

        spatial_time = self.time_embed_out(time_embed)
        [scale, shift] = _chunk_time3d(spatial_time, 2)

        spatial = self.ccl_manager.all_gather(spatial, dim=2, mesh_axis=tp_axis)

        spatial = spatial * (1 + scale) + shift

        return self.proj_out(spatial)


def _chunk_time3d(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = t.shape[-1] // count
    return [t[:, :, i * size : (i + 1) * size] for i in range(count)]
