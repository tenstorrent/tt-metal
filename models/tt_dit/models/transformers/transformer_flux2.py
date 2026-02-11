# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn

from ...blocks.attention import Attention
from ...blocks.transformer_block import TransformerBlock
from ...layers.embeddings import CombinedTimestepGuidanceTextProjEmbeddings
from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear, prepare_chunked_linear_output
from ...layers.module import Module, ModuleList
from ...layers.normalization import DistributedLayerNorm
from ...utils.substate import rename_substate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager
    from ...utils.padding import PaddingConfig


class Flux2Modulation(Module):
    def __init__(self, dim: int, *, mod_param_sets: int, tp_axis: int | None, device: ttnn.MeshDevice) -> None:
        super().__init__()

        self.linear = ColParallelLinear(
            dim, dim * 3 * mod_param_sets, bias=False, mesh_axis=tp_axis, mesh_device=device
        )

        self._mod_param_sets = mod_param_sets
        self._tp_factor = device.shape[tp_axis] if tp_axis is not None else 1

    def forward(self, x: torch.Tensor, *, skip_act_fn: bool = False) -> tuple[torch.Tensor, ...]:
        assert len(x.shape) == 3

        if not skip_act_fn:
            x = ttnn.silu(x)

        x = self.linear(x)
        return ttnn.chunk(x, 3 * self._mod_param_sets, dim=-1)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        prepare_chunked_linear_output(
            state, prefix="linear", device_count=self._tp_factor, chunks=3 * self._mod_param_sets
        )


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/transformers/transformer_flux.py
class Flux2SingleTransformerBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        head_dim: int,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
    ) -> None:
        super().__init__()

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        mlp_hidden_dim = 3 * dim

        self.attn = Attention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            added_kv_proj_dim=0,
            pre_only=True,
            proj_bias=False,
            eps=1e-6,
            mesh_device=device,
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
            mesh_axis=tp_axis,
            mesh_device=device,
            ccl_manager=ccl_manager,
        )

        # Shard output, since size of input dimension << size of output dimension.
        self.proj_mlp = ColParallelLinear(
            dim,
            mlp_hidden_dim,
            bias=False,
            activation_fn="swiglu",
            mesh_device=device,
            mesh_axis=tp_axis,
        )

        # Shard input, since size of input dimension >> size of output dimension.
        self.proj_out = RowParallelLinear(
            dim + mlp_hidden_dim,
            dim,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )

        self._dim = dim
        self._mlp_hidden_dim = mlp_hidden_dim
        self._tp_axis = tp_axis
        self._tp_factor = parallel_config.tensor_parallel.factor
        self._ccl_manager = ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        qkv_mlp_weight = state.pop("attn.to_qkv_mlp_proj.weight", None)
        if qkv_mlp_weight is not None:
            state["attn.to_q.weight"] = qkv_mlp_weight[: self._dim]
            state["attn.to_k.weight"] = qkv_mlp_weight[self._dim : 2 * self._dim]
            state["attn.to_v.weight"] = qkv_mlp_weight[2 * self._dim : 3 * self._dim]

            w = qkv_mlp_weight[3 * self._dim :]
            # swiglu implementation uses different ordering
            state["proj_mlp.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)

        proj_out_weight = state.pop("attn.to_out.weight", None)
        if proj_out_weight is not None:
            state["proj_out.weight"] = _prepare_weight_for_concatenated_input(
                proj_out_weight,
                [self._dim, self._mlp_hidden_dim],
                device_count=self._tp_factor,
            )

    # Since we do not have operations to concatenate and slice a tensor along a sharded dimension,
    # we keep the spatial and prompt tensors separate for now.
    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        time_embed: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        temb_mod_params: tuple[ttnn.Tensor, ...],
        skip_time_embed_activation_fn: bool = False,
        spatial_sequence_length: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if not skip_time_embed_activation_fn:
            time_embed = ttnn.silu(time_embed)

        x = ttnn.squeeze(self.norm(ttnn.unsqueeze(spatial, 0)), 0)
        c = ttnn.squeeze(self.norm(ttnn.unsqueeze(prompt, 0)), 0)

        shift_msa, scale_msa, gate_msa = temb_mod_params
        x = x * (1 + scale_msa) + shift_msa
        c = c * (1 + scale_msa) + shift_msa

        x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
        c = self._ccl_manager.all_gather_persistent_buffer(c, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        x_mlp = self.proj_mlp(x)
        c_mlp = self.proj_mlp(c)

        x, c = self.attn.forward(
            spatial=x,
            prompt=c,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            spatial_sequence_length=spatial_sequence_length,
        )

        x = ttnn.concat([x, x_mlp], dim=-1)
        c = ttnn.concat([c, c_mlp], dim=-1)
        del x_mlp, c_mlp

        x = gate_msa * self.proj_out(x)
        c = gate_msa * self.proj_out(c)

        return spatial + x, prompt + c


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/transformers/transformer_flux.py
class Flux2Transformer(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_layers: int,
        num_single_layers: int,
        attention_head_dim: int,
        num_attention_heads: int,
        joint_attention_dim: int,
        out_channels: int,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        tp_axis = parallel_config.tensor_parallel.mesh_axis

        # self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)

        self.time_guidance_embed = CombinedTimestepGuidanceTextProjEmbeddings(
            embedding_dim=inner_dim,
            pooled_projection_dim=0,
            bias=False,
            with_guidance=True,
            mesh_device=device,
        )

        self.context_embedder = ColParallelLinear(
            joint_attention_dim, inner_dim, bias=False, mesh_device=device, mesh_axis=tp_axis
        )

        # Shard output, since size of input dimension << size of output dimension.
        self.x_embedder = ColParallelLinear(in_channels, inner_dim, bias=False, mesh_device=device, mesh_axis=tp_axis)

        self.transformer_blocks = ModuleList(
            TransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_pre_only=False,
                attention_proj_bias=False,
                ff_activation_fn="swiglu",
                ff_mult=3,
                ff_bias=False,
                time_norm_affine=False,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                mesh_device=device,
            )
            for i in range(num_layers)
        )

        self.single_transformer_blocks = ModuleList(
            Flux2SingleTransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                device=device,
            )
            for i in range(num_single_layers)
        )

        self.time_embed_out = Linear(inner_dim, 2 * inner_dim, bias=False, mesh_device=device)

        self.norm_out = DistributedLayerNorm(
            inner_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )

        self.proj_out = Linear(inner_dim, out_channels, bias=False, mesh_device=device)

        self.double_stream_modulation_img = Flux2Modulation(inner_dim, mod_param_sets=2, tp_axis=tp_axis, device=device)
        self.double_stream_modulation_txt = Flux2Modulation(inner_dim, mod_param_sets=2, tp_axis=tp_axis, device=device)
        self.single_stream_modulation = Flux2Modulation(inner_dim, mod_param_sets=1, tp_axis=tp_axis, device=device)

        self.device = device
        self._ccl_manager = ccl_manager
        self._tp_axis = tp_axis

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm_out.linear", "time_embed_out")
        rename_substate(state, "norm_out.norm", "norm_out")

        for i in range(len(self.transformer_blocks)):
            prefix = f"transformer_blocks.{i}."

            # swiglu implementation uses different ordering
            w = state.pop(f"{prefix}ff.linear_in.weight", None)
            if w is not None:
                state[f"{prefix}ff.net.0.proj.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)
            w = state.pop(f"{prefix}ff_context.linear_in.weight", None)
            if w is not None:
                state[f"{prefix}ff_context.net.0.proj.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)

            rename_substate(state, f"{prefix}ff.linear_out", f"{prefix}ff.net.2")
            rename_substate(state, f"{prefix}ff_context.linear_out", f"{prefix}ff_context.net.2")

    # We do not shard the last dimension of spatial, because its dimension is less than the tile
    # size for a device count of four and more. This requires padding, which is not currently
    # supported by `reduce_scatter_minimal_async`.
    def forward(
        self,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
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
        time_embed = self.time_guidance_embed(timestep=timestep, guidance=guidance)
        ttnn.silu(time_embed, output_tensor=time_embed)
        time_embed = time_embed.reshape([time_embed.shape[-2], 1, time_embed.shape[-1]])

        double_stream_mod_img = self.double_stream_modulation_img(time_embed, skip_act_fn=True)
        double_stream_mod_txt = self.double_stream_modulation_txt(time_embed, skip_act_fn=True)
        single_stream_mod = self.single_stream_modulation(time_embed, skip_act_fn=True)

        spatial = self.x_embedder(spatial)
        prompt = self.context_embedder(prompt)

        for i, block in enumerate(self.transformer_blocks, start=1):
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                spatial_sequence_length=spatial_sequence_length,
                skip_time_embed_activation_fn=True,
            )

            if i % 6 == 0:
                ttnn.ReadDeviceProfiler(spatial.device())

        prompt = ttnn.clone(prompt, dtype=spatial.dtype)

        for i, block in enumerate(self.single_transformer_blocks, start=1):
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                temb_mod_params=single_stream_mod,
                spatial_sequence_length=spatial_sequence_length,
                skip_time_embed_activation_fn=True,
            )

            if i % 6 == 0:
                ttnn.ReadDeviceProfiler(spatial.device())

        spatial = ttnn.squeeze(self.norm_out(ttnn.unsqueeze(spatial, 0)), 0)

        spatial_time = self.time_embed_out(time_embed)
        [scale, shift] = ttnn.chunk(spatial_time, 2, dim=-1)

        spatial = self._ccl_manager.all_gather_persistent_buffer(
            spatial, dim=2, mesh_axis=self._tp_axis, use_hyperparams=True
        )

        spatial = spatial * (1 + scale) + shift

        return self.proj_out(spatial)


def _prepare_weight_for_concatenated_input(
    weight: torch.Tensor,
    sizes: Sequence[int],
    *,
    device_count: int,
) -> torch.Tensor:
    weights = weight.split(sizes, dim=1)
    weights = [w.unflatten(1, [device_count, -1]) for w in weights]
    return torch.cat(weights, dim=2).flatten(1, 2)
