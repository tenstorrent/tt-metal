# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import ttnn

from ...layers.attention import all_gather
from ...layers.embeddings import PatchEmbed
from ...layers.feedforward import FeedForward
from ...layers.linear import ColParallelLinear, Linear
from ...layers.module import Module, ModuleList, Parameter
from ...layers.normalization import DistributedLayerNorm
from ...layers.transformer_block import TransformerBlock
from ...utils.substate import rename_substate
from ...utils.tensor import bf16_tensor

if TYPE_CHECKING:
    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager
    from ...utils.padding import PaddingConfig


class TimeTextProjection(Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        pooled_projection_dim: int,
        time_embed_dim: int = 256,
        mesh_device: ttnn.MeshDevice | None = None,
    ) -> None:
        super().__init__()

        self.mesh_device = mesh_device

        self.timestep_embedder = FeedForward(
            dim=time_embed_dim, inner_dim=4 * time_embed_dim, dim_out=embedding_dim, mesh_device=mesh_device
        )
        self.text_embedder = FeedForward(
            dim=pooled_projection_dim,
            inner_dim=4 * pooled_projection_dim,
            dim_out=embedding_dim,
            mesh_device=mesh_device,
        )

        self.time_proj_factor = self._create_time_proj_factor(time_embed_dim)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "timestep_embedder.linear_1", "timestep_embedder.ff1")
        rename_substate(state, "timestep_embedder.linear_2", "timestep_embedder.ff2")
        rename_substate(state, "text_embedder.linear_1", "text_embedder.ff1")
        rename_substate(state, "text_embedder.linear_2", "text_embedder.ff2")

    def _create_time_proj_factor(self, num_channels: int) -> ttnn.Tensor:
        assert num_channels % 2 == 0
        half_dim = num_channels // 2

        max_period = 10000

        exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        exponent = exponent / half_dim
        factor = torch.exp(exponent)

        return ttnn.unsqueeze_to_4D(bf16_tensor(factor, device=self.mesh_device))

    # In order to avoid calling `unsqueeze` in this function, we expect already unsqueezed rank two
    # `timestep` tensor.
    def forward(
        self,
        *,
        timestep: ttnn.Tensor,
        pooled_projection: ttnn.Tensor,
    ) -> ttnn.Tensor:
        batch_size = pooled_projection.shape[0]

        assert len(pooled_projection.shape) == 2
        assert timestep.shape == [batch_size, 1]
        assert timestep.dtype == ttnn.float32, "timesteps require float32 precision"

        emb = timestep * self.time_proj_factor
        c = ttnn.cos(emb)
        s = ttnn.sin(emb)
        timesteps_proj = ttnn.concat([c, s], dim=-1)
        timesteps_emb = self.timestep_embedder(timesteps_proj)

        text_emb = self.text_embedder(pooled_projection)

        return timesteps_emb + text_emb


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/transformers/transformer_flux.py
class MotifTransformer(Module):
    ENCODED_TEXT_DIM = 4096
    SD3_LATENT_CHANNEL = 16

    def __init__(
        self,
        *,
        patch_size: int,
        num_layers: int,
        attention_head_dim: int,
        num_attention_heads: int,
        pooled_projection_dim: int,
        pos_embed_max_size: int,
        modulation_dim: int,
        time_embed_dim: int,
        register_token_num: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
    ) -> None:
        super().__init__()

        in_channels = self.SD3_LATENT_CHANNEL
        out_channels = self.SD3_LATENT_CHANNEL
        inner_dim = num_attention_heads * attention_head_dim
        sample_size = 64

        self.patch_size = patch_size
        self.num_layers = num_layers
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            pos_embed_max_size=pos_embed_max_size,
            mesh_device=mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            sp_mesh_axis=parallel_config.sequence_parallel.mesh_axis,
        )

        self.time_text_embed = TimeTextProjection(
            embedding_dim=modulation_dim,
            time_embed_dim=time_embed_dim,
            pooled_projection_dim=pooled_projection_dim,
            mesh_device=mesh_device,
        )

        self.context_embedder = ColParallelLinear(
            self.ENCODED_TEXT_DIM,
            inner_dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        self.transformer_blocks = ModuleList(
            TransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_pre_only=i == num_layers - 1,
                ff_activation_fn="linear-silu",
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                mesh_device=mesh_device,
            )
            for i in range(num_layers)
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
        # self.norm_out = AdaLayerNormContinuous(inner_dim, modulation_dim, elementwise_affine=False, eps=1e-6)

        self.proj_out = Linear(
            inner_dim,
            patch_size * patch_size * out_channels,
            mesh_device=mesh_device,
        )

        self.register_tokens = Parameter(
            shape=[1, register_token_num, inner_dim],
            device=mesh_device,
        )

        # TODO: mesh sharding?
        self.t_token_proj = Linear(modulation_dim, inner_dim, mesh_device=mesh_device)

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
        time_embed = self.time_text_embed(timestep=timestep, guidance=guidance, pooled_projection=pooled)
        ttnn.silu(time_embed, output_tensor=time_embed)
        time_embed = time_embed.reshape([time_embed.shape[-2], 1, time_embed.shape[-1]])

        spatial = self.pos_embed(spatial)
        prompt = self.context_embedder(prompt)

        # TODO
        # hidden_states = torch.cat((self.register_tokens.expand(hidden_states.shape[0], -1, -1), hidden_states), dim=1)
        # t_token = self.t_token_proj(temb).unsqueeze(1)
        # encoder_hidden_states = torch.cat([encoder_hidden_states, t_token], dim=1)

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

        # hidden_states = hidden_states[:, self.register_tokens.shape[1] :]

        # Flux:
        # spatial = ttnn.squeeze(self.norm_out(ttnn.unsqueeze(spatial, 0)), 0)
        # spatial_time = self.time_embed_out(time_embed)
        # [scale, shift] = _chunk_time3d(spatial_time, 2)
        # spatial = all_gather(
        #     spatial, dim=2, parallel_factor=self.parallel_config.tensor_parallel, ccl_manager=self.ccl_manager
        # )
        # spatial = spatial * (1 + scale) + shift
        # return self.proj_out(spatial)

        spatial_time = self.norm_out_linear(
            ttnn.silu(time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG), core_grid=self.core_grid
        )
        scale, shift = chunk_time(spatial_time, 2)

        spatial = all_gather(
            spatial, dim=1, parallel_factor=self.parallel_config.sequence_parallel, ccl_manager=self.ccl_manager
        )
        spatial = all_gather(
            spatial, dim=2, parallel_factor=self.parallel_config.tensor_parallel, ccl_manager=self.ccl_manager
        )

        spatial = self.norm_out_norm(spatial) * (1 + scale) + shift

        return self.proj_out(spatial, core_grid=self.core_grid, compute_kernel_config=self.hifi_compute_kernel_config)

        # NOTE: While we should be able to gather on sequence after norm and proj, it leads to
        # terrible outputs for 2x2sp1tp0. Need to debug.

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        ### First, convert Motif state dict to diffusers compatible (as far as possible) state dict.

        renames = {
            "pos_embed": "pos_embed.pos_embed",
            "text_cond.projection.weight": "context_embedder.weight",
            "text_cond.projection.bias": "context_embedder.bias",
            "final_modulation.weight": "norm_out.linear.weight",
            "final_modulation.bias": "norm_out.linear.bias",
            "final_linear_SD3.weight": "proj_out.weight",
            "final_linear_SD3.bias": "proj_out.bias",
            "patching.projection_SD3.weight": "pos_embed.proj.weight",
            "patching.projection_SD3.bias": "pos_embed.proj.bias",
            "time_emb.time_emb.linear_1.weight": "time_text_embed.timestep_embedder.linear_1.weight",
            "time_emb.time_emb.linear_1.bias": "time_text_embed.timestep_embedder.linear_1.bias",
            "time_emb.time_emb.linear_2.weight": "time_text_embed.timestep_embedder.linear_2.weight",
            "time_emb.time_emb.linear_2.bias": "time_text_embed.timestep_embedder.linear_2.bias",
            "time_emb.pooled_text_emb.linear_1.weight": "time_text_embed.text_embedder.linear_1.weight",
            "time_emb.pooled_text_emb.linear_1.bias": "time_text_embed.text_embedder.linear_1.bias",
            "time_emb.pooled_text_emb.linear_2.weight": "time_text_embed.text_embedder.linear_2.weight",
            "time_emb.pooled_text_emb.linear_2.bias": "time_text_embed.text_embedder.linear_2.bias",
        }

        block_renames = {
            "attn.o_proj.weight": "attn.to_out.0.weight",
            "attn.add_o_proj.weight": "attn.to_add_out.weight",
            "attn.q_norm_x.weight": "attn.norm_q.weight",
            "attn.k_norm_x.weight": "attn.norm_k.weight",
            "attn.q_norm_c.weight": "attn.norm_added_q.weight",
            "attn.k_norm_c.weight": "attn.norm_added_k.weight",
            "attn.q_scale": "attn.added_head_factors",
            "affine_params_c.projection.weight": "norm1_context.linear.weight",
            "affine_params_c.projection.bias": "norm1_context.linear.bias",
            "affine_params_x.projection.weight": "norm1.linear.weight",
            "affine_params_x.projection.bias": "norm1.linear.bias",
            "mlp_3_c.gate_proj.weight": "ff_context.net.0.proj.weight",
            "mlp_3_c.gate_proj.bias": "ff_context.net.0.proj.bias",
            "mlp_3_c.down_proj.weight": "ff_context.net.2.weight",
            "mlp_3_c.down_proj.bias": "ff_context.net.2.bias",
            "mlp_3_x.gate_proj.weight": "ff.net.0.proj.weight",
            "mlp_3_x.gate_proj.bias": "ff.net.0.proj.bias",
            "mlp_3_x.down_proj.weight": "ff.net.2.weight",
            "mlp_3_x.down_proj.bias": "ff.net.2.bias",
        }

        for src, dst in renames.items():
            state[dst] = state.pop(src)

        for i in range(self.num_layers):
            src_prefix = f"mmdit_blocks.{i}."
            dst_prefix = f"transformer_blocks.{i}."

            for src, dst in block_renames.items():
                state[f"{dst_prefix}{dst}"] = state.pop(f"{src_prefix}{src}")

            x_weight = state.pop(f"{src_prefix}linear_1_x.weight")
            x_bias = state.pop(f"{src_prefix}linear_1_x.bias")

            q_weight = state.pop(f"{src_prefix}attn.q_proj.weight")
            state[f"{dst_prefix}attn.to_q.weight"] = q_weight @ x_weight
            state[f"{dst_prefix}attn.to_q.bias"] = q_weight @ x_bias
            k_weight = state.pop(f"{src_prefix}attn.k_proj.weight")
            state[f"{dst_prefix}attn.to_k.weight"] = k_weight @ x_weight
            state[f"{dst_prefix}attn.to_k.bias"] = k_weight @ x_bias
            v_weight = state.pop(f"{src_prefix}attn.v_proj.weight")
            state[f"{dst_prefix}attn.to_v.weight"] = v_weight @ x_weight
            state[f"{dst_prefix}attn.to_v.bias"] = v_weight @ x_bias

            c_weight = state.pop(f"{src_prefix}linear_1_c.weight")
            c_bias = state.pop(f"{src_prefix}linear_1_c.bias")

            add_q_weight = state.pop(f"{src_prefix}attn.add_q_proj.weight")
            state[f"{dst_prefix}attn.add_q_proj.weight"] = add_q_weight @ c_weight
            state[f"{dst_prefix}attn.add_q_proj.bias"] = add_q_weight @ c_bias
            add_k_weight = state.pop(f"{src_prefix}attn.add_k_proj.weight")
            state[f"{dst_prefix}attn.add_k_proj.weight"] = add_k_weight @ c_weight
            state[f"{dst_prefix}attn.add_k_proj.bias"] = add_k_weight @ c_bias
            add_v_weight = state.pop(f"{src_prefix}attn.add_v_proj.weight")
            state[f"{dst_prefix}attn.add_v_proj.weight"] = add_v_weight @ c_weight
            state[f"{dst_prefix}attn.add_v_proj.bias"] = add_v_weight @ c_bias

            state[f"{dst_prefix}attn.to_out.0.bias"] = torch.zeros_like(state[f"{dst_prefix}attn.to_out.0.weight"][0])
            state[f"{dst_prefix}attn.to_add_out.bias"] = torch.zeros_like(
                state[f"{dst_prefix}attn.to_add_out.weight"][0]
            )

            _convert_ada_norm(state, f"{dst_prefix}norm1.linear", pre_only=False)
            _convert_ada_norm(state, f"{dst_prefix}norm1_context.linear", pre_only=i == self.num_layers - 1)

        state["pos_embed.pos_embed"] = state["pos_embed.pos_embed"].unsqueeze(0)

        # Unused since we can set context_pre_only=True in the last block.
        last_block_prefix = f"transformer_blocks.{self.num_layers - 1}"
        del state[f"{last_block_prefix}.attn.to_add_out.weight"]
        del state[f"{last_block_prefix}.attn.to_add_out.bias"]
        del state[f"{last_block_prefix}.ff_context.net.0.proj.weight"]
        del state[f"{last_block_prefix}.ff_context.net.0.proj.bias"]
        del state[f"{last_block_prefix}.ff_context.net.2.weight"]
        del state[f"{last_block_prefix}.ff_context.net.2.bias"]

        ### Second, convert diffusers state dict to tt_dit state dict.

        rename_substate(state, "norm_out.linear", "time_embed_out")  # chunks=2 if sharded
        rename_substate(state, "norm_out.norm", "norm_out")


def _chunk_time3d(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = t.shape[-1] // count
    return [t[:, :, i * size : (i + 1) * size] for i in range(count)]


def _convert_ada_norm(state: dict[str, torch.Tensor], prefix: str, *, pre_only: bool) -> None:
    ws = state[f"{prefix}.weight"].chunk(6)
    bs = state[f"{prefix}.bias"].chunk(6)

    if pre_only:
        state[f"{prefix}.weight"] = torch.concat(ws[:2])
        state[f"{prefix}.bias"] = torch.concat(bs[:2])
    else:
        state[f"{prefix}.weight"] = torch.concat([ws[1], ws[0], ws[2], ws[4], ws[3], ws[5]])
        state[f"{prefix}.bias"] = torch.concat([bs[1], bs[0], bs[2] + 1, bs[4], bs[3] - 1, bs[5]])
