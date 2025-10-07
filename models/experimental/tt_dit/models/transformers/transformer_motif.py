# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import ttnn

from ...blocks.transformer_block import TransformerBlock
from ...layers.embeddings import PatchEmbed
from ...layers.feedforward import FeedForward
from ...layers.linear import ColParallelLinear, Linear
from ...layers.module import Module, ModuleList, Parameter
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
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()

        self.mesh_device = mesh_device

        self.timestep_embedder = FeedForward(
            dim=time_embed_dim,
            inner_dim=4 * time_embed_dim,
            dim_out=embedding_dim,
            activation_fn="silu",
            mesh_device=mesh_device,
        )
        self.text_embedder = FeedForward(
            dim=pooled_projection_dim,
            inner_dim=4 * pooled_projection_dim,
            dim_out=embedding_dim,
            activation_fn="silu",
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
        assert timestep.shape in ([1, 1], [batch_size, 1])
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
    LATENT_CHANNELS = 16

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
        latents_height: int,
        latents_width: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
    ) -> None:
        super().__init__()

        in_channels = self.LATENT_CHANNELS
        out_channels = self.LATENT_CHANNELS
        inner_dim = num_attention_heads * attention_head_dim

        self.patch_size = patch_size
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.latents_height = latents_height
        self.latents_width = latents_width
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.register_token_num = register_token_num
        self.ccl_manager = ccl_manager

        sp_axis = parallel_config.sequence_parallel.mesh_axis
        sp_factor = parallel_config.sequence_parallel.factor
        tp_axis = parallel_config.tensor_parallel.mesh_axis

        raw_spatial_sequence_length = (latents_height // patch_size) * (latents_width // patch_size)

        self.spatial_sequence_length = raw_spatial_sequence_length + register_token_num
        self.spatial_sequence_padding = TransformerBlock.spatial_sequence_padding_length(
            length=self.spatial_sequence_length, sp_factor=sp_factor
        )
        self.padded_spatial_sequence_length = self.spatial_sequence_length + self.spatial_sequence_padding

        self.pos_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=inner_dim,
            pos_embed_max_size=pos_embed_max_size,
            patch_size=patch_size,
            width=latents_width,
            height=latents_height,
            sequence_padding=(register_token_num, self.spatial_sequence_padding),
            mesh_device=mesh_device,
            tp_mesh_axis=tp_axis,
            sp_mesh_axis=sp_axis,
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
                modulation_dim=modulation_dim,
                context_pre_only=i == num_layers - 1,
                context_head_scaling=True,
                add_attention_to_output=False,
                ff_activation_fn="silu",
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                mesh_device=mesh_device,
            )
            for i in range(num_layers)
        )

        self.time_embed_out = Linear(
            modulation_dim,
            2 * inner_dim,
            mesh_device=mesh_device,
        )

        self.proj_out = Linear(
            inner_dim,
            patch_size * patch_size * out_channels,
            mesh_device=mesh_device,
        )

        self.register_tokens = Parameter(
            total_shape=[1, self.padded_spatial_sequence_length, inner_dim],
            device=mesh_device,
            mesh_axes=[None, sp_axis, tp_axis],
        )

        self.register_tokens_mask = Parameter(
            total_shape=[1, self.padded_spatial_sequence_length, inner_dim],
            dtype=ttnn.bfloat4_b,
            device=mesh_device,
            mesh_axes=[None, sp_axis, tp_axis],
        )

        self.t_token_proj = ColParallelLinear(
            modulation_dim,
            inner_dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

    # We do not shard the last dimension of spatial, because its dimension is less than the tile
    # size for a device count of four and more. This requires padding, which is not currently
    # supported by `reduce_scatter_minimal_async`.
    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Run the model forward.

        Args:
            spatial: Tensor with shape [batch_size, padded_spatial_sequence_length / sp_factor, channels].
            prompt: Tensor with shape [batch_size, prompt_sequence_length, joint_attention_dim].
            pooled: Tensor with shape [batch_size, pooled_projection_dim].
            timestep: Tensor with shape [batch_size, 1].
        """
        batch_size, _, _ = spatial.shape
        assert len(prompt.shape) == 3
        assert len(pooled.shape) == 2
        assert timestep.shape in ([1, 1], [batch_size, 1])

        assert prompt.shape[0] == batch_size
        assert pooled.shape[0] == batch_size

        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        time_embed = self.time_text_embed(timestep=timestep, pooled_projection=pooled)
        time_embed = time_embed.reshape([batch_size, 1, time_embed.shape[-1]])
        time_embed_silu = ttnn.silu(time_embed)

        spatial = self.pos_embed.forward(spatial, already_unfolded=True)
        spatial = spatial.reshape([spatial.shape[-3], spatial.shape[-2], -1])  # fix shape since pos_embed unsqueezes
        prompt = self.context_embedder(prompt)

        # the sequence already includes space for register tokens
        # register_tokens = ttnn.repeat(self.register_tokens.data, [batch_size, 1, 1])
        # spatial = ttnn.concat([register_tokens, spatial], dim=1)
        spatial = spatial * self.register_tokens_mask.data + self.register_tokens.data

        # append time token
        t_token = self.t_token_proj(time_embed)
        t_token = ttnn.clone(t_token, dtype=prompt.dtype)
        prompt = ttnn.concat([prompt, t_token], dim=1)

        for i, block in enumerate(self.transformer_blocks, start=1):
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed_silu,
                spatial_sequence_length=self.spatial_sequence_length,
                skip_time_embed_activation_fn=True,
            )

            if i % 6 == 0:
                ttnn.ReadDeviceProfiler(spatial.device())

        # we keep the register tokens in the sequence
        # spatial = spatial[:, self.register_tokens.shape[1] :]

        spatial = self.ccl_manager.all_gather_persistent_buffer(spatial, dim=2, mesh_axis=tp_axis, use_hyperparams=True)

        # same as in SD3 but without norm and silu
        spatial_time = self.time_embed_out(time_embed)  # , core_grid=self.core_grid)
        scale, shift = _chunk_time3d(spatial_time, 2)

        spatial = spatial * (1 + scale) + shift

        return self.proj_out(
            spatial  # , core_grid=self.core_grid, compute_kernel_config=self.hifi_compute_kernel_config
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm_out.linear", "time_embed_out")  # chunks=2 if sharded
        rename_substate(state, "norm_out.norm", "norm_out")

        tokens = state.pop("register_tokens", None)
        if tokens is not None:
            _, token_num, _ = tokens.shape
            assert token_num == self.register_token_num

            mask = torch.zeros_like(tokens)
            padding = self.padded_spatial_sequence_length - self.register_token_num

            state["register_tokens"] = torch.nn.functional.pad(tokens, (0, 0, 0, padding))
            state["register_tokens_mask"] = torch.nn.functional.pad(mask, (0, 0, 0, padding), value=1)

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        # N, H, W, C -> N, (H / P) * (W / P), P * P * C
        batch_size, height, width, channels = latents.shape
        patch = self.patch_size

        latents = latents.reshape([batch_size, height // patch, patch, width // patch, patch, channels])
        spatial = latents.transpose(2, 3).flatten(3, 5).flatten(1, 2)

        return torch.nn.functional.pad(spatial, [0, 0, self.register_token_num, self.spatial_sequence_padding])

    def unpatchify(self, spatial: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        # N, (H / P) * (W / P), P * P * C -> N, H, W, C
        batch_size, _, _ = spatial.shape
        patch = self.patch_size
        sequence_length = (height // patch) * (width // patch)

        spatial = spatial[:, self.register_token_num : self.register_token_num + sequence_length, :]

        spatial = spatial.reshape([batch_size, height // patch, width // patch, patch, patch, -1])
        return spatial.transpose(2, 3).flatten(3, 4).flatten(1, 2)


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


def convert_motif_attention_state(
    state: dict[str, torch.Tensor],
    *,
    prefix: str = "",
    is_last_block: bool,
    x_weight: torch.Tensor,
    x_bias: torch.Tensor,
    c_weight: torch.Tensor,
    c_bias: torch.Tensor,
) -> None:
    renames = {
        "o_proj.weight": "to_out.0.weight",
        "add_o_proj.weight": "to_add_out.weight",
        "q_norm_x.weight": "norm_q.weight",
        "k_norm_x.weight": "norm_k.weight",
        "q_norm_c.weight": "norm_added_q.weight",
        "k_norm_c.weight": "norm_added_k.weight",
        "q_scale": "context_head_factors",
    }

    for src, dst in renames.items():
        state[f"{prefix}{dst}"] = state.pop(f"{prefix}{src}")

    q_weight = state.pop(f"{prefix}q_proj.weight")
    state[f"{prefix}to_q.weight"] = q_weight @ x_weight
    state[f"{prefix}to_q.bias"] = q_weight @ x_bias
    k_weight = state.pop(f"{prefix}k_proj.weight")
    state[f"{prefix}to_k.weight"] = k_weight @ x_weight
    state[f"{prefix}to_k.bias"] = k_weight @ x_bias
    v_weight = state.pop(f"{prefix}v_proj.weight")
    state[f"{prefix}to_v.weight"] = v_weight @ x_weight
    state[f"{prefix}to_v.bias"] = v_weight @ x_bias

    add_q_weight = state.pop(f"{prefix}add_q_proj.weight")
    state[f"{prefix}add_q_proj.weight"] = add_q_weight @ c_weight
    state[f"{prefix}add_q_proj.bias"] = add_q_weight @ c_bias
    add_k_weight = state.pop(f"{prefix}add_k_proj.weight")
    state[f"{prefix}add_k_proj.weight"] = add_k_weight @ c_weight
    state[f"{prefix}add_k_proj.bias"] = add_k_weight @ c_bias
    add_v_weight = state.pop(f"{prefix}add_v_proj.weight")
    state[f"{prefix}add_v_proj.weight"] = add_v_weight @ c_weight
    state[f"{prefix}add_v_proj.bias"] = add_v_weight @ c_bias

    state[f"{prefix}to_out.0.bias"] = torch.zeros_like(state[f"{prefix}to_out.0.weight"][0])
    state[f"{prefix}to_add_out.bias"] = torch.zeros_like(state[f"{prefix}to_add_out.weight"][0])

    # Unused since we can set context_pre_only=True in the last block.
    if is_last_block:
        del state[f"{prefix}to_add_out.weight"]
        del state[f"{prefix}to_add_out.bias"]


def convert_motif_transformer_block_state(
    state: dict[str, torch.Tensor], *, prefix: str = "", is_last_block: bool
) -> None:
    renames = {
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
        state[f"{prefix}{dst}"] = state.pop(f"{prefix}{src}")

    x_weight = state.pop(f"{prefix}linear_1_x.weight")
    x_bias = state.pop(f"{prefix}linear_1_x.bias")
    c_weight = state.pop(f"{prefix}linear_1_c.weight")
    c_bias = state.pop(f"{prefix}linear_1_c.bias")

    convert_motif_attention_state(
        state,
        prefix=f"{prefix}attn.",
        x_weight=x_weight,
        x_bias=x_bias,
        c_weight=c_weight,
        c_bias=c_bias,
        is_last_block=is_last_block,
    )

    _convert_ada_norm(state, f"{prefix}norm1.linear", pre_only=False)
    _convert_ada_norm(state, f"{prefix}norm1_context.linear", pre_only=is_last_block)

    # Unused since we can set context_pre_only=True in the last block.
    if is_last_block:
        del state[f"{prefix}ff_context.net.0.proj.weight"]
        del state[f"{prefix}ff_context.net.0.proj.bias"]
        del state[f"{prefix}ff_context.net.2.weight"]
        del state[f"{prefix}ff_context.net.2.bias"]


def convert_motif_transformer_state(state: dict[str, torch.Tensor], *, num_layers: int) -> None:
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

    for src, dst in renames.items():
        state[dst] = state.pop(src)

    state["pos_embed.pos_embed"] = state["pos_embed.pos_embed"].unsqueeze(0)

    for i in range(num_layers):
        rename_substate(state, f"mmdit_blocks.{i}", f"transformer_blocks.{i}")
        convert_motif_transformer_block_state(
            state,
            prefix=f"transformer_blocks.{i}.",
            is_last_block=i == num_layers - 1,
        )
