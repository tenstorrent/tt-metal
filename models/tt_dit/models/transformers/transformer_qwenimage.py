# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

from ...blocks.transformer_block import TransformerBlock
from ...layers.embeddings import SD35CombinedTimestepTextProjEmbeddings
from ...layers.linear import ColParallelLinear, Linear
from ...layers.module import Module, ModuleList
from ...layers.normalization import DistributedLayerNorm, RMSNorm
from ...utils.substate import rename_substate

if TYPE_CHECKING:
    import torch

    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager
    from ...utils.padding import PaddingConfig


class QwenImageTransformerBlock(TransformerBlock):
    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "img_mod.1", "norm1.linear")
        rename_substate(state, "img_norm1", "norm1.norm")
        rename_substate(state, "img_norm2", "norm2")
        rename_substate(state, "txt_mod.1", "norm1_context.linear")
        rename_substate(state, "txt_norm1", "norm1_context.norm")
        rename_substate(state, "img_mlp", "ff")
        rename_substate(state, "txt_norm2", "norm2_context")
        rename_substate(state, "txt_mlp", "ff_context")

        super()._prepare_torch_state(state)


# adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage.py
class QwenImageTransformer(Module):
    def __init__(
        self,
        *,
        patch_size: int,
        in_channels: int,
        num_layers: int,
        attention_head_dim: int,
        num_attention_heads: int,
        joint_attention_dim: int,
        out_channels: int,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        # FSDP: shard weights on sequence parallel axis to reduce memory
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.time_text_embed = SD35CombinedTimestepTextProjEmbeddings(
            embedding_dim=inner_dim, pooled_projection_dim=0, mesh_device=device
        )

        self.txt_norm = RMSNorm(joint_attention_dim, bias=False, norm_eps=1e-6, mesh_device=device)

        self.txt_in = ColParallelLinear(  # context_embedder in Flux.1
            joint_attention_dim,
            inner_dim,
            mesh_device=device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        # Shard output, since size of input dimension << size of output dimension.
        self.img_in = ColParallelLinear(  # x_embedder in Flux.1
            in_channels,
            inner_dim,
            mesh_device=device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.transformer_blocks = ModuleList(
            QwenImageTransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_pre_only=False,
                # ff_activation_fn="gelu-approximate",  # this is what the original model uses
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                mesh_device=device,
                is_fsdp=is_fsdp,
            )
            for i in range(num_layers)
        )

        self.time_embed_out = Linear(inner_dim, 2 * inner_dim, mesh_device=device)

        self.norm_out = DistributedLayerNorm(
            inner_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_device=device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.proj_out = Linear(
            inner_dim,
            patch_size * patch_size * out_channels,
            mesh_device=device,
        )

        self._patch_size = patch_size
        self._tp_axis = parallel_config.tensor_parallel.mesh_axis
        self._ccl_manager = ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm_out.linear", "time_embed_out")
        rename_substate(state, "norm_out.norm", "norm_out")

    # We do not shard the last dimension of spatial, because its dimension is less than the tile
    # size for a device count of four or more. This requires padding, which is not currently
    # supported by `reduce_scatter_minimal_async`.
    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
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
            spatial_rope: Tuple of two tensors with shape [spatial_sequence_length / sp_factor, head_dim].
            prompt_rope: Tuple of two tensors with shape [prompt_sequence_length, head_dim] (sequence is not sharded!).
        """
        time_embed = self.time_text_embed(timestep=timestep)
        ttnn.silu(time_embed, output_tensor=time_embed)
        time_embed = time_embed.reshape([time_embed.shape[-2], 1, time_embed.shape[-1]])

        spatial = self.img_in(spatial)

        prompt = self.txt_norm(prompt)
        prompt = self.txt_in(prompt)

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

            if i % 6 == 0:
                ttnn.ReadDeviceProfiler(spatial.device())

        # TODO: remove unsqueeze/squeeze when DistributedLayerNorm allows it
        spatial = ttnn.squeeze(self.norm_out(ttnn.unsqueeze(spatial, 0)), 0)

        spatial_time = self.time_embed_out(time_embed)
        [scale, shift] = _chunk_time3d(spatial_time, 2)

        spatial = self._ccl_manager.all_gather_persistent_buffer(
            spatial, dim=2, mesh_axis=self._tp_axis, use_hyperparams=True
        )

        spatial = spatial * (1 + scale) + shift

        return self.proj_out(spatial)

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        # N, H, W, C -> N, (H / P) * (W / P), C * P * P
        batch_size, height, width, channels = latents.shape
        patch = self._patch_size

        if height % patch != 0 or width % patch != 0:
            msg = f"height ({height}) and width ({width}) must be divisible by patch_size ({patch})"
            raise ValueError(msg)

        latents = latents.reshape([batch_size, height // patch, patch, width // patch, patch, channels])
        return latents.permute(0, 1, 3, 5, 2, 4).flatten(3, 5).flatten(1, 2)

    def unpatchify(self, spatial: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        # N, (H / P) * (W / P), C * P * P -> N, H, W, C
        batch_size, _, _ = spatial.shape
        patch = self._patch_size

        if height % patch != 0 or width % patch != 0:
            msg = f"height ({height}) and width ({width}) must be divisible by patch_size ({patch})"
            raise ValueError(msg)

        spatial = spatial.reshape([batch_size, height // patch, width // patch, -1, patch, patch])
        return spatial.permute(0, 1, 4, 2, 5, 3).flatten(3, 4).flatten(1, 2)


def _chunk_time3d(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = t.shape[-1] // count
    return [t[:, :, i * size : (i + 1) * size] for i in range(count)]
