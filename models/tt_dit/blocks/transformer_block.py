# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

from ..layers.feedforward import ParallelFeedForward
from ..layers.linear import ColParallelLinear, prepare_chunked_linear_output
from ..layers.module import Module
from ..layers.normalization import DistributedLayerNorm
from ..utils.substate import rename_substate
from .attention import Attention

if TYPE_CHECKING:
    import torch

    from ..parallel.config import DiTParallelConfig
    from ..parallel.manager import CCLManager
    from ..utils.padding import PaddingConfig


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
class TransformerBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        modulation_dim: int | None = None,
        num_heads: int,
        head_dim: int,
        context_pre_only: bool,
        add_attention_to_output: bool = True,
        context_head_scaling: bool = False,
        ff_activation_fn: str = "gelu",
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
        attention_k_chunk_size: int = 512,
        attention_q_chunk_size: int = 128,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        if modulation_dim is None:
            modulation_dim = dim

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.add_attention_to_output = add_attention_to_output

        self.context_pre_only = context_pre_only
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        # FSDP: shard weights on sequence parallel axis to reduce memory
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.norm1_linear = ColParallelLinear(
            modulation_dim,
            6 * dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
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
            modulation_dim,
            context_norm_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
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

        self.attn = Attention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            added_kv_proj_dim=dim,
            context_pre_only=context_pre_only,
            context_head_scaling=context_head_scaling,
            eps=1e-6,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
            k_chunk_size=attention_k_chunk_size,
            q_chunk_size=attention_q_chunk_size,
            is_fsdp=is_fsdp,
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
            activation_fn=ff_activation_fn,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
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
                activation_fn=ff_activation_fn,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                fsdp_mesh_axis=fsdp_mesh_axis,
                ccl_manager=ccl_manager,
            )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm1.linear", "norm1_linear")
        rename_substate(state, "norm1.norm", "norm1_norm")
        rename_substate(state, "norm1_context.linear", "norm1_context_linear")
        rename_substate(state, "norm1_context.norm", "norm1_context_norm")
        rename_substate(state, "ff.net.0.proj", "ff.ff1")
        rename_substate(state, "ff.net.2", "ff.ff2")
        rename_substate(state, "ff_context.net.0.proj", "ff_context.ff1")
        rename_substate(state, "ff_context.net.2", "ff_context.ff2")

        prepare_chunked_linear_output(
            state,
            prefix="norm1_linear",
            device_count=self.parallel_config.tensor_parallel.factor,
            chunks=6,
        )
        prepare_chunked_linear_output(
            state,
            prefix="norm1_context_linear",
            device_count=self.parallel_config.tensor_parallel.factor,
            chunks=2 if self.context_pre_only else 6,
        )

    def forward(
        self,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        time_embed: ttnn.Tensor,
        spatial_sequence_length: int,
        *,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        skip_time_embed_activation_fn: bool = False,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """Run the model forward.

        Args:
            spatial: Tensor with shape [batch_size, spatial_sequence_length / sp_factor, query_dim / tp_factor].
            prompt: Tensor with shape [batch_size, prompt_sequence_length, query_dim / tp_factor] (sequence is not sharded!).
            time_embed: Tensor with shape [batch_size, 1, query_dim].
            spatial_rope: Tuple of two tensors with shape [spatial_sequence_length / sp_factor, head_dim].
            prompt_rope: Tuple of two tensors with shape [prompt_sequence_length, head_dim] (sequence is not sharded!).
        """
        assert len(spatial.shape) == 3
        assert len(prompt.shape) == 3

        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        if not skip_time_embed_activation_fn:
            time_embed = ttnn.silu(time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        spatial_time = self.norm1_linear(time_embed)
        prompt_time = self.norm1_context_linear(time_embed)

        (
            spatial_shift_attn,
            spatial_scale_attn,
            spatial_gate_attn,
            spatial_shift_ff,
            spatial_scale_ff,
            spatial_gate_ff,
        ) = _chunk_time3d(spatial_time, 6)

        spatial_normed = ttnn.squeeze(
            self.norm1_norm(
                ttnn.unsqueeze(spatial, 0),
                dynamic_weight=(1 + spatial_scale_attn),
                dynamic_bias=spatial_shift_attn,
            ),
            0,
        )

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

        prompt_normed = ttnn.squeeze(
            self.norm1_context_norm(
                ttnn.unsqueeze(prompt, 0),
                dynamic_weight=(1 + prompt_scale_attn),
                dynamic_bias=prompt_shift_attn,
            ),
            0,
        )

        # Gather spatial, prompt before attention
        spatial_normed = self.ccl_manager.all_gather_persistent_buffer(
            spatial_normed, dim=2, mesh_axis=tp_axis, use_hyperparams=True
        )
        prompt_normed = self.ccl_manager.all_gather_persistent_buffer(
            prompt_normed, dim=2, mesh_axis=tp_axis, use_hyperparams=True
        )

        spatial_attn, prompt_attn = self.attn.forward(
            spatial=spatial_normed,
            prompt=prompt_normed,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            spatial_sequence_length=spatial_sequence_length,
        )
        spatial_attn = spatial_attn * spatial_gate_attn
        prompt_attn = prompt_attn * prompt_gate_attn if prompt_gate_attn is not None else None

        spatial_plus_attn = spatial + spatial_attn
        if self.add_attention_to_output:
            spatial = spatial_plus_attn

        spatial_normed = ttnn.squeeze(
            self.norm2(
                ttnn.unsqueeze(spatial_plus_attn, 0),
                dynamic_weight=(1 + spatial_scale_ff),
                dynamic_bias=spatial_shift_ff,
            ),
            0,
        )

        spatial_normed = self.ccl_manager.all_gather_persistent_buffer(
            spatial_normed, dim=2, mesh_axis=tp_axis, use_hyperparams=True
        )

        spatial_ff = ttnn.squeeze(self.ff(ttnn.unsqueeze(spatial_normed, 0)), 0)
        spatial_ff = spatial_ff * spatial_gate_ff

        spatial = spatial + spatial_ff

        if self.context_pre_only:
            return spatial, None

        prompt_plus_attn = prompt + prompt_attn
        if self.add_attention_to_output:
            prompt = prompt_plus_attn

        prompt_normed = ttnn.squeeze(
            self.norm2_context(
                ttnn.unsqueeze(prompt_plus_attn, 0),
                dynamic_weight=(1 + prompt_scale_ff),
                dynamic_bias=prompt_shift_ff,
            ),
            0,
        )

        prompt_normed = self.ccl_manager.all_gather_persistent_buffer(
            prompt_normed, dim=2, mesh_axis=tp_axis, use_hyperparams=True
        )

        prompt_ff = ttnn.squeeze(self.ff_context(ttnn.unsqueeze(prompt_normed, 0)), 0)
        prompt_ff = prompt_ff * prompt_gate_ff

        prompt = prompt + prompt_ff

        return spatial, prompt


def _chunk_time3d(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = t.shape[-1] // count
    return [t[:, :, i * size : (i + 1) * size] for i in range(count)]
