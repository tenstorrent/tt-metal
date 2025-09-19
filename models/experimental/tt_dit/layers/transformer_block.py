# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

from ..layers.feedforward import ParallelFeedForward
from ..layers.linear import ColParallelLinear
from ..layers.normalization import DistributedLayerNorm
from ..utils.substate import rename_substate
from .attention import Attention, all_gather
from .module import Module

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
        ff_activation_fn: str = "gelu",
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
        init: bool = False,
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

        self.norm1_linear = ColParallelLinear(
            modulation_dim,
            6 * dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            init=init,
        )

        self.norm1_norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            init=init,
        )

        context_norm_dim = 6 * dim if not context_pre_only else 2 * dim
        self.norm1_context_linear = ColParallelLinear(
            modulation_dim,
            context_norm_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            init=init,
        )
        self.norm1_context_norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            init=init,
        )

        self.attn = Attention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            added_kv_proj_dim=dim,
            context_pre_only=context_pre_only,
            eps=1e-6,
            mesh_device=mesh_device,
            init=init,
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
            init=init,
        )

        self.ff = ParallelFeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn=ff_activation_fn,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            init=init,
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
                init=init,
            )
            self.ff_context = ParallelFeedForward(
                dim=dim,
                dim_out=dim,
                activation_fn=ff_activation_fn,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
                init=init,
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

        def _shuffle_ada_norm_linear(prefix: str) -> None:
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

            weight_key = f"{prefix}.weight"
            bias_key = f"{prefix}.bias"

            if weight_key in state:
                in_dim = state[weight_key].shape[1]

                state[weight_key] = _shuffle(state[weight_key], in_dim)
                if bias_key in state:
                    state[bias_key] = _shuffle(state[bias_key].reshape(-1, 1), in_dim).squeeze()

        _shuffle_ada_norm_linear("norm1_linear")
        _shuffle_ada_norm_linear("norm1_context_linear")

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
        spatial_normed = all_gather(
            spatial_normed, dim=2, parallel_factor=self.parallel_config.tensor_parallel, ccl_manager=self.ccl_manager
        )
        prompt_normed = all_gather(
            prompt_normed, dim=2, parallel_factor=self.parallel_config.tensor_parallel, ccl_manager=self.ccl_manager
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

        spatial_normed = ttnn.squeeze(self.norm2(ttnn.unsqueeze(spatial_plus_attn, 0)), 0)
        spatial_normed = spatial_normed * (1 + spatial_scale_ff) + spatial_shift_ff

        spatial_normed = all_gather(
            spatial_normed, dim=2, parallel_factor=self.parallel_config.tensor_parallel, ccl_manager=self.ccl_manager
        )

        spatial_ff = ttnn.squeeze(self.ff(ttnn.unsqueeze(spatial_normed, 0), core_grid=self.core_grid), 0)
        spatial_ff = spatial_ff * spatial_gate_ff

        spatial = spatial + spatial_ff

        if self.context_pre_only:
            return spatial, None

        prompt_plus_attn = prompt + prompt_attn
        if self.add_attention_to_output:
            prompt = prompt_plus_attn

        prompt_normed = ttnn.squeeze(self.norm2_context(ttnn.unsqueeze(prompt_plus_attn, 0)), 0)
        prompt_normed = prompt_normed * (1 + prompt_scale_ff) + prompt_shift_ff

        prompt_normed = all_gather(
            prompt_normed, dim=2, parallel_factor=self.parallel_config.tensor_parallel, ccl_manager=self.ccl_manager
        )

        prompt_ff = ttnn.squeeze(self.ff_context(ttnn.unsqueeze(prompt_normed, 0), core_grid=self.core_grid), 0)
        prompt_ff = prompt_ff * prompt_gate_ff

        prompt = prompt + prompt_ff

        return spatial, prompt


def _chunk_time3d(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = t.shape[-1] // count
    return [t[:, :, i * size : (i + 1) * size] for i in range(count)]
