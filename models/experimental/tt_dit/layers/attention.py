# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn

from ..utils.padding import PaddingConfig, pad_weight_tensor
from ..utils.substate import pop_substate
from .linear import ColParallelLinear
from .module import Module, Parameter
from .normalization import RMSNorm

if TYPE_CHECKING:
    from ..parallel.config import DiTParallelConfig, ParallelFactor
    from ..parallel.manager import CCLManager


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
class Attention(Module):
    def __init__(
        self,
        *,
        query_dim: int,
        head_dim: int,
        heads: int,
        out_dim: int,
        added_kv_proj_dim: int,
        context_pre_only: bool = False,
        pre_only: bool = False,
        eps: float,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
        init: bool = False,
        use_spatial_weights_for_prompt: bool = False,
        added_head_scaling: bool = False,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.pre_only = pre_only
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.padding_config = padding_config
        self.use_spatial_weights_for_prompt = use_spatial_weights_for_prompt

        self.padded_heads = padding_config.target_heads if padding_config is not None else heads
        self.n_local_heads = self.padded_heads // self.parallel_config.tensor_parallel.factor

        common_args = dict(mesh_device=mesh_device, init=init)
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        padded_inner_dim = head_dim * self.padded_heads

        self.to_qkv = ColParallelLinear(query_dim, 3 * padded_inner_dim, mesh_axis=tp_axis, **common_args)

        self.norm_q = RMSNorm(embedding_dim=head_dim, norm_eps=eps, bias=False, **common_args)
        self.norm_k = RMSNorm(embedding_dim=head_dim, norm_eps=eps, bias=False, **common_args)

        self.to_out = (
            ColParallelLinear(padded_inner_dim, out_dim, mesh_axis=tp_axis, **common_args)
            if not self.pre_only
            else None
        )

        if use_spatial_weights_for_prompt:
            self.add_qkv_proj = self.to_qkv
            self.norm_added_q = self.norm_q
            self.norm_added_k = self.norm_k
            self.to_add_out = self.to_out
        elif added_kv_proj_dim > 0:
            self.add_qkv_proj = ColParallelLinear(
                added_kv_proj_dim, 3 * padded_inner_dim, mesh_axis=tp_axis, **common_args
            )

            self.norm_added_q = RMSNorm(embedding_dim=head_dim, norm_eps=eps, bias=False, **common_args)
            self.norm_added_k = RMSNorm(embedding_dim=head_dim, norm_eps=eps, bias=False, **common_args)

            self.to_add_out = (
                ColParallelLinear(padded_inner_dim, out_dim, mesh_axis=tp_axis, **common_args)
                if not context_pre_only
                else None
            )
        else:
            self.add_qkv_proj = None
            self.norm_added_q = None
            self.norm_added_k = None
            self.to_add_out = None

        self.added_head_factors = (
            Parameter(
                shape=[self.n_local_heads, 1],
                device=mesh_device,
                init=init and torch.ones([self.n_local_heads, 1]),
            )
            if added_head_scaling and self.add_qkv_proj is not None
            else None
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight, bias = self._reshape_and_merge_qkv(
            pop_substate(state, "to_q"),
            pop_substate(state, "to_k"),
            pop_substate(state, "to_v"),
        )
        if weight is not None:
            state["to_qkv.weight"] = weight
        if bias is not None:
            state["to_qkv.bias"] = bias

        weight, bias = self._reshape_and_merge_qkv(
            pop_substate(state, "add_q_proj"),
            pop_substate(state, "add_k_proj"),
            pop_substate(state, "add_v_proj"),
        )
        if weight is not None:
            state["add_qkv_proj.weight"] = weight
        if bias is not None:
            state["add_qkv_proj.bias"] = bias

        if "to_out.0.weight" in state:
            weight = state.pop("to_out.0.weight")
            if self.padding_config is not None:
                weight = pad_weight_tensor(weight, self.padding_config, pad_input_dim=True)
            state["to_out.weight"] = weight
        if "to_out.0.bias" in state:
            state["to_out.bias"] = state.pop("to_out.0.bias")

        if "to_add_out.weight" in state:
            weight = state.pop("to_add_out.weight")
            if self.padding_config is not None:
                weight = pad_weight_tensor(weight, self.padding_config, pad_input_dim=True)
            state["to_add_out.weight"] = weight
        if "to_add_out.bias" in state:
            state["to_add_out.bias"] = state.pop("to_add_out.bias")

        if "added_head_factors" in state:
            factors = state["added_head_factors"].reshape([-1, 1])
            if self.padding_config is not None:
                pad = (0, 0, 0, self.padding_config.self.head_padding)
                factors = torch.nn.functional.pad(factors, pad)
            state["added_head_factors"] = factors

    def _reshape_and_merge_qkv(
        self,
        q_state: dict[str, torch.Tensor],
        k_state: dict[str, torch.Tensor],
        v_state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # Rearrange QKV projections such column-fracturing shards the heads
        def _merge_tensors(q, k, v):
            n_dev = self.parallel_config.tensor_parallel.factor
            q, k, v = q.T, k.T, v.T
            # Pad QKV weights and biases to match the padded heads
            if self.padding_config is not None:
                q = pad_weight_tensor(q, self.padding_config, pad_output_dim=True)
                k = pad_weight_tensor(k, self.padding_config, pad_output_dim=True)
                v = pad_weight_tensor(v, self.padding_config, pad_output_dim=True)
            q = q.reshape(q.shape[0], n_dev, self.n_local_heads, self.head_dim)
            k = k.reshape(k.shape[0], n_dev, self.n_local_heads, self.head_dim)
            v = v.reshape(v.shape[0], n_dev, self.n_local_heads, self.head_dim)
            qkv = torch.cat([q, k, v], dim=2)
            qkv = qkv.reshape(qkv.shape[0], 3 * self.padded_heads * self.head_dim)
            qkv = qkv.T
            return qkv

        if "weight" in q_state and "weight" in k_state and "weight" in v_state:
            weight = _merge_tensors(q_state["weight"], k_state["weight"], v_state["weight"])
        else:
            weight = None

        if "bias" in q_state and "bias" in k_state and "bias" in v_state:
            bias = _merge_tensors(
                q_state["bias"].unsqueeze(-1), k_state["bias"].unsqueeze(-1), v_state["bias"].unsqueeze(-1)
            )
            bias = bias.squeeze(-1)
        else:
            bias = None

        return weight, bias

    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor | None = None,
        spatial_sequence_length: int,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """Forward pass of the model.

        Args:
            spatial: Tensor with shape [batch_size, spatial_sequence_length / sp_factor, query_dim].
            prompt: Tensor with shape [batch_size, prompt_sequence_length, query_dim] (not sharded!).
            spatial_rope: Tuple of two tensors with shape [spatial_sequence_length / sp_factor, head_dim].
            prompt_rope: Tuple of two tensors with shape [prompt_sequence_length, head_dim] (not sharded!).
        """
        assert len(spatial.shape) == 3
        if prompt is not None:
            assert len(prompt.shape) == 3
        for t in spatial_rope or ():
            assert len(t.shape) == 2
        for t in prompt_rope or ():
            assert len(t.shape) == 2

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

        qkv = self.to_qkv(
            spatial, core_grid=core_grid
        )  # [batch_size, spatial_sequence_length / sp_factor, 3 * n_local_heads * head_dim (in this order)]
        local_heads = self.n_local_heads
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=local_heads, transpose_key=False
        )  # [batch_size, n_local_heads, spatial_sequence_length / sp_factor, head_dim]

        q = self.norm_q(q)
        k = self.norm_k(k)

        if spatial_rope is not None:
            q = _apply_rope(q, spatial_rope)
            k = _apply_rope(k, spatial_rope)

        if self.add_qkv_proj is not None:
            add_qkv = self.add_qkv_proj(prompt, core_grid=core_grid)
            add_q, add_k, add_v = ttnn.transformer.split_query_key_value_and_split_heads(
                add_qkv, num_heads=local_heads, transpose_key=False
            )
            add_q = self.norm_added_q(add_q)
            add_k = self.norm_added_k(add_k)

            if self.added_head_factors is not None:
                add_q = add_q * self.added_head_factors.data

            if prompt_rope is not None:
                add_q = _apply_rope(add_q, prompt_rope)
                add_k = _apply_rope(add_k, prompt_rope)
        else:
            shape = [1, self.n_local_heads, 0, self.head_dim]
            add_q = add_k = add_v = ttnn.zeros(shape, device=self.mesh_device, layout=q.layout, dtype=q.dtype)

        k_chunk_size = 512
        while q.shape[-2] % k_chunk_size != 0:
            k_chunk_size //= 2

        full_grid = self.mesh_device.compute_with_storage_grid_size()
        sdpa_worker_grid = (full_grid.x, full_grid.y - 1)

        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_worker_grid,
            q_chunk_size=128,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,  # NOTE: False is more correct
        )
        sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,  # NOTE: Set to True if there's a correctness issue
        )

        if self.parallel_config.sequence_parallel.factor > 1:
            spatial, prompt, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q,
                k,
                v,
                add_q,
                add_k,
                add_v,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                    k.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
                ),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                    v.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
                ),
                joint_strategy="rear",
                logical_n=spatial_sequence_length,
                program_config=sdpa_program_config,
                compute_kernel_config=sdpa_compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.sequence_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
                mesh_device=self.mesh_device,
                topology=self.ccl_manager.topology,
                subdevice_id=self.ccl_manager.ccl_sub_device_id,
                ccl_core_grid_offset=(0, sdpa_worker_grid[1]),
            )
        else:
            spatial, prompt = ttnn.transformer.joint_scaled_dot_product_attention(
                q,
                k,
                v,
                add_q,
                add_k,
                add_v,
                joint_strategy="rear",
                program_config=sdpa_program_config,
                compute_kernel_config=sdpa_compute_kernel_config,
            )

        spatial = ttnn.transformer.concatenate_heads(spatial)
        if prompt is not None:
            prompt = ttnn.transformer.concatenate_heads(prompt)

        if self.to_out is not None:
            spatial = all_gather(
                spatial, dim=2, parallel_factor=self.parallel_config.tensor_parallel, ccl_manager=self.ccl_manager
            )

            spatial = self.to_out(spatial, core_grid=core_grid)

        if self.to_add_out is not None:
            prompt = all_gather(
                prompt, dim=2, parallel_factor=self.parallel_config.tensor_parallel, ccl_manager=self.ccl_manager
            )
            prompt = self.to_add_out(prompt, core_grid=core_grid)

        return spatial, prompt


def _apply_rope(x: ttnn.Tensor, freqs_cis: tuple[ttnn.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
    cos, sin = freqs_cis
    cos = cos.reshape([1, 1, *cos.shape])
    sin = sin.reshape([1, 1, *sin.shape])

    return x * cos + ttnn.alt_complex_rotate90(x) * sin


# TODO (Friedrich): move to CCLManager
# TODO (Friedrich): change parallel_factor to mesh_axis and obtain factor from mesh shape?
def all_gather(x: ttnn.Tensor, /, *, dim: int, parallel_factor: ParallelFactor, ccl_manager: CCLManager) -> ttnn.Tensor:
    if parallel_factor.factor <= 1:
        return x

    # all_gather_async currently supports tensors of rank 4 only
    rank = len(x.shape)
    if rank < 4:
        shape = [1] * (4 - rank) + list(x.shape)
        x = ttnn.reshape(x, shape)
        if dim >= 0:
            dim += 4 - rank

    x = ttnn.experimental.all_gather_async(
        x,
        persistent_output_buffer=ccl_manager.get_ag_ping_pong_buffer(x.shape, dim, parallel_factor.mesh_axis),
        dim=dim,
        multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(parallel_factor.mesh_axis),
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        cluster_axis=parallel_factor.mesh_axis,
        **ccl_manager.get_ag_hyperparams(x.shape),
    )

    if rank < 4:
        shape = list(x.shape)[4 - rank :]
        x = ttnn.reshape(x, shape)

    return x
