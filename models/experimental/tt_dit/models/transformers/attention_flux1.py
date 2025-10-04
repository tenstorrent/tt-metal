# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn

from ...layers.linear import ColParallelLinear
from ...layers.module import Module
from ...layers.normalization import RMSNorm
from ...utils.padding import PaddingConfig, pad_weight_tensor
from ...utils.substate import substate

if TYPE_CHECKING:
    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
class Flux1Attention(Module):
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
        use_spatial_weights_for_prompt: bool = False,
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

        common_args = dict(mesh_device=mesh_device)
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

    # TODO: migrate to _prepare_torch_state
    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        def pad_dense_out(state):
            # Pad dense output weights and biases to match the padded heads
            weight = state["weight"].T
            bias = state["bias"]
            if self.padding_config is not None:
                weight = pad_weight_tensor(weight, self.padding_config, pad_input_dim=True)
            weight = weight.T
            return {"weight": weight, "bias": bias}

        self.norm_q.load_state_dict(substate(state_dict, "norm_q"))
        self.norm_k.load_state_dict(substate(state_dict, "norm_k"))
        qkv_state = self._reshape_and_merge_qkv(
            substate(state_dict, "to_q"), substate(state_dict, "to_k"), substate(state_dict, "to_v")
        )
        self.to_qkv.load_state_dict(qkv_state)

        if not self.pre_only:
            self.to_out.load_state_dict(pad_dense_out(substate(state_dict, "to_out.0")))

        if self.add_qkv_proj is not None and not self.use_spatial_weights_for_prompt:
            add_qkv_state = self._reshape_and_merge_qkv(
                substate(state_dict, "add_q_proj"),
                substate(state_dict, "add_k_proj"),
                substate(state_dict, "add_v_proj"),
            )
            self.add_qkv_proj.load_state_dict(add_qkv_state)
            if self.to_add_out is not None:
                self.to_add_out.load_state_dict(pad_dense_out(substate(state_dict, "to_add_out")))
            self.norm_added_q.load_state_dict(substate(state_dict, "norm_added_q"))
            self.norm_added_k.load_state_dict(substate(state_dict, "norm_added_k"))

    def _reshape_and_merge_qkv(
        self,
        q_state: dict[str, torch.Tensor],
        k_state: dict[str, torch.Tensor],
        v_state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
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

        weight = _merge_tensors(q_state["weight"], k_state["weight"], v_state["weight"])

        out_state = {"weight": weight}
        if "bias" in q_state:
            bias = _merge_tensors(
                q_state["bias"].unsqueeze(-1), k_state["bias"].unsqueeze(-1), v_state["bias"].unsqueeze(-1)
            )
            bias = bias.squeeze(-1)
            out_state["bias"] = bias
        return out_state

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
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        qkv = self.to_qkv(spatial, core_grid=core_grid)
        local_heads = self.n_local_heads
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=local_heads, transpose_key=False
        )

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
            spatial = self.ccl_manager.all_gather(spatial, dim=2, mesh_axis=tp_axis)
            spatial = self.to_out(spatial, core_grid=core_grid)

        if self.to_add_out is not None:
            prompt = self.ccl_manager.all_gather(prompt, dim=2, mesh_axis=tp_axis)
            prompt = self.to_add_out(prompt, core_grid=core_grid)

        return spatial, prompt


def _apply_rope(x: ttnn.Tensor, freqs_cis: tuple[ttnn.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
    cos, sin = freqs_cis
    cos = cos.reshape([1, 1, *cos.shape])
    sin = sin.reshape([1, 1, *sin.shape])

    return x * cos + ttnn.alt_complex_rotate90(x) * sin
