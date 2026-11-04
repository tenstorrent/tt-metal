# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from ...layers.linear import ColParallelLinear
from ...layers.module import Module
from ...layers.normalization import RMSNorm
from ...parallel.config import DiTParallelConfig
from ...parallel.manager import CCLManager
from ...utils.sdpa_recipe import prepare_recipe_inputs, recipe_config, resolve_recipe
from ...utils.substate import pop_substate, rename_substate


class MochiAttention(Module):
    sdpa_precision_default = ttnn.SDPAPrecision.STANDARD

    def __init__(
        self,
        *,
        query_dim: int,
        added_kv_proj_dim: int,
        heads: int,
        head_dim: int,
        bias: bool = False,
        added_proj_bias: bool = True,
        out_dim: int | None = None,
        out_context_dim: int | None = None,
        out_bias: bool = True,
        context_pre_only: bool = False,
        eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
        sdpa_precision: ttnn.SDPAPrecision | None = None,
        sdpa_kv_dtype: ttnn.DataType | None = None,
    ) -> None:
        super().__init__()

        self.inner_dim = out_dim if out_dim is not None else head_dim * heads
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim else query_dim
        self.context_pre_only = context_pre_only

        self.heads = out_dim // head_dim if out_dim is not None else heads
        self.query_dim = query_dim
        self.head_dim = head_dim
        self.added_kv_proj_dim = added_kv_proj_dim
        self.sdpa_precision, self.sdpa_kv_dtype = resolve_recipe(
            sdpa_precision, sdpa_kv_dtype, default=self.sdpa_precision_default, head_dim=head_dim, model="Mochi"
        )

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.n_local_heads = self.heads // self.parallel_config.tensor_parallel.factor

        fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        rms_kwargs = {
            "embedding_dim": head_dim,
            "norm_eps": eps,
            "norm_elementwise_affine": True,
            "bias": False,
            "mesh_device": mesh_device,
        }

        self.norm_q = RMSNorm(**rms_kwargs)
        self.norm_k = RMSNorm(**rms_kwargs)

        self.norm_added_q = RMSNorm(**rms_kwargs)
        self.norm_added_k = RMSNorm(**rms_kwargs)

        self.to_qkv = ColParallelLinear(
            query_dim,
            3 * self.inner_dim,
            bias=bias,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        assert self.context_pre_only is not None, "context_pre_only should be a boolean"

        self.add_qkv_proj = ColParallelLinear(
            self.added_kv_proj_dim,
            3 * self.inner_dim,
            bias=added_proj_bias,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.to_out = ColParallelLinear(
            self.inner_dim,
            self.out_dim,
            bias=out_bias,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        if not self.context_pre_only:
            self.to_add_out = ColParallelLinear(
                self.inner_dim,
                self.out_context_dim,
                bias=out_bias,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                fsdp_mesh_axis=fsdp_mesh_axis,
                ccl_manager=ccl_manager,
            )

        full_grid = self.mesh_device.compute_with_storage_grid_size()
        self.sdpa_worker_grid = (full_grid.x, full_grid.y - 1)
        self.sdpa_program_config = recipe_config(full_grid)
        self.ring_sdpa_program_config = recipe_config(self.sdpa_worker_grid)

        self.rope_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.rmsnorm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        def reshape_and_merge_qkv(q_state, k_state, v_state):
            # Rearrange QKV projections such column-fracturing shards the heads
            def _merge_tensors(q, k, v):
                n_dev = self.parallel_config.tensor_parallel.factor
                q, k, v = q.T, k.T, v.T
                q = q.reshape(q.shape[0], n_dev, self.n_local_heads, self.head_dim)
                k = k.reshape(k.shape[0], n_dev, self.n_local_heads, self.head_dim)
                v = v.reshape(v.shape[0], n_dev, self.n_local_heads, self.head_dim)
                qkv = torch.cat([q, k, v], dim=2)
                qkv = qkv.reshape(qkv.shape[0], 3 * self.heads * self.head_dim)
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

        qkv_state = reshape_and_merge_qkv(
            pop_substate(state, "to_q"), pop_substate(state, "to_k"), pop_substate(state, "to_v")
        )
        state["to_qkv.weight"] = qkv_state["weight"]
        if "bias" in qkv_state:
            state["to_qkv.bias"] = qkv_state["bias"]

        add_qkv_state = reshape_and_merge_qkv(
            pop_substate(state, "add_q_proj"),
            pop_substate(state, "add_k_proj"),
            pop_substate(state, "add_v_proj"),
        )
        state["add_qkv_proj.weight"] = add_qkv_state["weight"]
        if "bias" in add_qkv_state:
            state["add_qkv_proj.bias"] = add_qkv_state["bias"]

        rename_substate(state, "to_out.0", "to_out")

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        N: int,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """
        spatial_1BND: fractured N on SP, replicated D on TP
        prompt_1BLP: replicated on SP, replicated D on TP

        Outputs:
        spatial_1BND: fractured N on SP, replicated D on TP
        prompt_1BLP: replicated on SP, replicated D on TP
        """

        # Project spatial
        qkv_1BNF = self.to_qkv(spatial_1BND, compute_kernel_config=self.mm_compute_kernel_config)
        q_BHNE, k_BHNE, v_BHNE = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.squeeze(qkv_1BNF, 0), num_heads=self.n_local_heads, transpose_key=False
        )

        # Norm spatial
        q_BHNE = self.norm_q(q_BHNE, compute_kernel_config=self.rmsnorm_compute_kernel_config)
        k_BHNE = self.norm_k(k_BHNE, compute_kernel_config=self.rmsnorm_compute_kernel_config)

        # Project prompt
        add_qkv_1BLF = self.add_qkv_proj(prompt_1BLP, compute_kernel_config=self.mm_compute_kernel_config)
        add_q_BHLE, add_k_BHLE, add_v_BHLE = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.squeeze(add_qkv_1BLF, 0), num_heads=self.n_local_heads, transpose_key=False
        )

        # Norm prompt
        add_q_BHLE = self.norm_added_q(add_q_BHLE, compute_kernel_config=self.rmsnorm_compute_kernel_config)
        add_k_BHLE = self.norm_added_k(add_k_BHLE, compute_kernel_config=self.rmsnorm_compute_kernel_config)

        # Rope
        q_BHNE = ttnn.experimental.rotary_embedding_llama(
            q_BHNE, rope_cos, rope_sin, trans_mat, compute_kernel_config=self.rope_compute_kernel_config
        )
        k_BHNE = ttnn.experimental.rotary_embedding_llama(
            k_BHNE, rope_cos, rope_sin, trans_mat, compute_kernel_config=self.rope_compute_kernel_config
        )

        # FAST: prepare spatial and prompt Q/K/V separately after norm/RoPE, before ring
        # communication (no-op otherwise).
        q_BHNE, k_BHNE, v_BHNE = prepare_recipe_inputs(self.sdpa_precision, self.sdpa_kv_dtype, q_BHNE, k_BHNE, v_BHNE)
        add_q_BHLE, add_k_BHLE, add_v_BHLE = prepare_recipe_inputs(
            self.sdpa_precision, self.sdpa_kv_dtype, add_q_BHLE, add_k_BHLE, add_v_BHLE
        )

        if self.parallel_config.sequence_parallel.factor > 1:
            spatial_BHNE, prompt_BHLE, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                add_q_BHLE,
                add_k_BHLE,
                add_v_BHLE,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                    k_BHNE.shape,
                    2,
                    self.parallel_config.sequence_parallel.mesh_axis,
                    dtype=k_BHNE.dtype,
                ),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                    v_BHNE.shape,
                    2,
                    self.parallel_config.sequence_parallel.mesh_axis,
                    dtype=v_BHNE.dtype,
                ),
                joint_strategy="rear",
                logical_n=N,
                program_config=self.ring_sdpa_program_config,
                precision=self.sdpa_precision,
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.sequence_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
                mesh_device=self.mesh_device,
                topology=self.ccl_manager.topology,
                subdevice_id=self.ccl_manager.ccl_sub_device_id,
                ccl_core_grid_offset=(0, self.sdpa_worker_grid[1]),
            )
        else:
            spatial_BHNE, prompt_BHLE = ttnn.transformer.joint_scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                add_q_BHLE,
                add_k_BHLE,
                add_v_BHLE,
                joint_strategy="rear",
                program_config=self.sdpa_program_config,
                precision=self.sdpa_precision,
            )

        spatial_1BND = ttnn.transformer.concatenate_heads(spatial_BHNE)
        spatial_1BND = ttnn.unsqueeze(spatial_1BND, 0)

        if self.parallel_config.tensor_parallel.factor > 1:
            # Gather spatial on TP axis before projection
            spatial_1BND = ttnn.experimental.all_gather_async(
                spatial_1BND,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial_1BND.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                **self.ccl_manager.get_ag_hyperparams(spatial_1BND.shape),
            )

        spatial_1BND = self.to_out(spatial_1BND, compute_kernel_config=self.mm_compute_kernel_config)

        if self.parallel_config.tensor_parallel.factor > 1:
            # Gather spatial on TP axis after projection
            spatial_1BND = ttnn.experimental.all_gather_async(
                spatial_1BND,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial_1BND.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                # **self.ccl_manager.get_ag_hyperparams(spatial_1BND.shape),
            )

        prompt_out = None
        if not self.context_pre_only:
            prompt_1BLD = ttnn.transformer.concatenate_heads(prompt_BHLE)
            prompt_1BLD = ttnn.unsqueeze(prompt_1BLD, 0)
            if self.parallel_config.tensor_parallel.factor > 1:
                # Gather prompt on TP axis before projection
                prompt_1BLD = ttnn.experimental.all_gather_async(
                    prompt_1BLD,
                    persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                        prompt_1BLD.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                    ),
                    dim=3,
                    multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                        self.parallel_config.tensor_parallel.mesh_axis
                    ),
                    num_links=self.ccl_manager.num_links,
                    topology=self.ccl_manager.topology,
                    cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                    # **self.ccl_manager.get_ag_hyperparams(prompt_1BLD.shape),
                )
            prompt_1BLP = self.to_add_out(prompt_1BLD, compute_kernel_config=self.mm_compute_kernel_config)

            if self.parallel_config.tensor_parallel.factor > 1:
                # Gather prompt on TP axis after projection
                prompt_1BLP = ttnn.experimental.all_gather_async(
                    prompt_1BLP,
                    persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                        prompt_1BLP.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                    ),
                    dim=3,
                    multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                        self.parallel_config.tensor_parallel.mesh_axis
                    ),
                    num_links=self.ccl_manager.num_links,
                    topology=self.ccl_manager.topology,
                    cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                    # **self.ccl_manager.get_ag_hyperparams(prompt_1BLP.shape),
                )
            prompt_out = prompt_1BLP

        return spatial_1BND, prompt_out
