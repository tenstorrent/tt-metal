# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ...layers.normalization import RMSNorm
from ...layers.linear import ColParallelLinear
from ...utils.substate import substate
from ...utils.padding import pad_weight_tensor


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
class SD35JointAttention:
    def __init__(
        self,
        query_dim,
        head_dim,
        heads,
        out_dim=None,
        bias=False,
        out_bias=True,
        context_pre_only=None,
        eps=1e-5,
        mesh_device=None,
        init=False,
        ccl_manager=None,
        parallel_config=None,
        padding_config=None,
    ):
        self.query_dim = query_dim
        self.head_dim = head_dim
        self.heads = heads
        self.padding_config = padding_config
        self.padded_heads = padding_config.target_heads if padding_config is not None else heads

        self.out_dim = out_dim if out_dim is not None else query_dim
        # Note: added_kv_proj_dim should be passed as parameter, using query_dim as default
        self.added_kv_proj_dim = query_dim
        self.context_pre_only = context_pre_only
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.n_local_heads = self.padded_heads // self.parallel_config.tensor_parallel.factor

        self.inner_dim = out_dim if out_dim is not None else head_dim * self.heads
        self.padded_inner_dim = head_dim * self.padded_heads
        rms_kwargs = {
            "embedding_dim": head_dim,
            "norm_eps": eps,
            "norm_elementwise_affine": True,
            "bias": False,
            "mesh_device": mesh_device,
            "init": init,
        }

        self.norm_q = RMSNorm(**rms_kwargs)
        self.norm_k = RMSNorm(**rms_kwargs)

        # Fused QKV projection
        self.to_qkv = ColParallelLinear(
            query_dim,
            3 * self.padded_inner_dim,
            bias=bias,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            init=init,
        )

        # Implementing joint attention
        self.add_qkv_proj = ColParallelLinear(
            self.added_kv_proj_dim,
            3 * self.padded_inner_dim,
            bias=bias,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            init=init,
        )

        self.to_out = ColParallelLinear(
            self.padded_inner_dim,
            self.out_dim,
            bias=out_bias,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            init=init,
        )

        if self.context_pre_only is not None and not self.context_pre_only:
            # TODO: Use `out_context_dim` parameter if given
            self.to_add_out = ColParallelLinear(
                self.padded_inner_dim,
                self.out_dim,
                bias=out_bias,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                init=init,
            )

        self.norm_added_q = RMSNorm(**rms_kwargs)
        self.norm_added_k = RMSNorm(**rms_kwargs)

        full_grid = self.mesh_device.compute_with_storage_grid_size()
        self.sdpa_worker_grid = (full_grid.x, full_grid.y - 1)
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_worker_grid,
            q_chunk_size=128,
            k_chunk_size=512,
            exp_approx_mode=False,  # NOTE: False is more correct
        )
        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,  # NOTE: Set to True if there's a correctness issue
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def load_state_dict(self, state_dict):
        def reshape_and_merge_qkv(q_state, k_state, v_state):
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
        qkv_state = reshape_and_merge_qkv(
            substate(state_dict, "to_q"), substate(state_dict, "to_k"), substate(state_dict, "to_v")
        )
        self.to_qkv.load_state_dict(qkv_state)
        add_qkv_state = reshape_and_merge_qkv(
            substate(state_dict, "add_q_proj"), substate(state_dict, "add_k_proj"), substate(state_dict, "add_v_proj")
        )
        self.add_qkv_proj.load_state_dict(add_qkv_state)
        self.to_out.load_state_dict(pad_dense_out(substate(state_dict, "to_out.0")))
        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out.load_state_dict(pad_dense_out(substate(state_dict, "to_add_out")))
        self.norm_added_q.load_state_dict(substate(state_dict, "norm_added_q"))
        self.norm_added_k.load_state_dict(substate(state_dict, "norm_added_k"))

    def __call__(self, spatial_1BND, prompt_1BLD, N):
        """
        Inputs are replicated
        Outputs are width-fractured
        """

        qkv_1BNF = self.to_qkv(spatial_1BND, core_grid=self.core_grid)
        local_heads = self.n_local_heads
        q_BHNE, k_BHNE, v_BHNE = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.squeeze(qkv_1BNF, 0), num_heads=local_heads, transpose_key=False
        )

        q_BHNE = self.norm_q(q_BHNE)
        k_BHNE = self.norm_k(k_BHNE)

        add_qkv_1BLF = self.add_qkv_proj(prompt_1BLD, core_grid=self.core_grid)
        add_q_BHLE, add_k_BHLE, add_v_BHLE = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.squeeze(add_qkv_1BLF, 0), num_heads=local_heads, transpose_key=False
        )
        add_q_BHLE = self.norm_added_q(add_q_BHLE)
        add_k_BHLE = self.norm_added_k(add_k_BHLE)

        if self.parallel_config.sequence_parallel.factor > 1:
            spatial_BHNE, prompt_BHLE, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                add_q_BHLE,
                add_k_BHLE,
                add_v_BHLE,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                    k_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
                ),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                    v_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
                ),
                joint_strategy="rear",
                logical_n=N,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
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
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )

        spatial_1BND = ttnn.transformer.concatenate_heads(spatial_BHNE)
        spatial_1BND = ttnn.unsqueeze(spatial_1BND, 0)

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_1BND = ttnn.experimental.all_gather_async(
                spatial_1BND,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial_1BND.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                **self.ccl_manager.get_ag_hyperparams(spatial_1BND.shape),
            )

        spatial_1BND = self.to_out(spatial_1BND, core_grid=self.core_grid)

        prompt_out = None
        if self.context_pre_only is not None and not self.context_pre_only:
            prompt_1BLD = ttnn.transformer.concatenate_heads(prompt_BHLE)
            prompt_1BLD = ttnn.unsqueeze(prompt_1BLD, 0)
            if self.parallel_config.tensor_parallel.factor > 1:
                prompt_1BLD = ttnn.experimental.all_gather_async(
                    prompt_1BLD,
                    persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                        prompt_1BLD.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                    ),
                    dim=3,
                    multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                    num_links=self.ccl_manager.num_links,
                    topology=self.ccl_manager.topology,
                    cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                    **self.ccl_manager.get_ag_hyperparams(prompt_1BLD.shape),
                )
            prompt_1BLD = self.to_add_out(prompt_1BLD, core_grid=self.core_grid)
            prompt_out = prompt_1BLD

        return spatial_1BND, prompt_out
