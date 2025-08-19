# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from ...layers.normalization import RMSNorm
from ...layers.linear import ColParallelLinear
from ...utils.substate import substate
from ...layers.feedforward import ParallelFeedForward
from .attention_mochi import MochiAttention


class MochiTransformerBlock:
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        pooled_projection_dim: int,
        activation_fn: str = "swiglu",
        context_pre_only: bool = False,
        eps: float = 1e-6,
        mesh_device=None,
        init=False,
        ccl_manager=None,
        parallel_config=None,
    ):
        self.context_pre_only = context_pre_only
        self.ff_inner_dim = (4 * dim * 2) // 3
        self.ff_context_inner_dim = (4 * pooled_projection_dim * 2) // 3
        self.dim = dim
        self.text_dim = pooled_projection_dim

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        rms_zero_kwargs = {
            "embedding_dim": dim,
            "norm_eps": eps,
            "norm_elementwise_affine": False,
            "bias": False,
            "mesh_device": mesh_device,
            "init": init,
        }

        self.norm1_linear = ColParallelLinear(
            dim,
            4 * dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            init=init,
        )
        self.norm1_norm = RMSNorm(**rms_zero_kwargs)

        if not context_pre_only:
            self.norm1_context_linear = ColParallelLinear(
                dim,
                4 * pooled_projection_dim,
                bias=True,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                init=init,
            )
        else:
            self.norm1_context_linear = ColParallelLinear(
                dim,
                pooled_projection_dim,
                bias=True,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                init=init,
            )
        self.norm1_context_norm = RMSNorm(**rms_zero_kwargs)

        self.attn1 = MochiAttention(
            query_dim=dim,
            added_kv_proj_dim=pooled_projection_dim,
            heads=num_attention_heads,
            head_dim=attention_head_dim,
            bias=False,
            added_proj_bias=False,
            out_dim=dim,
            out_context_dim=pooled_projection_dim,
            context_pre_only=context_pre_only,
            eps=1e-5,
            mesh_device=mesh_device,
            init=init,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        self.norm2_norm = RMSNorm(**rms_zero_kwargs)
        self.norm2_context_norm = RMSNorm(**rms_zero_kwargs) if not self.context_pre_only else None

        self.norm3_norm = RMSNorm(**rms_zero_kwargs)
        self.norm3_context_norm = RMSNorm(**rms_zero_kwargs) if not self.context_pre_only else None

        self.ff = ParallelFeedForward(
            dim,
            inner_dim=self.ff_inner_dim,
            activation_fn=activation_fn,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            init=init,
        )

        self.ff_context = None
        if not context_pre_only:
            self.ff_context = ParallelFeedForward(
                pooled_projection_dim,
                inner_dim=self.ff_context_inner_dim,
                activation_fn=activation_fn,
                bias=False,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
                init=init,
            )

        self.norm4_norm = RMSNorm(**rms_zero_kwargs)
        self.norm4_context_norm = RMSNorm(**rms_zero_kwargs) if not self.context_pre_only else None

        self.temb_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.ff_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Found that HiFi4 RMSNorm hurts accuracy, so let RMSNorm use default
        # self.rms_compute_kernel_config = ttnn.init_device_compute_kernel_config(
        #     mesh_device.arch(),
        #     math_fidelity=ttnn.MathFidelity.HiFi4,
        #     math_approx_mode=False,
        #     fp32_dest_acc_en=True,
        #     packer_l1_acc=True,
        # )
        self.rms_compute_kernel_config = None

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def load_state_dict(self, state_dict):
        self.norm1_linear.load_state_dict(substate(state_dict, "norm1.linear"))
        context_linear_key = "norm1_context.linear" if not self.context_pre_only else "norm1_context.linear_1"
        self.norm1_context_linear.load_state_dict(substate(state_dict, context_linear_key))
        self.attn1.load_state_dict(substate(state_dict, "attn1"))

        def rename_ff_state(state):
            out_state = {
                f"{replacement}{k[len(prefix):]}": v
                for k, v in state.items()
                for prefix, replacement in [("net.0.proj", "ff1"), ("net.2", "ff2")]
                if prefix in k
            }
            return out_state

        self.ff.load_state_dict(rename_ff_state(substate(state_dict, "ff")))
        if not self.context_pre_only:
            self.ff_context.load_state_dict(rename_ff_state(substate(state_dict, "ff_context")))

    def __call__(self, spatial_1BND, prompt_1BLP, temb_11BD, N, rope_cos, rope_sin, trans_mat):
        """
        spatial_1BND: fractured N on SP, replicated D on TP
        prompt_1BLP: replicated on SP, replicated D on TP
        temb_11BD: replicated on SP, replicated D on TP
        N: logical sequence length of the spatial input
        rope_cos_BANH: fractured N on SP, A (num_heads) on TP
        rope_sin_BANH: fractured N on SP, A (num_heads) on TP
        trans_mat: replicated on SP, replicated D on TP

        Outputs:
        spatial_1BND: fractured N on SP, fractured D on TP
        prompt_1BLP: replicated on SP, fractured D on TP
        """

        # NOTE: SILU is not very accurate, should try to move to host?
        silu_temb_11BD = ttnn.silu(temb_11BD)

        # Returns 4*dim fractured on TP
        mod_spatial_11BZ = self.norm1_linear(
            silu_temb_11BD, core_grid=self.core_grid, compute_kernel_config=self.temb_compute_kernel_config
        )
        if self.parallel_config.tensor_parallel.factor > 1:
            mod_spatial_11BZ = ttnn.experimental.all_gather_async(
                mod_spatial_11BZ,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    mod_spatial_11BZ.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        scale_msa_11BD = mod_spatial_11BZ[:, :, :, : self.dim]
        gate_msa_11BD = mod_spatial_11BZ[:, :, :, self.dim : 2 * self.dim]
        scale_mlp_11BD = mod_spatial_11BZ[:, :, :, 2 * self.dim : 3 * self.dim]
        gate_mlp_11BD = mod_spatial_11BZ[:, :, :, 3 * self.dim :]

        # Norm spatial input (MochiRMSNormZero)
        spatial_normed_1BND = self.norm1_norm(spatial_1BND, compute_kernel_config=self.rms_compute_kernel_config) * (
            1.0 + scale_msa_11BD
        )

        # Returns 4*dim if not context_pre_only, dim if context_pre_only, fractured TP
        mod_prompt_11BZ = self.norm1_context_linear(
            silu_temb_11BD, core_grid=self.core_grid, compute_kernel_config=self.temb_compute_kernel_config
        )
        if self.parallel_config.tensor_parallel.factor > 1:
            mod_prompt_11BZ = ttnn.experimental.all_gather_async(
                mod_prompt_11BZ,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    mod_prompt_11BZ.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        if not self.context_pre_only:
            prompt_scale_msa_11BD = mod_prompt_11BZ[:, :, :, : self.text_dim]
            prompt_gate_msa_11BD = mod_prompt_11BZ[:, :, :, self.text_dim : 2 * self.text_dim]
            prompt_scale_mlp_11BD = mod_prompt_11BZ[:, :, :, 2 * self.text_dim : 3 * self.text_dim]
            prompt_gate_mlp_11BD = mod_prompt_11BZ[:, :, :, 3 * self.text_dim :]
        else:
            prompt_scale_msa_11BD = mod_prompt_11BZ

        # Norm prompt input (MochiRMSNormZero)
        prompt_normed_1BLP = self.norm1_context_norm(
            prompt_1BLP, compute_kernel_config=self.rms_compute_kernel_config
        ) * (1.0 + prompt_scale_msa_11BD)

        spatial_attn_1BND, prompt_attn_1BLP = self.attn1(
            spatial_1BND=spatial_normed_1BND,
            prompt_1BLP=prompt_normed_1BLP,
            N=N,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
        )

        # ModulatedRMSNorm
        spatial_attn_mod_1BND = self.norm2_norm(
            spatial_attn_1BND, compute_kernel_config=self.rms_compute_kernel_config
        ) * ttnn.tanh(gate_msa_11BD, accuracy=True)

        # Residual
        spatial_1BND = spatial_1BND + spatial_attn_mod_1BND

        # Norm spatial input (MochiRMSNormZero) for FF
        spatial_normed_1BND = self.norm3_norm(spatial_1BND, compute_kernel_config=self.rms_compute_kernel_config) * (
            1.0 + scale_mlp_11BD
        )

        # TODO: Pass core_grid, compute_kernel_config for correctness check
        spatial_ff_1BND = self.ff(
            spatial_normed_1BND, core_grid=self.core_grid, compute_kernel_config=self.ff_compute_kernel_config
        )

        # Gather spatial FF output
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_ff_1BND = ttnn.experimental.all_gather_async(
                spatial_ff_1BND,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial_ff_1BND.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        spatial_ff_mod_1BND = self.norm4_norm(
            spatial_ff_1BND, compute_kernel_config=self.rms_compute_kernel_config
        ) * ttnn.tanh(gate_mlp_11BD, accuracy=True)

        # Residual
        spatial_1BND = spatial_1BND + spatial_ff_mod_1BND

        if not self.context_pre_only:
            # Norm attention output (MochiRMSNormZero)
            prompt_attn_mod_1BLP = self.norm2_context_norm(
                prompt_attn_1BLP, compute_kernel_config=self.rms_compute_kernel_config
            ) * ttnn.tanh(prompt_gate_msa_11BD, accuracy=True)

            # Residual
            prompt_1BLP = prompt_1BLP + prompt_attn_mod_1BLP

            # Norm prompt input (MochiRMSNormZero) for FF
            prompt_normed_1BLP = self.norm3_context_norm(
                prompt_1BLP, compute_kernel_config=self.rms_compute_kernel_config
            ) * (1.0 + prompt_scale_mlp_11BD)

            # TODO: Pass core_grid, compute_kernel_config for correctness check
            prompt_ff_1BLP = self.ff_context(
                prompt_normed_1BLP, core_grid=self.core_grid, compute_kernel_config=self.ff_compute_kernel_config
            )

            # Gather prompt FF output
            if self.parallel_config.tensor_parallel.factor > 1:
                prompt_ff_1BLP = ttnn.experimental.all_gather_async(
                    prompt_ff_1BLP,
                    persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                        prompt_ff_1BLP.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                    ),
                    dim=3,
                    multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                    num_links=self.ccl_manager.num_links,
                    topology=self.ccl_manager.topology,
                    cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                )

            prompt_ff_mod_1BLP = self.norm4_context_norm(
                prompt_ff_1BLP, compute_kernel_config=self.rms_compute_kernel_config
            ) * ttnn.tanh(prompt_gate_mlp_11BD, accuracy=True)

            # Residual
            prompt_1BLP = prompt_1BLP + prompt_ff_mod_1BLP

        return spatial_1BND, prompt_1BLP
