# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from ...layers.normalization import DistributedLayerNorm
from ...layers.linear import ColParallelLinear
from ...layers.feedforward import ParallelFeedForward
from ...utils.substate import substate
from .attention_sd35 import SD35JointAttention


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
class SD35TransformerBlock:
    def __init__(
        self,
        dim,
        num_heads,
        head_dim,
        context_pre_only,
        use_dual_attention=False,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
        init=False,
    ):
        assert not use_dual_attention, "Expecting dual attention"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.context_pre_only = context_pre_only
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        # TODO: Shuffle norm linear weights to match tensor parallelism
        self.norm1_linear = ColParallelLinear(
            dim,
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

        # TODO: Shuffle norm linear weights to match tensor parallelism
        context_norm_dim = 6 * dim if not context_pre_only else 2 * dim
        self.norm1_context_linear = ColParallelLinear(
            dim,
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

        self.attn = SD35JointAttention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            bias=True,
            context_pre_only=context_pre_only,
            eps=1e-6,
            mesh_device=mesh_device,
            init=init,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
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
            activation_fn="gelu",
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
                activation_fn="gelu",
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
                init=init,
            )

        # self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        #     math_fidelity=ttnn.MathFidelity.HiFi2,
        #     math_approx_mode=False,
        #     fp32_dest_acc_en=False,  # NOTE: Set to True if there's a correctness issue
        # )

    def load_state_dict(self, state_dict):
        def _shuffle_ada_norm_linear(linear_state):
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

            in_dim = linear_state["weight"].shape[1]
            weight = _shuffle(linear_state["weight"], in_dim)
            out_state = {"weight": weight}
            if "bias" in linear_state:
                bias = _shuffle(linear_state["bias"].reshape(-1, 1), in_dim)
                bias = bias.squeeze()
                out_state["bias"] = bias
            return out_state

        def rename_ff_state(state):
            out_state = {
                f"{replacement}{k[len(prefix):]}": v
                for k, v in state.items()
                for prefix, replacement in [("net.0.proj", "ff1"), ("net.2", "ff2")]
                if prefix in k
            }
            return out_state

        self.norm1_linear.load_state_dict(_shuffle_ada_norm_linear(substate(state_dict, "norm1.linear")))
        self.norm1_norm.load_state_dict(substate(state_dict, "norm1.norm"))
        self.norm1_context_linear.load_state_dict(
            _shuffle_ada_norm_linear(substate(state_dict, "norm1_context.linear"))
        )
        self.norm1_context_norm.load_state_dict(substate(state_dict, "norm1_context.norm"))
        self.attn.load_state_dict(substate(state_dict, "attn"))
        self.norm2.load_state_dict(substate(state_dict, "norm2"))
        self.ff.load_state_dict(rename_ff_state(substate(state_dict, "ff")))
        if not self.context_pre_only:
            self.norm2_context.load_state_dict(substate(state_dict, "norm2_context"))
            self.ff_context.load_state_dict(rename_ff_state(substate(state_dict, "ff_context")))

    def __call__(self, spatial_1BND, prompt_1BLD, time_embed_11BE, N, L):
        """
        spatial_1BND: fractured N on SP, fractured D on TP
        prompt_1BLD: replicated on SP, fractured D on TP
        time_embed_11BE: replicated
        """

        time_embed_11BE = ttnn.silu(time_embed_11BE, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        spatial_time_11BF = self.norm1_linear(time_embed_11BE)
        prompt_time_11BE = self.norm1_context_linear(time_embed_11BE)

        (
            spatial_shift_attn,
            spatial_scale_attn,
            spatial_gate_attn,
            spatial_shift_ff,
            spatial_scale_ff,
            spatial_gate_ff,
        ) = chunk_time(spatial_time_11BF, 6)

        spatial_normed_1BND = self.norm1_norm(spatial_1BND)
        spatial_normed_1BND = spatial_normed_1BND * (1 + spatial_scale_attn) + spatial_shift_attn

        if self.context_pre_only:
            prompt_scale_attn, prompt_shift_attn = chunk_time(prompt_time_11BE, 2)
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
            ) = chunk_time(prompt_time_11BE, 6)

        prompt_normed_1BLD = self.norm1_context_norm(prompt_1BLD)
        prompt_normed_1BLD = prompt_normed_1BLD * (1 + prompt_scale_attn) + prompt_shift_attn

        if self.parallel_config.tensor_parallel.factor > 1:
            # Gather spatial, prompt before attention
            spatial_normed_1BND = ttnn.experimental.all_gather_async(
                spatial_normed_1BND,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial_normed_1BND.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                # chunks_per_sync=16,
                # num_workers_per_link=3,
                # num_buffers_per_channel=2,
            )
            prompt_normed_1BLD = ttnn.experimental.all_gather_async(
                prompt_normed_1BLD,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    prompt_normed_1BLD.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                # chunks_per_sync=10,
                # num_workers_per_link=2,
                # num_buffers_per_channel=2,
            )

        spatial_attn_1BLD, prompt_attn_1BLD = self.attn(spatial_normed_1BND, prompt_normed_1BLD, N)
        spatial_attn_1BLD = spatial_attn_1BLD * spatial_gate_attn
        prompt_attn_1BLD = prompt_attn_1BLD * prompt_gate_attn if prompt_gate_attn is not None else None

        # residual
        spatial_1BND = spatial_1BND + spatial_attn_1BLD

        spatial_normed_1BND = self.norm2(spatial_1BND)
        spatial_normed_1BND = spatial_normed_1BND * (1 + spatial_scale_ff) + spatial_shift_ff

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_normed_1BND = ttnn.experimental.all_gather_async(
                spatial_normed_1BND,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial_normed_1BND.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                # chunks_per_sync=16,
                # num_workers_per_link=3,
                # num_buffers_per_channel=2,
            )

        spatial_ff_1BND = self.ff(spatial_normed_1BND)
        spatial_ff_1BND = spatial_ff_1BND * spatial_gate_ff

        spatial_1BND += spatial_ff_1BND

        if self.context_pre_only:
            return spatial_1BND, None

        prompt_1BLD += prompt_attn_1BLD

        prompt_normed_1BLD = self.norm2_context(prompt_1BLD)
        prompt_normed_1BLD = prompt_normed_1BLD * (1 + prompt_scale_ff) + prompt_shift_ff

        if self.parallel_config.tensor_parallel.factor > 1:
            prompt_normed_1BLD = ttnn.experimental.all_gather_async(
                prompt_normed_1BLD,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    prompt_normed_1BLD.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                # chunks_per_sync=10,
                # num_workers_per_link=2,
                # num_buffers_per_channel=2,
            )

        prompt_ff_1BLD = self.ff_context(prompt_normed_1BLD)
        prompt_ff_1BLD = prompt_ff_1BLD * prompt_gate_ff

        prompt_1BLD += prompt_ff_1BLD

        return spatial_1BND, prompt_1BLD


def chunk_time(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = t.shape[-1] // count
    return [t[:, :, :, i * size : (i + 1) * size] for i in range(count)]
