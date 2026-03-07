# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import torch

import ttnn


class MoEGatePrefill:
    """MoE gate module from DeepSeek-R1."""

    def __init__(self, config, mesh_device):
        self.dim = config.dim
        self.topk = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.mesh_device = mesh_device
        self.seq_len_per_chip = config.sp_dim
        self.num_chips_in_row = mesh_device.get_num_devices()
        self.num_chips_in_col = mesh_device.get_num_devices()

        self.core_grid = config.core_grid

        self.n_routed_experts = 256
        self.experts_per_chip = 8
        self.mm_compute_config = config.mm_configs["DEFAULT_COMPUTE_CONFIG"]
        self.mm_program_config = config.mm_configs["DEFAULT_PROGRAM_CONFIG"]
        self.ccl_config = config.ccl_config

        self.weight = ttnn.from_torch(
            torch.zeros([config.dim, config.n_routed_experts]),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0),
                mesh_shape=mesh_device.shape,
            ),
        )

        self.bias = ttnn.from_torch(
            # ttnn.experimental.deepseek_grouped_gate() requires bias to be broadcasted already
            torch.zeros([config.n_routed_experts]).repeat(config.sp_dim).view(config.sp_dim, -1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        self.experts_in_dispatch_row = ttnn.ones(
            [config.n_routed_experts],
            device=mesh_device,
        )

    def linear(self, x: ttnn.Tensor):
        return ttnn.matmul(
            x, self.weight, compute_kernel_config=self.mm_compute_config, program_config=self.mm_program_config
        )

    def all_reduce(self, x: ttnn.Tensor):
        return ttnn.all_reduce(
            x,
            cluster_axis=self.ccl_config["TP_AXIS"],
            num_links=self.ccl_config["NUM_LINKS"],
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def get_onehot_expert_selection(self, global_expert_indices):
        global_onehot_experts = ttnn.zeros(
            shape=[self.seq_len_per_chip, self.n_routed_experts],
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=global_expert_indices.device(),
        )
        global_updates = ttnn.ones_like(global_onehot_experts)
        global_onehot_experts = ttnn.scatter(
            global_onehot_experts, 1, global_expert_indices, global_updates, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        return global_onehot_experts

    def cumulative_sum_across_columns(self, input_tensor: ttnn.Tensor):
        input_tensor = ttnn.unsqueeze(input_tensor, dim=0)

        gathered = ttnn.all_gather(
            input_tensor,
            dim=0,
            cluster_axis=self.ccl_config["DISPATCH_AXIS"],
            num_links=self.ccl_config["NUM_LINKS"],
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        cumsum_result = ttnn.cumsum(gathered, dim=0)
        ttnn.deallocate(gathered)
        cumsum_result = ttnn.to_layout(cumsum_result, ttnn.ROW_MAJOR_LAYOUT)
        cumsum_result = ttnn.pad(cumsum_result, padding=[(1, 0), (0, 0)], value=0)  # add zeros at the beginning
        cumsum_result = cumsum_result * self.experts_in_dispatch_row
        return cumsum_result

    def forward(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        logits = self.all_reduce(self.linear(x))
        ttnn_scores, ttnn_top_k_experts_indices = ttnn.experimental.deepseek_grouped_gate(
            logits,
            self.bias,
            n_groups=8,
            summed_experts_per_group=2,
            topk_groups=4,
            n_activated_experts=8,
            route_scale=1.0,
            epsilon=1e-20,
        )
        global_onehot_experts = self.get_onehot_expert_selection(ttnn_top_k_experts_indices)
        expert_histograms = ttnn.sum(global_onehot_experts, dim=0)
        dispatch_offsets = self.cumulative_sum_across_columns(expert_histograms)

        return (ttnn_scores, ttnn_top_k_experts_indices, logits, dispatch_offsets)

    def __call__(self, x):
        return self.forward(x)
