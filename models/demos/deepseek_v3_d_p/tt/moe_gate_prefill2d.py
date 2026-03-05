# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import ttnn


class MoEGatePrefill:
    """MoE gate module from DeepSeek-R1."""

    def __init__(self, config, seq_len, mesh2d):
        self.dim = config.dim
        self.topk = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.mesh2d = mesh2d
        # self.row_submesh = row_submesh
        self.seq_len_per_chip = seq_len
        self.num_chips_in_row = mesh2d.get_num_devices()
        self.num_chips_in_col = mesh2d.get_num_devices()

        self.core_grid = config.core_grid

        self.n_routed_experts = 256
        self.experts_per_chip = 8
        self.mm_compute_config = config.mm_configs["DEFAULT_COMPUTE_CONFIG"]
        self.mm_program_config = config.mm_configs["DEFAULT_PROGRAM_CONFIG"]
        self.ccl_config = config.ccl_config

        self.weight = ttnn.zeros([config.n_routed_experts, config.dim], device=self.mesh2d, layout=ttnn.TILE_LAYOUT)
        self.bias = (
            ttnn.zeros([seq_len, 256], device=self.mesh2d, layout=ttnn.TILE_LAYOUT) if config.dim == 7168 else None
        )

    def linear(self, x: ttnn.Tensor):
        return ttnn.matmul(
            x, self.weight, compute_kernel_config=self.mm_compute_config, program_config=self.mm_program_config
        )

    def all_reduce(self, x: ttnn.Tensor):
        return ttnn.all_reduce(
            x,
            cluster_axis=1,  # self.ccl_config["CLUSTER_AXIS"],
            num_links=2,  # self.ccl_config["NUM_LINKS"],
            topology=ttnn.Topology.Linear,  # self.ccl_config["COL_TOPOLOGY"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def get_onehot_expert_selection(self, global_expert_indices):
        global_onehot_experts = ttnn.zeros(
            shape=[self.seq_len_per_chip, self.n_routed_experts],
            dtype=ttnn.uint16,
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
            cluster_axis=0,
            num_links=4,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        reshaped = ttnn.reshape(gathered, (self.mesh2d.shape[1], self.n_routed_experts))
        cumsum_result = ttnn.cumsum(reshaped, dim=0)

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
