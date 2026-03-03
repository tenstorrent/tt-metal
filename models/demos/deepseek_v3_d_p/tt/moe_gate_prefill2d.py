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
            num_links=4,  # self.ccl_config["NUM_LINKS"],
            topology=self.ccl_config["COL_TOPOLOGY"],
        )

    def all_gather_for_dispatch(self, x: ttnn.Tensor):
        return ttnn.all_gather(
            x,
            cluster_axis=0,  # self.ccl_config["CLUSTER_AXIS"],
            num_links=4,  # self.ccl_config["NUM_LINKS"],
            topology=self.ccl_config["ROW_TOPOLOGY"],
        )

    def get_local_expert_histogram(self, local_onehot_experts):
        for token in range(self.seq_len_per_chip):
            for topk_indice in range(self.topk):
                routed_expert = local_expert_indices[token, topk_indice]
                local_histogram[0, routed_expert] += 1

        return local_histogram

    def get_onehot_expert_selection(self, global_expert_indices):
        global_onehot_experts = ttnn.zeros(
            shape=[self.seq_len_per_chip, self.n_routed_experts],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=global_expert_indices.device(),
        )

        global_updates = ttnn.full_like(global_onehot_experts, fill_value=1.0)
        global_onehot_experts = ttnn.scatter_add(global_onehot_experts, 1, global_expert_indices, global_updates)
        return global_onehot_experts

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

        self.get_local_expert_histogram(per_device_onehot_experts[0])

        return (
            ttnn_scores,
            ttnn_top_k_experts_indices,
            logits,
        )

    def __call__(self, x):
        return self.forward(x)
