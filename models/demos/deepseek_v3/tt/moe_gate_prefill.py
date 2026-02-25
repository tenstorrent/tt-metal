# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import ttnn


class MoEGatePrefill:
    """MoE gate module from DeepSeek-R1."""

    def __init__(self, config, seq_len, device):
        self.dim = config.dim
        self.topk = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.device = device
        self.mm_compute_config = config.mm_configs["DEFAULT_COMPUTE_CONFIG"]
        self.mm_program_config = config.mm_configs["DEFAULT_PROGRAM_CONFIG"]
        self.ccl_config = config.ccl_config
        self.weight = ttnn.zeros([config.n_routed_experts, config.dim], device=self.device, layout=ttnn.TILE_LAYOUT)
        self.bias = (
            ttnn.zeros([seq_len, 256], device=self.device, layout=ttnn.TILE_LAYOUT) if config.dim == 7168 else None
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
            topology=self.ccl_config["TOPOLOGY"],
        )

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

        return ttnn_scores, ttnn_top_k_experts_indices, logits

    def __call__(self, x):
        return self.forward(x)
