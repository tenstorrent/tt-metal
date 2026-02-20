# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import ttnn

# from models.demos.deepseek_v3.utils.config_dataclass import (
#     BinaryOpConfig,
#     FromWeightConfig,
#     LinearConfig,
#     LinearFallbackConfig,
#     MeshDeviceStub,
#     MulConfig,
#     ReshapeConfig,
#     ScatterConfig,
#     TopKConfig,
#     TopKFallbackConfig,
# )

# from models.demos.deepseek_v3.utils.run_config import (
#     ModelDecodeConfig,
#     ModelPrefillConfig,
#     RunDecodeConfig,
#     RunPrefillConfig,
#     WeightConfig,
# )

# class Gate(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.dim = args.dim
#         self.topk = args.n_activated_experts
#         self.n_groups = args.n_expert_groups
#         self.topk_groups = args.n_limited_groups
#         self.score_func = args.score_func
#         self.route_scale = args.route_scale
#         self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
#         self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         scores = linear(x, self.weight)
#         if self.score_func == "softmax":
#             scores = scores.softmax(dim=-1, dtype=torch.float32)
#         else:
#             scores = scores.sigmoid()
#         original_scores = scores
#         if self.bias is not None:
#             scores = scores + self.bias
#         if self.n_groups > 1:
#             scores = scores.view(x.size(0), self.n_groups, -1)
#             if self.bias is None:
#                 group_scores = scores.amax(dim=-1)
#             else:
#                 group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
#             indices = group_scores.topk(self.topk_groups, dim=-1)[1]
#             mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
#             scores = (scores * mask.unsqueeze(-1)).flatten(1)
#         indices = torch.topk(scores, self.topk, dim=-1)[1]
#         weights = original_scores.gather(1, indices)
#         if self.score_func == "sigmoid":
#             weights /= weights.sum(dim=-1, keepdim=True)
#         weights *= self.route_scale
#         return weights.type_as(x), indices


class MoEGatePrefill:
    """MoE gate module from DeepSeek-R1."""

    def __init__(self, config, device):
        self.dim = config.dim
        self.topk = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.seq_len = config.max_seq_len
        self.device = device
        self.topology = config.topology

        self.weight = ttnn.zeros([config.n_routed_experts, config.dim], device=self.device, layout=ttnn.TILE_LAYOUT)
        self.bias = (
            ttnn.zeros([self.seq_len, 256], device=self.device, layout=ttnn.TILE_LAYOUT) if config.dim == 7168 else None
        )

    def linear(self, x):
        # breakpoint()
        return ttnn.all_reduce(ttnn.matmul(x, self.weight), cluster_axis=1, num_links=4, topology=self.topology)

    def forward(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        logits = self.linear(x)

        # Sigmoid activation
        # scores = ttnn.sigmoid(logits)
        # ttnn.deallocate(logits)

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
