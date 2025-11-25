# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple
import ttnn

from .module import Module


def top_k_gate(
    logits: ttnn.Tensor, k: int, expert_dim: int, clamp_min: float = 1e-8
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """
    Top-K gate implementation for something like `easy_topk` in Hunyuan-Image 3.0:
    https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/hunyuan_image_3/hunyuan.py#L234

    Args:
        logits: Logits tensor of shape (batch, seq_len, num_experts)
        k: Number of experts to select per token
        expert_dim: Dimension of the expert dimension, usually the last dimension
        clamp_min: Minimum value to clamp the expert weights sum to avoid division by zero

    Returns:
        Tuple of:
            - router_scores: Sparse scores tensor with non-zero values only for selected experts
            - expert_weights: Normalized weights for selected experts (used for weighted combination)
            - expert_indices: Indices of selected experts for each token
    """

    compute_config = ttnn.init_device_compute_kernel_config(
        logits.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    scores = ttnn.softmax(logits, dim=expert_dim, numeric_stable=True, compute_kernel_config=compute_config)

    expert_weights_1, expert_indices = ttnn.topk(scores, k=k, dim=expert_dim, sorted=True)
    router_scores = ttnn.scatter(ttnn.zeros_like(logits), dim=expert_dim, index=expert_indices, src=expert_weights_1)

    expert_weights_sum = ttnn.clamp(ttnn.sum(expert_weights_1, dim=expert_dim, keepdim=True), min=clamp_min)
    expert_weights = ttnn.div(expert_weights_1, expert_weights_sum)
    return router_scores, expert_weights, expert_indices


def group_top_k_gate(
    logits: ttnn.Tensor, k: int, group_size: int, expert_dim: int, clamp_min: float = 1e-8
) -> ttnn.Tensor:
    return logits


class MoELayer(Module):

    """
    MoE layer
    """

    def __init__(self, mesh_device, state_dict, args, layer_num, dtype):
        super().__init__()
        self.mesh_device = mesh_device
        self.state_dict = state_dict
        self.args = args
        self.layer_num = layer_num
        self.dtype = dtype

    def forward(self, x: ttnn.Tensor, compute_kernel_config=None) -> ttnn.Tensor:
        return x
