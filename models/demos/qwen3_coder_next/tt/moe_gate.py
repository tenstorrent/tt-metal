# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MoE routing gate for Qwen3-Coder-Next (LightweightModule / ttnn).

Top-10 expert selection from 512 total experts with normalized probabilities.
Router weight is replicated across all devices.

State dict key: model.layers.{layer}.mlp.gate.weight  (num_experts=512, hidden_size=2048)
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class MoEGate(LightweightModule):
    """Top-K expert routing gate using ttnn for the projection and torch for top-k.

    The router projection (hidden_size -> num_experts) runs on device via ttnn.linear.
    Top-k selection and softmax normalization run on host (small tensor: batch*seq x 512).
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, dtype):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.num_experts = args.num_experts  # 512
        self.num_experts_per_tok = args.num_experts_per_tok  # 10
        self.norm_topk_prob = args.norm_topk_prob

        # Load router weight from state_dict: shape (num_experts, hidden_size)
        # For ttnn.linear we need the weight transposed: (hidden_size, num_experts)
        # Support both with and without "model." prefix
        gate_key = f"model.layers.{layer_num}.mlp.gate.weight"
        if gate_key not in state_dict:
            gate_key = f"layers.{layer_num}.mlp.gate.weight"
        gate_weight = state_dict[gate_key]  # (num_experts, hidden_size)
        gate_weight_t = gate_weight.transpose(-2, -1).contiguous()  # (hidden_size, num_experts)

        # Store as 4D ttnn tensor [1, 1, hidden_size, num_experts] on device
        self.gate_weight = ttnn.as_tensor(
            gate_weight_t.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def forward(self, x):
        """Compute top-k expert routing.

        Args:
            x: ttnn tensor of shape (1, 1, batch*seq, hidden_size) on device.

        Returns:
            Tuple of torch tensors (on host):
                - topk_weights: (batch*seq, num_experts_per_tok) normalized probabilities
                - topk_indices: (batch*seq, num_experts_per_tok) expert indices
        """
        # Router projection on device: (1, 1, B*S, hidden_size) @ (1, 1, hidden_size, num_experts)
        # -> (1, 1, B*S, num_experts)
        router_logits = ttnn.linear(x, self.gate_weight)

        # Transfer to host for top-k (small tensor: B*S x 512 floats)
        if hasattr(self.mesh_device, "get_num_devices") and self.mesh_device.get_num_devices() > 1:
            router_logits_torch = ttnn.to_torch(
                router_logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            )[
                0:1
            ]  # Take first device (replicated)
        else:
            router_logits_torch = ttnn.to_torch(router_logits)  # (1, 1, B*S, num_experts)
        router_logits_torch = router_logits_torch.squeeze(0).squeeze(0)  # (B*S, num_experts)
        ttnn.deallocate(router_logits)

        # Softmax over experts
        routing_weights = torch.softmax(router_logits_torch.float(), dim=-1)

        # Top-k selection
        topk_weights, topk_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)

        # Normalize top-k probabilities
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_indices
