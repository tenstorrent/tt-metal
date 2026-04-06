# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Router: RMSNorm -> scale -> linear -> per-expert-scale -> softmax -> topk -> normalize -> per_expert_scale

Gemma4 uses softmax-THEN-topk (opposite of GPT-OSS).

HF weight names (from state_dict with "router." prefix stripped):
  proj.weight: [num_experts, hidden_size] = [128, 2816]
  scale: [hidden_size] = [2816]
  per_expert_scale: [num_experts] = [128]

The router norm has no learned weight (with_scale=False).
"""

import torch

import ttnn
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.demos.gemma4.utils.substate import substate


class Gemma4Router:
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None):
        self.mesh_device = mesh_device
        self.num_experts = hf_config.num_experts
        self.top_k = hf_config.top_k_experts
        self.hidden_size = hf_config.hidden_size
        self.scalar_root_size = self.hidden_size**-0.5

        # RMSNorm with no learned scale (with_scale=False)
        self.norm = RMSNorm(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict={},
            tensor_cache_path=f"{tensor_cache_path}/norm" if tensor_cache_path else None,
            with_scale=False,
        )

        if state_dict:
            # scale: [hidden_size] -> [1, 1, 1, hidden_size] for broadcast multiply
            scale_weight = state_dict["scale"].reshape(1, 1, 1, -1)
            # proj.weight: [num_experts, hidden_size] -> transpose to [1, 1, hidden_size, num_experts]
            proj_weight = substate(state_dict, "proj")["weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            # per_expert_scale: [num_experts] - kept on CPU for topk indexing
            self.per_expert_scale_torch = state_dict["per_expert_scale"]
        else:
            scale_weight = None
            proj_weight = None
            self.per_expert_scale_torch = None

        self.scale = ttnn.as_tensor(
            scale_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "scale"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.proj_weight = ttnn.as_tensor(
            proj_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "proj.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, hidden_states):
        """
        Route tokens to experts.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device, TILE_LAYOUT
                Note: HF flattens to [B*S, H] before calling router. We keep 4D.

        Returns:
            top_k_weights: [seq_len, top_k] torch tensor (CPU) - routing weights after normalize + per_expert_scale
            top_k_indices: [seq_len, top_k] torch tensor (CPU) - selected expert indices
        """
        # 1. RMSNorm (no learned weight)
        normed = self.norm.forward(hidden_states)

        # 2. Scale: normed * scale * scalar_root_size
        scaled = ttnn.mul(normed, self.scale)
        normed.deallocate(True)
        scaled = ttnn.mul(scaled, self.scalar_root_size)

        # 3. Linear projection -> [1, 1, seq_len, num_experts]
        expert_scores = ttnn.linear(scaled, self.proj_weight)
        scaled.deallocate(True)

        # Move to CPU for softmax + topk (these are small tensors, num_experts=128)
        # On mesh devices, extract device 0 (all devices have same replicated data)
        scores_for_cpu = expert_scores
        if hasattr(self.mesh_device, "shape"):
            scores_for_cpu = ttnn.get_device_tensors(expert_scores)[0]
        expert_scores_torch = ttnn.to_torch(scores_for_cpu).squeeze(0).squeeze(0)  # [seq_len, num_experts]
        expert_scores.deallocate(True)

        # 4. Softmax
        router_probs = torch.nn.functional.softmax(expert_scores_torch.float(), dim=-1)

        # 5. TopK
        top_k_weights, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)

        # 6. Normalize weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # 7. Apply per-expert scale
        if self.per_expert_scale_torch is not None:
            top_k_weights = top_k_weights * self.per_expert_scale_torch[top_k_indices].float()

        return top_k_weights.to(torch.bfloat16), top_k_indices
