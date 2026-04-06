# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Router: RMSNorm -> scale -> linear -> softmax -> topk -> normalize -> per_expert_scale

Gemma4 uses softmax-THEN-topk (opposite of GPT-OSS).

On-device operations: norm, scale, linear, softmax, topk
CPU operations: normalize, per_expert_scale (small [seq_len, top_k] tensors)
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
            # per_expert_scale: kept on CPU for post-topk indexing
            self.per_expert_scale_torch = state_dict["per_expert_scale"]
        else:
            scale_weight = None
            proj_weight = None
            self.per_expert_scale_torch = None

        is_mesh = hasattr(mesh_device, "shape")
        replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        self.scale = ttnn.as_tensor(
            scale_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "scale"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.proj_weight = ttnn.as_tensor(
            proj_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "proj.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, hidden_states):
        """
        Route tokens to experts.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device, TILE_LAYOUT

        Returns:
            top_k_weights: [seq_len, top_k] torch tensor (CPU)
            top_k_indices: [seq_len, top_k] torch tensor (CPU)
        """
        # 1. RMSNorm (no learned weight) — on device
        normed = self.norm.forward(hidden_states)

        # 2. Scale: normed * scale * scalar_root_size — on device
        scaled = ttnn.mul(normed, self.scale)
        normed.deallocate(True)
        scaled = ttnn.mul(scaled, self.scalar_root_size)

        # 3. Linear projection -> [1, 1, seq_len, num_experts] — on device
        expert_scores = ttnn.linear(scaled, self.proj_weight)
        scaled.deallocate(True)

        # 4. Softmax — on device
        router_probs = ttnn.softmax(expert_scores, dim=-1)
        expert_scores.deallocate(True)

        # 5. TopK — on device
        top_k_values, top_k_indices_tt = ttnn.topk(router_probs, k=self.top_k, dim=-1)
        router_probs.deallocate(True)

        # Move small [seq_len, top_k] tensors to CPU for normalize + per_expert_scale
        if hasattr(self.mesh_device, "shape"):
            top_k_values = ttnn.get_device_tensors(top_k_values)[0]
            top_k_indices_tt = ttnn.get_device_tensors(top_k_indices_tt)[0]

        top_k_weights = ttnn.to_torch(top_k_values).squeeze(0).squeeze(0).float()
        top_k_indices = ttnn.to_torch(top_k_indices_tt).squeeze(0).squeeze(0).to(torch.int64)

        # 6. Normalize weights to sum to 1 — CPU (small tensor)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # 7. Apply per-expert scale — CPU (small tensor, requires indexing)
        if self.per_expert_scale_torch is not None:
            top_k_weights = top_k_weights * self.per_expert_scale_torch[top_k_indices].float()

        return top_k_weights.to(torch.bfloat16), top_k_indices
