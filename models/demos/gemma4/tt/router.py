# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Router: RMSNorm -> scale -> linear -> softmax -> topk -> normalize -> per_expert_scale -> scatter

Fully on-device. Returns dense routing weights [1,1,S,E] on device for sparse_matmul.
Gemma4 uses softmax-THEN-topk (opposite of GPT-OSS).
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

        self.norm = RMSNorm(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict={},
            tensor_cache_path=f"{tensor_cache_path}/norm" if tensor_cache_path else None,
            with_scale=False,
        )

        if state_dict:
            scale_weight = state_dict["scale"].reshape(1, 1, 1, -1)
            proj_weight = substate(state_dict, "proj")["weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
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
        Route tokens to experts. Returns dense routing weights on device.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device

        Returns:
            dense_routing: [1, 1, seq_len, num_experts] on device — weights at selected experts, zeros elsewhere
        """
        # 1. RMSNorm — on device
        normed = self.norm.forward(hidden_states)

        # 2. Scale — on device
        scaled = ttnn.mul(normed, self.scale)
        normed.deallocate(True)
        scaled = ttnn.mul(scaled, self.scalar_root_size)

        # 3. Linear projection → [1, 1, seq_len, num_experts] — on device
        expert_scores = ttnn.linear(scaled, self.proj_weight)
        scaled.deallocate(True)

        # 4. Softmax — on device
        router_probs = ttnn.softmax(expert_scores, dim=-1)
        expert_scores.deallocate(True)

        # 5. TopK — on device → values [1,1,S,k], indices [1,1,S,k]
        top_k_values, top_k_indices = ttnn.topk(router_probs, k=self.top_k, dim=-1)

        # 6-7. Normalize + per_expert_scale — CPU (small [S, top_k] tensors)
        # Move to CPU for indexing (per_expert_scale[indices])
        if hasattr(self.mesh_device, "shape"):
            tkv_cpu = ttnn.get_device_tensors(top_k_values)[0]
            tki_cpu = ttnn.get_device_tensors(top_k_indices)[0]
        else:
            tkv_cpu = top_k_values
            tki_cpu = top_k_indices

        weights_torch = ttnn.to_torch(tkv_cpu).float()
        indices_torch = ttnn.to_torch(tki_cpu).to(torch.int64)

        # Normalize
        weights_torch = weights_torch / weights_torch.sum(dim=-1, keepdim=True)

        # Per-expert scale
        if self.per_expert_scale_torch is not None:
            weights_torch = weights_torch * self.per_expert_scale_torch[indices_torch.squeeze(0).squeeze(0)].float()

        # 8. Scatter into dense [1,1,S,E] — on device
        # Create zeros and scatter scaled weights at selected expert positions
        weights_scaled = weights_torch.to(torch.bfloat16)
        zeros = torch.zeros(
            router_probs.shape[0],
            router_probs.shape[1],
            router_probs.shape[2],
            self.num_experts,
            dtype=torch.bfloat16,
        )
        # scatter_: put weights at index positions
        zeros.scatter_(-1, indices_torch.to(torch.int64), weights_scaled)

        is_mesh = hasattr(self.mesh_device, "shape")
        dense_routing = ttnn.from_torch(
            zeros,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None,
        )

        router_probs.deallocate(True)
        top_k_values.deallocate(True)
        top_k_indices.deallocate(True)

        return dense_routing
