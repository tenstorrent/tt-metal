# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE router: linear -> softmax -> topk -> (sum-normalize) -> scatter.

Fully on-device, trace-compatible. Returns dense routing weights [1,1,S,E] on device
(weights at the selected experts, zeros elsewhere) for sparse_matmul.

Simpler than gemma4's router: Qwen3-Next/Qwen3.5-MoE has NO router RMSNorm, NO input
pre-scale, and NO per-expert scale. It softmaxes over ALL experts, takes top-k, and
(when norm_topk_prob) divides the top-k weights by their sum. The router matmul +
softmax run with fp32 accumulate to match HF's fp32 routing precision.
"""

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc


class Qwen36Router:
    def __init__(self, mesh_device, config, state_dict, tensor_cache_path=None, dtype=ttnn.bfloat16):
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.norm_topk_prob = config.norm_topk_prob

        is_mesh = hasattr(mesh_device, "shape")
        replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        # HF mlp.gate.weight is [E, H] (nn.Linear out,in). Transpose to [1,1,H,E] for
        # ttnn.linear (in,out) and replicate on every device (router is tiny +
        # accuracy-sensitive, kept at bf16).
        proj_weight = None
        if state_dict:
            proj_weight = state_dict["weight"].to(torch.bfloat16).transpose(-2, -1).unsqueeze(0).unsqueeze(0)

        self.proj_weight = ttnn.as_tensor(
            proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_mapper,
            cache_file_name=(str(tensor_cache_path / "moe.router.weight") if tensor_cache_path else None),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # HiFi2 + fp32 dest accumulate so the router matmul matches HF's fp32 routing.
        self.compute_kernel_config = tpc.COMPUTE_HIFI2

    def __call__(self, hidden_states):
        """hidden_states: [1,1,S,H] (replicated full hidden). Returns [1,1,S,E]."""
        expert_scores = ttnn.linear(hidden_states, self.proj_weight, compute_kernel_config=self.compute_kernel_config)
        router_probs = ttnn.softmax(expert_scores, dim=-1)
        expert_scores.deallocate(True)

        top_k_values, top_k_indices = ttnn.topk(router_probs, k=self.top_k, dim=-1)

        # Sum-normalize the top-k weights so they sum to 1 per token (HF norm_topk_prob).
        if self.norm_topk_prob:
            top_k_sum = ttnn.sum(top_k_values, dim=-1, keepdim=True)
            top_k_values = ttnn.div(top_k_values, top_k_sum)
            top_k_sum.deallocate(True)

        dense_routing = ttnn.scatter(
            ttnn.zeros_like(router_probs),
            dim=-1,
            index=top_k_indices,
            src=top_k_values,
        )
        router_probs.deallocate(True)
        top_k_values.deallocate(True)
        top_k_indices.deallocate(True)
        return dense_routing
