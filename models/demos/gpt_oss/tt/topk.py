# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Top-K Router implementation for Mixture of Experts (MoE) in GPT-OSS.

This module implements the routing mechanism that selects which experts should
process each token in the MoE architecture. The router uses a learned linear
transformation followed by top-k selection to assign tokens to experts.

"""

import os

import torch

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name


def topk_router(g, experts_per_token, use_throughput_experts, softmax_compute_config=None):
    typecast_needed = False
    if g.dtype != ttnn.bfloat16:
        g_og = g
        typecast_needed = True
        g = ttnn.typecast(g, dtype=ttnn.bfloat16)

    expert_weights, expert_indices = ttnn.topk(g, k=experts_per_token, dim=-1, sorted=True)
    if typecast_needed:
        g.deallocate(True)
        g = g_og
    if softmax_compute_config is None:
        softmax_compute_config = ttnn.init_device_compute_kernel_config(
            g.device().arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    expert_weights = ttnn.softmax(
        expert_weights, dim=1, numeric_stable=True, compute_kernel_config=softmax_compute_config
    )
    if use_throughput_experts:
        return expert_indices, expert_weights
    else:
        return expert_indices, ttnn.scatter(ttnn.zeros_like(g), dim=1, index=expert_indices, src=expert_weights)


class TopKRouter:
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None):
        self.top_k = hf_config.num_experts_per_tok
        self.num_experts = hf_config.num_local_experts
        self.hidden_dim = hf_config.hidden_size
        self.weight = ttnn.as_tensor(
            state_dict["weight"].transpose(0, 1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bias = ttnn.as_tensor(
            state_dict["bias"].unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Keep compute_config=None for linear (known quality-safe default)
        # Custom compute configs were previously found to cause quality degradation
        self.compute_config = None

        # Cache softmax compute config (same as what topk_router creates per-call)
        self.softmax_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Fused op support: matmul + topk + softmax in one kernel
        # Enable via TOPK_FUSED_OP=1 environment variable
        self.use_fused_op = os.environ.get("TOPK_FUSED_OP", "0") == "1"
        self._fused_bias = None
        self._fused_output = None

    def _init_fused_op(self, device, B):
        """Lazily initialize fused op tensors (bias broadcast + output buffer)."""
        if self._fused_bias is None:
            # self.bias may be a multi-device tensor — extract one device's copy
            device_tensors = ttnn.get_device_tensors(self.bias)
            bias_torch = ttnn.to_torch(device_tensors[0])  # [1, num_experts]
            bias_bcast = bias_torch.to(torch.bfloat16).expand(B, -1).contiguous()
            self._fused_bias = ttnn.from_torch(
                bias_bcast,
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if self._fused_output is None:
            self._fused_output = ttnn.from_torch(
                torch.zeros(B, 64, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )

    def __call__(self, hidden_states, use_throughput_experts):
        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))

        # Fused op only supports decode mode (B=32, seq_len=1 → shape [32, hidden_dim])
        if self.use_fused_op and hidden_states.shape[0] == 32:
            return self._fused_call(hidden_states, use_throughput_experts)

        # Output to L1 instead of DRAM (saves DRAM write+read round-trip)
        router_logits = ttnn.linear(
            hidden_states,
            self.weight,
            bias=self.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )

        # No need for to_memory_config since output is already in L1
        # (topk supports L1 inputs)

        expert_indices, expert_weights = topk_router(
            router_logits, self.top_k, use_throughput_experts, self.softmax_compute_config
        )
        ttnn.deallocate(router_logits)
        return expert_indices, expert_weights

    def _fused_call(self, hidden_states, use_throughput_experts):
        """Forward pass using fused matmul+topk+softmax kernel."""
        # Typecast to bf16 if needed (fused op requires bf16 input)
        needs_typecast = hidden_states.dtype != ttnn.bfloat16
        if needs_typecast:
            hidden_states_bf16 = ttnn.typecast(hidden_states, dtype=ttnn.bfloat16)
        else:
            hidden_states_bf16 = hidden_states

        B = hidden_states_bf16.shape[0]
        device = hidden_states_bf16.device()

        # Initialize fused op buffers (once)
        self._init_fused_op(device, B)

        # Run fused matmul + topk + softmax
        result = ttnn.experimental.topk_router_gpt(
            hidden_states_bf16,
            weight_tensor=self.weight,
            bias_tensor=self._fused_bias,
            output_tensor=self._fused_output,
            k=self.top_k,
            num_experts=self.num_experts,
        )

        if needs_typecast:
            ttnn.deallocate(hidden_states_bf16)

        # Unpack: tile 0 (cols 0..k-1) = softmax weights, tile 1 (cols 32..32+k-1) = indices
        expert_weights = ttnn.slice(result, [0, 0], [B, self.top_k])
        expert_indices = ttnn.slice(result, [0, 32], [B, 32 + self.top_k])

        if use_throughput_experts:
            return expert_indices, expert_weights
        else:
            # Non-throughput: scatter weights to dense [B, num_experts] format
            zeros = ttnn.zeros(
                [B, self.num_experts],
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )
            return expert_indices, ttnn.scatter(zeros, dim=1, index=expert_indices, src=expert_weights)
