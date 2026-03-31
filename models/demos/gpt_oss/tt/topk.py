# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Top-K Router implementation for Mixture of Experts (MoE) in GPT-OSS.

This module implements the routing mechanism that selects which experts should
process each token in the MoE architecture. The router uses a learned linear
transformation followed by top-k selection to assign tokens to experts.

"""

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
        # The fused kernel uses 4 groups of 3 cores, one per N-tile (32 experts
        # each), so it requires exactly 128 experts. Enable automatically when possible.
        self.use_fused_op = self.num_experts == 128
        self._fused_bias = None
        # Keep the original unsharded bias for fused op initialization
        # (ttnn.as_tensor shards self.bias across the mesh, but the fused op
        # needs the full [1, num_experts] bias replicated on every device)
        if self.use_fused_op:
            self._bias_torch = state_dict["bias"].unsqueeze(0).to(torch.bfloat16)
        else:
            self._bias_torch = None

    def _init_fused_op(self, device, B):
        """Lazily initialize fused op tensors (bias broadcast + output pre-alloc)."""
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None

        if self._fused_bias is None:
            # Use the original unsharded bias (self._bias_torch is [1, num_experts])
            # and broadcast to [B, num_experts] so every tile row has the bias vector.
            bias_bcast = self._bias_torch.expand(B, -1).contiguous()
            self._fused_bias = ttnn.from_torch(
                bias_bcast,
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )

    def __call__(self, hidden_states, use_throughput_experts):
        # Compute actual token count from the tensor volume before reshape,
        # since shape[0] after reshape returns the tile-padded dimension
        # (e.g. 8 tokens padded to 32 in TILE_LAYOUT).
        actual_tokens = hidden_states.volume() // self.hidden_dim
        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))

        # Fused op only supports decode mode (B=32, seq_len=1 → shape [32, hidden_dim])
        if self.use_fused_op and actual_tokens == 32 and use_throughput_experts:
            return self._fused_call(hidden_states, use_throughput_experts)

        # Use L1 for decode (small tensors), DRAM for prefill (large sequences)
        is_decode = actual_tokens <= 128
        mem_config = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        router_logits = ttnn.linear(
            hidden_states,
            self.weight,
            bias=self.bias,
            memory_config=mem_config,
            compute_kernel_config=self.compute_config,
        )

        expert_indices, expert_weights = topk_router(
            router_logits, self.top_k, use_throughput_experts, self.softmax_compute_config
        )
        ttnn.deallocate(router_logits)
        return expert_indices, expert_weights

    def _fused_call(self, hidden_states, use_throughput_experts):
        """Forward pass using fused matmul+topk+softmax kernel.

        Note: Fused op only supports throughput experts (sparse [B,k] output).
        """
        if not use_throughput_experts:
            raise ValueError("Fused topk_router_gpt requires use_throughput_experts=True")

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
        indices_rm, weights_rm = ttnn.experimental.topk_router_gpt(
            hidden_states_bf16,
            weight_tensor=self.weight,
            bias_tensor=self._fused_bias,
            k=self.top_k,
            num_experts=self.num_experts,
        )

        if needs_typecast:
            ttnn.deallocate(hidden_states_bf16)

        # Kernel produces uint16 RM [B, k_padded] and bf16 RM [B, k_padded].
        # Slice to [B, top_k] in RM and return directly.
        # fused_decode.py handles RM input natively (zero-cost reshape to 4D).
        expert_indices = ttnn.slice(indices_rm, [0, 0], [B, self.top_k])
        expert_weights = ttnn.slice(weights_rm, [0, 0], [B, self.top_k])
        ttnn.deallocate(indices_rm)
        ttnn.deallocate(weights_rm)
        return expert_indices, expert_weights
