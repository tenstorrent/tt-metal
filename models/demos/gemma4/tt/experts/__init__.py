# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Routed Experts module.

Decode (seq_len=1): on-device via sparse_matmul
Prefill (seq_len>1): CPU fallback (sparse_matmul only supports batch=1)
"""

import torch
import torch.nn.functional as F

import ttnn

from .decode import decode_forward
from .prefill import prefill_forward
from .weights import ExpertWeights, load_expert_weights


class Gemma4ExpertConfig:
    """Configuration for the routed experts, derived from HF config."""

    def __init__(self, hf_config):
        self.hidden_size = hf_config.hidden_size
        self.num_experts = hf_config.num_experts
        self.top_k = hf_config.top_k_experts
        self.moe_intermediate_size = hf_config.moe_intermediate_size


class Gemma4Experts:
    def __init__(
        self,
        mesh_device,
        config,
        state_dict,
        ccl_manager,
        mesh_config,
        program_config,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config

        # Load weights to device for sparse_matmul (decode)
        if mesh_device is not None:
            self.weights = load_expert_weights(
                mesh_device=mesh_device,
                config=config,
                state_dict=state_dict,
                weight_dtype=weight_dtype,
                tensor_cache_path=tensor_cache_path,
            )
        else:
            self.weights = None

        # Keep CPU weights for prefill fallback
        if state_dict and "gate_up_proj" in state_dict:
            self.gate_up_proj_cpu = state_dict["gate_up_proj"]
            self.down_proj_cpu = state_dict["down_proj"]
        else:
            self.gate_up_proj_cpu = None
            self.down_proj_cpu = None

    def __call__(self, hidden_states, dense_routing):
        """
        Expert forward. Dispatches to on-device (decode) or CPU (prefill).

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device
            dense_routing: [1, 1, seq_len, num_experts] on device

        Returns:
            output: [1, 1, seq_len, hidden_size] on device
        """
        seq_len = hidden_states.shape[2]

        if self.weights is None:
            return self._cpu_forward(hidden_states, dense_routing)

        if seq_len == 1:
            return decode_forward(
                hidden_states=hidden_states,
                routing_weights=dense_routing,
                weights=self.weights,
                config=self.config,
            )
        else:
            # Prefill: on-device via sparse_matmul with tile-grouped sequence
            # Falls back to CPU if seq_len not tile-aligned
            if seq_len % 32 == 0:
                return prefill_forward(
                    hidden_states=hidden_states,
                    routing_weights=dense_routing,
                    weights=self.weights,
                    config=self.config,
                )
            else:
                return self._cpu_forward(hidden_states, dense_routing)

    def _cpu_forward(self, hidden_states, dense_routing):
        """CPU expert forward for prefill (seq_len > 1)."""
        if self.gate_up_proj_cpu is None:
            return hidden_states

        is_mesh = hasattr(self.mesh_device, "shape")
        h_cpu = ttnn.get_device_tensors(hidden_states)[0] if is_mesh else hidden_states
        r_cpu = ttnn.get_device_tensors(dense_routing)[0] if is_mesh else dense_routing

        x_torch = ttnn.to_torch(h_cpu).reshape(-1, self.config.hidden_size)
        routing_torch = ttnn.to_torch(r_cpu).reshape(-1, self.config.num_experts)
        seq_len = x_torch.shape[0]

        # Convert dense routing to sparse (indices + weights)
        top_k_weights, top_k_indices = torch.topk(routing_torch.float(), k=self.config.top_k, dim=-1)

        final = torch.zeros_like(x_torch)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_indices.long(), num_classes=self.config.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for eidx in expert_hit:
            eidx = eidx[0]
            if eidx == self.config.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            cur = x_torch[token_idx]
            gate, up = F.linear(cur.float(), self.gate_up_proj_cpu[eidx].float()).chunk(2, dim=-1)
            out = F.linear(F.gelu(gate, approximate="tanh") * up, self.down_proj_cpu[eidx].float())
            out = out * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))

        result_4d = final.reshape(1, 1, seq_len, self.config.hidden_size)
        result_tt = ttnn.from_torch(
            result_4d,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None,
        )
        return result_tt
