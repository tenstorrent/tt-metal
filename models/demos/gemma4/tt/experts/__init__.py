# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Routed Experts module.

Decode (seq_len=1): on-device via sparse_matmul with top-k sparsity
Prefill (seq_len>1): on-device via sparse_matmul with all-ones sparsity (gpt_oss pattern)
"""

import ttnn
from models.demos.gemma4.tt.optimization import profile_weight_dtype

from .decode import decode_forward
from .prefill import create_prefill_sparsity, prefill_forward
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

        gate_choice = profile_weight_dtype(
            "expert_gate",
            env_name="GEMMA4_EXPERT_GATE_WEIGHT_DTYPE",
            legacy_env_name="GEMMA4_EXPERT_WEIGHT_DTYPE",
        )
        up_choice = profile_weight_dtype(
            "expert_up",
            env_name="GEMMA4_EXPERT_UP_WEIGHT_DTYPE",
            legacy_env_name="GEMMA4_EXPERT_WEIGHT_DTYPE",
        )
        down_choice = profile_weight_dtype(
            "expert_down",
            env_name="GEMMA4_EXPERT_DOWN_WEIGHT_DTYPE",
            legacy_env_name="GEMMA4_EXPERT_WEIGHT_DTYPE",
        )
        # Load weights to device for sparse_matmul
        self.weights = load_expert_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            gate_weight_dtype=gate_choice.dtype,
            up_weight_dtype=up_choice.dtype,
            down_weight_dtype=down_choice.dtype,
            tensor_cache_path=tensor_cache_path,
            gate_cache_suffix=gate_choice.cache_suffix,
            up_cache_suffix=up_choice.cache_suffix,
            down_cache_suffix=down_choice.cache_suffix,
        )
        # Cache all-ones prefill sparsity (reused for every prefill call)
        self.prefill_sparsity = create_prefill_sparsity(mesh_device, config.num_experts)

    def __call__(self, hidden_states, dense_routing):
        """
        Expert forward — fully on-device.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device
            dense_routing: [1, 1, seq_len, num_experts] on device

        Returns:
            output: [1, 1, seq_len, hidden_size] on device
        """
        seq_len = hidden_states.shape[2]

        if seq_len == 1:
            return decode_forward(
                hidden_states=hidden_states,
                routing_weights=dense_routing,
                weights=self.weights,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
            )
        else:
            assert seq_len % 32 == 0, f"Prefill seq_len must be a multiple of 32, got {seq_len}"
            return prefill_forward(
                hidden_states=hidden_states,
                routing_weights=dense_routing,
                weights=self.weights,
                config=self.config,
                prefill_sparsity=self.prefill_sparsity,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
            )
