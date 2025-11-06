# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Generic MoE Experts Module.

This module provides a model-agnostic implementation of MoE experts.
Models provide their own ProgramConfig implementations for customization.

Usage:
    from models.demos.gpt_oss.tt.experts import Experts, ExpertConfig
    from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig

    config = ExpertConfig(...)
    program_config = GPTOSSProgramConfig()

    experts = Experts(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=program_config,
    )

    output = experts(hidden_states, routing_weights)
"""

import ttnn

from models.demos.gpt_oss.config import MeshConfig, ModeConfig

from .config import ExpertConfig, ProgramConfig
from .decode import decode_forward
from .prefill import prefill_forward
from .weights import load_expert_weights

__all__ = ["Experts", "ExpertConfig", "ProgramConfig"]


class Experts:
    """
    Generic MoE Expert implementation with automatic decode/prefill dispatch.

    This class provides a clean interface for expert layers. Models provide
    their own ProgramConfig implementations to customize behavior.
    """

    def __init__(
        self,
        mesh_device,
        config: ExpertConfig,
        state_dict,
        ccl_manager,
        mesh_config: MeshConfig,
        program_config: ProgramConfig,
        weight_dtype=ttnn.bfloat4_b,
        activation_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        """
        Initialize expert layers.

        Args:
            mesh_device: TTNN mesh device
            config: Expert configuration
            state_dict: Expert weights dictionary
            ccl_manager: Communication manager
            mesh_config: Mesh parallelization configuration
            program_config: Model-specific program configurations
            weight_dtype: Data type for weights (default: bfloat4_b)
            activation_dtype: Data type for activations (default: bfloat8_b)
            tensor_cache_path: Optional path for weight caching
        """
        self.config = config
        self.mesh_config = mesh_config
        self.mesh_device = mesh_device  # ✅ Store mesh_device
        self.ccl_manager = ccl_manager
        self.program_config = program_config
        self.activation_dtype = activation_dtype

        # Load weights
        self.weights = load_expert_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )

        # Cache prefill sparsity (created once, reused for all prefill calls)
        self.prefill_sparsity = self._create_prefill_sparsity()

        # For backward compatibility
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size

    def _create_prefill_sparsity(self):
        """Create prefill sparsity mask once and reuse."""
        import torch
        from models.demos.gpt_oss.config import Mode

        prefill_config = self.mesh_config.get_config(Mode.PREFILL)
        prefill_ep = prefill_config.ep
        tokens_per_ep = self.config.num_experts // prefill_ep

        sparsity = torch.zeros(1, 1, prefill_ep, self.config.num_experts)
        for i in range(prefill_ep):
            sparsity[:, :, i, i * tokens_per_ep : (i + 1) * tokens_per_ep] = torch.ones(1, 1, 1, tokens_per_ep)

        return ttnn.from_torch(
            sparsity,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                dims=(-2, None) if prefill_ep > 1 else (None, None),
                mesh_shape=self.mesh_device.shape,
                mesh_device=self.mesh_device,
            ),
        )

    def __call__(self, hidden_states, routing_weights):
        """
        Forward pass - automatically dispatches to decode or prefill.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            routing_weights: Router output [seq_len, num_experts]

        Returns:
            Expert output tensor [batch, seq_len, hidden_size]
        """
        # Determine mode based on sequence length
        seq_len = hidden_states.shape[1]
        is_decode = seq_len == 1

        if is_decode:
            return decode_forward(
                hidden_states=hidden_states,
                routing_weights=routing_weights,
                weights=self.weights,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,  # ✅ Pass mesh_device
                ccl_manager=self.ccl_manager,
                program_config=self.program_config,
                activation_dtype=self.activation_dtype,
            )
        else:
            return prefill_forward(
                hidden_states=hidden_states,
                routing_weights=routing_weights,
                weights=self.weights,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,  # ✅ Pass mesh_device
                ccl_manager=self.ccl_manager,
                program_config=self.program_config,
                activation_dtype=self.activation_dtype,
                prefill_sparsity=self.prefill_sparsity,  # ✅ Pass cached sparsity
            )
