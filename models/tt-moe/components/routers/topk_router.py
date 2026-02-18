# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
TopK Router implementation for GPT-OSS MoE models.

This router selects the top-k experts for each token using a learned linear
transformation followed by softmax normalization. Unlike the MoEGateRouter
used by DeepSeek, this router doesn't apply score correction bias.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger

import ttnn

from .base_router import BaseRouter


class TopKRouter(BaseRouter):
    """
    TopK Expert Router for GPT-OSS style Mixture of Experts.

    This router implements the GPT-OSS routing mechanism:
    1. Linear projection from hidden_dim to num_experts
    2. Top-k selection without score correction
    3. Softmax normalization of selected weights
    4. Optional scatter for sparse weight format
    """

    def __init__(
        self,
        mesh_device,
        config: Dict[str, Any],
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        tensor_cache_path: Optional[str] = None,
    ):
        """
        Initialize TopK router.

        Args:
            mesh_device: TTNN mesh device
            config: Configuration dictionary containing:
                - num_experts: Total number of experts
                - num_experts_per_tok: Number of experts to select (k)
                - hidden_size: Input hidden dimension
            state_dict: Router weights with keys 'weight' and 'bias'
            tensor_cache_path: Optional cache path
        """
        # Call BaseRouter with the correct signature (config, mesh_device)
        super().__init__(config, mesh_device)

        # Store additional parameters not handled by BaseRouter
        self.mesh_device = mesh_device  # Store mesh_device as instance variable
        self.state_dict = state_dict
        self.tensor_cache_path = tensor_cache_path

        # Router configuration
        self.num_experts = config["num_experts"]
        self.num_experts_per_tok = config["num_experts_per_tok"]
        self.hidden_size = config["hidden_size"]

        # Compute kernel configuration for numerical stability
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Load weights if provided
        if state_dict is not None:
            self.load_weights(state_dict)
        else:
            self.weight = None
            self.bias = None

    def load_weights(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load router weights from state dictionary.

        GPT-OSS uses 'weight' and 'bias' keys directly under the router,
        and the weight needs to be transposed for the linear operation.

        Args:
            state_dict: Dictionary with 'weight' and 'bias' tensors
        """
        # Extract router weights (may be nested under 'router' key)
        if "router" in state_dict:
            weight_tensor = state_dict["router"]["weight"]
            bias_tensor = state_dict["router"]["bias"]
        elif "weight" in state_dict and "bias" in state_dict:
            weight_tensor = state_dict["weight"]
            bias_tensor = state_dict["bias"]
        else:
            # Try with model layer prefix (e.g., model.layers.0.mlp.router.weight)
            router_keys = [k for k in state_dict.keys() if "router" in k]
            if router_keys:
                weight_key = next((k for k in router_keys if k.endswith("weight")), None)
                bias_key = next((k for k in router_keys if k.endswith("bias")), None)
                if weight_key and bias_key:
                    weight_tensor = state_dict[weight_key]
                    bias_tensor = state_dict[bias_key]
                else:
                    raise ValueError(f"Could not find router weight/bias in state dict. Keys: {router_keys}")
            else:
                raise ValueError("Could not find router weights in state dict")

        # Transpose weight for linear operation (GPT-OSS convention)
        # Shape: [num_experts, hidden_size] -> [hidden_size, num_experts]
        weight_tensor = weight_tensor.transpose(0, 1)

        # Add batch dimension to bias
        if len(bias_tensor.shape) == 1:
            bias_tensor = bias_tensor.unsqueeze(0)

        # Convert to TTNN tensors
        self.weight = ttnn.from_torch(
            weight_tensor,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        self.bias = ttnn.from_torch(
            bias_tensor,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        logger.debug(f"TopKRouter: Loaded weights with shape {weight_tensor.shape}")

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        is_decode: bool = True,
        mode: Optional[str] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Route tokens to top-k experts.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            is_decode: Whether in decode mode (seq_len=1)
            mode: Optional mode string ("decode" or "prefill")

        Returns:
            Tuple of:
                - expert_indices: Selected expert indices [batch*seq_len, num_experts_per_tok]
                - expert_weights: Normalized weights in dense format
        """
        if self.weight is None or self.bias is None:
            raise ValueError("Router weights not loaded. Call load_weights() first.")

        # Get input shape and preserve batch dimensions
        input_shape = hidden_states.shape

        # Flatten batch and sequence dimensions for linear projection
        # Input could be [1, seq_len, batch_size, hidden_dim] or similar 4D shape
        if len(input_shape) == 4:
            batch_seq_len = input_shape[1] * input_shape[2]
            # Reshape to [1, 1, batch*seq, hidden_dim] for linear projection
            hidden_states = ttnn.reshape(hidden_states, (1, 1, batch_seq_len, self.hidden_size))
        elif len(input_shape) == 3:
            batch_seq_len = input_shape[0] * input_shape[1]
            # Reshape to [1, 1, batch*seq, hidden_dim] for linear projection
            hidden_states = ttnn.reshape(hidden_states, (1, 1, batch_seq_len, self.hidden_size))
        else:
            # Already in correct shape
            batch_seq_len = input_shape[0]

        # Linear projection to get router logits
        # Memory config based on decode mode (like GPT-OSS)
        mem_config = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG

        router_logits = ttnn.linear(
            hidden_states,
            self.weight,
            bias=self.bias,
            memory_config=mem_config,
            # Don't use custom compute config for linear to avoid quality issues
            compute_kernel_config=None,
        )

        # Convert to DRAM if needed for topk (topk doesn't support sharded inputs)
        if is_decode:
            router_logits = ttnn.to_memory_config(router_logits, ttnn.DRAM_MEMORY_CONFIG)

        # Select top-k experts
        expert_indices, expert_weights = self._topk_selection(router_logits, self.num_experts_per_tok)

        # Clean up
        ttnn.deallocate(router_logits)

        return expert_indices, expert_weights

    def _topk_selection(self, logits: ttnn.Tensor, k: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Perform top-k selection and normalization.

        Args:
            logits: Router logits [batch*seq_len, num_experts]
            k: Number of experts to select

        Returns:
            Tuple of (expert_indices, expert_weights)
        """
        # Handle dtype conversion if needed
        typecast_needed = False
        if logits.dtype != ttnn.bfloat16:
            logits_orig = logits
            typecast_needed = True
            logits = ttnn.typecast(logits, dtype=ttnn.bfloat16)

        # Top-k selection
        expert_weights, expert_indices = ttnn.topk(logits, k=k, dim=-1, sorted=True)

        # Clean up if we did typecast
        if typecast_needed:
            ttnn.deallocate(logits)
            logits = logits_orig

        # Convert indices to uint16 (required for all-to-all dispatch)
        expert_indices = ttnn.typecast(expert_indices, dtype=ttnn.uint16)

        # Softmax normalization with numerical stability
        expert_weights = ttnn.softmax(
            expert_weights, dim=-1, numeric_stable=True, compute_kernel_config=self.compute_config
        )

        # Return dense format for all experts (all-to-all dispatch)
        return expert_indices, expert_weights

    def get_expert_capacity(self, seq_len: int, batch_size: int) -> int:
        """
        Calculate expert capacity for load balancing.

        For TopK routing, each token selects exactly k experts,
        so capacity is deterministic.

        Args:
            seq_len: Sequence length
            batch_size: Batch size

        Returns:
            Expert capacity (tokens per expert)
        """
        total_tokens = seq_len * batch_size
        # Each token goes to k experts, so total expert assignments
        total_assignments = total_tokens * self.num_experts_per_tok
        # Distributed evenly across all experts
        capacity_per_expert = total_assignments // self.num_experts
        # Add some buffer for imbalanced routing
        return int(capacity_per_expert * 1.25)
