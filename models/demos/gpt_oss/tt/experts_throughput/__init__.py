# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Throughput-optimized MoE Experts Module with all_to_all operations.

This module provides a high-throughput implementation of MoE experts using
all_to_all_dispatch and all_to_all_combine operations to dynamically batch
tokens across 32 Galaxy devices (4 experts per device).

The key difference from the standard experts module is:
- Standard: Each device processes all experts, uses sparse matmul for routing
- Throughput: Experts distributed across devices, tokens routed via all_to_all

This approach enables handling multiple batches efficiently by dynamically
creating batches on each device based on expert routing decisions.

Usage:
    from models.demos.gpt_oss.tt.experts_throughput import (
        ThroughputExperts,
        ThroughputExpertConfig,
        ThroughputProgramConfig,
    )

    config = ThroughputExpertConfig.from_hf_config(hf_config, mesh_device)
    program_config = ThroughputProgramConfig()

    experts = ThroughputExperts(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        program_config=program_config,
    )

    output = experts(hidden_states, topk_indices, topk_weights)
"""

import ttnn
from typing import Optional
from models.demos.gpt_oss.config import MeshConfig
from .config import (
    ThroughputExpertConfig,
    ThroughputProgramConfig,
    AllToAllDispatchConfig,
    AllToAllCombineConfig,
    create_expert_mapping_tensors,
    create_remap_topk_mask,
)
from .decode import decode_forward
from .prefill import prefill_forward_chunked
from .weights import (
    ThroughputExpertWeights,
    load_throughput_expert_weights,
)

__all__ = [
    "ThroughputExperts",
    "ThroughputExpertConfig",
    "ThroughputProgramConfig",
    "ThroughputExpertWeights",
    "AllToAllDispatchConfig",
    "AllToAllCombineConfig",
]


class ThroughputExperts:
    """
    Throughput-optimized MoE Expert implementation using all_to_all operations.

    This class distributes experts across devices (4 per device on a 32-device
    Galaxy system) and uses all_to_all_dispatch and all_to_all_combine to
    dynamically route tokens to their assigned experts.

    The workflow is:
    1. Router computes top-k experts per token (external)
    2. all_to_all_dispatch sends tokens to devices hosting their experts
    3. Each device runs MLP on its local experts
    4. all_to_all_combine returns results to original token positions
    5. Results weighted by routing weights and summed

    Attributes:
        config: Expert configuration
        weights: Loaded expert weights (sharded by device)
        program_config: Matmul program configurations
        expert_mapping_tensors: Device-expert mapping for routing
        remap_topk_mask: Mask for expert remapping
        dispatch_config: all_to_all_dispatch configuration
        combine_config: all_to_all_combine configuration
    """

    def __init__(
        self,
        mesh_device,
        config: ThroughputExpertConfig,
        state_dict: dict,
        ccl_manager,
        mesh_config: MeshConfig,
        program_config: ThroughputProgramConfig = None,
        weight_dtype=ttnn.bfloat4_b,
        tensor_cache_path: str = None,
        dispatch_cluster_axis: Optional[int] = None,
        decode_memory_config: ttnn.MemoryConfig = None,
        prefill_memory_config: ttnn.MemoryConfig = None,
    ):
        """
        Initialize throughput experts.

        Args:
            mesh_device: TTNN mesh device (expected 32 devices for Galaxy)
            config: Expert configuration
            state_dict: Expert weights dictionary
            program_config: Optional custom program config (uses defaults if None)
            weight_dtype: Data type for weights (default: bfloat4_b)
            tensor_cache_path: Optional path for weight caching
            dispatch_cluster_axis: Mesh axis for all_to_all operations (default: 0 for rows)
            decode_memory_config: Memory config for decode (default: L1)
            prefill_memory_config: Memory config for prefill (default: DRAM)

        Raises:
            ValueError: If mesh configuration is incompatible with all_to_all operations
        """
        # Validate mesh configuration for all_to_all operations
        mesh_shape = mesh_device.shape

        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.config = config
        self.program_config = program_config or ThroughputProgramConfig()

        # Memory configurations
        decode_memory_config = decode_memory_config or ttnn.L1_MEMORY_CONFIG
        prefill_memory_config = prefill_memory_config or ttnn.DRAM_MEMORY_CONFIG

        # Create all_to_all configurations
        self.dispatch_config_decode = AllToAllDispatchConfig(
            cluster_axis=dispatch_cluster_axis,
            memory_config=decode_memory_config,
        )
        self.dispatch_config_prefill = AllToAllDispatchConfig(
            cluster_axis=dispatch_cluster_axis,
            memory_config=prefill_memory_config,
        )
        self.combine_config_decode = AllToAllCombineConfig(
            cluster_axis=dispatch_cluster_axis,
            memory_config=decode_memory_config,
        )
        self.combine_config_prefill = AllToAllCombineConfig(
            cluster_axis=dispatch_cluster_axis,
            memory_config=prefill_memory_config,
        )

        # Load weights
        self.weights = load_throughput_expert_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )

        # Create mapping tensors for all_to_all routing
        self.expert_mapping_tensors = create_expert_mapping_tensors(
            num_devices=config.num_devices,
            num_experts_per_device=config.num_experts_per_device,
            mesh_device=mesh_device,
            cluster_axis=dispatch_cluster_axis,
            mesh_shape=mesh_shape,
        )

        # Create remap mask (rows is dispatch dimension)
        num_dispatch_device_rows = mesh_device.shape[0]
        self.remap_topk_mask = create_remap_topk_mask(
            num_dispatch_device_rows=num_dispatch_device_rows,
            num_experts=config.num_experts,
            mesh_device=mesh_device,
        )

        # For backward compatibility with existing code
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.num_experts_per_device = config.num_experts_per_device

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        topk_expert_indices: ttnn.Tensor,
        topk_expert_weights: ttnn.Tensor,
        is_decode: bool = True,
        chunk_size: int = 128,  # TODO: increasing this causes diverging outputs for last mesh row (https://github.com/tenstorrent/tt-metal/issues/36335)
    ) -> ttnn.Tensor:
        """
        Forward pass - automatically dispatches to decode or prefill.

        Args:
            hidden_states: Input tensor [batch/seq, 1, 1, hidden_size]
            topk_expert_indices: Top-k expert indices per token
                [batch/seq, 1, 1, num_experts_per_tok]
            topk_expert_weights: Dense routing scores for top-k experts
                [batch/seq, 1, 1, num_experts_per_tok]
            chunk_size: Chunk size for prefill (default: 2048)

        Returns:
            Expert output tensor [batch/seq, 1, 1, hidden_size]
        """

        if is_decode:
            return self.forward_decode(
                hidden_states,
                topk_expert_indices,
                topk_expert_weights,
            )
        else:
            return self.forward_prefill(
                hidden_states,
                topk_expert_indices,
                topk_expert_weights,
                chunk_size=chunk_size,
            )

    def forward_decode(
        self,
        hidden_states: ttnn.Tensor,
        topk_expert_indices: ttnn.Tensor,
        topk_expert_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Decode forward pass.

        Args:
            hidden_states: Input [batch_per_device, 1, 1, hidden_size]
            topk_expert_indices: Expert indices [batch_per_device, 1, 1, k]
            topk_expert_weights: Routing weights [batch_per_device, 1, 1, k]

        Returns:
            Output [batch_per_device, 1, 1, hidden_size]
        """
        return decode_forward(
            hidden_states=hidden_states,
            topk_expert_indices=topk_expert_indices,
            topk_expert_weights=topk_expert_weights,
            weights=self.weights,
            config=self.config,
            expert_mapping_tensors=self.expert_mapping_tensors,
            remap_topk_mask=self.remap_topk_mask,
            dispatch_config=self.dispatch_config_decode,
            combine_config=self.combine_config_decode,
            program_config=self.program_config,
            mesh_device=self.mesh_device,
            mesh_config=self.mesh_config,
            ccl_manager=self.ccl_manager,
        )

    def forward_prefill(
        self,
        hidden_states: ttnn.Tensor,
        topk_expert_indices: ttnn.Tensor,
        topk_expert_weights: ttnn.Tensor,
        chunk_size: int,
    ) -> ttnn.Tensor:
        """
        Prefill forward pass.

        Args:
            hidden_states: Input [seq_per_device, 1, 1, hidden_size]
            topk_expert_indices: Expert indices [seq_per_device, 1, 1, k]
            topk_expert_weights: Routing weights [seq_per_device, 1, 1, k]
            chunk_size: Chunk size for long sequences

        Returns:
            Output [seq_per_device, 1, 1, hidden_size]
        """
        return prefill_forward_chunked(
            hidden_states=hidden_states,
            topk_expert_indices=topk_expert_indices,
            topk_expert_weights=topk_expert_weights,
            weights=self.weights,
            config=self.config,
            expert_mapping_tensors=self.expert_mapping_tensors,
            remap_topk_mask=self.remap_topk_mask,
            dispatch_config=self.dispatch_config_prefill,
            combine_config=self.combine_config_prefill,
            program_config=self.program_config,
            mesh_device=self.mesh_device,
            mesh_config=self.mesh_config,
            ccl_manager=self.ccl_manager,
            chunk_size=chunk_size,
        )

    @classmethod
    def from_hf_model(
        cls,
        mesh_device,
        hf_config,
        hf_state_dict: dict,
        layer_idx: int = None,
        program_config: ThroughputProgramConfig = None,
        weight_dtype=ttnn.bfloat4_b,
        tensor_cache_path: str = None,
    ) -> "ThroughputExperts":
        """
        Create ThroughputExperts from a HuggingFace model.

        Args:
            mesh_device: TTNN mesh device
            hf_config: HuggingFace model config
            hf_state_dict: HuggingFace state dict
            layer_idx: Optional layer index for extracting layer-specific weights
            program_config: Optional program config
            weight_dtype: Weight data type
            tensor_cache_path: Optional cache path

        Returns:
            ThroughputExperts instance
        """
        config = ThroughputExpertConfig.from_hf_config(hf_config, mesh_device)

        # Extract expert weights from HF state dict
        # HF typically stores as model.layers.{i}.mlp.experts.{e}.{proj}.weight
        expert_weights = {}
        prefix = f"model.layers.{layer_idx}.mlp." if layer_idx is not None else ""

        # Try to find expert weights
        import torch

        # Check for fused gate_up_proj format (GPT-OSS style)
        fused_key = f"{prefix}experts.gate_up_proj"
        if fused_key in hf_state_dict:
            expert_weights["gate_up_proj"] = hf_state_dict[fused_key]
            expert_weights["down_proj"] = hf_state_dict[f"{prefix}experts.down_proj"]
            if f"{prefix}experts.gate_up_proj_bias" in hf_state_dict:
                expert_weights["gate_up_proj_bias"] = hf_state_dict[f"{prefix}experts.gate_up_proj_bias"]
                expert_weights["down_proj_bias"] = hf_state_dict[f"{prefix}experts.down_proj_bias"]
        else:
            # Individual expert format
            w1_list, w2_list, w3_list = [], [], []
            for i in range(config.num_experts):
                w1_list.append(hf_state_dict[f"{prefix}experts.{i}.gate_proj.weight"].t())
                w2_list.append(hf_state_dict[f"{prefix}experts.{i}.down_proj.weight"].t())
                w3_list.append(hf_state_dict[f"{prefix}experts.{i}.up_proj.weight"].t())

            # Stack and reshape to expected format
            expert_weights["gate_proj"] = torch.stack(w1_list, dim=0)
            expert_weights["down_proj"] = torch.stack(w2_list, dim=0)
            expert_weights["up_proj"] = torch.stack(w3_list, dim=0)

        return cls(
            mesh_device=mesh_device,
            config=config,
            state_dict=expert_weights,
            program_config=program_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )
