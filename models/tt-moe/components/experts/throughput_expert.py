# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

"""
ThroughputExpert implementation for GPT-OSS MoE.

This module implements the ThroughputExpert class that uses sparse matmul operations
and clamped SwiGLU activation for GPT-OSS models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from loguru import logger

import ttnn

# No longer need separate all-to-all imports as operations are integrated


# ============================================================================
# Data Classes
# ============================================================================


@dataclass(frozen=True)
class ThroughputExpertWeights:
    """Container for throughput expert weight tensors - matches reference implementation."""

    w1: ttnn.Tensor  # Gate projection
    w2: ttnn.Tensor  # Down projection
    w3: ttnn.Tensor  # Up projection
    w1_bias: ttnn.Tensor  # Gate projection bias
    w2_bias: ttnn.Tensor  # Down projection bias
    w3_bias: ttnn.Tensor  # Up projection bias


# ============================================================================
# Utility Functions
# ============================================================================


def even_int_div(a: int, b: int) -> int:
    """Integer division that raises an error if b does not divide a without a remainder."""
    assert a % b == 0
    return a // b


# Compute kernel configuration for GPT-OSS
COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


# ============================================================================
# ThroughputExpert Class
# ============================================================================


class ThroughputExpert:
    """
    Throughput-optimized expert implementation for GPT-OSS MoE.

    Key features:
    - Uses sparse matmul operations for efficiency
    - Clamped SwiGLU activation with configurable limits
    - Token remapping for sparsity patterns
    - All-reduce after combine for tensor parallel
    """

    @classmethod
    def _get_num_experts_per_device(cls, config: Any, mesh_device: ttnn.MeshDevice) -> int:
        """Calculate the number of experts per device."""
        if hasattr(config, "num_experts"):
            num_experts = config.num_experts
        else:
            num_experts = config.get("num_experts", 128)  # GPT-OSS default
        return even_int_div(num_experts, mesh_device.get_num_devices())

    @classmethod
    def is_device_supported(cls, mesh_device: ttnn.MeshDevice) -> bool:
        """
        Check if the device configuration is supported.

        Args:
            mesh_device: The mesh device to check.

        Returns:
            True if the device is supported, False otherwise.
        """
        # GPT-OSS typically runs on 32 devices (Galaxy)
        return mesh_device.get_num_devices() == 32

    @classmethod
    def _apply_clamped_swiglu(
        cls,
        gate: ttnn.Tensor,
        up: ttnn.Tensor,
        alpha: float = 1.702,
        limit: float = 7.0,
        memory_config: Optional[ttnn.MemoryConfig] = None,
    ) -> ttnn.Tensor:
        """
        Apply clamped SwiGLU activation.

        Formula: (up + 1) * (gate * sigmoid(gate * alpha))
        With clamping: gate clamped to (None, limit], up clamped to [-limit, limit]

        Args:
            gate: Gate projection output
            up: Up projection output
            alpha: Sigmoid scaling factor (default 1.702 for GPT-OSS)
            limit: Clamping limit (default 7.0)
            memory_config: Memory configuration for operations

        Returns:
            Activated tensor
        """
        # Clamp gate (max only) - matches reference implementation
        gate_clamped = ttnn.clamp(gate, min=None, max=limit)
        ttnn.deallocate(gate)

        # Clamp up (both min and max)
        up_clamped = ttnn.clamp(up, min=-limit, max=limit)
        ttnn.deallocate(up)

        # Compute gate_alpha = gate * alpha
        gate_alpha = ttnn.mul(gate_clamped, alpha)

        # Compute gate_sigmoid = sigmoid(gate_alpha)
        gate_sigmoid = ttnn.sigmoid(gate_alpha)
        ttnn.deallocate(gate_alpha)

        # Compute glu = gate * gate_sigmoid
        glu = ttnn.mul(gate_clamped, gate_sigmoid, memory_config=memory_config)
        ttnn.deallocate(gate_clamped)
        ttnn.deallocate(gate_sigmoid)

        # Add 1 to up: up = up + 1
        up_clamped = ttnn.add(up_clamped, 1.0, output_tensor=up_clamped)

        # Multiply: result = up * glu
        result = ttnn.mul(up_clamped, glu, memory_config=memory_config)
        ttnn.deallocate(up_clamped)
        ttnn.deallocate(glu)

        return result

    @classmethod
    def _create_model_config(cls, config: Any, mesh_device: ttnn.MeshDevice, mode: str) -> Dict:
        """
        Create model configuration for ThroughputExpert.

        Args:
            config: Configuration object/dict
            mesh_device: Mesh device
            mode: Mode (decode/prefill)

        Returns:
            Model configuration dictionary
        """
        # Extract parameters
        if hasattr(config, "__dict__"):
            hidden_size = getattr(config, "hidden_size", 2880)
            intermediate_size = getattr(config, "intermediate_size", 360)
            swiglu_alpha = getattr(config, "swiglu_alpha", 1.702)
            swiglu_limit = getattr(config, "swiglu_limit", 7.0)
            sparsity_block_size = getattr(config, "sparsity_block_size", 32)
        else:
            hidden_size = config.get("hidden_size", 2880)
            intermediate_size = config.get("intermediate_size", 360)
            swiglu_alpha = config.get("swiglu_alpha", 1.702)
            swiglu_limit = config.get("swiglu_limit", 7.0)
            sparsity_block_size = config.get("sparsity_block_size", 32)

        # Memory configuration
        if mode == "decode":
            memory_config = ttnn.L1_MEMORY_CONFIG
        else:
            memory_config = ttnn.DRAM_MEMORY_CONFIG

        return {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "memory_config": memory_config,
            "compute_kernel_config": COMPUTE_KERNEL_CONFIG_LOFI,
            "swiglu_alpha": swiglu_alpha,
            "swiglu_limit": swiglu_limit,
            "sparsity_block_size": sparsity_block_size,
            "num_experts_per_device": cls._get_num_experts_per_device(config, mesh_device),
        }

    @classmethod
    def convert_weights(
        cls, config: Any, state_dicts: Sequence[Dict], weight_cache_dir: Path, mesh_device: ttnn.MeshDevice
    ) -> ThroughputExpertWeights:
        """
        Convert and prepare weights for ThroughputExpert.

        Args:
            config: Configuration object
            state_dicts: List of state dictionaries containing weights
            weight_cache_dir: Directory to cache converted weights
            mesh_device: Mesh device

        Returns:
            ThroughputExpertWeights dataclass with converted weights
        """
        logger.info("Converting ThroughputExpert weights for GPT-OSS")

        # Get configuration parameters
        if hasattr(config, "__dict__"):
            hidden_size = getattr(config, "hidden_size", 2880)
            intermediate_size = getattr(config, "intermediate_size", 360)
            num_experts = getattr(config, "num_experts", 128)
        else:
            hidden_size = config.get("hidden_size", 2880)
            intermediate_size = config.get("intermediate_size", 360)
            num_experts = config.get("num_experts", 128)

        num_devices = mesh_device.get_num_devices()
        num_experts_per_device = num_experts // num_devices

        # Extract weights from state dict
        state_dict = state_dicts[0] if state_dicts else {}

        # Collect all expert weights into tensors
        # Shape: [1, num_experts, hidden_size, intermediate_size] for w1/w3
        # Shape: [1, num_experts, intermediate_size, hidden_size] for w2
        w1_all = []
        w3_all = []
        w2_all = []
        w1_bias_all = []
        w3_bias_all = []
        w2_bias_all = []

        # Check for fused gate_up projection first
        has_fused_gate_up = any(k.startswith("gate_up_proj") for k in state_dict.keys())

        if has_fused_gate_up:
            # Handle fused gate/up projection weights
            gate_up = state_dict.get("gate_up_proj")
            if gate_up is not None:
                # Unfuse: gate_up shape is [num_experts, hidden_size, 2 * intermediate_size]
                # Extract interleaved indices: even for gate (w1), odd for up (w3)
                w1 = gate_up[..., ::2].reshape(1, num_experts, hidden_size, intermediate_size)
                w3 = gate_up[..., 1::2].reshape(1, num_experts, hidden_size, intermediate_size)

                # Get biases if present
                gate_up_bias = state_dict.get("gate_up_proj_bias")
                if gate_up_bias is not None:
                    w1_bias = gate_up_bias[..., ::2].reshape(1, num_experts, 1, intermediate_size)
                    w3_bias = gate_up_bias[..., 1::2].reshape(1, num_experts, 1, intermediate_size)
                else:
                    w1_bias = torch.zeros(1, num_experts, 1, intermediate_size)
                    w3_bias = torch.zeros(1, num_experts, 1, intermediate_size)
            else:
                raise ValueError("Fused gate_up_proj expected but not found in state dict")
        else:
            # Handle separate gate and up projections
            for expert_idx in range(num_experts):
                # Try different key patterns for each expert
                gate_keys = [
                    f"experts.{expert_idx}.gate_proj.weight",
                    f"expert_{expert_idx}.gate_proj.weight",
                    f"mlp.experts.{expert_idx}.gate_proj.weight",
                ]
                up_keys = [
                    f"experts.{expert_idx}.up_proj.weight",
                    f"expert_{expert_idx}.up_proj.weight",
                    f"mlp.experts.{expert_idx}.up_proj.weight",
                ]
                down_keys = [
                    f"experts.{expert_idx}.down_proj.weight",
                    f"expert_{expert_idx}.down_proj.weight",
                    f"mlp.experts.{expert_idx}.down_proj.weight",
                ]

                # Find gate weight
                gate_weight = None
                for key in gate_keys:
                    if key in state_dict:
                        gate_weight = state_dict[key]
                        break
                if gate_weight is None:
                    logger.warning(f"Gate weight not found for expert {expert_idx}")
                    gate_weight = torch.zeros(hidden_size, intermediate_size)
                w1_all.append(gate_weight)

                # Find up weight
                up_weight = None
                for key in up_keys:
                    if key in state_dict:
                        up_weight = state_dict[key]
                        break
                if up_weight is None:
                    logger.warning(f"Up weight not found for expert {expert_idx}")
                    up_weight = torch.zeros(hidden_size, intermediate_size)
                w3_all.append(up_weight)

                # Find down weight
                down_weight = None
                for key in down_keys:
                    if key in state_dict:
                        down_weight = state_dict[key]
                        break
                if down_weight is None:
                    logger.warning(f"Down weight not found for expert {expert_idx}")
                    down_weight = torch.zeros(intermediate_size, hidden_size)
                w2_all.append(down_weight)

                # Biases (usually not present in GPT-OSS, but handle them)
                w1_bias_all.append(torch.zeros(1, intermediate_size))
                w3_bias_all.append(torch.zeros(1, intermediate_size))
                w2_bias_all.append(torch.zeros(1, hidden_size))

            # Stack all weights
            w1 = torch.stack(w1_all, dim=0).reshape(1, num_experts, hidden_size, intermediate_size)
            w3 = torch.stack(w3_all, dim=0).reshape(1, num_experts, hidden_size, intermediate_size)
            w1_bias = torch.stack(w1_bias_all, dim=0).reshape(1, num_experts, 1, intermediate_size)
            w3_bias = torch.stack(w3_bias_all, dim=0).reshape(1, num_experts, 1, intermediate_size)

        # Down projection is handled separately in both cases
        if not has_fused_gate_up:
            w2 = torch.stack(w2_all, dim=0).reshape(1, num_experts, intermediate_size, hidden_size)
            w2_bias = torch.stack(w2_bias_all, dim=0).reshape(1, num_experts, 1, hidden_size)
        else:
            w2 = state_dict.get("down_proj", torch.zeros(1, num_experts, intermediate_size, hidden_size))
            w2_bias = state_dict.get("down_proj_bias", torch.zeros(1, num_experts, 1, hidden_size))
            if w2.dim() == 3:
                w2 = w2.unsqueeze(0)
            # Handle different bias tensor dimensions
            if w2_bias.dim() == 2:
                # Shape: (num_experts, hidden_size) -> (1, num_experts, 1, hidden_size)
                w2_bias = w2_bias.unsqueeze(0).unsqueeze(2)
            elif w2_bias.dim() == 3:
                # Shape: (1, num_experts, hidden_size) -> (1, num_experts, 1, hidden_size)
                w2_bias = w2_bias.unsqueeze(2)

        # Convert to ttnn tensors and shard across devices
        # Each device gets num_experts_per_device experts
        # Matching reference implementation's approach

        # Shard and convert w1 (gate projection)
        w1_tt = ttnn.as_tensor(
            w1.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),  # Shard expert dimension
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Shard and convert w3 (up projection)
        w3_tt = ttnn.as_tensor(
            w3.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),  # Shard expert dimension
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Shard and convert w2 (down projection)
        w2_tt = ttnn.as_tensor(
            w2.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),  # Shard expert dimension
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Shard and convert biases - reshape for proper broadcasting
        # Reshape biases to have expert dimension for sharding
        w1_bias_tt = ttnn.as_tensor(
            w1_bias.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-3),  # Expert dimension after reshape
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        w3_bias_tt = ttnn.as_tensor(
            w3_bias.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-3),  # Expert dimension after reshape
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        w2_bias_tt = ttnn.as_tensor(
            w2_bias.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-3),  # Expert dimension after reshape
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return ThroughputExpertWeights(
            w1=w1_tt, w2=w2_tt, w3=w3_tt, w1_bias=w1_bias_tt, w2_bias=w2_bias_tt, w3_bias=w3_bias_tt
        )

    @classmethod
    def decode_model_config(cls, config: Any, mesh_device: ttnn.MeshDevice) -> Dict:
        """
        Create decode configuration for ThroughputExpert.

        Args:
            config: Configuration object
            mesh_device: Mesh device

        Returns:
            Decode configuration dictionary
        """
        return cls._create_model_config(config, mesh_device, "decode")

    @classmethod
    def prefill_model_config(cls, config: Any, mesh_device: ttnn.MeshDevice) -> Dict:
        """
        Create prefill configuration for ThroughputExpert.

        Args:
            config: Configuration object
            mesh_device: Mesh device

        Returns:
            Prefill configuration dictionary
        """
        return cls._create_model_config(config, mesh_device, "prefill")

    @classmethod
    def forward_decode(
        cls,
        hidden_states: ttnn.Tensor,
        topk_expert_indices: ttnn.Tensor,
        topk_expert_weights: ttnn.Tensor,
        config: Dict,
        expert_mapping_tensors: ttnn.Tensor,
        remap_topk_mask: Optional[ttnn.Tensor],
        mesh_device: ttnn.MeshDevice,
    ) -> ttnn.Tensor:
        """
        Forward pass for ThroughputExpert in decode mode.

        This method implements the full pipeline:
        1. All-to-all dispatch
        2. MOE expert token remap for sparsity pattern
        3. Sparse matmul for gate/up projections
        4. Clamped SwiGLU activation
        5. Sparse matmul for down projection
        6. All-to-all combine
        7. Apply routing weights
        8. All-reduce (if needed)

        Args:
            hidden_states: Original input hidden states [batch, 1, seq, H]
            topk_expert_indices: Expert routing indices [batch, 1, seq, K]
            topk_expert_weights: Routing weights [K, 1, seq*batch, H]
            config: Model configuration dict including "weights" key with ThroughputExpertWeights
            expert_mapping_tensors: Expert to device mapping [1, 1, E, D]
            remap_topk_mask: Optional remap mask [1, dispatch_rows, 1, E]
            mesh_device: Mesh device for operations

        Returns:
            Expert output tensor [1, 1, seq*batch, H]
        """
        memory_config = config.get("memory_config", ttnn.L1_MEMORY_CONFIG)
        compute_kernel_config = config.get("compute_kernel_config", COMPUTE_KERNEL_CONFIG_LOFI)
        swiglu_alpha = config.get("swiglu_alpha", 1.702)
        swiglu_limit = config.get("swiglu_limit", 7.0)
        sparsity_block_size = config.get("sparsity_block_size", 32)
        hidden_size = config.get("hidden_size", 2880)
        intermediate_size = config.get("intermediate_size", 360)
        num_experts_per_device = config.get("num_experts_per_device", 4)
        num_experts = config.get("num_experts", 128)
        num_experts_per_tok = config.get("num_experts_per_tok", 4)
        cluster_axis = config.get("cluster_axis", 0)
        dispatch_topology = config.get("dispatch_topology", "Linear")
        combine_topology = config.get("combine_topology", "Linear")

        # Get sequence length and batch size
        batch_size, _, seq_len, _ = hidden_states.shape
        tokens_per_device = batch_size * seq_len

        # Get total tokens after dispatch (scaled by number of devices)
        num_dispatch_devices = mesh_device.shape[cluster_axis] if hasattr(mesh_device, "shape") else 1
        total_tokens = tokens_per_device * num_dispatch_devices

        # ==========================================================================
        # STEP 1: PREPARE INPUTS FOR ALL_TO_ALL_DISPATCH
        # ==========================================================================
        # Convert to ROW_MAJOR layout as required by all-to-all dispatch
        hidden_rm = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_rm = ttnn.reshape(hidden_rm, shape=(1, 1, tokens_per_device, hidden_size))

        # Expert indices: [1, 1, tokens_per_device, K]
        # Convert to ROW_MAJOR first before typecast if needed (typecast requires ROW_MAJOR with padded dimension)
        topk_indices_rm = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
        topk_indices_rm = ttnn.reshape(topk_indices_rm, shape=(1, 1, tokens_per_device, num_experts_per_tok))

        # Ensure indices are uint16 (required for all_to_all_dispatch)
        # Typecast after converting to ROW_MAJOR and only if not already uint16
        if topk_indices_rm.dtype != ttnn.uint16:
            # Pad the last dimension if needed for typecast
            if num_experts_per_tok < 32:
                # Pad to 32 for typecast requirement (padding = [[top, bottom], [left, right]] for each dimension)
                padding_amount = 32 - num_experts_per_tok
                padded_indices = ttnn.pad(
                    topk_indices_rm, padding=[[0, 0], [0, 0], [0, 0], [0, padding_amount]], value=0.0
                )
                padded_indices = ttnn.typecast(padded_indices, dtype=ttnn.uint16)
                # Slice back to original size
                topk_indices_rm = ttnn.slice(
                    padded_indices, [0, 0, 0, 0], [1, 1, tokens_per_device, num_experts_per_tok]
                )
                ttnn.deallocate(padded_indices)
            else:
                topk_indices_rm = ttnn.typecast(topk_indices_rm, dtype=ttnn.uint16)

        # ==========================================================================
        # STEP 2: ALL_TO_ALL_DISPATCH - Route tokens to expert devices
        # ==========================================================================
        # Convert topology string to enum if needed
        if isinstance(dispatch_topology, str):
            dispatch_topology = getattr(ttnn.Topology, dispatch_topology)
        if isinstance(combine_topology, str):
            combine_topology = getattr(ttnn.Topology, combine_topology)

        dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(
            hidden_rm,
            topk_indices_rm,
            expert_mapping_tensors,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
            output_concat_dim=2,  # Concatenate on token dimension
            topology=dispatch_topology,
        )
        ttnn.deallocate(hidden_rm)
        ttnn.deallocate(topk_indices_rm)

        # ==========================================================================
        # STEP 3: MOE_EXPERT_TOKEN_REMAP - Create sparsity pattern
        # ==========================================================================
        # Create remap mask if not provided
        if remap_topk_mask is None:
            num_dispatch_rows = mesh_device.shape[cluster_axis] if cluster_axis == 0 else 1
            remap_topk_mask = ttnn.ones(
                (1, num_dispatch_rows, 1, num_experts),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Broadcast remap_topk_mask across token dimension
        # remap_topk_mask: [1, dispatch_rows, 1, num_experts]
        # -> repeat to [1, dispatch_rows, tokens_per_device, num_experts]
        # -> reshape to [1, 1, total_tokens, num_experts]
        remap_mask = ttnn.repeat(remap_topk_mask, ttnn.Shape((1, 1, tokens_per_device, 1)))
        remap_mask = ttnn.reshape(remap_mask, (1, 1, total_tokens, num_experts))

        # Use moe_expert_token_remap to create sparsity pattern
        _, sparsity = ttnn.moe_expert_token_remap(
            remap_mask, expert_mapping_tensors, dispatch_metadata, reduction_size=sparsity_block_size
        )
        ttnn.deallocate(remap_mask)

        # ==========================================================================
        # STEP 4: PREPARE DISPATCH OUTPUT FOR EXPERT COMPUTATION
        # ==========================================================================
        # Reshape dispatch output for sparse matmul
        post_dispatch = ttnn.reshape(dispatch_output, shape=(1, 1, total_tokens, hidden_size))
        post_dispatch_rm = post_dispatch
        post_dispatch = ttnn.to_layout(post_dispatch, ttnn.TILE_LAYOUT)
        ttnn.deallocate(post_dispatch_rm)  # This deallocates dispatch_output via the view

        # Reshape to sparse block format for matmul
        num_sparse_blocks = total_tokens // sparsity_block_size
        expert_input = ttnn.reshape(post_dispatch, shape=(1, num_sparse_blocks, sparsity_block_size, hidden_size))

        # ==========================================================================
        # STEP 5: EXPERT COMPUTATION - Gate/Up/Down projections with sparse matmul
        # ==========================================================================
        # Get expert weights from config
        # The weights should be a ThroughputExpertWeights object
        weights = config.get("weights")
        if weights is None:
            # Try individual keys for backward compatibility
            w1 = config.get("w1")
            w2 = config.get("w2")
            w3 = config.get("w3")
            w1_bias = config.get("w1_bias")
            w2_bias = config.get("w2_bias")
            w3_bias = config.get("w3_bias")

            if w1 is None or w2 is None or w3 is None:
                raise ValueError(
                    "ThroughputExpert requires weights to be provided. "
                    "Please load real GPT-OSS weights using convert_weights method."
                )
        else:
            # Extract from ThroughputExpertWeights object
            w1 = weights.w1
            w2 = weights.w2
            w3 = weights.w3
            w1_bias = weights.w1_bias
            w2_bias = weights.w2_bias
            w3_bias = weights.w3_bias

        # Get program configs - create default if not provided
        # Use EXACT configs from reference that work!
        gate_up_program_config = config.get("gate_up_program_config")
        down_program_config = config.get("down_program_config")

        if gate_up_program_config is None:
            # Use configuration from reference implementation
            import math

            core_x, core_y = 5, 9  # From reference (45 cores)
            n_tiles = math.ceil(intermediate_size / ttnn.TILE_SIZE)
            per_core_N = max(1, n_tiles // (core_x * core_y))

            # in0_block_w should divide K tiles
            # For GPT-OSS: hidden_size=2880 / 32 = 90 tiles
            # 10 divides 90 perfectly
            in0_block_w = 10  # From reference

            gate_up_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=per_core_N,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )

        if down_program_config is None:
            # Use configuration from reference implementation
            import math

            core_x, core_y = 5, 9  # From reference (45 cores)
            n_tiles = math.ceil(hidden_size / ttnn.TILE_SIZE)
            per_core_N = max(1, n_tiles // (core_x * core_y))

            # in0_block_w should divide K tiles
            # For down: intermediate_size=2880 / 32 = 90 tiles (after SwiGLU)
            # 10 divides 90 perfectly
            in0_block_w = 10  # From reference

            down_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=per_core_N,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )

        # Use sparse matmul operations exactly as in reference
        output_tile = ttnn.Tile([sparsity_block_size, ttnn.TILE_SIZE])

        # Gate projection with sparse matmul
        w1_out = ttnn.sparse_matmul(
            expert_input,
            w1,
            sparsity=sparsity,
            memory_config=memory_config,
            program_config=gate_up_program_config,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            output_tile=output_tile,
        )

        # Add bias if available - TEMPORARILY DISABLED for debugging
        if False and w1_bias is not None:
            # Reshape bias for broadcasting with sparse matmul output
            # w1_out shape: [1, num_blocks, 1, num_experts_per_device, block_size, intermediate]
            # w1_bias shape after sharding: [1, num_experts_per_device, 1, intermediate]
            # Need to reshape to: [1, 1, 1, num_experts_per_device, 1, intermediate]
            w1_bias_reshaped = ttnn.reshape(w1_bias, (1, 1, 1, num_experts_per_device, 1, intermediate_size))
            w1_out = ttnn.add(w1_out, w1_bias_reshaped, output_tensor=w1_out)
            ttnn.deallocate(w1_bias_reshaped)

        # Up projection with sparse matmul
        w3_out = ttnn.sparse_matmul(
            expert_input,
            w3,
            sparsity=sparsity,
            memory_config=memory_config,
            program_config=gate_up_program_config,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            output_tile=output_tile,
        )
        ttnn.deallocate(expert_input)

        # Add bias if available - TEMPORARILY DISABLED for debugging
        if False and w3_bias is not None:
            # Reshape bias for broadcasting with sparse matmul output
            # w3_out shape: [1, num_blocks, 1, num_experts_per_device, block_size, intermediate]
            # w3_bias shape after sharding: [1, num_experts_per_device, 1, intermediate]
            # Need to reshape to: [1, 1, 1, num_experts_per_device, 1, intermediate]
            w3_bias_reshaped = ttnn.reshape(w3_bias, (1, 1, 1, num_experts_per_device, 1, intermediate_size))
            w3_out = ttnn.add(w3_out, w3_bias_reshaped, output_tensor=w3_out)
            ttnn.deallocate(w3_bias_reshaped)

        # Apply clamped SwiGLU activation
        activated = cls._apply_clamped_swiglu(
            w1_out, w3_out, alpha=swiglu_alpha, limit=swiglu_limit, memory_config=memory_config
        )

        # Squeeze batch dimensions for down projection
        activated = ttnn.squeeze(activated, 0)
        activated = ttnn.squeeze(activated, 1)

        # Down projection with sparse matmul
        expert_output_sparse = ttnn.sparse_matmul(
            activated,
            w2,
            sparsity=sparsity,
            memory_config=memory_config,
            program_config=down_program_config,
            is_input_a_sparse=True,  # Input is sparse for down projection
            is_input_b_sparse=False,
            output_tile=output_tile,
        )
        ttnn.deallocate(activated)
        ttnn.deallocate(sparsity)

        # Add bias if available - TEMPORARILY DISABLED for debugging
        if False and w2_bias is not None:
            # After squeeze, output shape is [num_blocks, num_experts_per_device, block_size, hidden_size]
            # w2_bias shape after sharding: [1, num_experts_per_device, 1, hidden_size]
            # Need to reshape to: [1, num_experts_per_device, 1, hidden_size] (already correct)
            # But might need unsqueeze for the blocks dimension
            expert_output_sparse = ttnn.add(expert_output_sparse, w2_bias)

        # ==========================================================================
        # STEP 6: PREPARE EXPERT OUTPUT FOR ALL_TO_ALL_COMBINE
        # ==========================================================================
        # Permute from [blocks, experts, block_size, H] to [experts, blocks, block_size, H]
        expert_output = ttnn.permute(expert_output_sparse, (1, 0, 2, 3))
        ttnn.deallocate(expert_output_sparse)

        # Reshape to [experts_per_device, 1, total_tokens, H]
        expert_output = ttnn.reshape(expert_output, shape=(num_experts_per_device, 1, total_tokens, hidden_size))

        # Convert to ROW_MAJOR for all_to_all_combine
        expert_output_tiled = expert_output
        expert_output = ttnn.to_layout(expert_output, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(expert_output_tiled)

        # ==========================================================================
        # STEP 6: ALL_TO_ALL_COMBINE - Route expert outputs back to token positions
        # ==========================================================================
        combine_output = ttnn.all_to_all_combine(
            expert_output,
            dispatch_metadata,
            expert_mapping_tensors,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
            output_shard_dim=2,  # Shard on token dimension
            topology=combine_topology,
        )
        ttnn.deallocate(expert_output)
        ttnn.deallocate(dispatch_metadata)

        # ==========================================================================
        # STEP 7: APPLY ROUTING WEIGHTS AND REDUCE ACROSS EXPERTS
        # ==========================================================================
        # Combine output already has tokens on dim -2: [K, 1, tokens_per_device, H]
        post_combine = ttnn.to_layout(combine_output, ttnn.TILE_LAYOUT)
        ttnn.deallocate(combine_output)

        # Prepare routing weights for broadcasting:
        # topk_expert_weights is [K, 1, tokens_per_device, H] (already reshaped by moe_block)
        # We need [K, 1, tokens_per_device, 1] so it can broadcast across hidden_size
        topk_weights_rm = ttnn.to_layout(topk_expert_weights, ttnn.ROW_MAJOR_LAYOUT)
        # Slice to get just one value per token (weights are repeated across hidden dim)
        topk_weights_rm = ttnn.slice(topk_weights_rm, [0, 0, 0, 0], [num_experts_per_tok, 1, tokens_per_device, 1])
        topk_weights_reshaped = ttnn.to_layout(topk_weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(topk_weights_rm)

        # Weighted sum: sum_k(expert_output_k * routing_weight_k)
        weighted_output = ttnn.mul(post_combine, topk_weights_reshaped, memory_config=memory_config)
        ttnn.deallocate(post_combine)
        ttnn.deallocate(topk_weights_reshaped)

        # Sum across K experts (first dimension)
        output = ttnn.sum(weighted_output, dim=0, keepdim=True)
        ttnn.deallocate(weighted_output)

        # ==========================================================================
        # STEP 8: ALL-REDUCE ACROSS COLUMNS (if needed)
        # ==========================================================================
        # All-reduce across columns (cluster_axis=1) to aggregate expert outputs
        # This is needed when experts are sharded across multiple columns
        if mesh_device.shape[1] > 1:  # If we have multiple columns
            output_all_reduced = ttnn.all_reduce(
                output,
                num_links=1,
                topology=ttnn.Topology.Linear,  # Use Linear instead of Ring
                cluster_axis=1,  # Sum across columns
                memory_config=memory_config,
            )
            ttnn.deallocate(output)

            # The all_reduce should create a replicated tensor, but to be safe,
            # let's make sure it's in the right format for torch conversion
            return output_all_reduced
        else:
            # No all-reduce needed for single column
            return output

    @classmethod
    def forward_prefill(
        cls,
        hidden_states: ttnn.Tensor,
        topk_expert_indices: ttnn.Tensor,
        topk_expert_weights: ttnn.Tensor,
        config: Dict,
        expert_mapping_tensors: ttnn.Tensor,
        remap_topk_mask: Optional[ttnn.Tensor],
        mesh_device: ttnn.MeshDevice,
    ) -> ttnn.Tensor:
        """
        Forward pass for ThroughputExpert in prefill mode.

        Similar to decode but with different memory configurations and
        potentially different chunk sizes.

        Args:
            hidden_states: Original input hidden states [batch, 1, seq, H]
            topk_expert_indices: Expert routing indices [batch, 1, seq, K]
            topk_expert_weights: Routing weights [K, 1, seq*batch, H]
            config: Model configuration dict including "weights" key with ThroughputExpertWeights
            expert_mapping_tensors: Expert to device mapping [1, 1, E, D]
            remap_topk_mask: Optional remap mask [1, dispatch_rows, 1, E]
            mesh_device: Mesh device for operations

        Returns:
            Expert output tensor [1, 1, seq*batch, H]
        """
        # Prefill uses similar logic to decode but with DRAM memory
        # Update config for prefill
        prefill_config = config.copy()
        prefill_config["memory_config"] = ttnn.DRAM_MEMORY_CONFIG

        return cls.forward_decode(
            hidden_states,
            topk_expert_indices,
            topk_expert_weights,
            prefill_config,
            expert_mapping_tensors,
            remap_topk_mask,
            mesh_device,
        )

    @classmethod
    def forward(
        cls,
        hidden_states: ttnn.Tensor,
        topk_expert_indices: ttnn.Tensor,
        topk_expert_weights: ttnn.Tensor,
        config: Dict,
        expert_mapping_tensors: ttnn.Tensor,
        remap_topk_mask: Optional[ttnn.Tensor],
        mesh_device: ttnn.MeshDevice,
        mode: str = "decode",
    ) -> ttnn.Tensor:
        """
        Main forward pass that dispatches to decode or prefill.

        Args:
            hidden_states: Original input hidden states [batch, 1, seq, H]
            topk_expert_indices: Expert routing indices [batch, 1, seq, K]
            topk_expert_weights: Routing weights [K, 1, seq*batch, H]
            config: Model configuration dict including "weights" key with ThroughputExpertWeights
            expert_mapping_tensors: Expert to device mapping [1, 1, E, D]
            remap_topk_mask: Optional remap mask [1, dispatch_rows, 1, E]
            mesh_device: Mesh device for operations
            mode: Mode (decode/prefill)

        Returns:
            Expert output tensor [1, 1, seq*batch, H]
        """
        if mode == "decode":
            return cls.forward_decode(
                hidden_states,
                topk_expert_indices,
                topk_expert_weights,
                config,
                expert_mapping_tensors,
                remap_topk_mask,
                mesh_device,
            )
        elif mode == "prefill":
            return cls.forward_prefill(
                hidden_states,
                topk_expert_indices,
                topk_expert_weights,
                config,
                expert_mapping_tensors,
                remap_topk_mask,
                mesh_device,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
