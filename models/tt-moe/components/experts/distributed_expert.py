# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Distributed expert implementation for MoE.

This is a direct port of models/demos/deepseek_v3/tt/experts.py
with all utilities copied directly into this file.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from loguru import logger

import ttnn

# ============================================================================
# Utility Functions (copied from models/demos/deepseek_v3/utils/config_helpers.py)
# ============================================================================


def even_int_div(a: int, b: int) -> int:
    """Integer division that raises an error if b does not divide a without a remainder."""
    assert a % b == 0
    return a // b


def dequantize(tensor: torch.Tensor, inv_scale: torch.Tensor, block_shape: Sequence[int]) -> torch.Tensor:
    """Dequantize a pytorch tensor using the provided scale."""
    assert tensor.ndim == inv_scale.ndim
    assert len(block_shape) == tensor.ndim and all(
        inv_scale.shape[i] * block_shape[i] >= tensor.shape[i] for i in range(tensor.ndim)
    )
    for i, block_dim in enumerate(block_shape):
        inv_scale = inv_scale.repeat_interleave(block_dim, dim=i)

    # Convert to float32 for multiplication
    tensor_float = tensor.float()
    scale_float = inv_scale[tuple(slice(0, s) for s in tensor.shape)].float()
    result = tensor_float * scale_float

    # Clamp values to prevent overflow when converting to bfloat16
    # bfloat16 range is approximately ±3.4e38
    result = torch.clamp(result, min=-3.38e38, max=3.38e38)

    del inv_scale
    return result.to(torch.bfloat16)  # Return as bfloat16


# Compute kernel configuration (copied from config_helpers.py)
COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


# ============================================================================
# DistributedExpert Class (Direct port of Experts from experts.py)
# ============================================================================


class DistributedExpert:
    """
    Distributed expert implementation for MoE.

    This is a direct port of the DeepSeek Experts class, handling
    all experts' weights together and processing them in batch.
    Matches the exact interface of models/demos/deepseek_v3/tt/experts.py
    """

    @classmethod
    def _get_num_experts_per_device(cls, hf_config: Any, mesh_device: ttnn.MeshDevice) -> int:
        """Calculate the number of experts per device based on the total number of experts and the device shape."""
        # Support both HF config objects and plain dicts
        if hasattr(hf_config, "n_routed_experts"):
            n_routed_experts = hf_config.n_routed_experts
        else:
            n_routed_experts = hf_config.get("n_routed_experts", hf_config.get("num_experts", 256))
        return even_int_div(n_routed_experts, mesh_device.get_num_devices())

    @classmethod
    def is_device_supported(cls, mesh_device: ttnn.MeshDevice) -> bool:
        """
        As we only support 1D tensor parallelism, we only support 1D mesh devices.

        Args:
            mesh_device: The mesh device to check.

        Returns:
            True if the device is supported, False otherwise.
        """
        return mesh_device.shape[1] == 8

    @classmethod
    def _create_model_config(cls, hf_config: Any, mesh_device: ttnn.MeshDevice, mode: str) -> Dict:
        """Create model configuration for decode or prefill mode."""
        num_experts_per_device = cls._get_num_experts_per_device(hf_config, mesh_device)

        # Memory configurations based on mode
        if mode == "decode":
            input_memory_config = ttnn.L1_MEMORY_CONFIG
            output_memory_config = ttnn.L1_MEMORY_CONFIG
        else:
            input_memory_config = ttnn.DRAM_MEMORY_CONFIG
            output_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Create config dictionary that includes weight references
        config = {
            "mesh_device": mesh_device,
            "input_memory_config": input_memory_config,
            "output_memory_config": output_memory_config,
            "num_experts_per_device": num_experts_per_device,
        }

        # Add linear configs with proper structure
        for name in ["w1_experts", "w2_experts", "w3_experts"]:
            config[name] = {
                "memory_config": output_memory_config,
                "compute_kernel_config": COMPUTE_KERNEL_CONFIG_LOFI,
            }

        # Add mul config for activation
        config["mul_experts"] = {
            "memory_config": output_memory_config,
            "input_tensor_a_activations": [ttnn.UnaryOpType.SILU],
        }

        return config

    @classmethod
    def decode_model_config(cls, hf_config: Any, mesh_device: ttnn.MeshDevice) -> Dict:
        """Generate decode configuration for this module."""
        return cls._create_model_config(hf_config, mesh_device, "decode")

    @classmethod
    def prefill_model_config(cls, hf_config: Any, mesh_device: ttnn.MeshDevice) -> Dict:
        """Generate prefill configuration for this module."""
        return cls._create_model_config(hf_config, mesh_device, "prefill")

    @classmethod
    def create_state(cls, hf_config: Any, mesh_device: ttnn.MeshDevice) -> Dict:
        """Create state (empty for experts)."""
        return {}

    @classmethod
    def convert_weights(
        cls,
        hf_config: Any,
        state_dicts: Tuple[Optional[Dict[str, torch.Tensor]], ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> Dict:
        """
        Convert and prepare weights for the distributed experts.

        This method loads weights and returns them in the format expected by the forward pass.
        For testing, we return the weights directly as TTNN tensors.
        """
        # Support both HF config objects and plain dicts
        if hasattr(hf_config, "n_routed_experts"):
            n_routed_experts = hf_config.n_routed_experts
            use_quantized = hasattr(hf_config, "quantization_config")
            weight_block_size = (
                hf_config.quantization_config.get("weight_block_size", [1, 1]) if use_quantized else [1, 1]
            )
        else:
            n_routed_experts = hf_config.get("n_routed_experts", hf_config.get("num_experts", 256))
            use_quantized = hf_config.get("use_quantized_weights", False)
            weight_block_size = hf_config.get("weight_block_size", [1, 1])

        assert n_routed_experts % mesh_device.get_num_devices() == 0, (
            f"Number of experts ({n_routed_experts}) must be divisible by the number of devices "
            f"({mesh_device.get_num_devices()})"
        )

        (state_dict,) = state_dicts
        assert state_dict is not None

        weight_config = {}

        for hf_name, ttnn_name in [
            ("gate_proj", "w1_experts"),
            ("down_proj", "w2_experts"),
            ("up_proj", "w3_experts"),
        ]:
            # Stack weights from all experts
            weight_list = []
            scale_list = []

            for expert_id in range(n_routed_experts):
                weight_key = f"experts.{expert_id}.{hf_name}.weight"
                if weight_key in state_dict:
                    weight_list.append(state_dict[weight_key])

                    # Check for quantization scale
                    scale_key = f"experts.{expert_id}.{hf_name}.weight_scale_inv"
                    if use_quantized and scale_key in state_dict:
                        scale_list.append(state_dict[scale_key])

            if weight_list:
                stacked_weights = torch.stack(weight_list)

                # Apply dequantization if needed
                if use_quantized and scale_list:
                    stacked_scales = torch.stack(scale_list)
                    stacked_weights = dequantize(
                        stacked_weights,
                        stacked_scales,
                        (1, *weight_block_size),
                    )

                # Transpose for matmul: [num_experts, out_features, in_features] -> [1, num_experts, in_features, out_features]
                stacked_weights = stacked_weights.unsqueeze(0).transpose(-1, -2)

                # Convert to bfloat16 for TTNN
                stacked_weights = stacked_weights.to(torch.bfloat16)

                # Create TTNN tensor with sharding across devices
                weight_tensor = ttnn.from_torch(
                    stacked_weights,
                    device=mesh_device,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),  # Shard experts across devices
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                # Store the weight tensor in the config
                weight_config[ttnn_name] = {"input_tensor_b": weight_tensor}

        return weight_config

    @classmethod
    def _forward(cls, x: ttnn.Tensor, cfg: Dict) -> ttnn.Tensor:
        """
        Forward pass through distributed experts.

        This is a direct port of the reference Experts._forward method.

        Args:
            x: Input tensor [1, num_experts_per_device, num_tokens, hidden_size]
            cfg: Configuration dictionary with weights and memory configs

        Returns:
            Expert output [1, num_experts_per_device, num_tokens, hidden_size]
        """
        # Verify input memory config
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"

        # Get input shape
        _, _, num_tokens, hidden_size = x.shape

        # Debug logging
        debug_experts = os.getenv("DEEPSEEK_V3_DEBUG_EXPERTS") == "1" and num_tokens > 8192

        def _log_expert_stats(name: str, tensor: ttnn.Tensor) -> None:
            if not debug_experts:
                return
            try:
                mesh_device = cfg.get("mesh_device")
                if mesh_device is not None:
                    tensor_torch = ttnn.to_torch(
                        tensor,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape
                        ),
                    )
                else:
                    tensor_torch = ttnn.to_torch(tensor)
                finite_mask = torch.isfinite(tensor_torch)
                numel = tensor_torch.numel()
                finite_count = finite_mask.sum().item()
                nan_count = torch.isnan(tensor_torch).sum().item()
                inf_count = torch.isinf(tensor_torch).sum().item()
                logger.info(
                    f"DEBUG experts {name}: shape={tensor_torch.shape}, "
                    f"mean={tensor_torch.mean():.4f}, std={tensor_torch.std():.4f}, "
                    f"max={tensor_torch.abs().max():.4f}, "
                    f"finite={finite_count}/{numel}, nan={nan_count}, inf={inf_count}"
                )
            except Exception as exc:
                logger.warning(f"DEBUG experts {name}: failed to extract stats: {exc}")

        # Gate and up projections (exactly like reference implementation)
        # Debug: check if weights exist in config
        if "input_tensor_b" not in cfg.get("w1_experts", {}):
            logger.error(f"w1_experts missing input_tensor_b! Keys: {cfg.get('w1_experts', {}).keys()}")
            raise ValueError("w1_experts missing input_tensor_b weight tensor")

        w1_out = ttnn.linear(x, **cfg["w1_experts"])
        w3_out = ttnn.linear(x, **cfg["w3_experts"])
        _log_expert_stats("w1_out", w1_out)
        _log_expert_stats("w3_out", w3_out)

        # Apply activation and multiply (SwiGLU)
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul_experts"])
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        _log_expert_stats("activated", activated)

        # Down projection
        output = ttnn.linear(activated, **cfg["w2_experts"])
        ttnn.deallocate(activated)
        _log_expert_stats("w2_out", output)

        # Reshape for output (exactly like reference)
        output = ttnn.permute(output, (1, 0, 2, 3))
        output = ttnn.reshape(output, shape=(1, cfg["num_experts_per_device"], num_tokens, hidden_size))

        # Verify output memory config
        assert output.memory_config() == cfg["output_memory_config"]

        return output

    @classmethod
    def forward_decode(
        cls,
        hidden_states: ttnn.Tensor,
        topk_expert_indices: ttnn.Tensor,
        topk_expert_weights: ttnn.Tensor,
        cfg: Dict,
        expert_mapping_tensors: ttnn.Tensor,
        mesh_device: ttnn.MeshDevice,
    ) -> ttnn.Tensor:
        """
        Forward pass in decode mode with integrated all-to-all operations.

        Args:
            hidden_states: Original input hidden states [batch, 1, seq, H]
            topk_expert_indices: Expert routing indices [batch, 1, seq, K]
            topk_expert_weights: Routing weights [K, 1, seq*batch, H]
            cfg: Model configuration including weights
            expert_mapping_tensors: Expert to device mapping [1, 1, E, D]
            mesh_device: Mesh device for operations

        Returns:
            Expert output tensor [1, 1, seq*batch, H]
        """
        # Get configuration parameters
        memory_config = cfg.get("output_memory_config", ttnn.L1_MEMORY_CONFIG)
        hidden_size = cfg.get("hidden_size", 7168)
        num_experts_per_tok = cfg.get("num_experts_per_tok", 8)
        num_experts_per_device = cfg.get("num_experts_per_device", 8)
        cluster_axis = cfg.get("cluster_axis", 0)
        dispatch_topology = cfg.get("dispatch_topology", "Linear")
        combine_topology = cfg.get("combine_topology", "Linear")

        # Convert topology strings to enums if needed
        if isinstance(dispatch_topology, str):
            dispatch_topology = getattr(ttnn.Topology, dispatch_topology)
        if isinstance(combine_topology, str):
            combine_topology = getattr(ttnn.Topology, combine_topology)

        # Get sequence length and batch size
        batch_size, _, seq_len, _ = hidden_states.shape
        tokens_per_device = batch_size * seq_len

        # ==========================================================================
        # STEP 1: PREPARE INPUTS FOR ALL_TO_ALL_DISPATCH
        # ==========================================================================
        # Convert to ROW_MAJOR layout as required by all-to-all dispatch
        hidden_rm = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_rm = ttnn.reshape(hidden_rm, shape=(1, 1, tokens_per_device, hidden_size))

        # Expert indices: [1, 1, tokens_per_device, K]
        topk_indices_rm = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
        topk_indices_rm = ttnn.reshape(topk_indices_rm, shape=(1, 1, tokens_per_device, num_experts_per_tok))

        # ==========================================================================
        # STEP 2: ALL_TO_ALL_DISPATCH - Route tokens to expert devices
        # ==========================================================================
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
        # STEP 3: PREPARE FOR EXPERT COMPUTATION
        # ==========================================================================
        # Get total tokens after dispatch
        num_dispatch_devices = mesh_device.shape[cluster_axis] if hasattr(mesh_device, "shape") else 1
        total_tokens = tokens_per_device * num_dispatch_devices

        # Reshape for expert computation
        dispatch_output = ttnn.reshape(dispatch_output, shape=(1, 1, total_tokens, hidden_size))
        dispatch_output = ttnn.to_layout(dispatch_output, ttnn.TILE_LAYOUT)

        # Repeat for all experts on this device
        dispatch_output = ttnn.repeat(
            dispatch_output, ttnn.Shape((1, num_experts_per_device, 1, 1)), memory_config=memory_config
        )

        # ==========================================================================
        # STEP 4: RUN EXPERT COMPUTATION
        # ==========================================================================
        # Call the existing _forward method for expert computation
        experts_output = cls._forward(dispatch_output, cfg)

        # ==========================================================================
        # STEP 5: PREPARE FOR ALL_TO_ALL_COMBINE
        # ==========================================================================
        # Convert to ROW_MAJOR for all_to_all_combine
        experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
        # Reshape from [1, num_experts_per_device, total_tokens, H] to [num_experts_per_device, 1, total_tokens, H]
        experts_output = ttnn.reshape(experts_output, shape=(num_experts_per_device, 1, total_tokens, hidden_size))

        # Reshape dispatch_metadata for combine
        dispatch_metadata = ttnn.reshape(dispatch_metadata, shape=(1, 1, total_tokens, num_experts_per_tok))

        # ==========================================================================
        # STEP 6: ALL_TO_ALL_COMBINE - Route expert outputs back to token positions
        # ==========================================================================
        combine_output = ttnn.all_to_all_combine(
            experts_output,
            dispatch_metadata,
            expert_mapping_tensors,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
            output_shard_dim=2,  # Shard on token dimension
            topology=combine_topology,
        )
        ttnn.deallocate(experts_output)
        ttnn.deallocate(dispatch_metadata)

        # ==========================================================================
        # STEP 7: APPLY ROUTING WEIGHTS AND REDUCE ACROSS EXPERTS
        # ==========================================================================
        # Combine output shape: [K, 1, tokens_per_device, H]
        post_combine = ttnn.to_layout(combine_output, ttnn.TILE_LAYOUT)
        ttnn.deallocate(combine_output)

        # Apply routing weights and sum across experts
        # topk_expert_weights is already in shape [K, 1, tokens_per_device, H]
        weighted_output = ttnn.mul(post_combine, topk_expert_weights, memory_config=memory_config)
        ttnn.deallocate(post_combine)

        # Sum across K experts (first dimension)
        output = ttnn.sum(weighted_output, dim=0, keepdim=True)
        ttnn.deallocate(weighted_output)

        # Final shape: [1, 1, tokens_per_device, H]
        return output

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: Dict) -> ttnn.Tensor:
        """Forward pass in prefill mode."""
        return cls._forward(x, cfg)
