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
    math_fidelity=ttnn.MathFidelity.HiFi4,  # Use highest precision
    math_approx_mode=False,
    fp32_dest_acc_en=True,  # Enable FP32 accumulation for better precision
    packer_l1_acc=False,  # Don't use L1 for packing to avoid memory issues
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
        # Use DRAM to avoid L1 memory issues and numerical problems
        if mode == "decode":
            input_memory_config = ttnn.DRAM_MEMORY_CONFIG  # Changed from L1 to DRAM
            output_memory_config = ttnn.DRAM_MEMORY_CONFIG  # Changed from L1 to DRAM
        else:
            input_memory_config = ttnn.DRAM_MEMORY_CONFIG
            output_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Get activation configuration
        # Support both HF config objects and plain dicts
        activation_type = "swiglu"  # Default
        swiglu_alpha = 1.702  # Default
        swiglu_limit = 7.0  # Default

        if hasattr(hf_config, "activation"):
            activation_config = hf_config.activation
            # Handle nested activation dict
            if isinstance(activation_config, dict):
                activation_type = activation_config.get("type", "swiglu")
                swiglu_alpha = activation_config.get("alpha", activation_config.get("swiglu_alpha", 1.702))
                swiglu_limit = activation_config.get("gate_limit", activation_config.get("swiglu_limit", 7.0))
            else:
                # Handle flat string
                activation_type = activation_config
        elif isinstance(hf_config, dict):
            # Check for nested activation dict in config
            if "activation" in hf_config:
                activation_config = hf_config["activation"]
                if isinstance(activation_config, dict):
                    activation_type = activation_config.get("type", "swiglu")
                    swiglu_alpha = activation_config.get("alpha", activation_config.get("swiglu_alpha", 1.702))
                    swiglu_limit = activation_config.get("gate_limit", activation_config.get("swiglu_limit", 7.0))
                else:
                    activation_type = activation_config
            # Also check for flat config values
            activation_type = hf_config.get("activation_type", activation_type)
            swiglu_alpha = hf_config.get("swiglu_alpha", swiglu_alpha)
            swiglu_limit = hf_config.get("swiglu_limit", swiglu_limit)

        # Create config dictionary that includes weight references
        config = {
            "mesh_device": mesh_device,
            "input_memory_config": input_memory_config,
            "output_memory_config": output_memory_config,
            "num_experts_per_device": num_experts_per_device,
            "activation": activation_type,
            "swiglu_alpha": swiglu_alpha,
            "swiglu_limit": swiglu_limit,
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
                # Float8 tensors don't support min/max operations
                if stacked_weights.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    logger.info(
                        f"[DEBUG] {hf_name} weights stacked: shape {stacked_weights.shape}, "
                        + f"dtype {stacked_weights.dtype} (Float8 - range not available)"
                    )
                else:
                    logger.info(
                        f"[DEBUG] {hf_name} weights stacked: shape {stacked_weights.shape}, "
                        + f"range [{stacked_weights.min():.4f}, {stacked_weights.max():.4f}]"
                    )

                # Apply dequantization if needed
                if use_quantized and scale_list:
                    stacked_scales = torch.stack(scale_list)
                    stacked_weights = dequantize(
                        stacked_weights,
                        stacked_scales,
                        (1, *weight_block_size),
                    )
                    logger.info(
                        f"[DEBUG] {hf_name} weights after dequantization: "
                        + f"range [{stacked_weights.min():.4f}, {stacked_weights.max():.4f}]"
                    )

                # Transpose for matmul: [num_experts, out_features, in_features] -> [1, num_experts, in_features, out_features]
                stacked_weights = stacked_weights.unsqueeze(0).transpose(-1, -2)
                logger.info(f"[DEBUG] {hf_name} weights after transpose: shape {stacked_weights.shape}")

                # Convert to bfloat16 for TTNN
                stacked_weights = stacked_weights.to(torch.bfloat16)

                # Check for NaN/Inf before conversion
                if torch.isnan(stacked_weights).any() or torch.isinf(stacked_weights).any():
                    logger.error(f"[ERROR] {hf_name} weights contain NaN or Inf values before TTNN conversion!")
                    logger.error(f"  NaN count: {torch.isnan(stacked_weights).sum().item()}")
                    logger.error(f"  Inf count: {torch.isinf(stacked_weights).sum().item()}")

                # Create TTNN tensor with sharding across devices
                weight_tensor = ttnn.from_torch(
                    stacked_weights,
                    device=mesh_device,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),  # Shard experts across devices
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                logger.info(f"[DEBUG] {hf_name} weights converted to TTNN: shape {weight_tensor.shape}")

                # Store the weight tensor in the config
                weight_config[ttnn_name] = {"input_tensor_b": weight_tensor}
            else:
                logger.warning(f"[WARNING] No weights found for {hf_name}!")

        return weight_config

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
        Apply clamped SwiGLU activation for GPT-OSS.

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

        # Debug logging (also enable for GPT_OSS_DEBUG)
        debug_experts = (os.getenv("DEEPSEEK_V3_DEBUG_EXPERTS") == "1" and num_tokens > 8192) or os.getenv(
            "GPT_OSS_DEBUG"
        ) == "1"

        # Debug input tensor
        if debug_experts:
            try:
                # Check if tensor is on mesh device
                mesh_device = cfg.get("mesh_device")
                if mesh_device is not None:
                    x_torch = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
                else:
                    x_torch = ttnn.to_torch(x)
                logger.info(
                    f"[DEBUG] Input x to _forward: shape={x.shape}, torch_shape={x_torch.shape}, "
                    + f"range=[{x_torch.min():.4f}, {x_torch.max():.4f}], "
                    + f"mean={x_torch.mean():.4f}, std={x_torch.std():.4f}"
                )
                if torch.isnan(x_torch).any() or torch.isinf(x_torch).any():
                    logger.error(f"[ERROR] Input x contains NaN or Inf values!")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to convert input x to torch for debugging: {e}")

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
            logger.error(f"Full w1_experts config: {cfg.get('w1_experts', {})}")
            logger.error(f"Available cfg keys: {list(cfg.keys())}")
            raise ValueError("w1_experts missing input_tensor_b weight tensor")

        # Debug: Log weight tensor info before linear
        w1_weight = cfg["w1_experts"].get("input_tensor_b")
        if w1_weight is not None:
            logger.debug(f"[DEBUG] w1_experts weight tensor shape: {w1_weight.shape}")
        else:
            logger.error("[ERROR] w1_experts weight tensor is None!")

        # Debug: Check all config keys for w1_experts
        if debug_experts:
            logger.info(f"[DEBUG] w1_experts config keys: {list(cfg['w1_experts'].keys())}")
            for key, value in cfg["w1_experts"].items():
                if key != "input_tensor_b":
                    logger.info(f"  - {key}: {value}")

        # Check matmul type configuration
        matmul_type = cfg.get("matmul_type", "dense")  # Default to dense for backward compatibility

        if debug_experts:
            logger.info(f"[DEBUG] Using {matmul_type} matmul operations")

        # Use DRAM for linear operations to avoid L1 memory overflow and numerical issues
        w1_config = cfg["w1_experts"].copy()
        w1_config["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        w3_config = cfg["w3_experts"].copy()
        w3_config["memory_config"] = ttnn.DRAM_MEMORY_CONFIG

        if matmul_type == "sparse":
            # Import sparse matmul operations
            from .sparse_matmul_ops import SparseMatmulConfig, reshape_for_sparse_matmul, sparse_gate_up_projection

            # Create sparse configuration
            sparse_config = SparseMatmulConfig(cfg)
            if debug_experts:
                sparse_config.log_config()

            # Reshape input for sparse matmul
            x_sparse, num_blocks = reshape_for_sparse_matmul(x, sparse_config.block_size)

            # Generate sparsity pattern (simplified for now - all blocks active)
            # In production, this would analyze dispatch_metadata from routing
            sparsity = ttnn.ones(
                (num_blocks, cfg["num_experts_per_device"]),
                dtype=ttnn.uint8,
                device=x.device(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Get weights and biases
            w1_weight = cfg["w1_experts"]["input_tensor_b"]
            w3_weight = cfg["w3_experts"]["input_tensor_b"]
            w1_bias = cfg["w1_experts"].get("bias")
            w3_bias = cfg["w3_experts"].get("bias")

            # Perform sparse gate/up projections
            w1_out = sparse_gate_up_projection(
                x_sparse,
                w1_weight,
                w1_bias,
                sparsity,
                sparse_config.block_size,
                sparse_config.memory_config,
                sparse_config.gate_up_config,
            )
            w3_out = sparse_gate_up_projection(
                x_sparse,
                w3_weight,
                w3_bias,
                sparsity,
                sparse_config.block_size,
                sparse_config.memory_config,
                sparse_config.gate_up_config,
            )

            # Store sparsity for down projection
            cfg["_sparse_sparsity"] = sparsity
            cfg["_sparse_config"] = sparse_config
            cfg["_sparse_num_blocks"] = num_blocks
        else:
            # Use dense linear operations (default for DeepSeek)
            w1_out = ttnn.linear(x, **w1_config)
            w3_out = ttnn.linear(x, **w3_config)
        _log_expert_stats("w1_out", w1_out)
        _log_expert_stats("w3_out", w3_out)

        # Apply activation based on configuration
        activation_type = cfg.get("activation", "swiglu")
        if debug_experts:
            logger.info(f"[DEBUG] Using activation type: {activation_type}")

        if activation_type == "clamped_swiglu":
            # Use clamped SwiGLU for GPT-OSS
            activated = cls._apply_clamped_swiglu(
                gate=w1_out,
                up=w3_out,
                alpha=cfg.get("swiglu_alpha", 1.702),
                limit=cfg.get("swiglu_limit", 7.0),
                memory_config=cfg["output_memory_config"],
            )
        else:
            # Use simple SwiGLU for DeepSeek (default)
            activated = ttnn.mul(w1_out, w3_out, **cfg["mul_experts"])
            ttnn.deallocate(w1_out)
            ttnn.deallocate(w3_out)
        _log_expert_stats("activated", activated)

        # Down projection
        if matmul_type == "sparse":
            from .sparse_matmul_ops import sparse_down_projection

            # Get stored sparse configuration
            sparse_config = cfg.get("_sparse_config")
            sparsity = cfg.get("_sparse_sparsity")
            num_blocks = cfg.get("_sparse_num_blocks")

            if sparse_config and sparsity is not None:
                # Get weight and bias
                w2_weight = cfg["w2_experts"]["input_tensor_b"]
                w2_bias = cfg["w2_experts"].get("bias")

                # Perform sparse down projection
                output = sparse_down_projection(
                    activated,
                    w2_weight,
                    w2_bias,
                    sparsity,
                    sparse_config.block_size,
                    sparse_config.memory_config,
                    sparse_config.down_config,
                )

                # Clean up temporary sparse data
                ttnn.deallocate(sparsity)
                del cfg["_sparse_sparsity"]
                del cfg["_sparse_config"]
                del cfg["_sparse_num_blocks"]
            else:
                # Fallback to dense if sparse config missing
                logger.warning("[WARNING] Sparse config missing, falling back to dense")
                output = ttnn.linear(activated, **cfg["w2_experts"])
        else:
            # Use dense linear operations (default)
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
        # Need ROW_MAJOR layout AND uint16 dtype for all_to_all_dispatch

        # First reshape if needed
        indices_shape = topk_expert_indices.shape

        if (
            len(indices_shape) == 4
            and indices_shape[0] == 1
            and indices_shape[1] == 1
            and indices_shape[2] == tokens_per_device
            and indices_shape[3] == num_experts_per_tok
        ):
            # Already in the exact right shape [1, 1, tokens_per_device, K]
            reshaped_indices = topk_expert_indices
        elif len(indices_shape) == 4:
            # Shape is [batch, 1, seq, K], need to flatten to [1, 1, batch*seq, K]
            # This handles the GPT-OSS case where input is [4, 1, 1, 4] -> [1, 1, 4, 4]
            batch_dim = indices_shape[0]
            seq_dim = indices_shape[2]

            if batch_dim * seq_dim == tokens_per_device:
                # Can safely reshape
                # IMPORTANT: Keep in ROW_MAJOR layout for uint16 reshape
                # TILE_LAYOUT doesn't support uint16 reshape operations
                reshaped_indices = ttnn.reshape(
                    topk_expert_indices, shape=(1, 1, tokens_per_device, num_experts_per_tok)
                )
            else:
                raise ValueError(
                    f"Cannot reshape indices from {indices_shape} to (1, 1, {tokens_per_device}, {num_experts_per_tok})"
                )
        elif len(indices_shape) == 2:
            # Shape is [tokens, K], reshape to [1, 1, tokens, K]
            reshaped_indices = ttnn.reshape(topk_expert_indices, shape=(1, 1, tokens_per_device, num_experts_per_tok))
        else:
            raise ValueError(f"Unexpected indices shape: {indices_shape}")

        # Convert to ROW_MAJOR layout - this changes dtype but we'll fix it
        topk_indices_rm = ttnn.to_layout(reshaped_indices, ttnn.ROW_MAJOR_LAYOUT)

        # If num_experts_per_tok is not a multiple of 32, pad it for typecast
        if num_experts_per_tok < 32:
            # Pad to 32 for typecast compatibility
            padded_indices = ttnn.pad(
                topk_indices_rm,
                [1, 1, tokens_per_device, 32],  # Pad last dim to 32
                [0, 0, 0, 0],  # Start indices
                value=0.0,
            )
            # Now typecast to uint16
            padded_indices_uint16 = ttnn.typecast(padded_indices, dtype=ttnn.uint16)
            # Slice back to original size
            topk_indices_rm = ttnn.slice(
                padded_indices_uint16, [0, 0, 0, 0], [1, 1, tokens_per_device, num_experts_per_tok]
            )
        else:
            # Already a multiple of 32, can typecast directly
            topk_indices_rm = ttnn.typecast(topk_indices_rm, dtype=ttnn.uint16)

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

        # Check if we should use sparse computation (for GPT-OSS)
        use_sparse = cfg.get("use_sparse_matmul", False)

        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"[DEBUG] DistributedExpert forward_decode: use_sparse={use_sparse}")
        logger.info(f"[DEBUG] Config keys: {list(cfg.keys())}")
        logger.info(f"[DEBUG] Activation from cfg: {cfg.get('activation', 'Not found')}")

        if use_sparse:
            # ==========================================================================
            # STEP 3A: SPARSE PATH FOR GPT-OSS
            # ==========================================================================
            # Import sparse functions
            from .distributed_expert_sparse_fix import (
                apply_sparse_expert_computation,
                create_program_configs_for_gpt_oss,
                generate_sparsity_pattern_for_gpt_oss,
            )

            # Create program configs if not provided
            if "gate_up_program_config" not in cfg or "down_program_config" not in cfg:
                intermediate_size = cfg.get("intermediate_size", 2880)
                hidden_size = cfg.get("hidden_size", 2880)
                gate_up_config, down_config = create_program_configs_for_gpt_oss(intermediate_size, hidden_size)
                cfg["gate_up_program_config"] = gate_up_config
                cfg["down_program_config"] = down_config

            # Reshape dispatch output to TILE layout
            dispatch_output = ttnn.reshape(dispatch_output, shape=(1, 1, total_tokens, hidden_size))
            dispatch_output = ttnn.to_layout(dispatch_output, ttnn.TILE_LAYOUT)

            # Generate remap_topk_mask (needed for sparsity pattern)
            num_experts = cfg.get("num_experts", 128)
            num_dispatch_rows = mesh_device.shape[cluster_axis] if hasattr(mesh_device, "shape") else 1
            remap_topk_mask = ttnn.zeros(
                (1, num_dispatch_rows, 1, num_experts),
                dtype=ttnn.bfloat16,
                device=mesh_device,
                memory_config=memory_config,
            )

            # Generate sparsity pattern
            sparsity_block_size = cfg.get("sparsity_block_size", 32)
            sparsity = generate_sparsity_pattern_for_gpt_oss(
                dispatch_metadata=dispatch_metadata,
                expert_mapping_tensors=expert_mapping_tensors,
                remap_topk_mask=remap_topk_mask,
                tokens_per_device=tokens_per_device,
                total_tokens=total_tokens,
                num_experts=num_experts,
                sparsity_block_size=sparsity_block_size,
            )

            # Reshape input for sparse blocks
            num_sparse_blocks = total_tokens // sparsity_block_size
            x_sparse = ttnn.reshape(
                dispatch_output,
                shape=(1, num_sparse_blocks, sparsity_block_size, hidden_size),
            )

            # Apply sparse expert computation
            expert_output_sparse = apply_sparse_expert_computation(
                x=x_sparse,
                cfg=cfg,
                sparsity=sparsity,
                sparsity_block_size=sparsity_block_size,
            )

            ttnn.deallocate(sparsity)
            ttnn.deallocate(remap_topk_mask)

            # Permute and reshape for combine
            # From: [num_sparse_blocks, experts_per_device, block_size, H]
            # To: [experts_per_device, 1, total_tokens, H]
            experts_output = ttnn.permute(expert_output_sparse, (1, 0, 2, 3))
            ttnn.deallocate(expert_output_sparse)
            experts_output = ttnn.reshape(
                experts_output,
                shape=(
                    1,
                    num_experts_per_device,
                    total_tokens,
                    hidden_size,
                ),  # Swapped dimensions 0 and 1 to match reference
            )

        else:
            # ==========================================================================
            # STEP 3B: DENSE PATH (Original for DeepSeek)
            # ==========================================================================
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
        # all_to_all_combine expects [num_experts_per_device, 1, total_tokens, H]
        # Expert computation outputs [1, num_experts_per_device, total_tokens, H]
        # So we need to reshape for all_to_all_combine
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
    def forward_compute_only(
        cls,
        dispatch_output: ttnn.Tensor,
        cfg: Dict,
        num_tokens: Optional[int] = None,
        matmul_type: str = "dense",
    ) -> ttnn.Tensor:
        """
        Expert computation ONLY - without all-to-all operations.

        This method performs only the expert MLP computation on already-dispatched tokens.
        It does not include all_to_all_dispatch at the beginning or all_to_all_combine at the end.

        Args:
            dispatch_output: Already dispatched tokens from all_to_all_dispatch
                           Shape: [1, 1, total_tokens_on_device, hidden_size]
            cfg: Configuration dictionary with expert weights and settings
            num_tokens: Optional number of tokens (if None, inferred from input shape)
            matmul_type: Type of matmul to use ("dense" or "sparse")

        Returns:
            Expert output tensor ready for all_to_all_combine
            Shape: [num_experts_per_device, 1, total_tokens_on_device, hidden_size]
        """
        debug_experts = os.environ.get("DEEPSEEK_V3_DEBUG_EXPERTS", "") == "1"

        # Get dimensions
        hidden_size = cfg["hidden_size"]
        num_experts_per_device = cfg["num_experts_per_device"]

        if num_tokens is None:
            # Infer from input shape
            num_tokens = dispatch_output.shape[-2]

        if debug_experts:
            logger.info(f"[DEBUG] forward_compute_only: dispatch_output shape={dispatch_output.shape}")
            logger.info(f"[DEBUG] num_tokens={num_tokens}, hidden_size={hidden_size}")
            logger.info(f"[DEBUG] num_experts_per_device={num_experts_per_device}")
            logger.info(f"[DEBUG] matmul_type={matmul_type}")

        # STEP 1: Reshape for expert computation
        # From [1, 1, total_tokens, H] to [num_experts_per_device, tokens_per_expert, H]
        dispatch_reshaped = ttnn.reshape(
            dispatch_output, shape=(num_experts_per_device, num_tokens // num_experts_per_device, 1, hidden_size)
        )

        # STEP 2: Expert computation (Gate, Up, Activation, Down projections)
        experts_output = cls._forward(dispatch_reshaped, cfg)

        # STEP 3: Prepare output for all_to_all_combine
        # Output shape should be [num_experts_per_device, 1, total_tokens, H]
        # The _forward method already returns in the correct shape

        if debug_experts:
            logger.info(f"[DEBUG] experts_output shape={experts_output.shape}")
            _log_expert_stats("experts_output", experts_output)

        return experts_output

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: Dict) -> ttnn.Tensor:
        """Forward pass in prefill mode."""
        return cls._forward(x, cfg)
