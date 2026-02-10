# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Distributed expert implementation for MoE.

This implementation is based on models/demos/deepseek_v3/tt/experts.py
"""

import os

import torch
from loguru import logger

import ttnn

try:
    from ...utils.debug_logger import log_op, log_tensor_props
except ImportError:
    try:
        from models.tt_moe.utils.debug_logger import log_op, log_tensor_props
    except ImportError:
        from utils.debug_logger import log_op, log_tensor_props

try:
    from .base_expert import BaseExpert
except ImportError:
    from components.experts.base_expert import BaseExpert


class DistributedExpert(BaseExpert):
    """
    Distributed expert implementation for MoE.

    This expert type processes tokens with FFN experts distributed across devices.
    Based on the DeepSeek demo's Experts implementation.
    """

    # Additional weight configuration (extends base)
    WEIGHT_TORCH_DTYPE = torch.float8_e4m3fn

    def __init__(self, config: dict, mesh_device: ttnn.MeshDevice, ccl=None):
        """
        Initialize distributed expert.

        Args:
            config: Expert configuration containing:
                - intermediate_size: FFN intermediate dimension (moe_intermediate_size in HF)
                - hidden_size: Model hidden dimension
                - num_experts_per_device: Number of experts on each device
                - dispatch_cluster_axis: Mesh axis for expert parallelism
                - memory_config: Memory configuration string
                - activation: Activation function (swiglu, gelu, etc.)
                - n_routed_experts: Total number of routed experts (optional)
            mesh_device: TTNN mesh device
            ccl: CCL instance for collective operations
        """
        # Initialize base class - handles common config
        super().__init__(config, mesh_device)

        self.ccl = ccl

        # Expert distribution
        self.dispatch_cluster_axis = config["dispatch_cluster_axis"]
        self.n_routed_experts = config.get("n_routed_experts", None)
        if self.n_routed_experts:
            # Calculate experts per device based on total devices (like working implementation)
            num_devices = mesh_device.get_num_devices()
            assert self.n_routed_experts % num_devices == 0, (
                f"Number of experts ({self.n_routed_experts}) must be divisible by "
                f"the number of devices ({num_devices})"
            )
            self.num_experts_per_device = self.n_routed_experts // num_devices
        else:
            self.num_experts_per_device = config.get("num_experts_per_device", 1)
        self.activation = config.get("activation", "swiglu")

        # Expert weights (will be loaded later)
        self.w1_experts = None  # Gate projection for SwiGLU
        self.w2_experts = None  # Down projection
        self.w3_experts = None  # Up projection for SwiGLU

    def _setup_compute_config(self, config: dict):
        """Override base compute config to use LoFi for distributed experts."""
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def load_weights(self, state_dict: dict, weight_path: str = None):
        """
        Load expert FFN weights.

        This follows the DeepSeek approach of loading all experts' weights together.

        Args:
            state_dict: Dictionary containing expert weights either as:
                - Combined tensors: w1_experts, w2_experts, w3_experts
                - Per-expert tensors: experts.0.gate_proj.weight, etc.
            weight_path: Optional path for cached weights
        """
        # Check if weights are already in combined format
        if "w1_experts" in state_dict:
            # Already combined format
            self.w1_experts = state_dict["w1_experts"]
            self.w2_experts = state_dict["w2_experts"]
            self.w3_experts = state_dict["w3_experts"]
        else:
            # Per-expert format - need to combine
            # Collect all expert weights
            w1_list = []
            w2_list = []
            w3_list = []
            w1_scale_list = []
            w2_scale_list = []
            w3_scale_list = []

            # Determine which experts to load for this device
            # In distributed setup, each device only loads a subset of experts
            if self.n_routed_experts:
                # Calculate which experts belong to this device
                # For simplicity, assume device_id = 0 (in real setup, this would come from mesh device)
                # TODO: Get actual device ID from mesh device for proper distribution
                device_id = 0  # This should be obtained from mesh_device in production
                start_expert = device_id * self.num_experts_per_device
                end_expert = start_expert + self.num_experts_per_device
                expert_range = range(start_expert, end_expert)
            else:
                # If not specified, load all experts in state dict (backward compatibility)
                num_experts = 0
                for key in state_dict.keys():
                    if key.startswith("experts.") and ".gate_proj.weight" in key:
                        num_experts += 1
                expert_range = range(num_experts)

            for expert_id in expert_range:
                # Gate projection (w1)
                gate_key = f"experts.{expert_id}.gate_proj.weight"
                if gate_key in state_dict:
                    weight = state_dict[gate_key]
                    w1_list.append(weight)

                    # Check for quantization scale
                    scale_key = f"experts.{expert_id}.gate_proj.weight_scale_inv"
                    if self.use_quantized_weights and scale_key in state_dict:
                        w1_scale_list.append(state_dict[scale_key])

                # Down projection (w2)
                down_key = f"experts.{expert_id}.down_proj.weight"
                if down_key in state_dict:
                    weight = state_dict[down_key]
                    w2_list.append(weight)

                    scale_key = f"experts.{expert_id}.down_proj.weight_scale_inv"
                    if self.use_quantized_weights and scale_key in state_dict:
                        w2_scale_list.append(state_dict[scale_key])

                # Up projection (w3)
                up_key = f"experts.{expert_id}.up_proj.weight"
                if up_key in state_dict:
                    weight = state_dict[up_key]
                    w3_list.append(weight)

                    scale_key = f"experts.{expert_id}.up_proj.weight_scale_inv"
                    if self.use_quantized_weights and scale_key in state_dict:
                        w3_scale_list.append(state_dict[scale_key])

            # Stack expert weights
            if w1_list:
                w1_stacked = torch.stack(w1_list)
                w1_dequantized = False
                if self.use_quantized_weights and w1_scale_list:
                    w1_scale_stacked = torch.stack(w1_scale_list)
                    w1_stacked = self._dequantize(w1_stacked, w1_scale_stacked, (1, *self.weight_block_size))
                    w1_dequantized = True

                # Transpose for matmul (weights are stored as [out_features, in_features])
                w1_stacked = w1_stacked.unsqueeze(0).transpose(-1, -2)

                # Handle dtype conversion based on quantization state
                if w1_dequantized or w1_stacked.dtype == torch.float8_e4m3fn:
                    # If dequantized or float8, convert to bfloat16
                    w1_stacked = w1_stacked.to(torch.bfloat16)
                    w1_dtype = ttnn.bfloat16
                else:
                    w1_dtype = ttnn.bfloat16  # Default

                # Shard across devices if needed
                self.w1_experts = self._shard_and_convert_weights(w1_stacked, "w1_experts", dtype=w1_dtype)

            if w2_list:
                w2_stacked = torch.stack(w2_list)
                w2_dequantized = False
                if self.use_quantized_weights and w2_scale_list:
                    w2_scale_stacked = torch.stack(w2_scale_list)
                    w2_stacked = self._dequantize(w2_stacked, w2_scale_stacked, (1, *self.weight_block_size))
                    w2_dequantized = True

                w2_stacked = w2_stacked.unsqueeze(0).transpose(-1, -2)

                # Handle dtype conversion based on quantization state
                if w2_dequantized or w2_stacked.dtype == torch.float8_e4m3fn:
                    # If dequantized or float8, convert to bfloat16
                    w2_stacked = w2_stacked.to(torch.bfloat16)
                    w2_dtype = ttnn.bfloat16
                else:
                    # Use bfloat4_b for down projection if not dequantized
                    w2_dtype = ttnn.bfloat4_b

                self.w2_experts = self._shard_and_convert_weights(w2_stacked, "w2_experts", dtype=w2_dtype)

            if w3_list:
                w3_stacked = torch.stack(w3_list)
                w3_dequantized = False
                if self.use_quantized_weights and w3_scale_list:
                    w3_scale_stacked = torch.stack(w3_scale_list)
                    w3_stacked = self._dequantize(w3_stacked, w3_scale_stacked, (1, *self.weight_block_size))
                    w3_dequantized = True

                w3_stacked = w3_stacked.unsqueeze(0).transpose(-1, -2)

                # Handle dtype conversion based on quantization state
                if w3_dequantized or w3_stacked.dtype == torch.float8_e4m3fn:
                    # If dequantized or float8, convert to bfloat16
                    w3_stacked = w3_stacked.to(torch.bfloat16)
                    w3_dtype = ttnn.bfloat16
                else:
                    # Use bfloat8_b for up projection if not dequantized
                    w3_dtype = ttnn.bfloat8_b

                self.w3_experts = self._shard_and_convert_weights(w3_stacked, "w3_experts", dtype=w3_dtype)

    def _shard_and_convert_weights(self, weight_tensor: torch.Tensor, name: str, dtype=ttnn.bfloat16):
        """
        Shard weights across devices and convert to TTNN tensor.

        Args:
            weight_tensor: Combined weight tensor [1, num_experts, in_features, out_features]
            name: Name of the weight tensor
            dtype: TTNN dtype for the weights

        Returns:
            TTNN tensor with weights sharded across devices
        """
        # Use ttnn.from_torch like the working test does, not ttnn.as_tensor
        # This properly initializes the tensor with correct block configurations
        return ttnn.from_torch(
            weight_tensor,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),  # Replicate like working test
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _log_expert_stats(self, name: str, tensor: ttnn.Tensor, num_tokens: int):
        """Log expert tensor statistics for debugging."""
        debug_experts = os.getenv("DEEPSEEK_V3_DEBUG_EXPERTS") == "1" and num_tokens > 8192
        if not debug_experts:
            return

        try:
            # Convert to torch for stats
            mesh_device = self.config.get("mesh_device")
            if mesh_device is not None:
                tensor_torch = ttnn.to_torch(
                    tensor,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
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

    def forward(self, x: ttnn.Tensor, indices: ttnn.Tensor = None, weights: ttnn.Tensor = None, mode: str = "decode"):
        """
        Forward pass through distributed experts.

        This follows the DeepSeek experts.py implementation pattern.

        Args:
            x: Input tensor with expert-dispatched tokens
               Shape: [1, num_experts_per_device, num_tokens, hidden_size]
            indices: Expert indices from router (optional, for compatibility)
            weights: Expert weights from router (optional, for compatibility)
            mode: "decode" or "prefill" mode

        Returns:
            Expert output [1, num_experts_per_device, num_tokens, hidden_size]
        """
        # Get input shape
        _, _, num_tokens, hidden_size = x.shape

        # Dynamically select memory config based on mode
        # Following DeepSeek's logic from experts.py
        if mode == "decode":
            output_memory_config = ttnn.L1_MEMORY_CONFIG
        else:
            # For prefill mode, use DRAM
            output_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Check memory config
        # Note: Input memory config might differ from output
        # assert x.memory_config() == self.memory_config, (
        #     f"Input memory config {x.memory_config()} != expected {self.memory_config}"
        # )

        # Gate and up projections
        # Add debug logging for tensor shapes and memory configs
        logger.info(f"DEBUG: Input x shape: {x.shape}")
        logger.info(f"DEBUG: Input x memory_config: {x.memory_config()}")
        logger.info(f"DEBUG: w1_experts shape: {self.w1_experts.shape}")
        logger.info(f"DEBUG: w1_experts memory_config: {self.w1_experts.memory_config()}")
        logger.info(f"DEBUG: output_memory_config: {output_memory_config}")

        # Log input tensor
        log_tensor_props("Expert input x", x)
        log_tensor_props("w1_experts weight", self.w1_experts)
        log_tensor_props("w3_experts weight", self.w3_experts)

        # Gate and up projections using ttnn.linear like working version
        # Use keyword arguments to match the working implementation
        w1_out = ttnn.linear(
            x,
            input_tensor_b=self.w1_experts,  # Use keyword arg like working version
            memory_config=output_memory_config,
            compute_kernel_config=self.compute_config,
            program_config=None,  # Explicitly set to None like working version
        )
        log_op(
            "ttnn.linear (w1_experts)",
            inputs=x,
            config={"memory_config": str(output_memory_config), "compute_kernel_config": str(self.compute_config)},
            output=w1_out,
        )

        w3_out = ttnn.linear(
            x,
            input_tensor_b=self.w3_experts,  # Use keyword arg like working version
            memory_config=output_memory_config,
            compute_kernel_config=self.compute_config,
            program_config=None,  # Explicitly set to None like working version
        )
        log_op(
            "ttnn.linear (w3_experts)",
            inputs=x,
            config={"memory_config": str(output_memory_config), "compute_kernel_config": str(self.compute_config)},
            output=w3_out,
        )

        # Log stats for debugging
        self._log_expert_stats("w1_out", w1_out, num_tokens)
        self._log_expert_stats("w3_out", w3_out, num_tokens)

        # Apply SwiGLU activation: silu(w1_out) * w3_out
        if self.activation == "swiglu":
            activated = ttnn.mul(
                w1_out,
                w3_out,
                memory_config=output_memory_config,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            )
            log_op(
                "ttnn.mul (SwiGLU activation)",
                inputs={"w1_out": w1_out, "w3_out": w3_out},
                config={"memory_config": str(output_memory_config), "activation": "SILU"},
                output=activated,
            )
        else:
            # For other activations, would apply them here
            raise ValueError(f"Unsupported activation: {self.activation}")

        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        self._log_expert_stats("activated", activated, num_tokens)

        # Down projection
        log_tensor_props("w2_experts weight", self.w2_experts)
        output = ttnn.linear(
            activated,
            input_tensor_b=self.w2_experts,  # Use keyword arg like working version
            memory_config=output_memory_config,
            compute_kernel_config=self.compute_config,
            program_config=None,  # Explicitly set to None like working version
        )
        log_op(
            "ttnn.linear (w2_experts)",
            inputs=activated,
            config={"memory_config": str(output_memory_config), "compute_kernel_config": str(self.compute_config)},
            output=output,
        )
        ttnn.deallocate(activated)
        self._log_expert_stats("w2_out", output, num_tokens)

        # Reshape for output
        # The DeepSeek implementation does a permute and reshape here
        output = ttnn.permute(output, (1, 0, 2, 3))
        log_op("ttnn.permute", inputs=output, config={"perm": "(1, 0, 2, 3)"}, output=output)

        output = ttnn.reshape(output, shape=(1, self.num_experts_per_device, num_tokens, hidden_size))
        log_op(
            "ttnn.reshape",
            inputs=output,
            config={"shape": (1, self.num_experts_per_device, num_tokens, hidden_size)},
            output=output,
        )

        # Check output memory config
        assert (
            output.memory_config() == output_memory_config
        ), f"Output memory config {output.memory_config()} != expected {output_memory_config}"

        return output
