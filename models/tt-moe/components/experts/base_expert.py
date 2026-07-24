# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Base expert interface for unified MoE implementation."""

from abc import ABC, abstractmethod

import torch

import ttnn


class BaseExpert(ABC):
    """
    Abstract base class for MoE experts.

    All expert implementations must inherit from this class and implement
    the forward method which processes tokens assigned to experts.

    This base class provides common functionality including:
    - Weight quantization/dequantization
    - Memory configuration setup
    - Forward mode dispatching
    - Common constants and utilities
    """

    # Common constants across all experts
    USERS_PER_ROW = 32  # For decode mode sharding
    WEIGHT_DTYPE = ttnn.bfloat8_b
    WEIGHT_SCALE_INV_TORCH_DTYPE = torch.float32

    def __init__(self, config: dict, mesh_device: ttnn.MeshDevice):
        """
        Initialize common expert configuration.

        Args:
            config: Expert configuration dictionary from JSON
            mesh_device: TTNN mesh device for tensor placement
        """
        self.config = config
        self.mesh_device = mesh_device

        # Common configuration setup
        self.hidden_size = config.get("hidden_size", 7168)
        self.intermediate_size = config.get("intermediate_size", 2048)

        # Quantization configuration
        self.use_quantized_weights = config.get("use_quantized_weights", False)
        self.weight_block_size = config.get("weight_block_size", [128, 128])

        # Memory configuration
        memory_config_str = config.get("memory_config", "L1_MEMORY_CONFIG")
        self.memory_config = getattr(ttnn, memory_config_str)

        # Output memory configuration (may differ from input)
        output_memory_config_str = config.get("output_memory_config", memory_config_str)
        self.output_memory_config = getattr(ttnn, output_memory_config_str)

        # Compute kernel configuration
        self._setup_compute_config(config)

    def _setup_compute_config(self, config: dict):
        """
        Set up compute kernel configuration.

        Args:
            config: Configuration dictionary
        """
        compute_config_str = config.get("compute_kernel_config", "HIFI2")

        if hasattr(ttnn.MathFidelity, compute_config_str):
            math_fidelity = getattr(ttnn.MathFidelity, compute_config_str)
            self.compute_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=math_fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        else:
            # Default compute configuration
            self.compute_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )

    def _dequantize(self, quantized_tensor: torch.Tensor, scale_inv: torch.Tensor, block_size):
        """
        Dequantize a weight tensor using scale inverse.

        This is a common dequantization method used by both shared and distributed experts.

        Args:
            quantized_tensor: Quantized weight tensor
            scale_inv: Scale inverse tensor
            block_size: Block size for quantization (can be list or tuple)

        Returns:
            Dequantized tensor as float32
        """
        # Convert to 1.0 / scale_inv to get the actual scale
        scale = 1.0 / scale_inv

        # Handle both tuple and list block_size
        if isinstance(block_size, (list, tuple)):
            if len(block_size) == 2:
                # Simple 2D block size [height, width]
                block_height, block_width = block_size

                # Expand scale to match weight dimensions
                if scale.dim() < quantized_tensor.dim():
                    scale = scale.unsqueeze(-1).unsqueeze(-1)
                    scale = scale.repeat(1, 1, block_height, block_width)
                    scale = scale.reshape(quantized_tensor.shape)

            elif len(block_size) == 3:
                # 3D block size for expert weights (num_blocks, height, width)
                _, block_height, block_width = block_size

                # Expand scale to match quantized tensor dimensions
                if quantized_tensor.dim() == 3:  # [num_experts, out_features, in_features]
                    num_experts = quantized_tensor.shape[0]
                    out_features = quantized_tensor.shape[1]
                    in_features = quantized_tensor.shape[2]

                    # Repeat scale for each block
                    scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
                    scale_expanded = scale_expanded.repeat(1, 1, 1, block_height, block_width)

                    # Reshape to match weight dimensions
                    scale = scale_expanded.reshape(num_experts, out_features, in_features)
                else:
                    # For 2D tensors, handle as before
                    scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
                    scale = scale_expanded.repeat(1, 1, block_height, block_width)
                    scale = scale.reshape(quantized_tensor.shape)

        return quantized_tensor.to(torch.float32) * scale

    def forward_prefill(self, x: ttnn.Tensor, cfg: dict = None) -> ttnn.Tensor:
        """
        Forward pass for prefill mode.

        Default implementation that calls forward with mode="prefill".
        Subclasses can override for specialized prefill behavior.

        Args:
            x: Input tensor
            cfg: Optional configuration dict

        Returns:
            Expert output tensor
        """
        return self.forward(x, mode="prefill")

    def forward_decode(self, x: ttnn.Tensor, cfg: dict = None) -> ttnn.Tensor:
        """
        Forward pass for decode mode.

        Default implementation that calls forward with mode="decode".
        Subclasses can override for specialized decode behavior.

        Args:
            x: Input tensor
            cfg: Optional configuration dict

        Returns:
            Expert output tensor
        """
        return self.forward(x, mode="decode")

    @abstractmethod
    def forward(self, x: ttnn.Tensor, mode: str = "decode"):
        """
        Forward pass through the expert.

        This must be implemented by all subclasses with their specific
        expert computation logic.

        Args:
            x: Input tensor (may be dispatched tokens)
            mode: "decode" or "prefill" mode

        Returns:
            Expert output tensor
        """

    @abstractmethod
    def load_weights(self, state_dict: dict, weight_path: str = None):
        """
        Load expert weights from state dict or cached path.

        This must be implemented by subclasses as weight loading
        patterns differ between shared and distributed experts.

        Args:
            state_dict: Dictionary containing expert weights
            weight_path: Optional path for cached weights
        """
