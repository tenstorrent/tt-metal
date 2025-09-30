# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimization and quantization configurations for TTTv2"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import ttnn


class QuantizationType(Enum):
    """Supported quantization types"""

    INT8 = "int8"
    INT4 = "int4"
    BFLOAT8 = "bfloat8"
    BFLOAT4 = "bfloat4"
    MIXED = "mixed"
    NONE = "none"


class OptimizationLevel(Enum):
    """Optimization levels for performance/accuracy trade-offs"""

    O0 = 0  # No optimizations (highest accuracy)
    O1 = 1  # Basic optimizations
    O2 = 2  # Aggressive optimizations
    O3 = 3  # Maximum performance


@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization.

    Supports various quantization schemes for weights and activations.
    """

    enabled: bool = False
    weight_dtype: str = "bfloat8"
    activation_dtype: str = "bfloat16"
    kv_cache_dtype: str = "bfloat8"

    # Per-layer quantization settings
    per_layer_config: Dict[int, Dict[str, str]] = field(default_factory=dict)

    # Quantization parameters
    symmetric: bool = True
    per_channel: bool = True
    calibration_method: str = "minmax"  # Options: "minmax", "percentile", "kl"
    calibration_samples: int = 128

    # Mixed precision settings
    mixed_precision_layers: List[int] = field(default_factory=list)
    sensitive_layers: List[int] = field(default_factory=list)  # Keep in higher precision

    def get_weight_dtype(self, layer_idx: int) -> ttnn.DataType:
        """Get weight data type for a specific layer"""
        if layer_idx in self.sensitive_layers:
            return ttnn.bfloat16

        layer_config = self.per_layer_config.get(layer_idx, {})
        dtype_str = layer_config.get("weight_dtype", self.weight_dtype)

        return self._str_to_dtype(dtype_str)

    def get_activation_dtype(self, layer_idx: int) -> ttnn.DataType:
        """Get activation data type for a specific layer"""
        if layer_idx in self.sensitive_layers:
            return ttnn.bfloat16

        layer_config = self.per_layer_config.get(layer_idx, {})
        dtype_str = layer_config.get("activation_dtype", self.activation_dtype)

        return self._str_to_dtype(dtype_str)

    def _str_to_dtype(self, dtype_str: str) -> ttnn.DataType:
        """Convert string to TTNN data type"""
        dtype_map = {
            "float32": ttnn.float32,
            "float16": ttnn.float16,
            "bfloat16": ttnn.bfloat16,
            "bfloat8": ttnn.bfloat8_b,
            "bfloat4": ttnn.bfloat4_b,
            "int8": ttnn.int8,
            "uint8": ttnn.uint8,
        }
        return dtype_map.get(dtype_str, ttnn.bfloat16)


@dataclass
class OptimizationConfig:
    """
    Master optimization configuration for TTT models.

    Controls various performance optimizations and their parameters.
    """

    optimization_level: OptimizationLevel = OptimizationLevel.O1
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # Memory optimizations
    gradient_checkpointing: bool = False
    gradient_checkpointing_layers: List[int] = field(default_factory=list)
    memory_efficient_attention: bool = True
    kv_cache_compression: bool = False

    # Compute optimizations
    use_fused_ops: bool = True
    use_flash_attention: bool = True
    use_rotary_embedding_cache: bool = True
    compile_mode: str = "default"  # Options: "default", "reduce-overhead", "max-performance"

    # Sharding and parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel: bool = False
    activation_checkpointing: bool = False

    # Kernel configurations
    matmul_config: Dict[str, Any] = field(default_factory=dict)
    attention_config: Dict[str, Any] = field(default_factory=dict)
    layernorm_config: Dict[str, Any] = field(default_factory=dict)

    # Performance settings
    enable_profiling: bool = False
    enable_debug_mode: bool = False
    deterministic: bool = False

    def __post_init__(self):
        """Apply optimization level presets"""
        self._apply_optimization_level()

    def _apply_optimization_level(self):
        """Apply settings based on optimization level"""
        if self.optimization_level == OptimizationLevel.O0:
            # No optimizations
            self.use_fused_ops = False
            self.use_flash_attention = False
            self.memory_efficient_attention = False
            self.quantization.enabled = False

        elif self.optimization_level == OptimizationLevel.O1:
            # Basic optimizations
            self.use_fused_ops = True
            self.use_flash_attention = True
            self.memory_efficient_attention = True

        elif self.optimization_level == OptimizationLevel.O2:
            # Aggressive optimizations
            self.use_fused_ops = True
            self.use_flash_attention = True
            self.memory_efficient_attention = True
            self.kv_cache_compression = True
            self.compile_mode = "reduce-overhead"

        elif self.optimization_level == OptimizationLevel.O3:
            # Maximum performance
            self.use_fused_ops = True
            self.use_flash_attention = True
            self.memory_efficient_attention = True
            self.kv_cache_compression = True
            self.compile_mode = "max-performance"
            self.quantization.enabled = True
            self.quantization.weight_dtype = "bfloat8"
            self.quantization.kv_cache_dtype = "bfloat8"

    def get_matmul_config(self, m: int, k: int, n: int) -> Dict[str, Any]:
        """
        Get optimized matmul configuration for given dimensions.

        Args:
            m: First dimension
            k: Inner dimension
            n: Last dimension

        Returns:
            Matmul program configuration
        """
        # Default configuration
        config = {
            "compute_kernel_config": {
                "math_fidelity": ttnn.MathFidelity.HiFi4,
                "math_approx_mode": True,
                "fp32_dest_acc_en": True,
                "packer_l1_acc": True,
            }
        }

        # Apply custom configurations
        config.update(self.matmul_config)

        # Size-specific optimizations
        if m * n > 1024 * 1024:  # Large matmul
            config["program_config"] = {
                "type": "bmm",
                "output_block_w": 4,
                "output_block_h": 4,
            }
        else:  # Small matmul
            config["program_config"] = {
                "type": "1d_systolic",
            }

        return config

    def get_attention_config(self, seq_len: int, num_heads: int) -> Dict[str, Any]:
        """
        Get optimized attention configuration.

        Args:
            seq_len: Sequence length
            num_heads: Number of attention heads

        Returns:
            Attention configuration
        """
        config = {
            "use_flash_attention": self.use_flash_attention and seq_len > 512,
            "memory_efficient": self.memory_efficient_attention,
        }

        # Apply custom configurations
        config.update(self.attention_config)

        return config

    def should_checkpoint_layer(self, layer_idx: int) -> bool:
        """Check if a layer should use gradient checkpointing"""
        if not self.gradient_checkpointing:
            return False

        if self.gradient_checkpointing_layers:
            return layer_idx in self.gradient_checkpointing_layers

        # Default: checkpoint every other layer
        return layer_idx % 2 == 0


@dataclass
class ComputeOptimizationConfig:
    """Detailed compute optimization settings"""

    # Math configurations
    math_fidelity: str = "HiFi4"  # HiFi4, HiFi2, LoFi
    enable_math_approx: bool = True
    fp32_accumulation: bool = True

    # Buffer configurations
    l1_buffer_size: Optional[int] = None
    enable_double_buffering: bool = True
    circular_buffer_size: Optional[int] = None

    # Grid configurations
    grid_size: Optional[tuple] = None
    shard_orientation: str = "row_major"
    enable_harvesting: bool = True

    # Operation-specific configs
    conv_optimizations: Dict[str, Any] = field(default_factory=dict)
    pool_optimizations: Dict[str, Any] = field(default_factory=dict)
    reduce_optimizations: Dict[str, Any] = field(default_factory=dict)


def create_optimization_config(
    model_size: str = "small",
    target_hardware: str = "wormhole_b0",
    use_case: str = "inference",
) -> OptimizationConfig:
    """
    Create optimization configuration based on model and hardware.

    Args:
        model_size: Size category (small, medium, large, xlarge)
        target_hardware: Target hardware architecture
        use_case: Usage scenario (inference, training, fine-tuning)

    Returns:
        Configured OptimizationConfig
    """
    # Base configuration
    config = OptimizationConfig()

    # Model size specific settings
    if model_size == "small":
        config.optimization_level = OptimizationLevel.O1
        config.memory_efficient_attention = False
    elif model_size == "medium":
        config.optimization_level = OptimizationLevel.O2
        config.memory_efficient_attention = True
    elif model_size in ["large", "xlarge"]:
        config.optimization_level = OptimizationLevel.O3
        config.memory_efficient_attention = True
        config.kv_cache_compression = True
        config.gradient_checkpointing = True

    # Hardware specific settings
    if target_hardware == "wormhole_b0":
        config.matmul_config = {
            "compute_kernel_config": {
                "math_fidelity": ttnn.MathFidelity.HiFi4,
                "math_approx_mode": True,
            }
        }
    elif target_hardware == "blackhole":
        config.matmul_config = {
            "compute_kernel_config": {
                "math_fidelity": ttnn.MathFidelity.HiFi4,
                "fp32_dest_acc_en": True,
            }
        }

    # Use case specific settings
    if use_case == "training":
        config.gradient_checkpointing = True
        config.activation_checkpointing = True
        config.deterministic = True
    elif use_case == "inference":
        config.quantization.enabled = True
        config.compile_mode = "max-performance"

    return config
