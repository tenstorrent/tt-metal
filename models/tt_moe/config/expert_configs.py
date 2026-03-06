# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Unified configuration for MoE experts supporting both DeepSeek and GPT-OSS backends."""

from dataclasses import dataclass
from typing import Dict, Optional

import ttnn


@dataclass
class ExpertActivationConfig:
    """Configuration for expert activation function.

    Supports both SILU (DeepSeek) and clamped SwiGLU (GPT-OSS) activation types.
    """

    activation_type: str  # "silu" or "clamped_swiglu"

    # For clamped SwiGLU (GPT-OSS)
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0

    # Memory configuration for activation
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG


@dataclass
class AllToAllDispatchConfig:
    """Configuration for all_to_all_dispatch operation (GPT-OSS)."""

    cluster_axis: Optional[int] = None
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG

    def as_dict(self) -> Dict:
        """Convert to dict for ttnn.all_to_all_dispatch kwargs."""
        result = {"memory_config": self.memory_config}
        if self.cluster_axis is not None:
            result["cluster_axis"] = self.cluster_axis
        return result


@dataclass
class AllToAllCombineConfig:
    """Configuration for all_to_all_combine operation (GPT-OSS)."""

    cluster_axis: Optional[int] = None
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG

    def as_dict(self) -> Dict:
        """Convert to dict for ttnn.all_to_all_combine kwargs."""
        result = {"memory_config": self.memory_config}
        if self.cluster_axis is not None:
            result["cluster_axis"] = self.cluster_axis
        return result


@dataclass
class UnifiedExpertConfig:
    """Unified configuration for both DeepSeek and GPT-OSS backends.

    This configuration parameterizes the differences between backends:
    - Activation function: SILU (DeepSeek) vs Clamped SwiGLU (GPT-OSS)
    - Weight format: Separate w1/w3 (DeepSeek) vs Fused gate_up (GPT-OSS optional)
    - Token routing: None (DeepSeek) vs all_to_all (GPT-OSS)
    - Output shape and format
    - CCL operations: External (DeepSeek) vs Internal (GPT-OSS)
    """

    # Model dimensions
    hidden_size: int
    intermediate_size: int
    num_experts_per_device: int

    # Activation configuration
    activation: ExpertActivationConfig

    # Weight format
    use_fused_gate_up: bool = False  # True for GPT-OSS fused weights
    weight_dtype: ttnn.DataType = ttnn.bfloat8_b

    # Memory configurations
    input_memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG
    output_memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG

    # Compute kernel configuration
    compute_kernel_config: Optional[Dict] = None

    # All-to-all configuration (GPT-OSS only)
    use_all_to_all: bool = False
    dispatch_config: Optional[AllToAllDispatchConfig] = None
    combine_config: Optional[AllToAllCombineConfig] = None

    # Output format
    output_shape_format: str = "deepseek"  # "deepseek" or "gptoss"

    # All-reduce configuration (GPT-OSS only)
    use_experimental_all_reduce: bool = False
    all_reduce_num_links: int = 4
    all_reduce_cluster_axis: int = 1

    # Number of experts per token (K)
    num_experts_per_tok: int = 6  # Default for DeepSeek


def create_deepseek_expert_config(
    hf_config, mode: str = "decode", num_experts_per_device: int = 16
) -> UnifiedExpertConfig:
    """Create configuration for DeepSeek backend.

    Args:
        hf_config: HuggingFace configuration object
        mode: Either "decode" or "prefill"
        num_experts_per_device: Number of experts per device (default 16)

    Returns:
        UnifiedExpertConfig for DeepSeek backend
    """
    if mode == "decode":
        memory_config = ttnn.L1_MEMORY_CONFIG
    else:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    return UnifiedExpertConfig(
        hidden_size=hf_config.hidden_size,
        intermediate_size=getattr(hf_config, "moe_intermediate_size", getattr(hf_config, "intermediate_size", 4608)),
        num_experts_per_device=num_experts_per_device,
        activation=ExpertActivationConfig(activation_type="silu", memory_config=memory_config),
        use_fused_gate_up=False,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=memory_config,
        output_memory_config=memory_config,
        use_all_to_all=False,  # Handled externally in MoEBlock
        output_shape_format="deepseek",
        num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", 6),
    )


def create_gptoss_expert_config(
    hf_config, mode: str = "decode", num_experts_per_device: int = 4, use_fused_weights: bool = True
) -> UnifiedExpertConfig:
    """Create configuration for GPT-OSS backend.

    Args:
        hf_config: HuggingFace configuration object
        mode: Either "decode" or "prefill"
        num_experts_per_device: Number of experts per device (default 4 for 32 devices)
        use_fused_weights: Whether to use fused gate_up weights (default True)

    Returns:
        UnifiedExpertConfig for GPT-OSS backend
    """
    if mode == "decode":
        memory_config = ttnn.L1_MEMORY_CONFIG
    else:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    return UnifiedExpertConfig(
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_experts_per_device=num_experts_per_device,
        activation=ExpertActivationConfig(
            activation_type="clamped_swiglu", swiglu_alpha=1.702, swiglu_limit=7.0, memory_config=memory_config
        ),
        use_fused_gate_up=use_fused_weights,
        weight_dtype=ttnn.bfloat4_b,
        input_memory_config=memory_config,
        output_memory_config=memory_config,
        use_all_to_all=True,  # ThroughputExperts behavior
        dispatch_config=AllToAllDispatchConfig(cluster_axis=0, memory_config=memory_config),
        combine_config=AllToAllCombineConfig(cluster_axis=0, memory_config=memory_config),
        output_shape_format="gptoss",
        use_experimental_all_reduce=False,
        all_reduce_num_links=4,
        all_reduce_cluster_axis=1,
        num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", 8),
    )
