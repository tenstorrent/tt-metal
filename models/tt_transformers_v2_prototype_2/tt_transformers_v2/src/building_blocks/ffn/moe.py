"""
Mixture of Experts (MoE) implementation.

MoE enables sparse computation by routing tokens to different expert networks,
allowing for larger model capacity without proportional compute increase.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import ttnn

from .mlp import MLPImplConfig


@dataclass
class MoESpec:
    """
    Mathematical specification for Mixture of Experts.

    Attributes:
        num_experts: Number of expert networks
        num_experts_per_tok: Number of experts to use per token
        hidden_dim: Input/output dimension
        intermediate_dim: Hidden layer dimension per expert
        activation: Activation function type
        use_bias: Whether to use bias in linear projections
        normalize_expert_weights: Whether to normalize expert weights
        router_jitter_noise: Jitter noise for load balancing (training only)
        expert_capacity_factor: Factor to determine expert capacity
        router_type: Type of router ("top_k", "expert_choice", "soft")
    """

    num_experts: int
    num_experts_per_tok: int
    hidden_dim: int
    intermediate_dim: int
    activation: Literal["silu", "gelu", "relu"] = "silu"
    use_bias: bool = False
    normalize_expert_weights: bool = True
    router_jitter_noise: float = 0.0
    expert_capacity_factor: float = 1.0
    router_type: str = "top_k"

    def __post_init__(self):
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}")
        if self.num_experts_per_tok <= 0 or self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                f"num_experts_per_tok must be in (0, {self.num_experts}], " f"got {self.num_experts_per_tok}"
            )
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim must be positive, got {self.intermediate_dim}")

    def validate(self):
        """Validate spec constraints."""
        assert self.num_experts > 0, "num_experts must be positive"
        assert (
            0 < self.num_experts_per_tok <= self.num_experts
        ), f"num_experts_per_tok must be in (0, {self.num_experts}]"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.intermediate_dim > 0, "intermediate_dim must be positive"


@dataclass
class MoEImplConfig(MLPImplConfig):
    """
    TTNN-specific implementation configuration for MoE.

    Extends MLPImplConfig with MoE-specific configurations.

    Attributes:
        router_dtype: Data type for router computations
        expert_parallel_mode: How to parallelize experts ("none", "tensor", "expert")
        router_memory_config: Memory configuration for router
        load_balancing_loss_coef: Coefficient for load balancing loss
    """

    router_dtype: ttnn.DataType = ttnn.bfloat16
    expert_parallel_mode: str = "none"
    router_memory_config: Optional[ttnn.MemoryConfig] = None
    load_balancing_loss_coef: float = 0.01


@dataclass
class MoEOutput:
    """
    Output structure for MoE forward pass.

    Attributes:
        hidden_states: Output tensor
        router_logits: Raw router logits (for analysis/loss computation)
        expert_weights: Weights assigned to each expert
        expert_indices: Which experts were selected for each token
        load_balancing_loss: Optional load balancing loss
    """

    hidden_states: ttnn.Tensor
    router_logits: ttnn.Tensor
    expert_weights: ttnn.Tensor
    expert_indices: ttnn.Tensor
    load_balancing_loss: Optional[ttnn.Tensor] = None


def get_default_impl_config(
    spec: MoESpec, device: str, mode: Literal["prefill", "decode"], strategy: str = "default"
) -> MoEImplConfig:
    """
    Return default implementation configuration for MoE.

    Args:
        spec: MoE specification
        device: Target device (e.g., "N150", "T3000", "cpu")
        mode: Execution strategy mode (prefill or decode)
        strategy: Performance strategy hint

    Returns:
        Implementation configuration with device-specific defaults
    """
    if device.startswith("N150"):
        return MoEImplConfig(
            dtype=ttnn.bfloat16 if mode == "prefill" else ttnn.bfloat8_b,
            router_dtype=ttnn.bfloat16,
            use_1d_systolic_array=True,
            expert_parallel_mode="none",  # Could be "tensor" for larger models
        )
    elif device.startswith("T3000"):
        return MoEImplConfig(
            dtype=ttnn.bfloat16,
            router_dtype=ttnn.bfloat16,
            use_1d_systolic_array=True,
            expert_parallel_mode="tensor",  # T3000 can handle tensor parallelism
        )
    else:
        # CPU or default fallback
        return MoEImplConfig(
            use_1d_systolic_array=False,
            expert_parallel_mode="none",
        )


def prefill_forward(
    hidden_states: ttnn.Tensor,
    spec: MoESpec,
    impl_config: MoEImplConfig,
    router_weight: ttnn.Tensor,
    expert_weights: List[Tuple[ttnn.Tensor, ttnn.Tensor]],
    router_bias: Optional[ttnn.Tensor] = None,
    **kwargs,
) -> MoEOutput:
    """
    MoE forward pass for prefill mode.

    Routes each token to top-k experts and combines their outputs.

    Args:
        hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)
        spec: MoE specification
        impl_config: Implementation configuration
        router_weight: Router weight for expert selection
        expert_weights: List of (up_weight, down_weight) tuples for each expert
        router_bias: Optional router bias
        **kwargs: Additional arguments

    Returns:
        MoEOutput with results and auxiliary information
    """
    raise NotImplementedError("Refactor from TTTv1 models/tt_transformers/tt/mixtral_moe.py")


def decode_forward(
    hidden_states: ttnn.Tensor,
    spec: MoESpec,
    impl_config: MoEImplConfig,
    router_weight: ttnn.Tensor,
    expert_weights: List[Tuple[ttnn.Tensor, ttnn.Tensor]],
    router_bias: Optional[ttnn.Tensor] = None,
    **kwargs,
) -> MoEOutput:
    """
    MoE forward pass for decode mode.

    Routes each token to top-k experts and combines their outputs.

    Args:
        hidden_states: Input tensor of shape (batch, 1, hidden_dim)
        spec: MoE specification
        impl_config: Implementation configuration
        router_weight: Router weight for expert selection
        expert_weights: List of (up_weight, down_weight) tuples for each expert
        router_bias: Optional router bias
        **kwargs: Additional arguments

    Returns:
        MoEOutput with results and auxiliary information
    """
    raise NotImplementedError("Refactor from TTTv1 models/tt_transformers/tt/mixtral_moe.py")


def compute_load_balancing_loss(
    router_logits: ttnn.Tensor, expert_indices: ttnn.Tensor, spec: MoESpec, impl_config: MoEImplConfig
) -> ttnn.Tensor:
    """
    Compute auxiliary load balancing loss for MoE training.

    Args:
        router_logits: Raw router outputs
        expert_indices: Selected expert indices
        spec: MoE specification
        impl_config: Implementation configuration

    Returns:
        Load balancing loss scalar
    """
    raise NotImplementedError("Implement load balancing loss computation")
