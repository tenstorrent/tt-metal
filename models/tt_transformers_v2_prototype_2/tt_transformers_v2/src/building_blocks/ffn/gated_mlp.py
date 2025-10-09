"""
Gated MLP implementations (SwiGLU, GeGLU).

Gated MLPs use an additional gating mechanism to control information flow,
often resulting in better performance than standard MLPs.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import ttnn

from .mlp import MLPImplConfig


@dataclass
class GatedMLPSpec:
    """
    Mathematical specification for gated MLPs (SwiGLU, GeGLU).

    Attributes:
        hidden_dim: Input/output dimension
        intermediate_dim: Hidden layer dimension
        activation: Activation function type (swiglu, geglu, reglu)
        use_bias: Whether to use bias in linear projections
        dropout: Dropout rate (for training)
        gate_activation: Activation for the gating mechanism
    """

    hidden_dim: int
    intermediate_dim: int
    activation: Literal["swiglu", "geglu", "reglu"] = "swiglu"
    use_bias: bool = False
    dropout: float = 0.0
    gate_activation: Optional[str] = None

    def __post_init__(self):
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim must be positive, got {self.intermediate_dim}")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

        # Set default gate activations based on type
        if self.gate_activation is None:
            if self.activation == "swiglu":
                self.gate_activation = "silu"
            elif self.activation == "geglu":
                self.gate_activation = "gelu"
            elif self.activation == "reglu":
                self.gate_activation = "relu"

    def validate(self):
        """Validate spec constraints."""
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.intermediate_dim > 0, "intermediate_dim must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.activation in ["swiglu", "geglu", "reglu"], f"Invalid activation: {self.activation}"


@dataclass
class GatedMLPImplConfig(MLPImplConfig):
    """
    TTNN-specific implementation configuration for gated MLPs.

    Extends MLPImplConfig with gate-specific configurations.

    Attributes:
        gate_proj_memory_config: Specific memory config for gate projection
    """

    gate_proj_memory_config: Optional[ttnn.MemoryConfig] = None


def get_default_impl_config(
    spec: GatedMLPSpec, device: str, mode: Literal["prefill", "decode"], strategy: str = "default"
) -> GatedMLPImplConfig:
    """
    Return default implementation configuration for gated MLP.

    Args:
        spec: Gated MLP specification
        device: Target device (e.g., "N150", "T3000", "cpu")
        mode: Execution strategy mode (prefill or decode)
        strategy: Performance strategy hint

    Returns:
        Implementation configuration with device-specific defaults
    """
    if device.startswith("N150"):
        if mode == "prefill":
            return GatedMLPImplConfig(
                dtype=ttnn.bfloat16,
                use_1d_systolic_array=True,
            )
        else:  # decode
            return GatedMLPImplConfig(
                dtype=ttnn.bfloat8_b,
                use_1d_systolic_array=True,
            )
    elif device.startswith("T3000"):
        return GatedMLPImplConfig(
            dtype=ttnn.bfloat16,
            use_1d_systolic_array=True,
        )
    else:
        # CPU or default fallback
        return GatedMLPImplConfig(
            use_1d_systolic_array=False,
        )


def prefill_forward(
    hidden_states: ttnn.Tensor,
    spec: GatedMLPSpec,
    impl_config: GatedMLPImplConfig,
    up_weight: ttnn.Tensor,
    gate_weight: ttnn.Tensor,
    down_weight: ttnn.Tensor,
    up_bias: Optional[ttnn.Tensor] = None,
    gate_bias: Optional[ttnn.Tensor] = None,
    down_bias: Optional[ttnn.Tensor] = None,
    **kwargs,
) -> ttnn.Tensor:
    """
    Gated MLP forward pass for prefill mode.

    Computes: output = down_proj(activation(up_proj(x)) * gate_proj(x))

    Args:
        hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)
        spec: Gated MLP specification
        impl_config: Implementation configuration
        up_weight: Up projection weight
        gate_weight: Gate projection weight
        down_weight: Down projection weight
        up_bias: Optional up projection bias
        gate_bias: Optional gate projection bias
        down_bias: Optional down projection bias
        **kwargs: Additional arguments

    Returns:
        Output tensor of shape (batch, seq_len, hidden_dim)
    """
    raise NotImplementedError("Refactor from TTTv1 models/tt_transformers/tt/mlp.py")


def decode_forward(
    hidden_states: ttnn.Tensor,
    spec: GatedMLPSpec,
    impl_config: GatedMLPImplConfig,
    up_weight: ttnn.Tensor,
    gate_weight: ttnn.Tensor,
    down_weight: ttnn.Tensor,
    up_bias: Optional[ttnn.Tensor] = None,
    gate_bias: Optional[ttnn.Tensor] = None,
    down_bias: Optional[ttnn.Tensor] = None,
    **kwargs,
) -> ttnn.Tensor:
    """
    Gated MLP forward pass for decode mode.

    Computes: output = down_proj(activation(up_proj(x)) * gate_proj(x))

    Args:
        hidden_states: Input tensor of shape (batch, 1, hidden_dim)
        spec: Gated MLP specification
        impl_config: Implementation configuration
        up_weight: Up projection weight
        gate_weight: Gate projection weight
        down_weight: Down projection weight
        up_bias: Optional up projection bias
        gate_bias: Optional gate projection bias
        down_bias: Optional down projection bias
        **kwargs: Additional arguments

    Returns:
        Output tensor of shape (batch, 1, hidden_dim)
    """
    raise NotImplementedError("Refactor from TTTv1 models/tt_transformers/tt/mlp.py")


# Convenience class for specific gated MLP types
class SwiGLU(GatedMLPSpec):
    """SwiGLU activation (SiLU-gated linear unit) as used in LLaMA."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, **kwargs):
        super().__init__(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, activation="swiglu", **kwargs)


class GeGLU(GatedMLPSpec):
    """GeGLU activation (GELU-gated linear unit)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, **kwargs):
        super().__init__(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, activation="geglu", **kwargs)
