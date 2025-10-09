"""
Standard Multi-Layer Perceptron (MLP) implementation.

Provides the basic feed-forward network used in transformer models,
typically with two linear layers and an activation function.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import ttnn


@dataclass
class MLPSpec:
    """
    Mathematical specification for standard MLP.

    Attributes:
        hidden_dim: Input/output dimension
        intermediate_dim: Hidden layer dimension (typically 4x hidden_dim)
        activation: Activation function type
        use_bias: Whether to use bias in linear projections
        dropout: Dropout rate (for training)
    """

    hidden_dim: int
    intermediate_dim: int
    activation: Literal["relu", "gelu", "silu"] = "gelu"
    use_bias: bool = False
    dropout: float = 0.0

    def __post_init__(self):
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim must be positive, got {self.intermediate_dim}")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def validate(self):
        """Validate spec constraints."""
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.intermediate_dim > 0, "intermediate_dim must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"


@dataclass
class MLPImplConfig:
    """
    TTNN-specific implementation configuration for MLP.

    Attributes:
        dtype: Data type for computations
        output_dtype: Output data type (if different)
        memory_config: Memory configuration for TTNN operations
        compute_kernel_config: Kernel configuration for compute operations
        up_proj_memory_config: Specific memory config for up projection
        down_proj_memory_config: Specific memory config for down projection
        use_1d_systolic_array: Whether to use 1D systolic array for matmuls
        activation_memory_config: Memory config for activation function
    """

    dtype: ttnn.DataType = ttnn.bfloat16
    output_dtype: Optional[ttnn.DataType] = None
    memory_config: Optional[ttnn.MemoryConfig] = None
    compute_kernel_config: Optional[dict] = None
    up_proj_memory_config: Optional[ttnn.MemoryConfig] = None
    down_proj_memory_config: Optional[ttnn.MemoryConfig] = None
    use_1d_systolic_array: bool = True
    activation_memory_config: Optional[ttnn.MemoryConfig] = None

    def __post_init__(self):
        if self.output_dtype is None:
            self.output_dtype = self.dtype


def get_default_impl_config(
    spec: MLPSpec, device: str, mode: Literal["prefill", "decode"], strategy: str = "default"
) -> MLPImplConfig:
    """
    Return default implementation configuration for the given device and mode.

    Args:
        spec: MLP specification
        device: Target device (e.g., "N150", "T3000", "cpu")
        mode: Execution strategy mode (prefill or decode)
        strategy: Performance strategy hint

    Returns:
        Implementation configuration with device-specific defaults
    """
    if device.startswith("N150"):
        if mode == "prefill":
            return MLPImplConfig(
                dtype=ttnn.bfloat16,
                use_1d_systolic_array=True,
            )
        else:  # decode
            return MLPImplConfig(
                dtype=ttnn.bfloat8_b,
                use_1d_systolic_array=True,
            )
    elif device.startswith("T3000"):
        return MLPImplConfig(
            dtype=ttnn.bfloat16,
            use_1d_systolic_array=True,
        )
    else:
        # CPU or default fallback
        return MLPImplConfig(
            use_1d_systolic_array=False,
        )


def prefill_forward(
    hidden_states: ttnn.Tensor,
    spec: MLPSpec,
    impl_config: MLPImplConfig,
    up_weight: ttnn.Tensor,
    down_weight: ttnn.Tensor,
    up_bias: Optional[ttnn.Tensor] = None,
    down_bias: Optional[ttnn.Tensor] = None,
    **kwargs,
) -> ttnn.Tensor:
    """
    MLP forward pass for prefill mode (process entire sequence).

    Computes: output = down_proj(activation(up_proj(input)))

    Args:
        hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)
        spec: MLP specification
        impl_config: Implementation configuration
        up_weight: Up projection weight of shape (hidden_dim, intermediate_dim)
        down_weight: Down projection weight of shape (intermediate_dim, hidden_dim)
        up_bias: Optional up projection bias
        down_bias: Optional down projection bias
        **kwargs: Additional arguments for future extensions

    Returns:
        Output tensor of shape (batch, seq_len, hidden_dim)
    """
    raise NotImplementedError("Refactor from TTTv1 models/tt_transformers/tt/mlp.py")


def decode_forward(
    hidden_states: ttnn.Tensor,
    spec: MLPSpec,
    impl_config: MLPImplConfig,
    up_weight: ttnn.Tensor,
    down_weight: ttnn.Tensor,
    up_bias: Optional[ttnn.Tensor] = None,
    down_bias: Optional[ttnn.Tensor] = None,
    **kwargs,
) -> ttnn.Tensor:
    """
    MLP forward pass for decode mode (single token).

    Computes: output = down_proj(activation(up_proj(input)))

    Args:
        hidden_states: Input tensor of shape (batch, 1, hidden_dim)
        spec: MLP specification
        impl_config: Implementation configuration
        up_weight: Up projection weight
        down_weight: Down projection weight
        up_bias: Optional up projection bias
        down_bias: Optional down projection bias
        **kwargs: Additional arguments for future extensions

    Returns:
        Output tensor of shape (batch, 1, hidden_dim)
    """
    raise NotImplementedError("Refactor from TTTv1 models/tt_transformers/tt/mlp.py")
