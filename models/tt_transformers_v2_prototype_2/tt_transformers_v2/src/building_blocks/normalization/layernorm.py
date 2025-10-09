"""
LayerNorm (Layer Normalization) implementation.

LayerNorm normalizes across the feature dimension with learnable affine transform,
commonly used in transformer models.
"""

from dataclasses import dataclass
from typing import Optional

import ttnn


@dataclass
class LayerNormSpec:
    """
    Mathematical specification for LayerNorm.

    Attributes:
        normalized_shape: Shape of the features to normalize
        eps: Small value to prevent division by zero
        elementwise_affine: Whether to apply learnable affine transform
    """

    normalized_shape: int
    eps: float = 1e-5
    elementwise_affine: bool = True

    def __post_init__(self):
        if self.normalized_shape <= 0:
            raise ValueError(f"normalized_shape must be positive, got {self.normalized_shape}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")

    def validate(self):
        """Validate spec constraints."""
        assert self.normalized_shape > 0, "normalized_shape must be positive"
        assert self.eps > 0, "eps must be positive"


@dataclass
class LayerNormImplConfig:
    """
    TTNN-specific implementation configuration for LayerNorm.

    Attributes:
        dtype: Data type for computations
        memory_config: Memory configuration for TTNN operations
        compute_kernel_config: Kernel configuration for compute operations
    """

    dtype: ttnn.DataType = ttnn.bfloat16
    memory_config: Optional[ttnn.MemoryConfig] = None
    compute_kernel_config: Optional[dict] = None


def get_default_impl_config(
    spec: LayerNormSpec, device: str, mode: Optional[str] = None, strategy: str = "default"
) -> LayerNormImplConfig:
    """
    Return default implementation configuration for the given device and mode.

    Args:
        spec: LayerNorm specification
        device: Target device (e.g., "N150", "T3000", "cpu")
        mode: Execution mode (not used for normalization, kept for API consistency)
        strategy: Performance strategy hint

    Returns:
        Implementation configuration with device-specific defaults
    """
    if device.startswith("N150") or device.startswith("T3000"):
        return LayerNormImplConfig(
            dtype=ttnn.bfloat16,
            # Device-specific settings
        )
    else:
        return LayerNormImplConfig()


def forward(
    hidden_states: ttnn.Tensor,
    spec: LayerNormSpec,
    impl_config: LayerNormImplConfig,
    gamma: Optional[ttnn.Tensor] = None,
    beta: Optional[ttnn.Tensor] = None,
    **kwargs,
) -> ttnn.Tensor:
    """
    Apply LayerNorm to input tensor.

    Args:
        hidden_states: Input tensor of shape (batch, seq_len, normalized_shape)
        spec: LayerNorm specification
        impl_config: Implementation configuration
        gamma: Scale parameter of shape (normalized_shape,)
        beta: Shift parameter of shape (normalized_shape,)
        **kwargs: Additional arguments for future extensions

    Returns:
        Normalized tensor of same shape as input
    """
    raise NotImplementedError("Refactor from TTTv1")
