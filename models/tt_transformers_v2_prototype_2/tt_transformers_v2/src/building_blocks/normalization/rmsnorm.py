"""
RMSNorm (Root Mean Square Layer Normalization) implementation.

RMSNorm is a simplified version of LayerNorm that normalizes by RMS statistics
without mean centering, commonly used in models like LLaMA.
"""

from dataclasses import dataclass
from typing import Optional

import ttnn


@dataclass
class RMSNormSpec:
    """
    Mathematical specification for RMSNorm.

    Attributes:
        hidden_dim: Size of the input/output features
        eps: Small value to prevent division by zero
    """

    hidden_dim: int
    eps: float = 1e-6

    def __post_init__(self):
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")

    def validate(self):
        """Validate spec constraints."""
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.eps > 0, "eps must be positive"


@dataclass
class RMSNormImplConfig:
    """
    TTNN-specific implementation configuration for RMSNorm.

    Attributes:
        dtype: Data type for computations
        memory_config: Memory configuration for TTNN operations
        compute_kernel_config: Kernel configuration for compute operations
        use_fused_kernel: Whether to use fused TTNN kernel
    """

    dtype: ttnn.DataType = ttnn.bfloat16
    memory_config: Optional[ttnn.MemoryConfig] = None
    compute_kernel_config: Optional[dict] = None
    use_fused_kernel: bool = True


def get_default_impl_config(
    spec: RMSNormSpec, device: str, mode: Optional[str] = None, strategy: str = "default"
) -> RMSNormImplConfig:
    """
    Return default implementation configuration for the given device and mode.

    Args:
        spec: RMSNorm specification
        device: Target device (e.g., "N150", "T3000", "cpu")
        mode: Execution mode (not used for normalization, kept for API consistency)
        strategy: Performance strategy hint

    Returns:
        Implementation configuration with device-specific defaults
    """
    if device.startswith("N150"):
        return RMSNormImplConfig(
            dtype=ttnn.bfloat16,
            use_fused_kernel=True,
            # Device-specific tuning would go here
        )
    elif device.startswith("T3000"):
        return RMSNormImplConfig(
            dtype=ttnn.bfloat16,
            use_fused_kernel=True,
            # Different device-specific settings
        )
    else:
        # CPU or default fallback
        return RMSNormImplConfig(
            use_fused_kernel=False,
        )


def forward(
    hidden_states: ttnn.Tensor, spec: RMSNormSpec, impl_config: RMSNormImplConfig, gamma: ttnn.Tensor, **kwargs
) -> ttnn.Tensor:
    """
    Apply RMSNorm to input tensor.

    Args:
        hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)
        spec: RMSNorm specification
        impl_config: Implementation configuration
        gamma: Scale parameter of shape (hidden_dim,)
        **kwargs: Additional arguments for future extensions

    Returns:
        Normalized tensor of same shape as input
    """
    raise NotImplementedError("Refactor from TTTv1 models/common/rmsnorm.py")
