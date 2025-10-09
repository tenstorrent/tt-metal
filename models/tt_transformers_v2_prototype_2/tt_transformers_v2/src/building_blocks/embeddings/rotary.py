"""
Rotary Position Embeddings (RoPE) implementation.

RoPE applies rotational position embeddings to query and key tensors,
enabling better extrapolation to longer sequences.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import ttnn


@dataclass
class RoPESpec:
    """
    Specification for Rotary Position Embeddings.

    Attributes:
        dim: Dimension of the embeddings (usually head_dim)
        max_seq_len: Maximum sequence length
        base: Base for the geometric progression (theta)
        scaling_factor: Optional scaling factor for position indices
        rope_type: Type of RoPE ("default" or "llama3" style)
    """

    dim: int
    max_seq_len: int = 2048
    base: float = 10000.0
    scaling_factor: float = 1.0
    rope_type: str = "default"

    def __post_init__(self):
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.base <= 0:
            raise ValueError(f"base must be positive, got {self.base}")

    def validate(self):
        """Validate spec constraints."""
        assert self.dim > 0, "dim must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.base > 0, "base must be positive"


@dataclass
class RoPEImplConfig:
    """
    TTNN-specific implementation configuration for RoPE.

    Attributes:
        dtype: Data type for computations
        memory_config: Memory configuration for TTNN operations
        use_cached: Whether to use pre-computed sin/cos values
        cache_dtype: Data type for cached sin/cos values
    """

    dtype: ttnn.DataType = ttnn.bfloat16
    memory_config: Optional[ttnn.MemoryConfig] = None
    use_cached: bool = True
    cache_dtype: Optional[ttnn.DataType] = None

    def __post_init__(self):
        if self.cache_dtype is None:
            self.cache_dtype = self.dtype


def get_default_impl_config(
    spec: RoPESpec, device: str, mode: Optional[str] = None, strategy: str = "default"
) -> RoPEImplConfig:
    """
    Return default implementation configuration for the given device.

    Args:
        spec: RoPE specification
        device: Target device (e.g., "N150", "T3000", "cpu")
        mode: Execution mode (not typically used for RoPE)
        strategy: Performance strategy hint

    Returns:
        Implementation configuration with device-specific defaults
    """
    if device.startswith("N150") or device.startswith("T3000"):
        return RoPEImplConfig(
            dtype=ttnn.bfloat16,
            use_cached=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        return RoPEImplConfig(
            use_cached=True,
        )


def compute_frequencies(
    spec: RoPESpec, impl_config: RoPEImplConfig, device: Optional[ttnn.Device] = None
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Pre-compute sin and cos frequencies for RoPE.

    Args:
        spec: RoPE specification
        impl_config: Implementation configuration
        device: Target device for the tensors

    Returns:
        Tuple of (cos, sin) tensors of shape (max_seq_len, dim)
    """
    raise NotImplementedError("Refactor from TTTv1 models/tt_transformers/tt/rope.py")


def apply_rotary_embeddings(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    position_ids: Optional[ttnn.Tensor] = None,
    impl_config: Optional[RoPEImplConfig] = None,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.

    Args:
        query: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        key: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        cos: Cosine values for RoPE
        sin: Sine values for RoPE
        position_ids: Position indices (optional)
        impl_config: Implementation configuration

    Returns:
        Tuple of (rotated_query, rotated_key)
    """
    raise NotImplementedError("Refactor from TTTv1 models/tt_transformers/tt/rope.py")
