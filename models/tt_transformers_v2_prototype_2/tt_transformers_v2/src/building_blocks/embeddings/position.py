"""
Position embedding implementations.

Provides various positional embedding methods including learned and sinusoidal.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import ttnn


@dataclass
class PositionEmbeddingSpec:
    """
    Mathematical specification for positional embeddings.

    Attributes:
        max_seq_len: Maximum sequence length
        embedding_dim: Dimension of the embedding vectors
        embedding_type: Type of embeddings ("learned", "sinusoidal", "alibi")
        base: Base for sinusoidal embeddings
        scale: Scaling factor for embeddings
        trainable: Whether embeddings are trainable (for learned type)
    """

    max_seq_len: int
    embedding_dim: int
    embedding_type: Literal["learned", "sinusoidal", "alibi"] = "sinusoidal"
    base: float = 10000.0
    scale: float = 1.0
    trainable: bool = True

    def __post_init__(self):
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")

        # Set trainable based on type if not explicitly set
        if self.embedding_type == "sinusoidal" and self.trainable:
            self.trainable = False
        elif self.embedding_type == "learned" and not self.trainable:
            self.trainable = True

    def validate(self):
        """Validate spec constraints."""
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        assert self.embedding_type in [
            "learned",
            "sinusoidal",
            "alibi",
        ], f"Invalid embedding_type: {self.embedding_type}"


@dataclass
class PositionEmbeddingImplConfig:
    """
    TTNN-specific implementation configuration for position embeddings.

    Attributes:
        dtype: Data type for embeddings
        memory_config: Memory configuration for TTNN operations
        cache_embeddings: Whether to cache computed embeddings
        use_fused_add: Whether to use fused addition with token embeddings
    """

    dtype: ttnn.DataType = ttnn.bfloat16
    memory_config: Optional[ttnn.MemoryConfig] = None
    cache_embeddings: bool = True
    use_fused_add: bool = False


def get_default_impl_config(
    spec: PositionEmbeddingSpec, device: str, mode: Optional[str] = None, strategy: str = "default"
) -> PositionEmbeddingImplConfig:
    """
    Return default implementation configuration for the given device.

    Args:
        spec: Position embedding specification
        device: Target device (e.g., "N150", "T3000", "cpu")
        mode: Execution mode (not typically used)
        strategy: Performance strategy hint

    Returns:
        Implementation configuration with device-specific defaults
    """
    if device.startswith("N150") or device.startswith("T3000"):
        return PositionEmbeddingImplConfig(
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_embeddings=True,
        )
    else:
        return PositionEmbeddingImplConfig(
            cache_embeddings=True,
        )


def forward(
    positions: ttnn.Tensor,
    spec: PositionEmbeddingSpec,
    impl_config: PositionEmbeddingImplConfig,
    weight: Optional[ttnn.Tensor] = None,
    **kwargs,
) -> ttnn.Tensor:
    """
    Generate position embeddings for given positions.

    Args:
        positions: Position indices of shape (batch, seq_len) or (seq_len,)
        spec: Position embedding specification
        impl_config: Implementation configuration
        weight: Learned embeddings table (if spec.embedding_type == "learned")
        **kwargs: Additional arguments

    Returns:
        Position embeddings of shape matching positions + (embedding_dim,)
    """
    raise NotImplementedError("Implement position embedding generation")


def create_sinusoidal_embeddings(
    spec: PositionEmbeddingSpec, impl_config: PositionEmbeddingImplConfig, device: Optional[ttnn.Device] = None
) -> ttnn.Tensor:
    """
    Pre-compute sinusoidal position embeddings.

    Args:
        spec: Position embedding specification
        impl_config: Implementation configuration
        device: Target device

    Returns:
        Sinusoidal embeddings of shape (max_seq_len, embedding_dim)
    """
    raise NotImplementedError("Create sinusoidal position embeddings")


def create_alibi_slopes(num_heads: int, device: Optional[ttnn.Device] = None) -> ttnn.Tensor:
    """
    Create ALiBi (Attention with Linear Biases) slopes.

    Args:
        num_heads: Number of attention heads
        device: Target device

    Returns:
        ALiBi slopes tensor
    """
    raise NotImplementedError("Create ALiBi slopes for position bias")


def add_position_embeddings(
    token_embeddings: ttnn.Tensor,
    position_embeddings: ttnn.Tensor,
    impl_config: PositionEmbeddingImplConfig,
    scale: Optional[float] = None,
) -> ttnn.Tensor:
    """
    Add position embeddings to token embeddings.

    Args:
        token_embeddings: Token embeddings tensor
        position_embeddings: Position embeddings tensor
        impl_config: Implementation configuration
        scale: Optional scaling factor

    Returns:
        Combined embeddings
    """
    raise NotImplementedError("Add position to token embeddings")
