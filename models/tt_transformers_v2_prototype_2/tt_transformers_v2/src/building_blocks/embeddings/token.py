"""
Token embedding implementation.

Provides token embeddings that convert token IDs to dense vectors.
"""

from dataclasses import dataclass
from typing import Optional

import ttnn


@dataclass
class TokenEmbeddingSpec:
    """
    Mathematical specification for token embeddings.

    Attributes:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of the embedding vectors
        padding_idx: Index of padding token (if any)
        max_norm: Maximum norm for embeddings
        norm_type: Type of norm to use (if max_norm is set)
        scale_grad_by_freq: Scale gradients by word frequency
        sparse: Whether to use sparse gradients
    """

    vocab_size: int
    embedding_dim: int
    padding_idx: Optional[int] = None
    max_norm: Optional[float] = None
    norm_type: float = 2.0
    scale_grad_by_freq: bool = False
    sparse: bool = False

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.padding_idx is not None:
            if self.padding_idx < 0 or self.padding_idx >= self.vocab_size:
                raise ValueError(f"padding_idx must be in [0, {self.vocab_size}), got {self.padding_idx}")

    def validate(self):
        """Validate spec constraints."""
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        if self.padding_idx is not None:
            assert 0 <= self.padding_idx < self.vocab_size, f"padding_idx must be in [0, {self.vocab_size})"


@dataclass
class TokenEmbeddingImplConfig:
    """
    TTNN-specific implementation configuration for token embeddings.

    Attributes:
        dtype: Data type for embeddings
        output_dtype: Data type for output (if different from embeddings)
        memory_config: Memory configuration for TTNN operations
        embedding_memory_config: Specific memory config for embedding table
        use_embedding_table_on_device: Whether to keep embedding table on device
        prefetch_size: Number of embeddings to prefetch (for optimization)
    """

    dtype: ttnn.DataType = ttnn.bfloat16
    output_dtype: Optional[ttnn.DataType] = None
    memory_config: Optional[ttnn.MemoryConfig] = None
    embedding_memory_config: Optional[ttnn.MemoryConfig] = None
    use_embedding_table_on_device: bool = True
    prefetch_size: int = 0

    def __post_init__(self):
        if self.output_dtype is None:
            self.output_dtype = self.dtype


def get_default_impl_config(
    spec: TokenEmbeddingSpec, device: str, mode: Optional[str] = None, strategy: str = "default"
) -> TokenEmbeddingImplConfig:
    """
    Return default implementation configuration for the given device.

    Args:
        spec: Token embedding specification
        device: Target device (e.g., "N150", "T3000", "cpu")
        mode: Execution mode (not typically used for embeddings)
        strategy: Performance strategy hint

    Returns:
        Implementation configuration with device-specific defaults
    """
    if device.startswith("N150"):
        return TokenEmbeddingImplConfig(
            dtype=ttnn.bfloat16,
            use_embedding_table_on_device=True,
            embedding_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    elif device.startswith("T3000"):
        return TokenEmbeddingImplConfig(
            dtype=ttnn.bfloat16,
            use_embedding_table_on_device=True,
            embedding_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            prefetch_size=16,  # T3000 can benefit from prefetching
        )
    else:
        # CPU or default fallback
        return TokenEmbeddingImplConfig(
            use_embedding_table_on_device=False,
        )


def forward(
    input_ids: ttnn.Tensor,
    spec: TokenEmbeddingSpec,
    impl_config: TokenEmbeddingImplConfig,
    weight: ttnn.Tensor,
    **kwargs,
) -> ttnn.Tensor:
    """
    Apply token embeddings to input IDs.

    Args:
        input_ids: Input token IDs of shape (batch, seq_len)
        spec: Token embedding specification
        impl_config: Implementation configuration
        weight: Embedding table of shape (vocab_size, embedding_dim)
        **kwargs: Additional arguments for future extensions

    Returns:
        Embeddings tensor of shape (batch, seq_len, embedding_dim)
    """
    raise NotImplementedError("Refactor from TTTv1 models/tt_transformers/tt/embedding.py")


def create_embedding_table(
    spec: TokenEmbeddingSpec,
    impl_config: TokenEmbeddingImplConfig,
    device: Optional[ttnn.Device] = None,
    initializer: Optional[str] = "normal",
) -> ttnn.Tensor:
    """
    Create and initialize an embedding table.

    Args:
        spec: Token embedding specification
        impl_config: Implementation configuration
        device: Target device
        initializer: Initialization method ("normal", "uniform", "xavier")

    Returns:
        Initialized embedding table tensor
    """
    raise NotImplementedError("Create embedding table with proper initialization")
