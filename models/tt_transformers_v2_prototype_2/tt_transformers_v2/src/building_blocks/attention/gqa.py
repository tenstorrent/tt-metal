"""
Grouped-Query Attention (GQA) implementation.

GQA is a variant of multi-head attention where multiple query heads share
the same key-value heads, reducing memory and computation for KV cache.
"""

from dataclasses import dataclass

from .mha import MultiHeadAttentionSpec


@dataclass
class GroupedQueryAttentionSpec(MultiHeadAttentionSpec):
    """
    Specification for Grouped-Query Attention.

    This is essentially MultiHeadAttentionSpec with explicit GQA configuration.
    The key difference is that num_kv_heads < num_heads.
    """

    def __post_init__(self):
        super().__post_init__()
        # Ensure this is actually GQA (not MHA)
        if self.num_kv_heads == self.num_heads:
            raise ValueError(
                f"For GQA, num_kv_heads ({self.num_kv_heads}) must be less than "
                f"num_heads ({self.num_heads}). Use MultiHeadAttentionSpec for standard MHA."
            )

    def validate(self):
        """Validate GQA-specific constraints."""
        super().validate()
        assert self.num_kv_heads < self.num_heads, "For GQA, num_kv_heads must be less than num_heads"
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"


def create_gqa_spec(
    hidden_dim: int, num_heads: int, num_kv_heads: int, max_seq_len: int = 2048, **kwargs
) -> GroupedQueryAttentionSpec:
    """
    Convenience function to create a GQA specification.

    Args:
        hidden_dim: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (must be less than num_heads)
        max_seq_len: Maximum sequence length
        **kwargs: Additional arguments passed to GroupedQueryAttentionSpec

    Returns:
        GroupedQueryAttentionSpec instance

    Example:
        >>> # Create GQA with 32 query heads and 8 KV heads (4:1 ratio)
        >>> gqa_spec = create_gqa_spec(
        ...     hidden_dim=4096,
        ...     num_heads=32,
        ...     num_kv_heads=8
        ... )
    """
    if num_kv_heads >= num_heads:
        raise ValueError(f"For GQA, num_kv_heads ({num_kv_heads}) must be less than " f"num_heads ({num_heads})")

    return GroupedQueryAttentionSpec(
        hidden_dim=hidden_dim, num_heads=num_heads, num_kv_heads=num_kv_heads, max_seq_len=max_seq_len, **kwargs
    )


# GQA uses the same forward functions as MHA, just with different num_kv_heads
# Import them for convenience
