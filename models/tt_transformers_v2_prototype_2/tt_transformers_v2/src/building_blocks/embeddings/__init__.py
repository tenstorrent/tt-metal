"""
Embedding building blocks for transformers.

This module provides:
- Token embeddings
- Position embeddings (learned, sinusoidal, ALiBi)
- Rotary position embeddings (RoPE)
"""

__all__ = [
    # Token embeddings
    "TokenEmbedding",
    "TokenEmbeddingSpec",
    "TokenEmbeddingImplConfig",
    "token_embedding_forward",
    "get_default_token_impl_config",
    "create_embedding_table",
    # Position embeddings
    "PositionEmbedding",
    "PositionEmbeddingSpec",
    "PositionEmbeddingImplConfig",
    "position_embedding_forward",
    "get_default_position_impl_config",
    "create_sinusoidal_embeddings",
    "create_alibi_slopes",
    "add_position_embeddings",
    # Rotary embeddings
    "RoPE",
    "RoPESpec",
    "RoPEImplConfig",
    "compute_rope_frequencies",
    "apply_rotary_embeddings",
    "get_default_rope_impl_config",
]

# Token embedding imports
from .token import (
    TokenEmbeddingSpec,
    TokenEmbeddingImplConfig,
    forward as token_embedding_forward,
    get_default_impl_config as get_default_token_impl_config,
    create_embedding_table,
)

# Position embedding imports
from .position import (
    PositionEmbeddingSpec,
    PositionEmbeddingImplConfig,
    forward as position_embedding_forward,
    get_default_impl_config as get_default_position_impl_config,
    create_sinusoidal_embeddings,
    create_alibi_slopes,
    add_position_embeddings,
)

# Rotary embedding imports
from .rotary import (
    RoPESpec,
    RoPEImplConfig,
    compute_frequencies as compute_rope_frequencies,
    apply_rotary_embeddings,
    get_default_impl_config as get_default_rope_impl_config,
)

# Convenience class names
TokenEmbedding = TokenEmbeddingSpec
PositionEmbedding = PositionEmbeddingSpec
RoPE = RoPESpec
