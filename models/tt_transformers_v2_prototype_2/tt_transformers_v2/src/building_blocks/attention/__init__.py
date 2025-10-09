"""
Attention mechanisms for transformer models.

This module provides various attention implementations including:
- Multi-head attention (MHA)
- Grouped-query attention (GQA)
- Flash attention
- Sliding window attention
"""

__all__ = [
    # Multi-head attention
    "MultiHeadAttention",
    "MultiHeadAttentionSpec",
    "MultiHeadAttentionImplConfig",
    "mha_prefill_forward",
    "mha_decode_forward",
    "get_default_mha_impl_config",
    # Grouped-query attention
    "GroupedQueryAttention",
    "GroupedQueryAttentionSpec",
    "GroupedQueryAttentionImplConfig",
    "create_gqa_spec",
    # Flash attention
    "FlashAttentionImplConfig",
    "flash_attention_forward",
    "is_flash_attention_available",
    "get_default_flash_impl_config",
    # Sliding window attention
    "SlidingWindowAttention",
    "SlidingWindowAttentionSpec",
    "sliding_prefill_forward",
    "sliding_decode_forward",
    "create_sliding_window_mask",
]

# Multi-head attention imports
from .mha import (
    MultiHeadAttentionSpec,
    MultiHeadAttentionImplConfig,
    prefill_forward as mha_prefill_forward,
    decode_forward as mha_decode_forward,
    get_default_impl_config as get_default_mha_impl_config,
)

# Grouped-query attention imports
from .gqa import (
    GroupedQueryAttentionSpec,
    GroupedQueryAttentionImplConfig,
    create_gqa_spec,
    gqa_prefill_forward,
    gqa_decode_forward,
)

# Flash attention imports
from .flash import (
    FlashAttentionImplConfig,
    flash_attention_forward,
    is_flash_attention_available,
    get_default_flash_impl_config,
)

# Sliding window attention imports
from .sliding import (
    SlidingWindowAttentionSpec,
    prefill_forward as sliding_prefill_forward,
    decode_forward as sliding_decode_forward,
    create_sliding_window_mask,
)

# Convenience class names (following design proposal pattern)
MultiHeadAttention = MultiHeadAttentionSpec
GroupedQueryAttention = GroupedQueryAttentionSpec
SlidingWindowAttention = SlidingWindowAttentionSpec
