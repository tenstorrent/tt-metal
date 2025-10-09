"""
Multi-Head Attention (MHA) implementation.

This module provides the core attention mechanism used in transformer models,
supporting both standard MHA and Grouped-Query Attention (GQA) configurations.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import ttnn


@dataclass
class MultiHeadAttentionSpec:
    """
    Mathematical specification for multi-head attention.

    Attributes:
        hidden_dim: Model dimension (d_model)
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads (for GQA; None means MHA)
        head_dim: Dimension per head (computed from hidden_dim/num_heads if not specified)
        max_seq_len: Maximum sequence length
        attention_dropout: Dropout rate for attention weights
        use_bias: Whether to use bias in QKV projections
        scale_attention_scores: Whether to scale attention scores by 1/sqrt(head_dim)
    """

    hidden_dim: int
    num_heads: int
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_seq_len: int = 2048
    attention_dropout: float = 0.0
    use_bias: bool = False
    scale_attention_scores: bool = True

    def __post_init__(self):
        # Validation first
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")

        # Set default values
        if self.head_dim is None:
            if self.hidden_dim % self.num_heads != 0:
                raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})")
            self.head_dim = self.hidden_dim // self.num_heads

        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        # Additional validation
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")

    def validate(self):
        """Validate spec constraints."""
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.num_kv_heads > 0, "num_kv_heads must be positive"
        assert (
            self.hidden_dim % self.num_heads == 0
        ), f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        assert self.head_dim * self.num_heads == self.hidden_dim, f"head_dim * num_heads must equal hidden_dim"

    @property
    def num_q_heads_per_kv_head(self) -> int:
        """Number of query heads per key-value head (for GQA)."""
        return self.num_heads // self.num_kv_heads


@dataclass
class MultiHeadAttentionImplConfig:
    """
    TTNN-specific implementation configuration for multi-head attention.

    Attributes:
        qkv_dtype: Data type for QKV projections
        output_dtype: Data type for output projection
        compute_dtype: Data type for attention computation
        compute_kernel_config: Kernel configuration for compute operations
        cache_dtype: Data type for KV cache storage
        cache_memory_config: Memory configuration for KV cache
        qkv_memory_config: Memory configuration for QKV projections
        output_memory_config: Memory configuration for output projection
        softmax_memory_config: Memory configuration for softmax
        use_fused_qkv: Whether to use fused QKV projection
        block_size: Block size for blocked attention
    """

    qkv_dtype: ttnn.DataType = ttnn.bfloat16
    output_dtype: ttnn.DataType = ttnn.bfloat16
    compute_dtype: Optional[ttnn.DataType] = None
    compute_kernel_config: Optional[dict] = None
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    cache_memory_config: Optional[ttnn.MemoryConfig] = None
    qkv_memory_config: Optional[ttnn.MemoryConfig] = None
    output_memory_config: Optional[ttnn.MemoryConfig] = None
    softmax_memory_config: Optional[ttnn.MemoryConfig] = None
    use_fused_qkv: bool = True
    block_size: int = 64

    def __post_init__(self):
        if self.compute_dtype is None:
            self.compute_dtype = self.qkv_dtype


def get_default_impl_config(
    spec: MultiHeadAttentionSpec, device: str, mode: Literal["prefill", "decode"], strategy: str = "default"
) -> MultiHeadAttentionImplConfig:
    """
    Return default implementation configuration for the given device and mode.

    Args:
        spec: Multi-head attention specification
        device: Target device (e.g., "N150", "T3000", "cpu")
        mode: Execution strategy mode (prefill or decode)
        strategy: Performance strategy hint

    Returns:
        Implementation configuration with device-specific defaults
    """
    if device.startswith("N150"):
        if mode == "prefill":
            return MultiHeadAttentionImplConfig(
                qkv_dtype=ttnn.bfloat16,
                output_dtype=ttnn.bfloat16,
                use_fused_qkv=True,
                cache_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:  # decode
            return MultiHeadAttentionImplConfig(
                qkv_dtype=ttnn.bfloat16,
                output_dtype=ttnn.bfloat16,
                cache_dtype=ttnn.bfloat8_b,
                cache_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
    elif device.startswith("T3000"):
        if mode == "prefill":
            return MultiHeadAttentionImplConfig(
                qkv_dtype=ttnn.bfloat16,
            )
        else:
            return MultiHeadAttentionImplConfig(
                qkv_dtype=ttnn.bfloat16,
                cache_dtype=ttnn.bfloat8_b,
            )
    else:
        # CPU or default fallback
        return MultiHeadAttentionImplConfig()


def prefill_forward(
    hidden_states: ttnn.Tensor,
    spec: MultiHeadAttentionSpec,
    impl_config: MultiHeadAttentionImplConfig,
    q_weight: Optional[ttnn.Tensor] = None,
    k_weight: Optional[ttnn.Tensor] = None,
    v_weight: Optional[ttnn.Tensor] = None,
    qkv_weight: Optional[ttnn.Tensor] = None,
    output_weight: ttnn.Tensor = None,
    q_bias: Optional[ttnn.Tensor] = None,
    k_bias: Optional[ttnn.Tensor] = None,
    v_bias: Optional[ttnn.Tensor] = None,
    qkv_bias: Optional[ttnn.Tensor] = None,
    output_bias: Optional[ttnn.Tensor] = None,
    position_ids: Optional[ttnn.Tensor] = None,
    attention_mask: Optional[ttnn.Tensor] = None,
    **kwargs,
) -> Tuple[ttnn.Tensor, Dict[str, ttnn.Tensor]]:
    """
    Multi-head attention forward pass for prefill mode (process entire sequence).

    Args:
        hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)
        spec: Multi-head attention specification
        impl_config: Implementation configuration
        q_weight, k_weight, v_weight: Separate QKV weights (if not fused)
        qkv_weight: Fused QKV weight (if impl_config.use_fused_qkv)
        output_weight: Output projection weight
        q_bias, k_bias, v_bias, qkv_bias, output_bias: Optional biases
        position_ids: Position indices (if using positional encoding)
        attention_mask: Attention mask tensor
        **kwargs: Additional arguments for future extensions

    Returns:
        Tuple of:
            - Output tensor of shape (batch, seq_len, hidden_dim)
            - Cache dictionary with 'k' and 'v' tensors for decode
    """
    raise NotImplementedError("Refactor from TTTv1 models/tt_transformers/tt/attention.py")


def decode_forward(
    hidden_states: ttnn.Tensor,
    spec: MultiHeadAttentionSpec,
    impl_config: MultiHeadAttentionImplConfig,
    cache: Dict[str, ttnn.Tensor],
    q_weight: Optional[ttnn.Tensor] = None,
    k_weight: Optional[ttnn.Tensor] = None,
    v_weight: Optional[ttnn.Tensor] = None,
    qkv_weight: Optional[ttnn.Tensor] = None,
    output_weight: ttnn.Tensor = None,
    q_bias: Optional[ttnn.Tensor] = None,
    k_bias: Optional[ttnn.Tensor] = None,
    v_bias: Optional[ttnn.Tensor] = None,
    qkv_bias: Optional[ttnn.Tensor] = None,
    output_bias: Optional[ttnn.Tensor] = None,
    position_ids: Optional[ttnn.Tensor] = None,
    attention_mask: Optional[ttnn.Tensor] = None,
    cache_position: Optional[int] = None,
    **kwargs,
) -> ttnn.Tensor:
    """
    Multi-head attention forward pass for decode mode (single token with KV cache).

    Args:
        hidden_states: Input tensor of shape (batch, 1, hidden_dim)
        spec: Multi-head attention specification
        impl_config: Implementation configuration
        cache: KV cache dictionary with 'k' and 'v' tensors
        q_weight, k_weight, v_weight: Separate QKV weights (if not fused)
        qkv_weight: Fused QKV weight (if impl_config.use_fused_qkv)
        output_weight: Output projection weight
        q_bias, k_bias, v_bias, qkv_bias, output_bias: Optional biases
        position_ids: Position indices (if using positional encoding)
        attention_mask: Attention mask tensor
        cache_position: Current position in the cache
        **kwargs: Additional arguments for future extensions

    Returns:
        Output tensor of shape (batch, 1, hidden_dim)
    """
    raise NotImplementedError("Refactor from TTTv1 models/tt_transformers/tt/attention.py")
