"""
Sliding Window Attention implementation.

Sliding window attention limits attention to a local window, reducing
computational complexity while maintaining local context. Used in models
like Mistral.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import ttnn

from .mha import MultiHeadAttentionImplConfig, MultiHeadAttentionSpec
from .mha import get_default_impl_config as get_default_mha_impl_config


@dataclass
class SlidingWindowAttentionSpec(MultiHeadAttentionSpec):
    """
    Specification for sliding window attention.

    Extends MultiHeadAttentionSpec with sliding window specific parameters.

    Attributes:
        sliding_window_size: Size of the attention window
        global_tokens: Number of global tokens that attend to all positions (optional)
    """

    sliding_window_size: int = 4096
    global_tokens: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.sliding_window_size <= 0:
            raise ValueError(f"sliding_window_size must be positive, got {self.sliding_window_size}")
        if self.global_tokens < 0:
            raise ValueError(f"global_tokens must be non-negative, got {self.global_tokens}")

    def validate(self):
        """Validate spec constraints."""
        super().validate()
        assert self.sliding_window_size > 0, "sliding_window_size must be positive"
        assert self.global_tokens >= 0, "global_tokens must be non-negative"


def get_default_impl_config(
    spec: SlidingWindowAttentionSpec, device: str, mode: Literal["prefill", "decode"], strategy: str = "default"
) -> MultiHeadAttentionImplConfig:
    """
    Return default implementation configuration for sliding window attention.

    Args:
        spec: Sliding window attention specification
        device: Target device (e.g., "N150", "T3000", "cpu")
        mode: Execution strategy mode (prefill or decode)
        strategy: Performance strategy hint

    Returns:
        Implementation configuration with device-specific defaults
    """
    # Start with base MHA config and customize for sliding window
    config = get_default_mha_impl_config(spec, device, mode, strategy)

    # Sliding window specific optimizations
    if device.startswith("N150") or device.startswith("T3000"):
        # Can use smaller block size due to local attention pattern
        config.block_size = 32

    return config


def create_sliding_window_mask(
    seq_len: int, window_size: int, dtype: ttnn.DataType = ttnn.bfloat16, device: Optional[ttnn.Device] = None
) -> ttnn.Tensor:
    """
    Create a sliding window attention mask.

    Args:
        seq_len: Sequence length
        window_size: Size of the sliding window
        dtype: Data type for the mask
        device: Target device

    Returns:
        Attention mask tensor with sliding window pattern
    """
    raise NotImplementedError("Create sliding window mask")


def prefill_forward(
    hidden_states: ttnn.Tensor, spec: SlidingWindowAttentionSpec, impl_config: MultiHeadAttentionImplConfig, **kwargs
) -> Tuple[ttnn.Tensor, Dict[str, ttnn.Tensor]]:
    """
    Sliding window attention forward pass for prefill mode.

    This is a specialized version that applies sliding window masking.

    Args:
        hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)
        spec: Sliding window attention specification
        impl_config: Implementation configuration
        **kwargs: Additional arguments passed to base attention

    Returns:
        Tuple of:
            - Output tensor of shape (batch, seq_len, hidden_dim)
            - Cache dictionary with 'k' and 'v' tensors for decode
    """
    raise NotImplementedError("Implement sliding window prefill using MHA with custom mask")


def decode_forward(
    hidden_states: ttnn.Tensor,
    spec: SlidingWindowAttentionSpec,
    impl_config: MultiHeadAttentionImplConfig,
    cache: Dict[str, ttnn.Tensor],
    cache_position: int,
    **kwargs,
) -> ttnn.Tensor:
    """
    Sliding window attention forward pass for decode mode.

    Only attends to tokens within the sliding window during decode.

    Args:
        hidden_states: Input tensor of shape (batch, 1, hidden_dim)
        spec: Sliding window attention specification
        impl_config: Implementation configuration
        cache: KV cache dictionary with 'k' and 'v' tensors
        cache_position: Current position in the cache
        **kwargs: Additional arguments passed to base attention

    Returns:
        Output tensor of shape (batch, 1, hidden_dim)
    """
    raise NotImplementedError("Implement sliding window decode with limited KV cache access")
