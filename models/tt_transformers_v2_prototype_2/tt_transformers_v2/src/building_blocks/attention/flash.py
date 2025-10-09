"""
Flash Attention implementation.

Flash attention is a memory-efficient attention algorithm that reduces memory
footprint from O(N²) to O(N) while maintaining exact attention computation.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import ttnn

from .mha import MultiHeadAttentionSpec


@dataclass
class FlashAttentionImplConfig:
    """
    TTNN-specific implementation configuration for Flash Attention.

    Attributes:
        dtype: Data type for computations
        compute_dtype: Data type for attention computation
        block_size_q: Block size for queries
        block_size_kv: Block size for keys/values
        num_warps: Number of warps for kernel execution
        pre_norm: Whether to apply pre-normalization
        causal: Whether to use causal masking
        window_size: Optional sliding window size
        alibi_slopes: Optional ALiBi slopes for position bias
    """

    dtype: ttnn.DataType = ttnn.bfloat16
    compute_dtype: Optional[ttnn.DataType] = None
    block_size_q: int = 128
    block_size_kv: int = 128
    num_warps: int = 8
    pre_norm: bool = False
    causal: bool = True
    window_size: Optional[int] = None
    alibi_slopes: Optional[ttnn.Tensor] = None

    def __post_init__(self):
        if self.compute_dtype is None:
            self.compute_dtype = self.dtype


def get_default_flash_impl_config(
    spec: MultiHeadAttentionSpec, device: str, mode: Literal["prefill", "decode"], strategy: str = "default"
) -> FlashAttentionImplConfig:
    """
    Return default Flash Attention configuration for the given device.

    Args:
        spec: Multi-head attention specification
        device: Target device (e.g., "N150", "T3000", "cpu")
        mode: Execution strategy mode (prefill or decode)
        strategy: Performance strategy hint

    Returns:
        Flash attention specific configuration
    """
    if device.startswith("N150"):
        if mode == "prefill":
            return FlashAttentionImplConfig(
                dtype=ttnn.bfloat16,
                block_size_q=128,
                block_size_kv=128,
                causal=True,
            )
        else:  # Flash attention typically not used for decode
            raise ValueError("Flash attention is typically only used in prefill mode")
    elif device.startswith("T3000"):
        if mode == "prefill":
            return FlashAttentionImplConfig(
                dtype=ttnn.bfloat16,
                block_size_q=256,  # Larger blocks for T3000
                block_size_kv=256,
                causal=True,
            )
        else:
            raise ValueError("Flash attention is typically only used in prefill mode")
    else:
        # CPU fallback - Flash attention requires specific hardware support
        raise NotImplementedError("Flash attention requires hardware support")


def flash_attention_forward(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    spec: MultiHeadAttentionSpec,
    impl_config: FlashAttentionImplConfig,
    attention_mask: Optional[ttnn.Tensor] = None,
    **kwargs,
) -> ttnn.Tensor:
    """
    Flash attention forward pass.

    Implements the memory-efficient flash attention algorithm.

    Args:
        query: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        key: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        value: Value tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        spec: Multi-head attention specification
        impl_config: Flash attention implementation configuration
        attention_mask: Optional attention mask
        **kwargs: Additional arguments

    Returns:
        Output tensor of shape (batch, seq_len, num_heads, head_dim)
    """
    raise NotImplementedError("Implement Flash Attention kernel")


def is_flash_attention_available(device: str) -> bool:
    """
    Check if Flash Attention is available on the given device.

    Args:
        device: Target device identifier

    Returns:
        True if Flash Attention is supported, False otherwise
    """
    # Flash attention requires specific hardware capabilities
    supported_devices = ["N150", "N300", "T3000", "TG", "TGG"]
    return any(device.startswith(d) for d in supported_devices)
