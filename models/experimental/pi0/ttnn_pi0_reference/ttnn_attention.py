# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Attention mask utilities for TTNN PI0 implementation.

This module provides functions for creating and managing attention masks
that control which tokens can attend to which other tokens in the transformer.

Attention patterns in PI0:
    - Prefix (images + language): Full bidirectional attention
    - Suffix (state + actions): Causal attention
    - Cross-attention: Suffix can attend to prefix, but not vice versa
"""

from typing import Optional, Tuple

import torch

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None


def make_att_2d_masks_torch(
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Create 2D attention masks from padding and attention masks.
    
    Tokens can attend to valid input tokens which have a cumulative
    mask_ar smaller or equal to theirs.
    
    Examples:
        [[0 0 0 1 1 1]]: prefix-lm attention. First 3 tokens attend
                        bidirectionally, last 3 have causal attention.
        [[1 1 1 1 1 1]]: pure causal attention.
    
    Args:
        pad_masks: bool[B, N] - True if part of input, False if padding
        att_masks: bool[B, N] - 1 where previous tokens cannot depend on it
    
    Returns:
        2D boolean attention mask of shape (B, N, N)
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2D, got {att_masks.ndim}D")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2D, got {pad_masks.ndim}D")
    
    # Cumulative sum creates attention boundaries
    cumsum = torch.cumsum(att_masks.long(), dim=1)
    
    # Token i can attend to token j if cumsum[j] <= cumsum[i]
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    
    # Apply padding mask: can only attend to non-padding tokens
    pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
    
    return att_2d_masks & pad_2d_masks


def make_att_2d_masks_ttnn(
    pad_masks: "ttnn.Tensor",
    att_masks: "ttnn.Tensor",
    device: Optional["ttnn.Device"] = None,
) -> "ttnn.Tensor":
    """
    Create 2D attention masks from padding and attention masks (TTNN version).
    
    Args:
        pad_masks: TTNN tensor (B, N) - True if valid token
        att_masks: TTNN tensor (B, N) - attention boundary markers
        device: TTNN device
    
    Returns:
        2D TTNN attention mask of shape (B, N, N)
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")
    
    # Convert to torch, compute, convert back
    # (TTNN cumsum and broadcasting are less flexible)
    pad_masks_torch = ttnn.to_torch(pad_masks)
    att_masks_torch = ttnn.to_torch(att_masks)
    
    result = make_att_2d_masks_torch(pad_masks_torch, att_masks_torch)
    
    if device is None:
        device = pad_masks.device()
    
    return ttnn.from_torch(
        result,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def prepare_attention_masks_4d_torch(
    att_2d_masks: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert 2D attention masks to 4D format for transformer models.
    
    Args:
        att_2d_masks: 2D boolean mask of shape (B, N, N)
        dtype: Output dtype
    
    Returns:
        4D attention mask of shape (B, 1, N, N) with values:
            - 0.0 for positions that can attend
            - -2.3819763e38 for positions that cannot attend (masked)
    """
    # Add head dimension
    att_4d = att_2d_masks[:, None, :, :]
    
    # Convert to float with masking values
    # True (can attend) -> 0.0, False (masked) -> large negative
    mask_value = -2.3819763e38
    return torch.where(att_4d, torch.zeros_like(att_4d, dtype=dtype), 
                       torch.full_like(att_4d, mask_value, dtype=dtype))


def prepare_attention_masks_4d_ttnn(
    att_2d_masks: "ttnn.Tensor",
    device: Optional["ttnn.Device"] = None,
) -> "ttnn.Tensor":
    """
    Convert 2D attention masks to 4D format (TTNN version).
    
    Args:
        att_2d_masks: 2D TTNN mask of shape (B, N, N)
        device: TTNN device
    
    Returns:
        4D TTNN attention mask of shape (B, 1, N, N)
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")
    
    # Get shape and add head dimension
    shape = att_2d_masks.shape
    batch_size, seq_len = shape[0], shape[1]
    
    # Reshape to 4D: (B, N, N) -> (B, 1, N, N)
    att_4d = ttnn.reshape(att_2d_masks, (batch_size, 1, seq_len, seq_len))
    
    # Apply masking values using ttnn.where
    mask_value = -2.3819763e38
    zeros = ttnn.zeros_like(att_4d)
    mask_fill = ttnn.full_like(att_4d, mask_value)
    
    return ttnn.where(att_4d, zeros, mask_fill)


def combine_prefix_suffix_masks_torch(
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    suffix_pad_masks: torch.Tensor,
    suffix_att_masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine prefix and suffix attention masks for joint processing.
    
    Args:
        prefix_pad_masks: Padding masks for prefix (images + language)
        prefix_att_masks: Attention masks for prefix
        suffix_pad_masks: Padding masks for suffix (state + actions)
        suffix_att_masks: Attention masks for suffix
    
    Returns:
        Tuple of (combined_pad_masks, combined_att_masks)
    """
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    return pad_masks, att_masks


def create_causal_mask_torch(
    seq_len: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Create a simple causal attention mask.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        dtype: Data type for mask
    
    Returns:
        2D boolean mask where each token can only attend to previous tokens
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=dtype, device=device))


def create_causal_mask_ttnn(
    seq_len: int,
    device: "ttnn.Device",
) -> "ttnn.Tensor":
    """
    Create a causal attention mask (TTNN version).
    
    Args:
        seq_len: Sequence length
        device: TTNN device
    
    Returns:
        TTNN causal mask of shape (seq_len, seq_len)
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")
    
    # Create on CPU then transfer
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32))
    
    return ttnn.from_torch(
        mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def create_prefix_suffix_cross_mask_torch(
    prefix_len: int,
    suffix_len: int,
    batch_size: int = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create attention mask for prefix-suffix cross-attention.
    
    Suffix tokens can attend to all prefix tokens and causally to suffix.
    Prefix tokens can only attend to each other (bidirectional).
    
    Args:
        prefix_len: Length of prefix sequence
        suffix_len: Length of suffix sequence
        batch_size: Batch size
        device: Device to create mask on
    
    Returns:
        2D attention mask of shape (B, prefix_len + suffix_len, prefix_len + suffix_len)
    """
    total_len = prefix_len + suffix_len
    
    # Start with full mask (all False = no attention)
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    
    # Prefix tokens attend to all prefix tokens (bidirectional)
    mask[:prefix_len, :prefix_len] = True
    
    # Suffix tokens attend to all prefix tokens
    mask[prefix_len:, :prefix_len] = True
    
    # Suffix tokens attend causally to suffix tokens
    suffix_causal = torch.tril(torch.ones(suffix_len, suffix_len, dtype=torch.bool, device=device))
    mask[prefix_len:, prefix_len:] = suffix_causal
    
    # Expand for batch
    return mask.unsqueeze(0).expand(batch_size, -1, -1)


class AttentionMaskUtils:
    """
    Utility class for creating and managing attention masks.
    
    Provides a clean interface for all attention mask operations.
    """
    
    @staticmethod
    def make_att_2d_masks(pad_masks, att_masks):
        """Create 2D attention masks. Auto-selects torch or ttnn."""
        if isinstance(pad_masks, torch.Tensor):
            return make_att_2d_masks_torch(pad_masks, att_masks)
        else:
            return make_att_2d_masks_ttnn(pad_masks, att_masks)
    
    @staticmethod
    def prepare_attention_masks_4d(att_2d_masks, **kwargs):
        """Convert 2D to 4D attention masks. Auto-selects torch or ttnn."""
        if isinstance(att_2d_masks, torch.Tensor):
            return prepare_attention_masks_4d_torch(att_2d_masks, **kwargs)
        else:
            return prepare_attention_masks_4d_ttnn(att_2d_masks, **kwargs)
    
    @staticmethod
    def combine_prefix_suffix_masks(
        prefix_pad_masks,
        prefix_att_masks,
        suffix_pad_masks,
        suffix_att_masks,
    ):
        """Combine prefix and suffix masks."""
        if isinstance(prefix_pad_masks, torch.Tensor):
            return combine_prefix_suffix_masks_torch(
                prefix_pad_masks, prefix_att_masks,
                suffix_pad_masks, suffix_att_masks,
            )
        else:
            # TTNN version
            pad_masks = ttnn.concat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = ttnn.concat([prefix_att_masks, suffix_att_masks], dim=1)
            return pad_masks, att_masks
    
    @staticmethod
    def create_causal_mask(seq_len, device=None):
        """Create causal mask. Auto-selects torch or ttnn based on device type."""
        if device is None or isinstance(device, torch.device):
            return create_causal_mask_torch(seq_len, device)
        else:
            return create_causal_mask_ttnn(seq_len, device)


# Default exports
make_att_2d_masks = make_att_2d_masks_torch
prepare_attention_masks_4d = prepare_attention_masks_4d_torch
combine_prefix_suffix_masks = combine_prefix_suffix_masks_torch
create_causal_mask = create_causal_mask_torch

