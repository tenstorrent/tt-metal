"""
Attention mask utilities for PI-Zero PyTorch model.

This module provides functions for creating and managing attention masks that
control which tokens can attend to which other tokens in the transformer.

Use Case:
    - Creates 2D attention masks from padding and attention masks
    - Prepares 4D attention masks for transformer models
    - Combines prefix and suffix attention masks
    - Ensures proper causal attention and prevents prefix from attending to suffix
"""

import torch
from torch import Tensor


class AttentionMaskUtils:
    """
    Utility class for creating and managing attention masks.
    
    Use Case:
        Provides a clean interface for all attention mask operations,
        making it easier to understand and maintain the attention logic.
    """
    
    @staticmethod
    def make_att_2d_masks(pad_masks, att_masks):
        """
        Create 2D attention masks from padding and attention masks.
        
        Copied from big_vision. Tokens can attend to valid input tokens which
        have a cumulative mask_ar smaller or equal to theirs.
        
        Examples:
            [[1 1 1 1 1 1]]: pure causal attention.
            [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend
                between themselves and the last 3 tokens have causal attention.
            [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks.
        
        Args:
            pad_masks: bool[B, N] true if part of input, false if padding
            att_masks: int32[B, N] mask that's 1 where previous tokens cannot
                      depend on it and 0 where it shares the same attention mask
        
        Returns:
            2D boolean attention mask of shape (B, N, N)
        
        Use Case:
            Creates the core attention pattern that controls information flow
            in the transformer. Critical for ensuring prefix tokens can attend
            to each other but suffix tokens have causal attention.
        """
        if att_masks.ndim != 2:
            raise ValueError(f"att_masks must be 2D, got {att_masks.ndim}D")
        if pad_masks.ndim != 2:
            raise ValueError(f"pad_masks must be 2D, got {pad_masks.ndim}D")

        cumsum = torch.cumsum(att_masks, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks
    
    @staticmethod
    def prepare_attention_masks_4d(att_2d_masks):
        """
        Convert 2D attention masks to 4D format for transformer models.
        
        Args:
            att_2d_masks: 2D boolean mask of shape (B, N, N)
        
        Returns:
            4D attention mask of shape (B, 1, N, N) with values:
            - 0.0 for positions that can attend
            - -2.3819763e38 for positions that cannot attend (masked)
        
        Use Case:
            Transformers expect 4D attention masks with a head dimension.
            This function converts our 2D masks to the required format and
            applies the masking value used by the model.
        """
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)
    
    @staticmethod
    def combine_prefix_suffix_masks(prefix_pad_masks, prefix_att_masks, suffix_pad_masks, suffix_att_masks):
        """
        Combine prefix and suffix attention masks for joint processing.
        
        Args:
            prefix_pad_masks: Padding masks for prefix (images + language)
            prefix_att_masks: Attention masks for prefix
            suffix_pad_masks: Padding masks for suffix (state + actions)
            suffix_att_masks: Attention masks for suffix
        
        Returns:
            Tuple of (combined_pad_masks, combined_att_masks)
        
        Use Case:
            When processing prefix and suffix together in the forward pass,
            we need to combine their masks. This ensures proper attention
            patterns across the entire sequence.
        """
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        return pad_masks, att_masks
    
    @staticmethod
    def create_causal_mask(seq_len, device):
        """
        Create a simple causal attention mask.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
        
        Returns:
            2D boolean mask where each token can only attend to previous tokens
        
        Use Case:
            Creates a standard causal mask for autoregressive models.
            Used as a building block for more complex attention patterns.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask


# Standalone function version for backward compatibility
def make_att_2d_masks(pad_masks, att_masks):
    """
    Standalone function version of make_att_2d_masks.
    
    This is provided for backward compatibility and direct usage.
    See AttentionMaskUtils.make_att_2d_masks for details.
    """
    return AttentionMaskUtils.make_att_2d_masks(pad_masks, att_masks)

