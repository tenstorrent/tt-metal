# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for attention mask utilities.
"""

import pytest
import torch

from ..ttnn_attention import (
    make_att_2d_masks_torch,
    prepare_attention_masks_4d_torch,
    combine_prefix_suffix_masks_torch,
    create_causal_mask_torch,
    create_prefix_suffix_cross_mask_torch,
)


class TestMakeAtt2DMasks:
    """Tests for 2D attention mask creation."""
    
    def test_bidirectional_attention(self):
        """Test all zeros creates bidirectional attention."""
        batch_size, seq_len = 2, 4
        pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        att_masks = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        result = make_att_2d_masks_torch(pad_masks, att_masks)
        
        # All tokens can attend to all tokens
        assert result.shape == (batch_size, seq_len, seq_len)
        assert result.all()
    
    def test_causal_attention(self):
        """Test all ones creates causal attention."""
        batch_size, seq_len = 1, 4
        pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        att_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        result = make_att_2d_masks_torch(pad_masks, att_masks)
        
        # Should be lower triangular
        expected = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        assert torch.equal(result[0], expected)
    
    def test_prefix_lm_attention(self):
        """Test prefix-LM attention pattern."""
        batch_size, seq_len = 1, 6
        pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        # First 3 bidirectional, last 3 causal
        att_masks = torch.tensor([[False, False, False, True, True, True]])
        
        result = make_att_2d_masks_torch(pad_masks, att_masks)
        
        # First 3 tokens attend to each other
        assert result[0, 0:3, 0:3].all()
        
        # Token 3 can attend to tokens 0-3
        assert result[0, 3, 0:4].all()
        assert not result[0, 3, 4:6].any()
    
    def test_with_padding(self):
        """Test padding is properly masked."""
        batch_size, seq_len = 1, 4
        # Last token is padding
        pad_masks = torch.tensor([[True, True, True, False]])
        att_masks = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        result = make_att_2d_masks_torch(pad_masks, att_masks)
        
        # No token attends to padding
        assert not result[0, :, 3].any()
        # Padding token doesn't attend to anything
        assert not result[0, 3, :].any()


class TestPrepare4DMasks:
    """Tests for 4D attention mask preparation."""
    
    def test_output_shape(self):
        """Test output shape is correct."""
        batch_size, seq_len = 2, 4
        att_2d = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
        
        result = prepare_attention_masks_4d_torch(att_2d)
        
        assert result.shape == (batch_size, 1, seq_len, seq_len)
    
    def test_mask_values(self):
        """Test mask values are correct."""
        batch_size, seq_len = 1, 2
        att_2d = torch.tensor([[[True, False], [True, True]]])
        
        result = prepare_attention_masks_4d_torch(att_2d)
        
        # True -> 0.0, False -> large negative
        assert result[0, 0, 0, 0] == 0.0
        assert result[0, 0, 0, 1] < -1e30  # Large negative
        assert result[0, 0, 1, 0] == 0.0
        assert result[0, 0, 1, 1] == 0.0


class TestCombineMasks:
    """Tests for combining prefix and suffix masks."""
    
    def test_concatenation(self):
        """Test masks are properly concatenated."""
        batch_size = 2
        prefix_len, suffix_len = 3, 4
        
        prefix_pad = torch.ones(batch_size, prefix_len, dtype=torch.bool)
        prefix_att = torch.zeros(batch_size, prefix_len, dtype=torch.bool)
        suffix_pad = torch.ones(batch_size, suffix_len, dtype=torch.bool)
        suffix_att = torch.ones(batch_size, suffix_len, dtype=torch.bool)
        
        combined_pad, combined_att = combine_prefix_suffix_masks_torch(
            prefix_pad, prefix_att, suffix_pad, suffix_att
        )
        
        assert combined_pad.shape == (batch_size, prefix_len + suffix_len)
        assert combined_att.shape == (batch_size, prefix_len + suffix_len)
        
        # Check values
        assert combined_att[0, 0:prefix_len].sum() == 0  # Prefix is bidirectional
        assert combined_att[0, prefix_len:].sum() == suffix_len  # Suffix is causal


class TestCausalMask:
    """Tests for causal mask creation."""
    
    def test_shape(self):
        """Test output shape."""
        seq_len = 4
        mask = create_causal_mask_torch(seq_len)
        
        assert mask.shape == (seq_len, seq_len)
    
    def test_lower_triangular(self):
        """Test mask is lower triangular."""
        seq_len = 4
        mask = create_causal_mask_torch(seq_len)
        
        expected = torch.tril(torch.ones(seq_len, seq_len))
        assert torch.equal(mask, expected)


class TestPrefixSuffixCrossMask:
    """Tests for prefix-suffix cross-attention mask."""
    
    def test_shape(self):
        """Test output shape."""
        prefix_len, suffix_len = 5, 3
        batch_size = 2
        
        mask = create_prefix_suffix_cross_mask_torch(prefix_len, suffix_len, batch_size)
        
        assert mask.shape == (batch_size, prefix_len + suffix_len, prefix_len + suffix_len)
    
    def test_prefix_bidirectional(self):
        """Test prefix tokens can attend bidirectionally."""
        prefix_len, suffix_len = 3, 2
        
        mask = create_prefix_suffix_cross_mask_torch(prefix_len, suffix_len)
        
        # Prefix tokens attend to all prefix tokens
        assert mask[0, :prefix_len, :prefix_len].all()
    
    def test_suffix_attends_to_prefix(self):
        """Test suffix tokens can attend to all prefix tokens."""
        prefix_len, suffix_len = 3, 2
        
        mask = create_prefix_suffix_cross_mask_torch(prefix_len, suffix_len)
        
        # Suffix tokens attend to all prefix tokens
        assert mask[0, prefix_len:, :prefix_len].all()
    
    def test_suffix_causal(self):
        """Test suffix tokens have causal attention within suffix."""
        prefix_len, suffix_len = 3, 4
        
        mask = create_prefix_suffix_cross_mask_torch(prefix_len, suffix_len)
        
        # Suffix is causal
        suffix_part = mask[0, prefix_len:, prefix_len:]
        expected = torch.tril(torch.ones(suffix_len, suffix_len, dtype=torch.bool))
        assert torch.equal(suffix_part, expected)
    
    def test_prefix_doesnt_attend_to_suffix(self):
        """Test prefix tokens don't attend to suffix."""
        prefix_len, suffix_len = 3, 2
        
        mask = create_prefix_suffix_cross_mask_torch(prefix_len, suffix_len)
        
        # Prefix doesn't see suffix
        assert not mask[0, :prefix_len, prefix_len:].any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

