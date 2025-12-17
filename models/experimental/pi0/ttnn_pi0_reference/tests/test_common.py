# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for common utility functions.
"""

import math
import pytest
import torch

from ..ttnn_common import (
    create_sinusoidal_pos_embedding_torch,
    safe_cat_torch,
    compute_position_ids_torch,
    sample_noise_torch,
    sample_time_torch,
)


class TestSinusoidalEmbedding:
    """Tests for sinusoidal position embedding."""
    
    def test_output_shape(self):
        """Test output shape is correct."""
        batch_size = 4
        dim = 512
        time = torch.rand(batch_size)
        
        emb = create_sinusoidal_pos_embedding_torch(time, dim)
        
        assert emb.shape == (batch_size, dim)
    
    def test_dimension_must_be_even(self):
        """Test that odd dimension raises error."""
        time = torch.rand(4)
        
        with pytest.raises(ValueError, match="divisible by 2"):
            create_sinusoidal_pos_embedding_torch(time, 511)
    
    def test_different_timesteps_different_embeddings(self):
        """Test that different timesteps produce different embeddings."""
        dim = 256
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([0.5])
        t3 = torch.tensor([1.0])
        
        e1 = create_sinusoidal_pos_embedding_torch(t1, dim)
        e2 = create_sinusoidal_pos_embedding_torch(t2, dim)
        e3 = create_sinusoidal_pos_embedding_torch(t3, dim)
        
        assert not torch.allclose(e1, e2)
        assert not torch.allclose(e2, e3)
        assert not torch.allclose(e1, e3)
    
    def test_deterministic(self):
        """Test same input produces same output."""
        time = torch.tensor([0.25, 0.5, 0.75])
        dim = 128
        
        e1 = create_sinusoidal_pos_embedding_torch(time, dim)
        e2 = create_sinusoidal_pos_embedding_torch(time, dim)
        
        assert torch.allclose(e1, e2)


class TestSafeCat:
    """Tests for safe concatenation."""
    
    def test_basic_concatenation(self):
        """Test basic tensor concatenation."""
        t1 = torch.randn(2, 3)
        t2 = torch.randn(2, 4)
        
        result = safe_cat_torch([t1, t2], dim=-1)
        
        assert result.shape == (2, 7)
    
    def test_dtype_conversion(self):
        """Test that tensors are converted to first tensor's dtype."""
        t1 = torch.randn(2, 3, dtype=torch.float32)
        t2 = torch.randn(2, 4, dtype=torch.float16)
        
        result = safe_cat_torch([t1, t2], dim=-1)
        
        assert result.dtype == torch.float32
    
    def test_empty_list_raises(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="empty list"):
            safe_cat_torch([])


class TestPositionIds:
    """Tests for position ID computation."""
    
    def test_basic_computation(self):
        """Test basic position ID computation."""
        # All valid tokens
        pad_masks = torch.tensor([[True, True, True, True]])
        
        pos_ids = compute_position_ids_torch(pad_masks)
        
        expected = torch.tensor([[0, 1, 2, 3]])
        assert torch.equal(pos_ids, expected)
    
    def test_with_padding(self):
        """Test position IDs with padding."""
        # Last two tokens are padding
        pad_masks = torch.tensor([[True, True, False, False]])
        
        pos_ids = compute_position_ids_torch(pad_masks)
        
        # Padding positions have invalid IDs (negative or repeated)
        assert pos_ids[0, 0] == 0
        assert pos_ids[0, 1] == 1


class TestSampling:
    """Tests for sampling functions."""
    
    def test_noise_shape(self):
        """Test noise tensor shape."""
        shape = (2, 50, 32)
        noise = sample_noise_torch(shape)
        
        assert noise.shape == shape
    
    def test_noise_statistics(self):
        """Test noise is approximately standard normal."""
        shape = (100, 50, 32)
        noise = sample_noise_torch(shape)
        
        mean = noise.mean().item()
        std = noise.std().item()
        
        assert abs(mean) < 0.1  # Mean close to 0
        assert abs(std - 1.0) < 0.1  # Std close to 1
    
    def test_time_range(self):
        """Test sampled time is in valid range."""
        batch_size = 100
        time = sample_time_torch(batch_size)
        
        assert time.shape == (batch_size,)
        assert (time >= 0.001).all()
        assert (time <= 0.999).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

