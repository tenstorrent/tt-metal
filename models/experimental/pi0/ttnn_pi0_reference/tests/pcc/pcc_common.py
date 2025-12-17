# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for ttnn_common.py module.

Tests sinusoidal embeddings, safe_cat, and other utility functions.
"""

import torch

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    pytest = None

# Decorator for skipping tests when pytest is not available
def skipif_no_pytest(condition, reason):
    if PYTEST_AVAILABLE:
        return pytest.mark.skipif(condition, reason=reason)
    else:
        def decorator(func):
            return func
        return decorator

from . import TTNN_AVAILABLE, compute_pcc, check_pcc, torch_to_ttnn, ttnn_to_torch

if TTNN_AVAILABLE:
    import ttnn


class TestSinusoidalEmbeddingPCC:
    """PCC tests for sinusoidal position embeddings."""
    
    def test_torch_determinism(self):
        """Test PyTorch sinusoidal embedding is deterministic."""
        from ...ttnn_common import create_sinusoidal_pos_embedding_torch
        
        batch_size, dim = 8, 1024
        time = torch.rand(batch_size)
        
        emb1 = create_sinusoidal_pos_embedding_torch(time, dim)
        emb2 = create_sinusoidal_pos_embedding_torch(time, dim)
        
        assert check_pcc(emb1, emb2, threshold=1.0, test_name="sinusoidal_torch_determinism")
    
    def test_different_dimensions(self):
        """Test consistency across different dimensions."""
        from ...ttnn_common import create_sinusoidal_pos_embedding_torch
        
        time = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        
        for dim in [128, 256, 512, 1024, 2048]:
            emb1 = create_sinusoidal_pos_embedding_torch(time, dim)
            emb2 = create_sinusoidal_pos_embedding_torch(time, dim)
            
            pcc = compute_pcc(emb1, emb2)
            assert pcc > 0.9999, f"Dimension {dim}: PCC = {pcc}"
    
    def test_embedding_range(self):
        """Test embedding values are in reasonable range."""
        from ...ttnn_common import create_sinusoidal_pos_embedding_torch
        
        time = torch.rand(100)
        emb = create_sinusoidal_pos_embedding_torch(time, 512)
        
        # Sin/cos values should be in [-1, 1]
        assert emb.min() >= -1.1, f"Min value {emb.min()} out of range"
        assert emb.max() <= 1.1, f"Max value {emb.max()} out of range"
    
    @skipif_no_pytest(not TTNN_AVAILABLE, reason="TTNN not available")
    def test_ttnn_vs_torch(self):
        """Test TTNN sinusoidal embedding matches PyTorch."""
        from ...ttnn_common import (
            create_sinusoidal_pos_embedding_torch,
            create_sinusoidal_pos_embedding_ttnn,
        )
        
        device = ttnn.open_device(device_id=0)
        
        try:
            batch_size, dim = 4, 512
            time_torch = torch.rand(batch_size)
            
            # PyTorch reference
            torch_emb = create_sinusoidal_pos_embedding_torch(time_torch, dim)
            
            # TTNN
            time_ttnn = torch_to_ttnn(time_torch, device)
            ttnn_emb = create_sinusoidal_pos_embedding_ttnn(time_ttnn, dim, device=device)
            ttnn_emb_torch = ttnn_to_torch(ttnn_emb)
            
            assert check_pcc(
                torch_emb, ttnn_emb_torch,
                threshold=0.99,
                test_name="sinusoidal_ttnn_vs_torch"
            )
        finally:
            ttnn.close_device(device)


class TestSafeCatPCC:
    """PCC tests for safe concatenation."""
    
    def test_concatenation_consistency(self):
        """Test concatenation produces consistent results."""
        from ...ttnn_common import safe_cat_torch
        
        t1 = torch.randn(4, 10, 256)
        t2 = torch.randn(4, 5, 256)
        t3 = torch.randn(4, 8, 256)
        
        result1 = safe_cat_torch([t1, t2, t3], dim=1)
        result2 = safe_cat_torch([t1, t2, t3], dim=1)
        
        assert check_pcc(result1, result2, threshold=1.0, test_name="safe_cat_consistency")
    
    def test_dtype_preservation(self):
        """Test that dtype conversion maintains numerical values."""
        from ...ttnn_common import safe_cat_torch
        
        t1 = torch.randn(4, 10, 256, dtype=torch.float32)
        t2 = torch.randn(4, 5, 256, dtype=torch.float16)
        
        result = safe_cat_torch([t1, t2], dim=1)
        
        # Check first part matches original
        pcc = compute_pcc(t1, result[:, :10, :])
        assert pcc == 1.0, f"First tensor PCC = {pcc}"
    
    @skipif_no_pytest(not TTNN_AVAILABLE, reason="TTNN not available")
    def test_ttnn_concat(self):
        """Test TTNN concat matches PyTorch."""
        from ...ttnn_common import safe_cat_torch, safe_cat_ttnn
        
        device = ttnn.open_device(device_id=0)
        
        try:
            t1 = torch.randn(2, 10, 128)
            t2 = torch.randn(2, 5, 128)
            
            # PyTorch
            torch_result = safe_cat_torch([t1, t2], dim=1)
            
            # TTNN
            t1_ttnn = torch_to_ttnn(t1, device)
            t2_ttnn = torch_to_ttnn(t2, device)
            ttnn_result = safe_cat_ttnn([t1_ttnn, t2_ttnn], dim=1)
            ttnn_result_torch = ttnn_to_torch(ttnn_result)
            
            assert check_pcc(
                torch_result, ttnn_result_torch,
                threshold=0.999,
                test_name="concat_ttnn_vs_torch"
            )
        finally:
            ttnn.close_device(device)


class TestPositionIdsPCC:
    """PCC tests for position ID computation."""
    
    def test_position_ids_consistency(self):
        """Test position IDs are computed consistently."""
        from ...ttnn_common import compute_position_ids_torch
        
        pad_masks = torch.tensor([
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
        ])
        
        pos1 = compute_position_ids_torch(pad_masks)
        pos2 = compute_position_ids_torch(pad_masks)
        
        # Should be exactly equal (integer computation)
        assert torch.equal(pos1, pos2), "Position IDs not equal"
    
    def test_position_ids_values(self):
        """Test position IDs have correct values."""
        from ...ttnn_common import compute_position_ids_torch
        
        pad_masks = torch.tensor([[True, True, True, True]])
        pos = compute_position_ids_torch(pad_masks)
        
        expected = torch.tensor([[0, 1, 2, 3]])
        assert torch.equal(pos, expected), f"Expected {expected}, got {pos}"


class TestSamplingPCC:
    """PCC tests for sampling functions."""
    
    def test_noise_statistics(self):
        """Test noise has correct statistics."""
        from ...ttnn_common import sample_noise_torch
        
        # Large sample for statistical accuracy
        shape = (1000, 50, 32)
        noise = sample_noise_torch(shape)
        
        mean = noise.mean().item()
        std = noise.std().item()
        
        assert abs(mean) < 0.05, f"Mean {mean} too far from 0"
        assert abs(std - 1.0) < 0.05, f"Std {std} too far from 1.0"
    
    def test_time_distribution(self):
        """Test time sampling distribution."""
        from ...ttnn_common import sample_time_torch
        
        # Sample many times
        times = sample_time_torch(10000)
        
        # Should be in valid range
        assert times.min() >= 0.001
        assert times.max() <= 0.999
        
        # Mean should be around 0.6 for Beta(1.5, 1.0) scaled
        mean = times.mean().item()
        assert 0.4 < mean < 0.8, f"Mean {mean} outside expected range"


def run_pcc_common_tests():
    """Run all PCC tests for common module."""
    print("=" * 60)
    print("PCC Tests: ttnn_common.py")
    print("=" * 60)
    
    test_sin = TestSinusoidalEmbeddingPCC()
    test_sin.test_torch_determinism()
    test_sin.test_different_dimensions()
    test_sin.test_embedding_range()
    
    test_cat = TestSafeCatPCC()
    test_cat.test_concatenation_consistency()
    test_cat.test_dtype_preservation()
    
    test_pos = TestPositionIdsPCC()
    test_pos.test_position_ids_consistency()
    test_pos.test_position_ids_values()
    
    test_sample = TestSamplingPCC()
    test_sample.test_noise_statistics()
    test_sample.test_time_distribution()
    
    print("\n✓ All PCC tests for ttnn_common.py passed!")


if __name__ == "__main__":
    run_pcc_common_tests()

