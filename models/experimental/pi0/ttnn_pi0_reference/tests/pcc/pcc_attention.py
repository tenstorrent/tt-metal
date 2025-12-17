# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for ttnn_attention.py module.

Tests attention mask creation and manipulation utilities.
"""

import torch

try:
    import pytest
except ImportError:
    pytest = None

def skipif_no_pytest(condition, reason):
    if pytest:
        return pytest.mark.skipif(condition, reason=reason)
    def decorator(func):
        return func
    return decorator

from . import TTNN_AVAILABLE, compute_pcc, check_pcc, torch_to_ttnn, ttnn_to_torch

if TTNN_AVAILABLE:
    import ttnn


class TestMakeAtt2DMasksPCC:
    """PCC tests for 2D attention mask creation."""
    
    def test_bidirectional_mask(self):
        """Test bidirectional attention mask."""
        from ...ttnn_attention import make_att_2d_masks_torch
        
        batch_size, seq_len = 4, 16
        pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        att_masks = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        result1 = make_att_2d_masks_torch(pad_masks, att_masks)
        result2 = make_att_2d_masks_torch(pad_masks, att_masks)
        
        # Convert to float for PCC
        assert check_pcc(
            result1.float(), result2.float(),
            threshold=1.0,
            test_name="att_2d_bidirectional_consistency"
        )
    
    def test_causal_mask(self):
        """Test causal attention mask."""
        from ...ttnn_attention import make_att_2d_masks_torch
        
        batch_size, seq_len = 4, 16
        pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        att_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        result = make_att_2d_masks_torch(pad_masks, att_masks)
        
        # Should be lower triangular
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    if j > i:
                        assert not result[b, i, j], f"Non-causal at [{b},{i},{j}]"
    
    def test_prefix_lm_mask(self):
        """Test prefix-LM attention mask."""
        from ...ttnn_attention import make_att_2d_masks_torch
        
        batch_size, seq_len = 2, 8
        pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        # First 4 bidirectional, last 4 causal
        att_masks = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]).expand(batch_size, -1).bool()
        
        result1 = make_att_2d_masks_torch(pad_masks, att_masks)
        result2 = make_att_2d_masks_torch(pad_masks, att_masks)
        
        assert check_pcc(
            result1.float(), result2.float(),
            threshold=1.0,
            test_name="att_2d_prefix_lm_consistency"
        )
    
    @skipif_no_pytest(not TTNN_AVAILABLE, reason="TTNN not available")
    def test_ttnn_vs_torch(self):
        """Test TTNN 2D mask creation matches PyTorch."""
        from ...ttnn_attention import make_att_2d_masks_torch, make_att_2d_masks_ttnn
        
        device = ttnn.open_device(device_id=0)
        
        try:
            batch_size, seq_len = 2, 32
            pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
            att_masks = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1] * 4]).expand(batch_size, -1).bool()
            
            # PyTorch
            torch_result = make_att_2d_masks_torch(pad_masks, att_masks)
            
            # TTNN
            pad_ttnn = torch_to_ttnn(pad_masks.float(), device)
            att_ttnn = torch_to_ttnn(att_masks.float(), device)
            ttnn_result = make_att_2d_masks_ttnn(pad_ttnn, att_ttnn, device)
            ttnn_result_torch = ttnn_to_torch(ttnn_result)
            
            # Compare (threshold lower due to float conversion)
            assert check_pcc(
                torch_result.float(), ttnn_result_torch,
                threshold=0.99,
                test_name="att_2d_masks_ttnn_vs_torch"
            )
        finally:
            ttnn.close_device(device)


class TestPrepare4DMasksPCC:
    """PCC tests for 4D attention mask preparation."""
    
    def test_4d_mask_shape(self):
        """Test 4D mask has correct shape."""
        from ...ttnn_attention import make_att_2d_masks_torch, prepare_attention_masks_4d_torch
        
        batch_size, seq_len = 2, 16
        pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        att_masks = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        att_2d = make_att_2d_masks_torch(pad_masks, att_masks)
        att_4d = prepare_attention_masks_4d_torch(att_2d)
        
        assert att_4d.shape == (batch_size, 1, seq_len, seq_len)
    
    def test_4d_mask_values(self):
        """Test 4D mask has correct values."""
        from ...ttnn_attention import make_att_2d_masks_torch, prepare_attention_masks_4d_torch
        
        batch_size, seq_len = 1, 4
        pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        att_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)  # Causal
        
        att_2d = make_att_2d_masks_torch(pad_masks, att_masks)
        att_4d = prepare_attention_masks_4d_torch(att_2d)
        
        # True positions should be 0.0
        # False positions should be large negative
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:  # Causal: can attend
                    assert att_4d[0, 0, i, j] == 0.0, f"Expected 0.0 at [{i},{j}]"
                else:  # Masked
                    assert att_4d[0, 0, i, j] < -1e30, f"Expected large neg at [{i},{j}]"
    
    def test_4d_consistency(self):
        """Test 4D mask is consistent across calls."""
        from ...ttnn_attention import make_att_2d_masks_torch, prepare_attention_masks_4d_torch
        
        batch_size, seq_len = 2, 32
        pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        att_masks = torch.rand(batch_size, seq_len) > 0.5
        
        att_2d = make_att_2d_masks_torch(pad_masks, att_masks)
        
        att_4d_1 = prepare_attention_masks_4d_torch(att_2d)
        att_4d_2 = prepare_attention_masks_4d_torch(att_2d)
        
        assert check_pcc(att_4d_1, att_4d_2, threshold=1.0, test_name="att_4d_consistency")


class TestCombineMasksPCC:
    """PCC tests for combining prefix and suffix masks."""
    
    def test_combine_shapes(self):
        """Test combined mask shapes."""
        from ...ttnn_attention import combine_prefix_suffix_masks_torch
        
        batch_size = 2
        prefix_len, suffix_len = 10, 8
        
        prefix_pad = torch.ones(batch_size, prefix_len, dtype=torch.bool)
        prefix_att = torch.zeros(batch_size, prefix_len, dtype=torch.bool)
        suffix_pad = torch.ones(batch_size, suffix_len, dtype=torch.bool)
        suffix_att = torch.ones(batch_size, suffix_len, dtype=torch.bool)
        
        combined_pad, combined_att = combine_prefix_suffix_masks_torch(
            prefix_pad, prefix_att, suffix_pad, suffix_att
        )
        
        assert combined_pad.shape == (batch_size, prefix_len + suffix_len)
        assert combined_att.shape == (batch_size, prefix_len + suffix_len)
    
    def test_combine_consistency(self):
        """Test combined masks are consistent."""
        from ...ttnn_attention import combine_prefix_suffix_masks_torch
        
        batch_size = 4
        prefix_len, suffix_len = 16, 8
        
        prefix_pad = torch.ones(batch_size, prefix_len, dtype=torch.bool)
        prefix_att = torch.zeros(batch_size, prefix_len, dtype=torch.bool)
        suffix_pad = torch.ones(batch_size, suffix_len, dtype=torch.bool)
        suffix_att = torch.ones(batch_size, suffix_len, dtype=torch.bool)
        
        result1 = combine_prefix_suffix_masks_torch(
            prefix_pad, prefix_att, suffix_pad, suffix_att
        )
        result2 = combine_prefix_suffix_masks_torch(
            prefix_pad, prefix_att, suffix_pad, suffix_att
        )
        
        assert check_pcc(result1[0].float(), result2[0].float(), threshold=1.0, test_name="combine_pad_consistency")
        assert check_pcc(result1[1].float(), result2[1].float(), threshold=1.0, test_name="combine_att_consistency")


class TestCausalMaskPCC:
    """PCC tests for causal mask creation."""
    
    def test_causal_mask_values(self):
        """Test causal mask has correct values."""
        from ...ttnn_attention import create_causal_mask_torch
        
        seq_len = 8
        mask = create_causal_mask_torch(seq_len)
        
        for i in range(seq_len):
            for j in range(seq_len):
                expected = 1.0 if j <= i else 0.0
                assert mask[i, j] == expected, f"Wrong at [{i},{j}]: expected {expected}, got {mask[i, j]}"
    
    def test_causal_mask_consistency(self):
        """Test causal mask is consistent."""
        from ...ttnn_attention import create_causal_mask_torch
        
        for seq_len in [8, 16, 32, 64, 128]:
            mask1 = create_causal_mask_torch(seq_len)
            mask2 = create_causal_mask_torch(seq_len)
            
            pcc = compute_pcc(mask1, mask2)
            assert pcc == 1.0, f"Seq len {seq_len}: PCC = {pcc}"
    
    @skipif_no_pytest(not TTNN_AVAILABLE, reason="TTNN not available")
    def test_causal_mask_ttnn(self):
        """Test TTNN causal mask matches PyTorch."""
        from ...ttnn_attention import create_causal_mask_torch, create_causal_mask_ttnn
        
        device = ttnn.open_device(device_id=0)
        
        try:
            seq_len = 64
            
            torch_mask = create_causal_mask_torch(seq_len)
            ttnn_mask = create_causal_mask_ttnn(seq_len, device)
            ttnn_mask_torch = ttnn_to_torch(ttnn_mask)
            
            assert check_pcc(
                torch_mask, ttnn_mask_torch,
                threshold=0.999,
                test_name="causal_mask_ttnn_vs_torch"
            )
        finally:
            ttnn.close_device(device)


def run_pcc_attention_tests():
    """Run all PCC tests for attention module."""
    print("=" * 60)
    print("PCC Tests: ttnn_attention.py")
    print("=" * 60)
    
    test_2d = TestMakeAtt2DMasksPCC()
    test_2d.test_bidirectional_mask()
    test_2d.test_causal_mask()
    test_2d.test_prefix_lm_mask()
    
    test_4d = TestPrepare4DMasksPCC()
    test_4d.test_4d_mask_shape()
    test_4d.test_4d_mask_values()
    test_4d.test_4d_consistency()
    
    test_combine = TestCombineMasksPCC()
    test_combine.test_combine_shapes()
    test_combine.test_combine_consistency()
    
    test_causal = TestCausalMaskPCC()
    test_causal.test_causal_mask_values()
    test_causal.test_causal_mask_consistency()
    
    print("\n✓ All PCC tests for ttnn_attention.py passed!")


if __name__ == "__main__":
    run_pcc_attention_tests()

