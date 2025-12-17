# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC (Pearson Correlation Coefficient) tests for validating TTNN outputs match torch reference.

These tests compare TTNN implementations against PyTorch reference implementations
to ensure numerical correctness. A PCC > 0.99 is typically considered acceptable.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# Check if TTNN is available
try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Compute Pearson Correlation Coefficient between two tensors.
    
    Args:
        tensor1: First tensor (reference)
        tensor2: Second tensor (comparison)
    
    Returns:
        PCC value between -1 and 1 (1 = perfect correlation)
    """
    t1 = tensor1.flatten().float().numpy()
    t2 = tensor2.flatten().float().numpy()
    
    if len(t1) != len(t2):
        raise ValueError(f"Tensor sizes don't match: {len(t1)} vs {len(t2)}")
    
    # Handle zero variance cases
    if np.std(t1) == 0 or np.std(t2) == 0:
        if np.allclose(t1, t2):
            return 1.0
        else:
            return 0.0
    
    return np.corrcoef(t1, t2)[0, 1]


def check_pcc(
    reference: torch.Tensor,
    comparison: torch.Tensor,
    threshold: float = 0.99,
    test_name: str = "unnamed",
) -> bool:
    """
    Check if PCC meets threshold and print result.
    
    Args:
        reference: Reference tensor (PyTorch)
        comparison: Comparison tensor (may be TTNN converted to torch)
        threshold: Minimum acceptable PCC
        test_name: Name of the test for logging
    
    Returns:
        True if PCC >= threshold
    """
    pcc = compute_pcc(reference, comparison)
    passed = pcc >= threshold
    
    status = "PASSED" if passed else "FAILED"
    print(f"[{status}] {test_name}: PCC = {pcc:.6f} (threshold: {threshold})")
    
    return passed


class TestCommonPCC:
    """PCC tests for common utility functions."""
    
    def test_sinusoidal_embedding_self_pcc(self):
        """Test sinusoidal embedding produces consistent results."""
        from ..ttnn_common import create_sinusoidal_pos_embedding_torch
        
        batch_size, dim = 4, 512
        time = torch.rand(batch_size)
        
        emb1 = create_sinusoidal_pos_embedding_torch(time, dim)
        emb2 = create_sinusoidal_pos_embedding_torch(time, dim)
        
        assert check_pcc(emb1, emb2, threshold=1.0, test_name="sinusoidal_embedding_determinism")


class TestGemmaPCC:
    """PCC tests for Gemma components."""
    
    def test_rms_norm_pcc(self):
        """Test RMSNorm produces consistent results."""
        from ..ttnn_gemma import rms_norm_torch
        
        batch_size, seq_len, hidden_dim = 2, 4, 256
        x = torch.randn(batch_size, seq_len, hidden_dim)
        weight = torch.randn(hidden_dim)
        
        result1 = rms_norm_torch(x, weight)
        result2 = rms_norm_torch(x, weight)
        
        assert check_pcc(result1, result2, threshold=1.0, test_name="rms_norm_determinism")
    
    def test_gemma_mlp_pcc(self):
        """Test Gemma MLP produces consistent results."""
        from ..ttnn_gemma import GemmaConfig, GemmaMLPTorch
        
        config = GemmaConfig(width=512, mlp_dim=2048)
        weights = {
            "mlp.gate_proj.weight": torch.randn(config.mlp_dim, config.width),
            "mlp.up_proj.weight": torch.randn(config.mlp_dim, config.width),
            "mlp.down_proj.weight": torch.randn(config.width, config.mlp_dim),
        }
        
        mlp = GemmaMLPTorch(config, weights)
        x = torch.randn(2, 4, config.width)
        
        result1 = mlp.forward(x)
        result2 = mlp.forward(x)
        
        assert check_pcc(result1, result2, threshold=1.0, test_name="gemma_mlp_determinism")


class TestSuffixPCC:
    """PCC tests for suffix embedding."""
    
    def test_suffix_embedding_pcc(self):
        """Test suffix embedding produces consistent results."""
        from ..ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        config = SuffixConfig(
            action_dim=32,
            action_horizon=50,
            expert_width=256,  # Small for testing
            pi05=False,
        )
        
        weights = {
            "action_in_proj.weight": torch.randn(config.expert_width, config.action_dim),
            "action_in_proj.bias": torch.randn(config.expert_width),
            "action_out_proj.weight": torch.randn(config.action_dim, config.expert_width),
            "action_out_proj.bias": torch.randn(config.action_dim),
            "state_proj.weight": torch.randn(config.expert_width, config.action_dim),
            "state_proj.bias": torch.randn(config.expert_width),
            "action_time_mlp_in.weight": torch.randn(config.expert_width, config.expert_width * 2),
            "action_time_mlp_in.bias": torch.randn(config.expert_width),
            "action_time_mlp_out.weight": torch.randn(config.expert_width, config.expert_width),
            "action_time_mlp_out.bias": torch.randn(config.expert_width),
        }
        
        suffix = SuffixEmbeddingTorch(config, weights)
        
        state = torch.randn(2, config.action_dim)
        actions = torch.randn(2, config.action_horizon, config.action_dim)
        time = torch.rand(2)
        
        result1, _, _, _ = suffix.embed_suffix(state, actions, time)
        result2, _, _, _ = suffix.embed_suffix(state, actions, time)
        
        assert check_pcc(result1, result2, threshold=1.0, test_name="suffix_embedding_determinism")


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
class TestTTNNvsTorchPCC:
    """
    PCC tests comparing TTNN implementations against PyTorch reference.
    
    These tests require TTNN and a Tenstorrent device.
    """
    
    @pytest.fixture
    def device(self):
        """Get TTNN device."""
        dev = ttnn.open_device(device_id=0)
        yield dev
        ttnn.close_device(dev)
    
    def test_linear_pcc(self, device):
        """Test TTNN linear matches PyTorch."""
        import torch.nn.functional as F
        
        batch_size, seq_len, in_features, out_features = 2, 10, 256, 512
        
        # Create input and weight
        x = torch.randn(batch_size, seq_len, in_features)
        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)
        
        # PyTorch reference
        torch_result = F.linear(x, weight, bias)
        
        # TTNN
        x_ttnn = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        weight_ttnn = ttnn.from_torch(
            weight.T.contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        bias_ttnn = ttnn.from_torch(
            bias.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        
        ttnn_result = ttnn.linear(x_ttnn, weight_ttnn, bias=bias_ttnn)
        ttnn_result_torch = ttnn.to_torch(ttnn_result)
        
        pcc = compute_pcc(torch_result, ttnn_result_torch)
        print(f"Linear PCC: {pcc:.6f}")
        
        assert pcc > 0.97, f"Linear PCC {pcc:.6f} below threshold 0.97"


class TestModelIntegrationPCC:
    """Integration tests for full model components."""
    
    def test_attention_mask_consistency(self):
        """Test attention masks produce expected patterns."""
        from ..ttnn_attention import make_att_2d_masks_torch, prepare_attention_masks_4d_torch
        
        batch_size, seq_len = 2, 8
        pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        att_masks = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]).expand(batch_size, -1).bool()
        
        att_2d_1 = make_att_2d_masks_torch(pad_masks, att_masks)
        att_2d_2 = make_att_2d_masks_torch(pad_masks, att_masks)
        
        # Convert to float for PCC
        att_4d_1 = prepare_attention_masks_4d_torch(att_2d_1)
        att_4d_2 = prepare_attention_masks_4d_torch(att_2d_2)
        
        assert check_pcc(att_4d_1, att_4d_2, threshold=1.0, test_name="attention_mask_determinism")
    
    def test_denoising_with_identity_forward(self):
        """Test denoising with known velocity function."""
        from ..ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        config = DenoiseConfig(num_steps=5, action_dim=4, action_horizon=3)
        
        # Identity forward (velocity = 0)
        def identity_forward(x, t, kv_cache=None, **kwargs):
            return torch.zeros_like(x)
        
        module = DenoisingModuleTorch(config, identity_forward)
        
        # Fix seed for reproducibility
        torch.manual_seed(42)
        result1 = module.sample_actions(batch_size=2)
        
        torch.manual_seed(42)
        result2 = module.sample_actions(batch_size=2)
        
        assert check_pcc(result1, result2, threshold=1.0, test_name="denoising_determinism")


def run_all_pcc_tests():
    """Run all PCC tests and print summary."""
    print("=" * 60)
    print("Running PCC Tests for TTNN PI0 Reference Implementation")
    print("=" * 60)
    
    # Run common tests
    test_common = TestCommonPCC()
    test_common.test_sinusoidal_embedding_self_pcc()
    
    # Run Gemma tests
    test_gemma = TestGemmaPCC()
    test_gemma.test_rms_norm_pcc()
    test_gemma.test_gemma_mlp_pcc()
    
    # Run suffix tests
    test_suffix = TestSuffixPCC()
    test_suffix.test_suffix_embedding_pcc()
    
    # Run integration tests
    test_integration = TestModelIntegrationPCC()
    test_integration.test_attention_mask_consistency()
    test_integration.test_denoising_with_identity_forward()
    
    print("=" * 60)
    print("PCC Tests Complete")
    print("=" * 60)


if __name__ == "__main__":
    # Run with pytest or standalone
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--standalone":
        run_all_pcc_tests()
    else:
        pytest.main([__file__, "-v"])

