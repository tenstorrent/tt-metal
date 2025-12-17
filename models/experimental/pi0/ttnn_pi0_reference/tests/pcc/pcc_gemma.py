# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for ttnn_gemma.py module.

Tests Gemma transformer components: RMSNorm, RoPE, Multi-Query Attention,
GeGLU MLP, and full transformer blocks.
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


def create_attention_weights(config):
    """Create mock attention weights."""
    return {
        "self_attn.q_proj.weight": torch.randn(config.num_heads * config.head_dim, config.width),
        "self_attn.k_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
        "self_attn.v_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
        "self_attn.o_proj.weight": torch.randn(config.width, config.num_heads * config.head_dim),
    }


def create_mlp_weights(config):
    """Create mock MLP weights."""
    return {
        "mlp.gate_proj.weight": torch.randn(config.mlp_dim, config.width),
        "mlp.up_proj.weight": torch.randn(config.mlp_dim, config.width),
        "mlp.down_proj.weight": torch.randn(config.width, config.mlp_dim),
    }


def create_block_weights(config):
    """Create mock full block weights."""
    weights = {
        "input_layernorm.weight": torch.randn(config.width),
        "post_attention_layernorm.weight": torch.randn(config.width),
    }
    weights.update(create_attention_weights(config))
    weights.update(create_mlp_weights(config))
    return weights


class TestRMSNormPCC:
    """PCC tests for RMSNorm."""
    
    def test_rms_norm_consistency(self):
        """Test RMSNorm produces consistent results."""
        from ...ttnn_gemma import rms_norm_torch
        
        batch_size, seq_len, hidden_dim = 4, 16, 512
        x = torch.randn(batch_size, seq_len, hidden_dim)
        weight = torch.randn(hidden_dim)
        
        result1 = rms_norm_torch(x, weight)
        result2 = rms_norm_torch(x, weight)
        
        assert check_pcc(result1, result2, threshold=1.0, test_name="rms_norm_consistency")
    
    def test_rms_norm_normalization(self):
        """Test RMSNorm normalizes correctly."""
        from ...ttnn_gemma import rms_norm_torch
        
        batch_size, seq_len, hidden_dim = 4, 16, 512
        # Use large values to test normalization
        x = torch.randn(batch_size, seq_len, hidden_dim) * 100
        weight = torch.zeros(hidden_dim)  # weight + 1 = 1
        
        result = rms_norm_torch(x, weight)
        
        # Check RMS is approximately 1
        rms = (result ** 2).mean(dim=-1).sqrt()
        mean_rms = rms.mean().item()
        
        assert abs(mean_rms - 1.0) < 0.1, f"Mean RMS = {mean_rms}, expected ~1.0"
    
    def test_rms_norm_gemma_style(self):
        """Test RMSNorm uses Gemma-style (weight + 1)."""
        from ...ttnn_gemma import rms_norm_torch
        
        batch_size, seq_len, hidden_dim = 2, 4, 256
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        # With weight = 0, output should equal normalized x (since weight + 1 = 1)
        weight_zero = torch.zeros(hidden_dim)
        result_zero = rms_norm_torch(x, weight_zero)
        
        # With weight = 1, output should be 2x normalized x (since weight + 1 = 2)
        weight_one = torch.ones(hidden_dim)
        result_one = rms_norm_torch(x, weight_one)
        
        # result_one should be ~2x result_zero
        ratio = (result_one / result_zero).mean().item()
        assert abs(ratio - 2.0) < 0.1, f"Ratio = {ratio}, expected ~2.0"


class TestRoPEPCC:
    """PCC tests for Rotary Position Embeddings."""
    
    def test_rope_consistency(self):
        """Test RoPE precomputation is consistent."""
        from ...ttnn_gemma import precompute_freqs_cis_torch
        
        head_dim, max_seq_len = 256, 2048
        
        cos1, sin1 = precompute_freqs_cis_torch(head_dim, max_seq_len)
        cos2, sin2 = precompute_freqs_cis_torch(head_dim, max_seq_len)
        
        assert check_pcc(cos1, cos2, threshold=1.0, test_name="rope_cos_consistency")
        assert check_pcc(sin1, sin2, threshold=1.0, test_name="rope_sin_consistency")
    
    def test_rope_shape(self):
        """Test RoPE tensors have correct shape."""
        from ...ttnn_gemma import precompute_freqs_cis_torch
        
        head_dim, max_seq_len = 256, 2048
        cos, sin = precompute_freqs_cis_torch(head_dim, max_seq_len)
        
        assert cos.shape == (max_seq_len, head_dim // 2)
        assert sin.shape == (max_seq_len, head_dim // 2)
    
    def test_rope_bounded(self):
        """Test RoPE values are bounded."""
        from ...ttnn_gemma import precompute_freqs_cis_torch
        
        head_dim, max_seq_len = 256, 2048
        cos, sin = precompute_freqs_cis_torch(head_dim, max_seq_len)
        
        assert cos.min() >= -1.0 and cos.max() <= 1.0
        assert sin.min() >= -1.0 and sin.max() <= 1.0


class TestGemmaAttentionPCC:
    """PCC tests for Gemma Multi-Query Attention."""
    
    def test_attention_consistency(self):
        """Test attention produces consistent results."""
        from ...ttnn_gemma import GemmaConfig, GemmaAttentionTorch, precompute_freqs_cis_torch
        
        config = GemmaConfig(width=256, num_heads=4, num_kv_heads=1, head_dim=64)
        weights = create_attention_weights(config)
        attention = GemmaAttentionTorch(config, weights, layer_idx=0)
        
        cos, sin = precompute_freqs_cis_torch(config.head_dim, 1024)
        x = torch.randn(2, 16, config.width)
        
        out1, _ = attention.forward(x, cos, sin)
        out2, _ = attention.forward(x, cos, sin)
        
        assert check_pcc(out1, out2, threshold=1.0, test_name="attention_consistency")
    
    def test_attention_output_shape(self):
        """Test attention output has correct shape."""
        from ...ttnn_gemma import GemmaConfig, GemmaAttentionTorch, precompute_freqs_cis_torch
        
        config = GemmaConfig(width=512, num_heads=8, num_kv_heads=1, head_dim=64)
        weights = create_attention_weights(config)
        attention = GemmaAttentionTorch(config, weights, layer_idx=0)
        
        cos, sin = precompute_freqs_cis_torch(config.head_dim, 1024)
        
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, config.width)
        
        output, cache = attention.forward(x, cos, sin)
        
        assert output.shape == (batch_size, seq_len, config.width)
    
    def test_attention_kv_cache(self):
        """Test attention KV cache has correct shape."""
        from ...ttnn_gemma import GemmaConfig, GemmaAttentionTorch, precompute_freqs_cis_torch
        
        config = GemmaConfig(width=256, num_heads=4, num_kv_heads=1, head_dim=64)
        weights = create_attention_weights(config)
        attention = GemmaAttentionTorch(config, weights, layer_idx=0)
        
        cos, sin = precompute_freqs_cis_torch(config.head_dim, 1024)
        
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.width)
        
        _, cache = attention.forward(x, cos, sin, use_cache=True)
        
        assert cache is not None
        k_cache, v_cache = cache
        assert k_cache.shape == (batch_size, config.num_kv_heads, seq_len, config.head_dim)
        assert v_cache.shape == (batch_size, config.num_kv_heads, seq_len, config.head_dim)


class TestGemmaMLPPCC:
    """PCC tests for Gemma GeGLU MLP."""
    
    def test_mlp_consistency(self):
        """Test MLP produces consistent results."""
        from ...ttnn_gemma import GemmaConfig, GemmaMLPTorch
        
        config = GemmaConfig(width=256, mlp_dim=1024)
        weights = create_mlp_weights(config)
        mlp = GemmaMLPTorch(config, weights)
        
        x = torch.randn(2, 16, config.width)
        
        out1 = mlp.forward(x)
        out2 = mlp.forward(x)
        
        assert check_pcc(out1, out2, threshold=1.0, test_name="mlp_consistency")
    
    def test_mlp_output_shape(self):
        """Test MLP output has correct shape."""
        from ...ttnn_gemma import GemmaConfig, GemmaMLPTorch
        
        config = GemmaConfig(width=512, mlp_dim=2048)
        weights = create_mlp_weights(config)
        mlp = GemmaMLPTorch(config, weights)
        
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, config.width)
        
        output = mlp.forward(x)
        
        assert output.shape == (batch_size, seq_len, config.width)
    
    def test_mlp_geglu_activation(self):
        """Test MLP uses GeGLU (gate * up) pattern."""
        from ...ttnn_gemma import GemmaConfig, GemmaMLPTorch
        import torch.nn.functional as F
        
        config = GemmaConfig(width=256, mlp_dim=512)
        weights = create_mlp_weights(config)
        mlp = GemmaMLPTorch(config, weights)
        
        x = torch.randn(2, 4, config.width)
        
        # Manual computation
        gate = F.linear(x, weights["mlp.gate_proj.weight"])
        up = F.linear(x, weights["mlp.up_proj.weight"])
        hidden = F.gelu(gate, approximate="tanh") * up
        expected = F.linear(hidden, weights["mlp.down_proj.weight"])
        
        # MLP forward
        actual = mlp.forward(x)
        
        assert check_pcc(expected, actual, threshold=1.0, test_name="mlp_geglu_pattern")


class TestGemmaBlockPCC:
    """PCC tests for full Gemma transformer block."""
    
    def test_block_consistency(self):
        """Test block produces consistent results."""
        from ...ttnn_gemma import GemmaConfig, GemmaBlockTorch, precompute_freqs_cis_torch
        
        config = GemmaConfig(width=256, num_heads=4, num_kv_heads=1, head_dim=64, mlp_dim=512)
        weights = create_block_weights(config)
        block = GemmaBlockTorch(config, weights, layer_idx=0)
        
        cos, sin = precompute_freqs_cis_torch(config.head_dim, 1024)
        x = torch.randn(2, 16, config.width)
        
        out1, _ = block.forward(x, cos, sin)
        out2, _ = block.forward(x, cos, sin)
        
        assert check_pcc(out1, out2, threshold=1.0, test_name="block_consistency")
    
    def test_block_residual(self):
        """Test block includes residual connections."""
        from ...ttnn_gemma import GemmaConfig, GemmaBlockTorch, precompute_freqs_cis_torch
        
        config = GemmaConfig(width=256, num_heads=4, num_kv_heads=1, head_dim=64, mlp_dim=512)
        weights = create_block_weights(config)
        block = GemmaBlockTorch(config, weights, layer_idx=0)
        
        cos, sin = precompute_freqs_cis_torch(config.head_dim, 1024)
        x = torch.randn(2, 16, config.width)
        
        output, _ = block.forward(x, cos, sin)
        
        # Output should be correlated with input due to residual
        pcc = compute_pcc(x, output)
        assert pcc > 0.1, f"Residual connection weak: PCC = {pcc}"


class TestGemmaConfigsPCC:
    """PCC tests for Gemma configurations."""
    
    def test_gemma_2b_config(self):
        """Test Gemma 2B configuration."""
        from ...ttnn_gemma import GemmaConfig
        
        config = GemmaConfig.gemma_2b()
        
        assert config.width == 2048
        assert config.depth == 18
        assert config.mlp_dim == 16384
        assert config.num_heads == 8
        assert config.num_kv_heads == 1
        assert config.head_dim == 256
    
    def test_gemma_300m_config(self):
        """Test Gemma 300M configuration."""
        from ...ttnn_gemma import GemmaConfig
        
        config = GemmaConfig.gemma_300m()
        
        assert config.width == 1024
        assert config.depth == 18
        assert config.mlp_dim == 4096
        assert config.num_heads == 8
        assert config.num_kv_heads == 1
        assert config.head_dim == 256


def run_pcc_gemma_tests():
    """Run all PCC tests for gemma module."""
    print("=" * 60)
    print("PCC Tests: ttnn_gemma.py")
    print("=" * 60)
    
    test_norm = TestRMSNormPCC()
    test_norm.test_rms_norm_consistency()
    test_norm.test_rms_norm_normalization()
    test_norm.test_rms_norm_gemma_style()
    
    test_rope = TestRoPEPCC()
    test_rope.test_rope_consistency()
    test_rope.test_rope_shape()
    test_rope.test_rope_bounded()
    
    test_attn = TestGemmaAttentionPCC()
    test_attn.test_attention_consistency()
    test_attn.test_attention_output_shape()
    test_attn.test_attention_kv_cache()
    
    test_mlp = TestGemmaMLPPCC()
    test_mlp.test_mlp_consistency()
    test_mlp.test_mlp_output_shape()
    test_mlp.test_mlp_geglu_activation()
    
    test_block = TestGemmaBlockPCC()
    test_block.test_block_consistency()
    test_block.test_block_residual()
    
    test_config = TestGemmaConfigsPCC()
    test_config.test_gemma_2b_config()
    test_config.test_gemma_300m_config()
    
    print("\n✓ All PCC tests for ttnn_gemma.py passed!")


if __name__ == "__main__":
    run_pcc_gemma_tests()

