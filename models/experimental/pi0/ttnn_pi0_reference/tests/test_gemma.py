# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Gemma transformer blocks.
"""

import pytest
import torch

from ..ttnn_gemma import (
    GemmaConfig,
    rms_norm_torch,
    precompute_freqs_cis_torch,
    GemmaAttentionTorch,
    GemmaMLPTorch,
    GemmaBlockTorch,
)


class TestRMSNorm:
    """Tests for RMSNorm."""
    
    def test_output_shape(self):
        """Test output shape matches input."""
        batch_size, seq_len, hidden_dim = 2, 4, 256
        x = torch.randn(batch_size, seq_len, hidden_dim)
        weight = torch.randn(hidden_dim)
        
        result = rms_norm_torch(x, weight)
        
        assert result.shape == x.shape
    
    def test_normalization(self):
        """Test that output is normalized."""
        batch_size, seq_len, hidden_dim = 2, 4, 256
        x = torch.randn(batch_size, seq_len, hidden_dim) * 10  # Large values
        weight = torch.zeros(hidden_dim)  # Weight + 1 = 1, so just normalization
        
        result = rms_norm_torch(x, weight)
        
        # RMS should be approximately 1
        rms = (result ** 2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestRoPE:
    """Tests for Rotary Position Embeddings."""
    
    def test_freq_shapes(self):
        """Test frequency tensor shapes."""
        head_dim = 256
        max_seq_len = 1024
        
        cos, sin = precompute_freqs_cis_torch(head_dim, max_seq_len)
        
        assert cos.shape == (max_seq_len, head_dim // 2)
        assert sin.shape == (max_seq_len, head_dim // 2)
    
    def test_cos_sin_range(self):
        """Test cos/sin values are in [-1, 1]."""
        head_dim = 256
        max_seq_len = 1024
        
        cos, sin = precompute_freqs_cis_torch(head_dim, max_seq_len)
        
        assert (cos >= -1).all() and (cos <= 1).all()
        assert (sin >= -1).all() and (sin <= 1).all()


class TestGemmaAttention:
    """Tests for Gemma Multi-Query Attention."""
    
    @pytest.fixture
    def config(self):
        return GemmaConfig(
            width=512,
            num_heads=8,
            num_kv_heads=1,
            head_dim=64,
        )
    
    @pytest.fixture
    def attention(self, config):
        weights = {
            "self_attn.q_proj.weight": torch.randn(config.num_heads * config.head_dim, config.width),
            "self_attn.k_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
            "self_attn.v_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
            "self_attn.o_proj.weight": torch.randn(config.width, config.num_heads * config.head_dim),
        }
        return GemmaAttentionTorch(config, weights, layer_idx=0)
    
    def test_output_shape(self, attention, config):
        """Test attention output shape."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.width)
        cos, sin = precompute_freqs_cis_torch(config.head_dim, 1024)
        
        output, cache = attention.forward(x, cos, sin)
        
        assert output.shape == (batch_size, seq_len, config.width)
    
    def test_kv_cache(self, attention, config):
        """Test KV caching."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.width)
        cos, sin = precompute_freqs_cis_torch(config.head_dim, 1024)
        
        output, cache = attention.forward(x, cos, sin, use_cache=True)
        
        assert cache is not None
        k_cache, v_cache = cache
        assert k_cache.shape == (batch_size, config.num_kv_heads, seq_len, config.head_dim)


class TestGemmaMLP:
    """Tests for Gemma MLP."""
    
    @pytest.fixture
    def config(self):
        return GemmaConfig(width=512, mlp_dim=2048)
    
    @pytest.fixture
    def mlp(self, config):
        weights = {
            "mlp.gate_proj.weight": torch.randn(config.mlp_dim, config.width),
            "mlp.up_proj.weight": torch.randn(config.mlp_dim, config.width),
            "mlp.down_proj.weight": torch.randn(config.width, config.mlp_dim),
        }
        return GemmaMLPTorch(config, weights)
    
    def test_output_shape(self, mlp, config):
        """Test MLP output shape."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.width)
        
        output = mlp.forward(x)
        
        assert output.shape == (batch_size, seq_len, config.width)


class TestGemmaBlock:
    """Tests for complete Gemma block."""
    
    @pytest.fixture
    def config(self):
        return GemmaConfig(
            width=512,
            num_heads=8,
            num_kv_heads=1,
            head_dim=64,
            mlp_dim=2048,
        )
    
    @pytest.fixture
    def block(self, config):
        weights = {
            "input_layernorm.weight": torch.randn(config.width),
            "post_attention_layernorm.weight": torch.randn(config.width),
            "self_attn.q_proj.weight": torch.randn(config.num_heads * config.head_dim, config.width),
            "self_attn.k_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
            "self_attn.v_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
            "self_attn.o_proj.weight": torch.randn(config.width, config.num_heads * config.head_dim),
            "mlp.gate_proj.weight": torch.randn(config.mlp_dim, config.width),
            "mlp.up_proj.weight": torch.randn(config.mlp_dim, config.width),
            "mlp.down_proj.weight": torch.randn(config.width, config.mlp_dim),
        }
        return GemmaBlockTorch(config, weights, layer_idx=0)
    
    def test_output_shape(self, block, config):
        """Test block output shape."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.width)
        cos, sin = precompute_freqs_cis_torch(config.head_dim, 1024)
        
        output, cache = block.forward(x, cos, sin)
        
        assert output.shape == (batch_size, seq_len, config.width)
    
    def test_residual_connection(self, block, config):
        """Test that output includes residual."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.width)
        cos, sin = precompute_freqs_cis_torch(config.head_dim, 1024)
        
        output, _ = block.forward(x, cos, sin)
        
        # Output should not be zero if input was non-zero
        assert not torch.allclose(output, torch.zeros_like(output))


class TestGemmaConfigs:
    """Tests for Gemma configuration presets."""
    
    def test_gemma_2b_config(self):
        """Test Gemma 2B configuration values."""
        config = GemmaConfig.gemma_2b()
        
        assert config.width == 2048
        assert config.depth == 18
        assert config.mlp_dim == 16384
        assert config.num_heads == 8
        assert config.num_kv_heads == 1
        assert config.head_dim == 256
    
    def test_gemma_300m_config(self):
        """Test Gemma 300M configuration values."""
        config = GemmaConfig.gemma_300m()
        
        assert config.width == 1024
        assert config.depth == 18
        assert config.mlp_dim == 4096
        assert config.num_heads == 8
        assert config.num_kv_heads == 1
        assert config.head_dim == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

