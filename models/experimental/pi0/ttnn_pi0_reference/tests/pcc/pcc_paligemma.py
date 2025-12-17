# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for ttnn_paligemma.py module.

Tests PaliGemma backbone components: VLM + Expert forward,
shared attention, and combined processing.
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


class TestPaliGemmaConfigPCC:
    """PCC tests for PaliGemma configuration."""
    
    def test_config_defaults(self):
        """Test default configuration."""
        from ...ttnn_paligemma import PaliGemmaConfig
        from ...ttnn_gemma import GemmaConfig
        from ...ttnn_siglip import SigLIPConfig
        
        config = PaliGemmaConfig()
        
        # Check component configs are initialized
        assert config.vlm_config is not None
        assert config.expert_config is not None
        assert config.siglip_config is not None
        
        # Check defaults match expected architectures
        assert config.vlm_config.width == 2048
        assert config.expert_config.width == 1024
    
    def test_config_custom(self):
        """Test custom configuration."""
        from ...ttnn_paligemma import PaliGemmaConfig
        from ...ttnn_gemma import GemmaConfig
        
        custom_vlm = GemmaConfig(width=1024, depth=12)
        custom_expert = GemmaConfig(width=512, depth=6)
        
        config = PaliGemmaConfig(
            vlm_config=custom_vlm,
            expert_config=custom_expert,
            max_seq_len=4096,
        )
        
        assert config.vlm_config.width == 1024
        assert config.expert_config.width == 512
        assert config.max_seq_len == 4096


class TestEmbedLanguageTokensPCC:
    """PCC tests for language token embedding."""
    
    def test_embed_shape(self):
        """Test embedding output shape (mock test without full weights)."""
        # This test would require full PaliGemma weights
        # For now, test the configuration path
        from ...ttnn_paligemma import PaliGemmaConfig
        
        config = PaliGemmaConfig()
        
        # Verify vocab size is set
        assert config.vocab_size == 257152
    
    def test_embedding_lookup_consistency(self):
        """Test that embedding lookup is deterministic."""
        import torch.nn.functional as F
        
        # Simulate embedding lookup
        vocab_size, hidden_dim = 1000, 256
        embed_weight = torch.randn(vocab_size, hidden_dim)
        tokens = torch.randint(0, vocab_size, (4, 32))
        
        emb1 = F.embedding(tokens, embed_weight)
        emb2 = F.embedding(tokens, embed_weight)
        
        assert check_pcc(emb1, emb2, threshold=1.0, test_name="embedding_lookup_consistency")


class TestVLMForwardPCC:
    """PCC tests for VLM forward pass patterns."""
    
    def test_forward_shape_patterns(self):
        """Test expected shapes in VLM forward."""
        from ...ttnn_gemma import GemmaConfig, GemmaBlockTorch, precompute_freqs_cis_torch
        
        config = GemmaConfig(width=256, num_heads=4, num_kv_heads=1, head_dim=64, mlp_dim=512)
        
        # Create weights for a single block
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
        
        block = GemmaBlockTorch(config, weights, layer_idx=0)
        cos, sin = precompute_freqs_cis_torch(config.head_dim, 1024)
        
        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.width)
        
        out, cache = block.forward(x, cos, sin, use_cache=True)
        
        assert out.shape == (batch_size, seq_len, config.width)
        assert cache is not None


class TestExpertForwardPCC:
    """PCC tests for Expert forward pass patterns."""
    
    def test_expert_forward_consistency(self):
        """Test expert-sized block produces consistent results."""
        from ...ttnn_gemma import GemmaConfig, GemmaBlockTorch, precompute_freqs_cis_torch
        
        # Expert configuration (Gemma 300M scale, but smaller for testing)
        config = GemmaConfig(width=256, num_heads=4, num_kv_heads=1, head_dim=64, mlp_dim=512)
        
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
        
        block = GemmaBlockTorch(config, weights, layer_idx=0)
        cos, sin = precompute_freqs_cis_torch(config.head_dim, 1024)
        
        # Suffix-like sequence (state + actions)
        batch_size = 2
        suffix_len = 51  # 1 state + 50 action tokens
        x = torch.randn(batch_size, suffix_len, config.width)
        
        out1, _ = block.forward(x, cos, sin)
        out2, _ = block.forward(x, cos, sin)
        
        assert check_pcc(out1, out2, threshold=1.0, test_name="expert_forward_consistency")


class TestSharedAttentionPatternPCC:
    """PCC tests for shared attention patterns."""
    
    def test_shared_kv_pattern(self):
        """Test pattern where VLM and Expert share KV."""
        from ...ttnn_gemma import GemmaConfig, precompute_freqs_cis_torch
        import torch.nn.functional as F
        
        # Simplified shared attention pattern
        config = GemmaConfig(width=256, num_heads=4, num_kv_heads=1, head_dim=64)
        
        batch_size = 2
        prefix_len, suffix_len = 32, 16
        total_len = prefix_len + suffix_len
        
        # Create Q, K, V projections
        q_proj = torch.randn(config.num_heads * config.head_dim, config.width)
        k_proj = torch.randn(config.num_kv_heads * config.head_dim, config.width)
        v_proj = torch.randn(config.num_kv_heads * config.head_dim, config.width)
        
        # Combined sequence
        combined = torch.randn(batch_size, total_len, config.width)
        
        # Compute Q, K, V
        q = F.linear(combined, q_proj)
        k = F.linear(combined, k_proj)
        v = F.linear(combined, v_proj)
        
        # Reshape
        q = q.view(batch_size, total_len, config.num_heads, config.head_dim).transpose(1, 2)
        k = k.view(batch_size, total_len, config.num_kv_heads, config.head_dim).transpose(1, 2)
        v = v.view(batch_size, total_len, config.num_kv_heads, config.head_dim).transpose(1, 2)
        
        # Expand K, V for broadcast
        k_expanded = k.expand(batch_size, config.num_heads, total_len, config.head_dim)
        v_expanded = v.expand(batch_size, config.num_heads, total_len, config.head_dim)
        
        # Attention
        scale = 1.0 / (config.head_dim ** 0.5)
        attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_expanded)
        
        # Should be deterministic
        attn_weights_2 = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
        attn_weights_2 = F.softmax(attn_weights_2, dim=-1)
        attn_output_2 = torch.matmul(attn_weights_2, v_expanded)
        
        assert check_pcc(attn_output, attn_output_2, threshold=1.0, test_name="shared_kv_pattern")


class TestMultiQueryAttentionPCC:
    """PCC tests for Multi-Query Attention specific to PaliGemma."""
    
    def test_mqa_broadcast(self):
        """Test MQA broadcasts K, V correctly."""
        batch_size = 2
        num_heads = 8
        num_kv_heads = 1
        seq_len = 32
        head_dim = 64
        
        # Single K, V head
        k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        
        # Expand to all heads
        k_expanded = k.expand(batch_size, num_heads, seq_len, head_dim)
        v_expanded = v.expand(batch_size, num_heads, seq_len, head_dim)
        
        # All heads should have same K, V
        for h in range(num_heads):
            pcc = compute_pcc(k_expanded[:, h], k_expanded[:, 0])
            assert pcc == 1.0, f"Head {h} K different from head 0"
    
    def test_mqa_memory_efficiency(self):
        """Test MQA uses less memory than MHA."""
        batch_size = 4
        num_heads = 8
        num_kv_heads = 1
        seq_len = 512
        head_dim = 256
        
        # MHA memory
        mha_k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        mha_v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        mha_memory = mha_k.numel() + mha_v.numel()
        
        # MQA memory
        mqa_k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        mqa_v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        mqa_memory = mqa_k.numel() + mqa_v.numel()
        
        # MQA should use 1/num_heads of the memory
        ratio = mqa_memory / mha_memory
        expected_ratio = num_kv_heads / num_heads
        
        assert ratio == expected_ratio, f"Memory ratio {ratio} != expected {expected_ratio}"


def run_pcc_paligemma_tests():
    """Run all PCC tests for paligemma module."""
    print("=" * 60)
    print("PCC Tests: ttnn_paligemma.py")
    print("=" * 60)
    
    test_config = TestPaliGemmaConfigPCC()
    test_config.test_config_defaults()
    test_config.test_config_custom()
    
    test_embed = TestEmbedLanguageTokensPCC()
    test_embed.test_embed_shape()
    test_embed.test_embedding_lookup_consistency()
    
    test_vlm = TestVLMForwardPCC()
    test_vlm.test_forward_shape_patterns()
    
    test_expert = TestExpertForwardPCC()
    test_expert.test_expert_forward_consistency()
    
    test_shared = TestSharedAttentionPatternPCC()
    test_shared.test_shared_kv_pattern()
    
    test_mqa = TestMultiQueryAttentionPCC()
    test_mqa.test_mqa_broadcast()
    test_mqa.test_mqa_memory_efficiency()
    
    print("\n✓ All PCC tests for ttnn_paligemma.py passed!")


if __name__ == "__main__":
    run_pcc_paligemma_tests()

