# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for ttnn_siglip.py module.

Tests SigLIP vision tower components: patch embedding, attention,
MLP, transformer blocks, and multi-modal projector.
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


def create_siglip_attention_weights(config):
    """Create mock SigLIP attention weights."""
    return {
        "self_attn.q_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.k_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.v_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.out_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.q_proj.bias": torch.randn(config.hidden_size),
        "self_attn.k_proj.bias": torch.randn(config.hidden_size),
        "self_attn.v_proj.bias": torch.randn(config.hidden_size),
        "self_attn.out_proj.bias": torch.randn(config.hidden_size),
    }


def create_siglip_mlp_weights(config):
    """Create mock SigLIP MLP weights."""
    return {
        "mlp.fc1.weight": torch.randn(config.intermediate_size, config.hidden_size),
        "mlp.fc1.bias": torch.randn(config.intermediate_size),
        "mlp.fc2.weight": torch.randn(config.hidden_size, config.intermediate_size),
        "mlp.fc2.bias": torch.randn(config.hidden_size),
    }


def create_siglip_block_weights(config):
    """Create mock SigLIP block weights."""
    weights = {
        "layer_norm1.weight": torch.randn(config.hidden_size),
        "layer_norm1.bias": torch.randn(config.hidden_size),
        "layer_norm2.weight": torch.randn(config.hidden_size),
        "layer_norm2.bias": torch.randn(config.hidden_size),
    }
    weights.update(create_siglip_attention_weights(config))
    weights.update(create_siglip_mlp_weights(config))
    return weights


class TestPatchEmbeddingPCC:
    """PCC tests for patch embedding."""
    
    def test_patch_embedding_consistency(self):
        """Test patch embedding produces consistent results."""
        from ...ttnn_siglip import SigLIPConfig, PatchEmbeddingTorch
        
        config = SigLIPConfig(hidden_size=256, patch_size=14, image_size=224)
        weights = {
            "patch_embedding.weight": torch.randn(config.hidden_size, 3, config.patch_size, config.patch_size),
            "patch_embedding.bias": torch.randn(config.hidden_size),
        }
        patch_embed = PatchEmbeddingTorch(config, weights)
        
        images = torch.randn(2, 3, 224, 224)
        
        emb1 = patch_embed.forward(images)
        emb2 = patch_embed.forward(images)
        
        assert check_pcc(emb1, emb2, threshold=1.0, test_name="patch_embedding_consistency")
    
    def test_patch_embedding_shape(self):
        """Test patch embedding output shape."""
        from ...ttnn_siglip import SigLIPConfig, PatchEmbeddingTorch
        
        config = SigLIPConfig(hidden_size=256, patch_size=14, image_size=224)
        weights = {
            "patch_embedding.weight": torch.randn(config.hidden_size, 3, config.patch_size, config.patch_size),
            "patch_embedding.bias": torch.randn(config.hidden_size),
        }
        patch_embed = PatchEmbeddingTorch(config, weights)
        
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        emb = patch_embed.forward(images)
        
        num_patches = (224 // 14) ** 2  # 16 * 16 = 256
        assert emb.shape == (batch_size, num_patches, config.hidden_size)


class TestSigLIPAttentionPCC:
    """PCC tests for SigLIP attention."""
    
    def test_attention_consistency(self):
        """Test attention produces consistent results."""
        from ...ttnn_siglip import SigLIPConfig, SigLIPAttentionTorch
        
        config = SigLIPConfig(hidden_size=256, num_attention_heads=4)
        weights = create_siglip_attention_weights(config)
        attention = SigLIPAttentionTorch(config, weights)
        
        x = torch.randn(2, 64, config.hidden_size)
        
        out1 = attention.forward(x)
        out2 = attention.forward(x)
        
        assert check_pcc(out1, out2, threshold=1.0, test_name="siglip_attention_consistency")
    
    def test_attention_shape(self):
        """Test attention output shape."""
        from ...ttnn_siglip import SigLIPConfig, SigLIPAttentionTorch
        
        config = SigLIPConfig(hidden_size=512, num_attention_heads=8)
        weights = create_siglip_attention_weights(config)
        attention = SigLIPAttentionTorch(config, weights)
        
        batch_size, seq_len = 4, 256
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output = attention.forward(x)
        
        assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestSigLIPMLPPCC:
    """PCC tests for SigLIP MLP."""
    
    def test_mlp_consistency(self):
        """Test MLP produces consistent results."""
        from ...ttnn_siglip import SigLIPConfig, SigLIPMLPTorch
        
        config = SigLIPConfig(hidden_size=256, intermediate_size=1024)
        weights = create_siglip_mlp_weights(config)
        mlp = SigLIPMLPTorch(config, weights)
        
        x = torch.randn(2, 64, config.hidden_size)
        
        out1 = mlp.forward(x)
        out2 = mlp.forward(x)
        
        assert check_pcc(out1, out2, threshold=1.0, test_name="siglip_mlp_consistency")
    
    def test_mlp_shape(self):
        """Test MLP output shape."""
        from ...ttnn_siglip import SigLIPConfig, SigLIPMLPTorch
        
        config = SigLIPConfig(hidden_size=512, intermediate_size=2048)
        weights = create_siglip_mlp_weights(config)
        mlp = SigLIPMLPTorch(config, weights)
        
        batch_size, seq_len = 4, 256
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output = mlp.forward(x)
        
        assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestSigLIPBlockPCC:
    """PCC tests for SigLIP transformer block."""
    
    def test_block_consistency(self):
        """Test block produces consistent results."""
        from ...ttnn_siglip import SigLIPConfig, SigLIPBlockTorch
        
        config = SigLIPConfig(hidden_size=256, num_attention_heads=4, intermediate_size=512)
        weights = create_siglip_block_weights(config)
        block = SigLIPBlockTorch(config, weights)
        
        x = torch.randn(2, 64, config.hidden_size)
        
        out1 = block.forward(x)
        out2 = block.forward(x)
        
        assert check_pcc(out1, out2, threshold=1.0, test_name="siglip_block_consistency")
    
    def test_block_residual(self):
        """Test block includes residual connections."""
        from ...ttnn_siglip import SigLIPConfig, SigLIPBlockTorch
        
        config = SigLIPConfig(hidden_size=256, num_attention_heads=4, intermediate_size=512)
        weights = create_siglip_block_weights(config)
        block = SigLIPBlockTorch(config, weights)
        
        x = torch.randn(2, 64, config.hidden_size)
        output = block.forward(x)
        
        # Output should be correlated with input due to residual
        pcc = compute_pcc(x, output)
        assert pcc > 0.1, f"Residual connection weak: PCC = {pcc}"


class TestMultiModalProjectorPCC:
    """PCC tests for multi-modal projector."""
    
    def test_projector_consistency(self):
        """Test projector produces consistent results."""
        from ...ttnn_siglip import MultiModalProjectorTorch
        
        weights = {
            "linear.weight": torch.randn(2048, 1152),
            "linear.bias": torch.randn(2048),
        }
        projector = MultiModalProjectorTorch(weights)
        
        x = torch.randn(2, 256, 1152)
        
        out1 = projector.forward(x)
        out2 = projector.forward(x)
        
        assert check_pcc(out1, out2, threshold=1.0, test_name="projector_consistency")
    
    def test_projector_shape(self):
        """Test projector output shape."""
        from ...ttnn_siglip import MultiModalProjectorTorch
        
        vision_dim, language_dim = 1152, 2048
        weights = {
            "linear.weight": torch.randn(language_dim, vision_dim),
            "linear.bias": torch.randn(language_dim),
        }
        projector = MultiModalProjectorTorch(weights)
        
        batch_size, num_patches = 4, 256
        x = torch.randn(batch_size, num_patches, vision_dim)
        output = projector.forward(x)
        
        assert output.shape == (batch_size, num_patches, language_dim)


class TestSigLIPConfigPCC:
    """PCC tests for SigLIP configuration."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        from ...ttnn_siglip import SigLIPConfig
        
        config = SigLIPConfig()
        
        assert config.hidden_size == 1152
        assert config.num_hidden_layers == 27
        assert config.num_attention_heads == 16
        assert config.image_size == 224
        assert config.patch_size == 14
    
    def test_config_num_patches(self):
        """Test num_patches property."""
        from ...ttnn_siglip import SigLIPConfig
        
        config = SigLIPConfig(image_size=224, patch_size=14)
        assert config.num_patches == 256  # (224/14)^2
        
        config = SigLIPConfig(image_size=336, patch_size=14)
        assert config.num_patches == 576  # (336/14)^2
    
    def test_config_head_dim(self):
        """Test head_dim property."""
        from ...ttnn_siglip import SigLIPConfig
        
        config = SigLIPConfig(hidden_size=1152, num_attention_heads=16)
        assert config.head_dim == 72  # 1152 / 16


def run_pcc_siglip_tests():
    """Run all PCC tests for siglip module."""
    print("=" * 60)
    print("PCC Tests: ttnn_siglip.py")
    print("=" * 60)
    
    test_patch = TestPatchEmbeddingPCC()
    test_patch.test_patch_embedding_consistency()
    test_patch.test_patch_embedding_shape()
    
    test_attn = TestSigLIPAttentionPCC()
    test_attn.test_attention_consistency()
    test_attn.test_attention_shape()
    
    test_mlp = TestSigLIPMLPPCC()
    test_mlp.test_mlp_consistency()
    test_mlp.test_mlp_shape()
    
    test_block = TestSigLIPBlockPCC()
    test_block.test_block_consistency()
    test_block.test_block_residual()
    
    test_proj = TestMultiModalProjectorPCC()
    test_proj.test_projector_consistency()
    test_proj.test_projector_shape()
    
    test_config = TestSigLIPConfigPCC()
    test_config.test_config_defaults()
    test_config.test_config_num_patches()
    test_config.test_config_head_dim()
    
    print("\n✓ All PCC tests for ttnn_siglip.py passed!")


if __name__ == "__main__":
    run_pcc_siglip_tests()

