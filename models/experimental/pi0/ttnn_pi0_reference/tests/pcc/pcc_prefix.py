# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for ttnn_prefix.py module.

Tests prefix embedding components: image embedding, language embedding,
and their concatenation.
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


class TestPrefixEmbeddingPCC:
    """PCC tests for prefix embedding."""
    
    def test_embed_prefix_consistency(self):
        """Test prefix embedding is deterministic with mock functions."""
        from ...ttnn_prefix import PrefixConfig, PrefixEmbeddingTorch, MockEmbeddingFunctions
        
        config = PrefixConfig(vlm_width=512, num_image_tokens=64)
        mock = MockEmbeddingFunctions(hidden_dim=config.vlm_width, num_image_tokens=config.num_image_tokens)
        
        prefix = PrefixEmbeddingTorch(config)
        prefix.embed_image_fn = mock.embed_image
        prefix.embed_language_fn = mock.embed_language
        
        # Create test inputs
        torch.manual_seed(42)
        images = [torch.randn(2, 3, 224, 224)]
        img_masks = [torch.ones(2, dtype=torch.bool)]
        lang_tokens = torch.randint(0, 1000, (2, 16))
        lang_masks = torch.ones(2, 16, dtype=torch.bool)
        
        # Two forward passes with same seed
        torch.manual_seed(123)
        embs1, pad1, att1 = prefix.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        
        torch.manual_seed(123)
        embs2, pad2, att2 = prefix.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        
        assert check_pcc(embs1, embs2, threshold=1.0, test_name="embed_prefix_consistency")
        assert torch.equal(pad1, pad2), "Pad masks should be equal"
    
    def test_prefix_shape(self):
        """Test prefix embedding has correct shape."""
        from ...ttnn_prefix import PrefixConfig, PrefixEmbeddingTorch, MockEmbeddingFunctions
        
        config = PrefixConfig(vlm_width=512, num_image_tokens=64)
        mock = MockEmbeddingFunctions(hidden_dim=config.vlm_width, num_image_tokens=config.num_image_tokens)
        
        prefix = PrefixEmbeddingTorch(config)
        prefix.embed_image_fn = mock.embed_image
        prefix.embed_language_fn = mock.embed_language
        
        batch_size = 2
        num_images = 2
        lang_len = 16
        
        images = [torch.randn(batch_size, 3, 224, 224) for _ in range(num_images)]
        img_masks = [torch.ones(batch_size, dtype=torch.bool) for _ in range(num_images)]
        lang_tokens = torch.randint(0, 1000, (batch_size, lang_len))
        lang_masks = torch.ones(batch_size, lang_len, dtype=torch.bool)
        
        embs, pad, att = prefix.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        
        expected_len = num_images * config.num_image_tokens + lang_len
        assert embs.shape == (batch_size, expected_len, config.vlm_width)
        assert pad.shape == (batch_size, expected_len)
        assert att.shape == (batch_size, expected_len)


class TestMockEmbeddingFunctionsPCC:
    """PCC tests for mock embedding functions."""
    
    def test_mock_image_shape(self):
        """Test mock image embedding produces correct shape."""
        from ...ttnn_prefix import MockEmbeddingFunctions
        
        mock = MockEmbeddingFunctions(hidden_dim=512, num_image_tokens=64)
        
        batch_size = 4
        image = torch.randn(batch_size, 3, 224, 224)
        emb = mock.embed_image(image)
        
        assert emb.shape == (batch_size, 64, 512)
    
    def test_mock_language_shape(self):
        """Test mock language embedding produces correct shape."""
        from ...ttnn_prefix import MockEmbeddingFunctions
        
        mock = MockEmbeddingFunctions(hidden_dim=512, num_image_tokens=64)
        
        batch_size, seq_len = 4, 32
        tokens = torch.randint(0, 1000, (batch_size, seq_len))
        emb = mock.embed_language(tokens)
        
        assert emb.shape == (batch_size, seq_len, 512)


class TestImageEmbeddingPCC:
    """PCC tests for image embedding."""
    
    def test_embed_images_shapes(self):
        """Test image embedding produces correct shapes."""
        from ...ttnn_prefix import PrefixConfig, PrefixEmbeddingTorch, MockEmbeddingFunctions
        
        config = PrefixConfig(vlm_width=512, num_image_tokens=64)
        mock = MockEmbeddingFunctions(hidden_dim=config.vlm_width, num_image_tokens=config.num_image_tokens)
        
        prefix = PrefixEmbeddingTorch(config)
        prefix.embed_image_fn = mock.embed_image
        
        batch_size = 2
        images = [torch.randn(batch_size, 3, 224, 224), torch.randn(batch_size, 3, 224, 224)]
        img_masks = [torch.ones(batch_size, dtype=torch.bool), torch.ones(batch_size, dtype=torch.bool)]
        
        image_embs, expanded_masks = prefix.embed_images(images, img_masks)
        
        assert len(image_embs) == 2
        assert len(expanded_masks) == 2
        
        for emb, mask in zip(image_embs, expanded_masks):
            assert emb.shape == (batch_size, config.num_image_tokens, config.vlm_width)
            assert mask.shape == (batch_size, config.num_image_tokens)


class TestLanguageEmbeddingPCC:
    """PCC tests for language embedding."""
    
    def test_embed_language_scaling(self):
        """Test language embedding includes sqrt(dim) scaling."""
        from ...ttnn_prefix import PrefixConfig, PrefixEmbeddingTorch, MockEmbeddingFunctions
        import math
        
        config = PrefixConfig(vlm_width=512, num_image_tokens=64)
        mock = MockEmbeddingFunctions(hidden_dim=config.vlm_width, num_image_tokens=config.num_image_tokens)
        
        prefix = PrefixEmbeddingTorch(config)
        prefix.embed_language_fn = mock.embed_language
        
        tokens = torch.randint(0, 1000, (2, 16))
        masks = torch.ones(2, 16, dtype=torch.bool)
        
        # Get raw embedding
        torch.manual_seed(42)
        raw_emb = mock.embed_language(tokens)
        
        # Get scaled embedding
        torch.manual_seed(42)
        scaled_emb = prefix.embed_language(tokens, masks)
        
        # Check scaling
        expected_scale = math.sqrt(config.vlm_width)
        actual_scale = (scaled_emb / raw_emb).mean().item()
        
        assert abs(actual_scale - expected_scale) < 0.1, f"Expected scale {expected_scale}, got {actual_scale}"


class TestPrefixAttentionPatternPCC:
    """PCC tests for prefix attention patterns."""
    
    def test_prefix_uses_bidirectional(self):
        """Test that prefix attention mask uses bidirectional (all zeros)."""
        from ...ttnn_prefix import PrefixConfig, PrefixEmbeddingTorch, MockEmbeddingFunctions
        
        config = PrefixConfig(vlm_width=512, num_image_tokens=32)
        mock = MockEmbeddingFunctions(hidden_dim=config.vlm_width, num_image_tokens=config.num_image_tokens)
        
        prefix = PrefixEmbeddingTorch(config)
        prefix.embed_image_fn = mock.embed_image
        prefix.embed_language_fn = mock.embed_language
        
        images = [torch.randn(2, 3, 224, 224)]
        img_masks = [torch.ones(2, dtype=torch.bool)]
        lang_tokens = torch.randint(0, 1000, (2, 16))
        lang_masks = torch.ones(2, 16, dtype=torch.bool)
        
        _, _, att_masks = prefix.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        
        # All attention masks should be 0 (bidirectional)
        assert not att_masks.any(), "Prefix should use bidirectional attention (all zeros)"


def run_pcc_prefix_tests():
    """Run all PCC tests for prefix module."""
    print("=" * 60)
    print("PCC Tests: ttnn_prefix.py")
    print("=" * 60)
    
    test_prefix = TestPrefixEmbeddingPCC()
    test_prefix.test_embed_prefix_consistency()
    test_prefix.test_prefix_shape()
    
    test_mock = TestMockEmbeddingFunctionsPCC()
    test_mock.test_mock_image_shape()
    test_mock.test_mock_language_shape()
    
    test_image = TestImageEmbeddingPCC()
    test_image.test_embed_images_shapes()
    
    test_lang = TestLanguageEmbeddingPCC()
    test_lang.test_embed_language_scaling()
    
    test_att = TestPrefixAttentionPatternPCC()
    test_att.test_prefix_uses_bidirectional()
    
    print("\n✓ All PCC tests for ttnn_prefix.py passed!")


if __name__ == "__main__":
    run_pcc_prefix_tests()

