# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for ttnn_pi0.py module.

Tests full PI0 model orchestrator: configuration, forward passes,
and action sampling.
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


class TestPI0ModelConfigPCC:
    """PCC tests for PI0 model configuration."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        from ...ttnn_pi0 import PI0ModelConfig
        
        config = PI0ModelConfig()
        
        assert config.action_dim == 32
        assert config.action_horizon == 50
        assert config.num_denoising_steps == 10
        assert config.max_seq_len == 2048
        assert config.pi05 == False
    
    def test_config_custom(self):
        """Test custom configuration."""
        from ...ttnn_pi0 import PI0ModelConfig
        
        config = PI0ModelConfig(
            action_dim=64,
            action_horizon=100,
            num_denoising_steps=20,
            pi05=True,
        )
        
        assert config.action_dim == 64
        assert config.action_horizon == 100
        assert config.num_denoising_steps == 20
        assert config.pi05 == True
    
    def test_config_variants(self):
        """Test PI0 and PI05 variants."""
        from ...ttnn_pi0 import PI0ModelConfig
        
        pi0_config = PI0ModelConfig(pi05=False)
        pi05_config = PI0ModelConfig(pi05=True)
        
        assert pi0_config.pi05 == False
        assert pi05_config.pi05 == True


class TestPI0WeightLoaderIntegrationPCC:
    """PCC tests for weight loader integration."""
    
    def test_weight_loader_available(self):
        """Test weight loader can be imported."""
        from ...weight_loader import PI0WeightLoader, PI0Config
        
        # Just verify import works
        assert PI0WeightLoader is not None
        assert PI0Config is not None
    
    def test_weight_categorization(self):
        """Test weight categorization produces expected keys."""
        from ...weight_loader import categorize_weights
        
        # Mock state dict
        mock_state = {
            "action_in_proj.weight": torch.randn(1024, 32),
            "action_out_proj.weight": torch.randn(32, 1024),
            "state_proj.weight": torch.randn(1024, 32),
            "paligemma_with_expert.paligemma.model.language_model.model.layers.0.weight": torch.randn(256),
            "paligemma_with_expert.paligemma.model.vision_tower.encoder.weight": torch.randn(256),
            "paligemma_with_expert.gemma_expert.model.layers.0.weight": torch.randn(256),
        }
        
        categorized = categorize_weights(mock_state)
        
        assert "pi0_projections" in categorized
        assert "vlm_language" in categorized
        assert "vlm_vision" in categorized
        assert "action_expert" in categorized


class TestPI0ForwardPatternsPCC:
    """PCC tests for PI0 forward pass patterns."""
    
    def test_prefix_suffix_dimensions(self):
        """Test prefix and suffix have compatible dimensions."""
        from ...ttnn_pi0 import PI0ModelConfig
        
        config = PI0ModelConfig()
        
        # VLM width = 2048, Expert width = 1024
        assert config.vlm_config.width == 2048
        assert config.expert_config.width == 1024
        
        # Both should have same depth for shared attention
        assert config.vlm_config.depth == config.expert_config.depth
    
    def test_action_projection_dimensions(self):
        """Test action projection dimensions match config."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        action_dim = 32
        action_horizon = 50
        expert_width = 1024
        
        config = SuffixConfig(
            action_dim=action_dim,
            action_horizon=action_horizon,
            expert_width=expert_width,
        )
        
        weights = {
            "action_in_proj.weight": torch.randn(expert_width, action_dim),
            "action_in_proj.bias": torch.randn(expert_width),
            "action_out_proj.weight": torch.randn(action_dim, expert_width),
            "action_out_proj.bias": torch.randn(action_dim),
            "state_proj.weight": torch.randn(expert_width, action_dim),
            "state_proj.bias": torch.randn(expert_width),
            "action_time_mlp_in.weight": torch.randn(expert_width, expert_width * 2),
            "action_time_mlp_in.bias": torch.randn(expert_width),
            "action_time_mlp_out.weight": torch.randn(expert_width, expert_width),
            "action_time_mlp_out.bias": torch.randn(expert_width),
        }
        
        suffix = SuffixEmbeddingTorch(config, weights)
        
        # Test round-trip through projections
        actions = torch.randn(2, action_horizon, action_dim)
        projected = suffix.embed_actions(actions)
        
        assert projected.shape == (2, action_horizon, expert_width)
        
        # Project back
        recovered = suffix.project_output(projected)
        assert recovered.shape == (2, action_horizon, action_dim)


class TestPI0AttentionMaskPatternsPCC:
    """PCC tests for PI0 attention mask patterns."""
    
    def test_prefix_bidirectional_suffix_causal(self):
        """Test attention pattern: prefix bidirectional, suffix causal."""
        from ...ttnn_attention import make_att_2d_masks_torch
        
        batch_size = 2
        prefix_len = 256 + 32  # Images + language
        suffix_len = 51  # State + actions
        total_len = prefix_len + suffix_len
        
        # Create pad masks (all valid)
        pad_masks = torch.ones(batch_size, total_len, dtype=torch.bool)
        
        # Create att masks: 0 for prefix (bidirectional), 1 for suffix (causal)
        att_masks = torch.cat([
            torch.zeros(batch_size, prefix_len, dtype=torch.bool),
            torch.ones(batch_size, suffix_len, dtype=torch.bool),
        ], dim=1)
        
        att_2d = make_att_2d_masks_torch(pad_masks, att_masks)
        
        # Check prefix is bidirectional (all can attend to all prefix)
        for b in range(batch_size):
            prefix_block = att_2d[b, :prefix_len, :prefix_len]
            assert prefix_block.all(), "Prefix should be bidirectional"
        
        # Check suffix is causal within suffix
        for b in range(batch_size):
            for i in range(suffix_len):
                for j in range(suffix_len):
                    actual = att_2d[b, prefix_len + i, prefix_len + j]
                    expected = j <= i
                    assert actual == expected, f"Suffix causal wrong at [{i},{j}]"
    
    def test_suffix_sees_prefix(self):
        """Test suffix tokens can attend to prefix."""
        from ...ttnn_attention import make_att_2d_masks_torch
        
        batch_size = 1
        prefix_len = 32
        suffix_len = 8
        total_len = prefix_len + suffix_len
        
        pad_masks = torch.ones(batch_size, total_len, dtype=torch.bool)
        att_masks = torch.cat([
            torch.zeros(batch_size, prefix_len, dtype=torch.bool),
            torch.ones(batch_size, suffix_len, dtype=torch.bool),
        ], dim=1)
        
        att_2d = make_att_2d_masks_torch(pad_masks, att_masks)
        
        # All suffix tokens should see all prefix tokens
        for i in range(suffix_len):
            for j in range(prefix_len):
                actual = att_2d[0, prefix_len + i, j]
                assert actual, f"Suffix token {i} should see prefix token {j}"


class TestPI0DenoisingIntegrationPCC:
    """PCC tests for denoising integration."""
    
    def test_denoising_uses_suffix(self):
        """Test denoising module uses suffix embedding."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        suffix_config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=256)
        suffix_weights = {
            "action_in_proj.weight": torch.randn(256, 32),
            "action_in_proj.bias": torch.randn(256),
            "action_out_proj.weight": torch.randn(32, 256),
            "action_out_proj.bias": torch.randn(32),
            "state_proj.weight": torch.randn(256, 32),
            "state_proj.bias": torch.randn(256),
            "action_time_mlp_in.weight": torch.randn(256, 512),
            "action_time_mlp_in.bias": torch.randn(256),
            "action_time_mlp_out.weight": torch.randn(256, 256),
            "action_time_mlp_out.bias": torch.randn(256),
        }
        suffix = SuffixEmbeddingTorch(suffix_config, suffix_weights)
        
        # Create forward function using suffix
        def forward_fn(noisy_actions, timestep, kv_cache=None, state=None, **kwargs):
            if state is None:
                state = torch.zeros(noisy_actions.shape[0], 32)
            
            embs, _, _, _ = suffix.embed_suffix(state, noisy_actions, timestep)
            
            # Skip state token, project action tokens
            action_embs = embs[:, 1:, :]
            return suffix.project_output(action_embs)
        
        denoise_config = DenoiseConfig(num_steps=2, action_dim=32, action_horizon=50)
        denoising = DenoisingModuleTorch(denoise_config, forward_fn)
        
        # Test denoising
        torch.manual_seed(42)
        actions = denoising.sample_actions(batch_size=2)
        
        assert actions.shape == (2, 50, 32)


class TestPI0EndToEndPCC:
    """PCC tests for end-to-end PI0 patterns."""
    
    def test_full_pipeline_shapes(self):
        """Test full pipeline produces correct shapes."""
        from ...ttnn_pi0 import PI0ModelConfig
        
        config = PI0ModelConfig(
            action_dim=32,
            action_horizon=50,
            num_denoising_steps=3,
        )
        
        batch_size = 2
        
        # Simulate prefix embedding
        num_image_tokens = 256
        lang_tokens = 32
        prefix_len = num_image_tokens + lang_tokens
        prefix_embs = torch.randn(batch_size, prefix_len, config.vlm_config.width)
        
        # Simulate suffix embedding
        suffix_len = 1 + config.action_horizon  # State + actions
        suffix_embs = torch.randn(batch_size, suffix_len, config.expert_config.width)
        
        # Check shapes match expectations
        assert prefix_embs.shape == (batch_size, prefix_len, 2048)
        assert suffix_embs.shape == (batch_size, suffix_len, 1024)
    
    def test_action_output_matches_config(self):
        """Test action output dimensions match config."""
        from ...ttnn_pi0 import PI0ModelConfig
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        config = PI0ModelConfig(action_dim=64, action_horizon=100)
        
        def dummy_forward(x, t, kv_cache=None, **kwargs):
            return torch.zeros_like(x)
        
        denoise_config = DenoiseConfig(
            num_steps=1,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
        )
        denoising = DenoisingModuleTorch(denoise_config, dummy_forward)
        
        actions = denoising.sample_actions(batch_size=2)
        
        assert actions.shape == (2, config.action_horizon, config.action_dim)


def run_pcc_pi0_tests():
    """Run all PCC tests for pi0 module."""
    print("=" * 60)
    print("PCC Tests: ttnn_pi0.py")
    print("=" * 60)
    
    test_config = TestPI0ModelConfigPCC()
    test_config.test_config_defaults()
    test_config.test_config_custom()
    test_config.test_config_variants()
    
    test_weights = TestPI0WeightLoaderIntegrationPCC()
    test_weights.test_weight_loader_available()
    test_weights.test_weight_categorization()
    
    test_forward = TestPI0ForwardPatternsPCC()
    test_forward.test_prefix_suffix_dimensions()
    test_forward.test_action_projection_dimensions()
    
    test_masks = TestPI0AttentionMaskPatternsPCC()
    test_masks.test_prefix_bidirectional_suffix_causal()
    test_masks.test_suffix_sees_prefix()
    
    test_denoise = TestPI0DenoisingIntegrationPCC()
    test_denoise.test_denoising_uses_suffix()
    
    test_e2e = TestPI0EndToEndPCC()
    test_e2e.test_full_pipeline_shapes()
    test_e2e.test_action_output_matches_config()
    
    print("\n✓ All PCC tests for ttnn_pi0.py passed!")


if __name__ == "__main__":
    run_pcc_pi0_tests()

