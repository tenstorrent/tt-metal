# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for suffix embedding module.
"""

import pytest
import torch

from ..ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch


def create_mock_weights(config: SuffixConfig):
    """Create mock weights for testing."""
    return {
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


class TestSuffixEmbedding:
    """Tests for suffix embedding."""
    
    @pytest.fixture
    def config(self):
        return SuffixConfig(
            action_dim=32,
            action_horizon=50,
            expert_width=1024,
            pi05=False,
        )
    
    @pytest.fixture
    def suffix_embedding(self, config):
        weights = create_mock_weights(config)
        return SuffixEmbeddingTorch(config, weights)
    
    def test_embed_actions_shape(self, suffix_embedding, config):
        """Test action embedding output shape."""
        batch_size = 2
        noisy_actions = torch.randn(batch_size, config.action_horizon, config.action_dim)
        
        result = suffix_embedding.embed_actions(noisy_actions)
        
        assert result.shape == (batch_size, config.action_horizon, config.expert_width)
    
    def test_embed_state_shape(self, suffix_embedding, config):
        """Test state embedding output shape."""
        batch_size = 2
        state = torch.randn(batch_size, config.action_dim)
        
        result = suffix_embedding.embed_state(state)
        
        assert result.shape == (batch_size, 1, config.expert_width)
    
    def test_embed_timestep_shape(self, suffix_embedding, config):
        """Test timestep embedding output shape."""
        batch_size = 2
        timestep = torch.rand(batch_size)
        
        result = suffix_embedding.embed_timestep(timestep)
        
        assert result.shape == (batch_size, config.expert_width)
    
    def test_fuse_action_time_shape(self, suffix_embedding, config):
        """Test action-time fusion output shape."""
        batch_size = 2
        action_emb = torch.randn(batch_size, config.action_horizon, config.expert_width)
        time_emb = torch.randn(batch_size, config.expert_width)
        
        result, adarms = suffix_embedding.fuse_action_time(action_emb, time_emb)
        
        assert result.shape == (batch_size, config.action_horizon, config.expert_width)
        assert adarms is None  # PI0 mode returns None for adaRMS
    
    def test_embed_suffix_shape(self, suffix_embedding, config):
        """Test full suffix embedding output shape."""
        batch_size = 2
        state = torch.randn(batch_size, config.action_dim)
        noisy_actions = torch.randn(batch_size, config.action_horizon, config.action_dim)
        timestep = torch.rand(batch_size)
        
        suffix_embs, pad_masks, att_masks, adarms = suffix_embedding.embed_suffix(
            state, noisy_actions, timestep
        )
        
        # Suffix has state + actions = 1 + action_horizon tokens
        expected_len = 1 + config.action_horizon
        
        assert suffix_embs.shape == (batch_size, expected_len, config.expert_width)
        assert pad_masks.shape == (batch_size, expected_len)
        assert att_masks.shape == (batch_size, expected_len)
    
    def test_project_output_shape(self, suffix_embedding, config):
        """Test output projection shape."""
        batch_size = 2
        expert_output = torch.randn(batch_size, config.action_horizon, config.expert_width)
        
        result = suffix_embedding.project_output(expert_output)
        
        assert result.shape == (batch_size, config.action_horizon, config.action_dim)


class TestSuffixEmbeddingPI05:
    """Tests for PI05 mode suffix embedding."""
    
    @pytest.fixture
    def config_pi05(self):
        return SuffixConfig(
            action_dim=32,
            action_horizon=50,
            expert_width=1024,
            pi05=True,
        )
    
    @pytest.fixture
    def suffix_embedding_pi05(self, config_pi05):
        weights = create_mock_weights(config_pi05)
        return SuffixEmbeddingTorch(config_pi05, weights)
    
    def test_embed_state_returns_none(self, suffix_embedding_pi05, config_pi05):
        """Test that state embedding returns None in PI05 mode."""
        state = torch.randn(2, config_pi05.action_dim)
        
        result = suffix_embedding_pi05.embed_state(state)
        
        assert result is None
    
    def test_fuse_returns_adarms(self, suffix_embedding_pi05, config_pi05):
        """Test that fuse_action_time returns adaRMS conditioning in PI05 mode."""
        batch_size = 2
        action_emb = torch.randn(batch_size, config_pi05.action_horizon, config_pi05.expert_width)
        time_emb = torch.randn(batch_size, config_pi05.expert_width)
        
        result, adarms = suffix_embedding_pi05.fuse_action_time(action_emb, time_emb)
        
        # In PI05, adarms should be the time embedding
        assert adarms is not None
        assert adarms.shape == (batch_size, config_pi05.expert_width)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

