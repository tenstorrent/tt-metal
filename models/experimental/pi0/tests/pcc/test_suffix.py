# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for Suffix Embedding.

Tests the suffix embedding module that processes state, noisy actions,
and timestep for the action expert transformer.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.pi0.common.configs import SuffixConfig
from models.experimental.pi0.reference.torch_suffix import SuffixEmbedding as SuffixEmbeddingTorch
from models.experimental.pi0.tt.ttnn_suffix import TtSuffixEmbedding, convert_suffix_weights_to_ttnn


@pytest.fixture
def suffix_config():
    """Default Suffix config for testing."""
    return SuffixConfig(
        action_dim=32,
        action_horizon=50,
        expert_width=1024,
        state_dim=32,
        time_emb_dim=1024,
        pi05=False,
    )


def create_suffix_weights(config: SuffixConfig):
    """Create random weights for suffix embedding."""
    return {
        "action_in_proj.weight": torch.randn(config.expert_width, config.action_dim),
        "action_in_proj.bias": torch.randn(config.expert_width),
        "action_out_proj.weight": torch.randn(config.action_dim, config.expert_width),
        "action_out_proj.bias": torch.randn(config.action_dim),
        "state_proj.weight": torch.randn(config.expert_width, config.state_dim),
        "state_proj.bias": torch.randn(config.expert_width),
        "action_time_mlp_in.weight": torch.randn(config.time_emb_dim, config.expert_width + config.time_emb_dim),
        "action_time_mlp_in.bias": torch.randn(config.time_emb_dim),
        "action_time_mlp_out.weight": torch.randn(config.expert_width, config.time_emb_dim),
        "action_time_mlp_out.bias": torch.randn(config.expert_width),
    }


class TestSuffixEmbedding:
    """PCC tests for Suffix Embedding."""
    
    def test_suffix_embed_state_pcc(self, device, suffix_config):
        """Test state embedding Torch vs TTNN."""
        config = suffix_config
        weights = create_suffix_weights(config)
        
        # Create input
        state = torch.randn(1, config.state_dim)
        
        # Torch forward
        suffix_torch = SuffixEmbeddingTorch(config, weights)
        out_torch = suffix_torch.embed_state(state)
        
        # TTNN forward
        weights_ttnn = convert_suffix_weights_to_ttnn(weights, device)
        suffix_ttnn = TtSuffixEmbedding(config, weights_ttnn, device)
        state_ttnn = ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn = suffix_ttnn.embed_state(state_ttnn)
        out_ttnn_torch = ttnn.to_torch(out_ttnn)
        
        assert_with_pcc(out_torch, out_ttnn_torch, pcc=0.95)
    
    def test_suffix_embed_actions_pcc(self, device, suffix_config):
        """Test action embedding Torch vs TTNN."""
        config = suffix_config
        weights = create_suffix_weights(config)
        
        # Create input
        noisy_actions = torch.randn(1, config.action_horizon, config.action_dim)
        
        # Torch forward
        suffix_torch = SuffixEmbeddingTorch(config, weights)
        out_torch = suffix_torch.embed_actions(noisy_actions)
        
        # TTNN forward
        weights_ttnn = convert_suffix_weights_to_ttnn(weights, device)
        suffix_ttnn = TtSuffixEmbedding(config, weights_ttnn, device)
        actions_ttnn = ttnn.from_torch(noisy_actions, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn = suffix_ttnn.embed_actions(actions_ttnn)
        out_ttnn_torch = ttnn.to_torch(out_ttnn)
        
        assert_with_pcc(out_torch, out_ttnn_torch, pcc=0.95)
    
    def test_suffix_project_output_pcc(self, device, suffix_config):
        """Test output projection Torch vs TTNN."""
        config = suffix_config
        weights = create_suffix_weights(config)
        
        # Create input (simulated expert output)
        expert_output = torch.randn(1, config.action_horizon, config.expert_width)
        
        # Torch forward
        suffix_torch = SuffixEmbeddingTorch(config, weights)
        out_torch = suffix_torch.project_output(expert_output)
        
        # TTNN forward
        weights_ttnn = convert_suffix_weights_to_ttnn(weights, device)
        suffix_ttnn = TtSuffixEmbedding(config, weights_ttnn, device)
        expert_ttnn = ttnn.from_torch(expert_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn = suffix_ttnn.project_output(expert_ttnn)
        out_ttnn_torch = ttnn.to_torch(out_ttnn)
        
        assert_with_pcc(out_torch, out_ttnn_torch, pcc=0.95)
