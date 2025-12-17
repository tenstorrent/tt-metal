# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for ttnn_suffix.py module.

Tests suffix embedding components: action projection, state projection,
time embedding, and action-time fusion.
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


def create_suffix_weights(config):
    """Create mock weights for suffix embedding."""
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


class TestActionEmbeddingPCC:
    """PCC tests for action embedding."""
    
    def test_action_embedding_consistency(self):
        """Test action embedding is deterministic."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=512)
        weights = create_suffix_weights(config)
        suffix = SuffixEmbeddingTorch(config, weights)
        
        actions = torch.randn(4, config.action_horizon, config.action_dim)
        
        emb1 = suffix.embed_actions(actions)
        emb2 = suffix.embed_actions(actions)
        
        assert check_pcc(emb1, emb2, threshold=1.0, test_name="action_embedding_consistency")
    
    def test_action_embedding_shape(self):
        """Test action embedding has correct shape."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=1024)
        weights = create_suffix_weights(config)
        suffix = SuffixEmbeddingTorch(config, weights)
        
        batch_size = 4
        actions = torch.randn(batch_size, config.action_horizon, config.action_dim)
        emb = suffix.embed_actions(actions)
        
        assert emb.shape == (batch_size, config.action_horizon, config.expert_width)


class TestStateEmbeddingPCC:
    """PCC tests for state embedding."""
    
    def test_state_embedding_consistency(self):
        """Test state embedding is deterministic."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=512, pi05=False)
        weights = create_suffix_weights(config)
        suffix = SuffixEmbeddingTorch(config, weights)
        
        state = torch.randn(4, config.action_dim)
        
        emb1 = suffix.embed_state(state)
        emb2 = suffix.embed_state(state)
        
        assert check_pcc(emb1, emb2, threshold=1.0, test_name="state_embedding_consistency")
    
    def test_state_embedding_pi05_none(self):
        """Test state embedding returns None in PI05 mode."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=512, pi05=True)
        weights = create_suffix_weights(config)
        suffix = SuffixEmbeddingTorch(config, weights)
        
        state = torch.randn(4, config.action_dim)
        emb = suffix.embed_state(state)
        
        assert emb is None, "PI05 mode should return None for state embedding"


class TestTimeEmbeddingPCC:
    """PCC tests for timestep embedding."""
    
    def test_time_embedding_consistency(self):
        """Test timestep embedding is deterministic."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=512)
        weights = create_suffix_weights(config)
        suffix = SuffixEmbeddingTorch(config, weights)
        
        timestep = torch.rand(4)
        
        emb1 = suffix.embed_timestep(timestep)
        emb2 = suffix.embed_timestep(timestep)
        
        assert check_pcc(emb1, emb2, threshold=1.0, test_name="time_embedding_consistency")
    
    def test_time_embedding_different_values(self):
        """Test different timesteps produce different embeddings."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=512)
        weights = create_suffix_weights(config)
        suffix = SuffixEmbeddingTorch(config, weights)
        
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([0.5])
        t3 = torch.tensor([1.0])
        
        e1 = suffix.embed_timestep(t1)
        e2 = suffix.embed_timestep(t2)
        e3 = suffix.embed_timestep(t3)
        
        # Different timesteps should produce different embeddings
        pcc_12 = compute_pcc(e1, e2)
        pcc_23 = compute_pcc(e2, e3)
        
        assert pcc_12 < 0.99, f"t=0 and t=0.5 too similar: PCC={pcc_12}"
        assert pcc_23 < 0.99, f"t=0.5 and t=1.0 too similar: PCC={pcc_23}"


class TestActionTimeFusionPCC:
    """PCC tests for action-time fusion MLP."""
    
    def test_fusion_consistency(self):
        """Test action-time fusion is deterministic."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=512, pi05=False)
        weights = create_suffix_weights(config)
        suffix = SuffixEmbeddingTorch(config, weights)
        
        action_emb = torch.randn(4, config.action_horizon, config.expert_width)
        time_emb = torch.randn(4, config.expert_width)
        
        fused1, _ = suffix.fuse_action_time(action_emb, time_emb)
        fused2, _ = suffix.fuse_action_time(action_emb, time_emb)
        
        assert check_pcc(fused1, fused2, threshold=1.0, test_name="action_time_fusion_consistency")
    
    def test_fusion_shape(self):
        """Test fusion output has correct shape."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=512, pi05=False)
        weights = create_suffix_weights(config)
        suffix = SuffixEmbeddingTorch(config, weights)
        
        batch_size = 4
        action_emb = torch.randn(batch_size, config.action_horizon, config.expert_width)
        time_emb = torch.randn(batch_size, config.expert_width)
        
        fused, adarms = suffix.fuse_action_time(action_emb, time_emb)
        
        assert fused.shape == (batch_size, config.action_horizon, config.expert_width)
        assert adarms is None  # PI0 mode


class TestFullSuffixEmbeddingPCC:
    """PCC tests for complete suffix embedding."""
    
    def test_embed_suffix_consistency(self):
        """Test full suffix embedding is deterministic."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=512, pi05=False)
        weights = create_suffix_weights(config)
        suffix = SuffixEmbeddingTorch(config, weights)
        
        state = torch.randn(4, config.action_dim)
        actions = torch.randn(4, config.action_horizon, config.action_dim)
        timestep = torch.rand(4)
        
        embs1, pad1, att1, _ = suffix.embed_suffix(state, actions, timestep)
        embs2, pad2, att2, _ = suffix.embed_suffix(state, actions, timestep)
        
        assert check_pcc(embs1, embs2, threshold=1.0, test_name="embed_suffix_consistency")
        assert torch.equal(pad1, pad2), "Pad masks should be equal"
        assert torch.equal(att1, att2), "Att masks should be equal"
    
    def test_embed_suffix_shape(self):
        """Test suffix embedding has correct shape."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=1024, pi05=False)
        weights = create_suffix_weights(config)
        suffix = SuffixEmbeddingTorch(config, weights)
        
        batch_size = 4
        state = torch.randn(batch_size, config.action_dim)
        actions = torch.randn(batch_size, config.action_horizon, config.action_dim)
        timestep = torch.rand(batch_size)
        
        embs, pad, att, _ = suffix.embed_suffix(state, actions, timestep)
        
        # State token + action tokens
        expected_len = 1 + config.action_horizon
        assert embs.shape == (batch_size, expected_len, config.expert_width)
        assert pad.shape == (batch_size, expected_len)
        assert att.shape == (batch_size, expected_len)


class TestOutputProjectionPCC:
    """PCC tests for output projection."""
    
    def test_project_output_consistency(self):
        """Test output projection is deterministic."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=512)
        weights = create_suffix_weights(config)
        suffix = SuffixEmbeddingTorch(config, weights)
        
        expert_output = torch.randn(4, config.action_horizon, config.expert_width)
        
        out1 = suffix.project_output(expert_output)
        out2 = suffix.project_output(expert_output)
        
        assert check_pcc(out1, out2, threshold=1.0, test_name="project_output_consistency")
    
    def test_project_output_shape(self):
        """Test output projection has correct shape."""
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
        
        config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=512)
        weights = create_suffix_weights(config)
        suffix = SuffixEmbeddingTorch(config, weights)
        
        batch_size = 4
        expert_output = torch.randn(batch_size, config.action_horizon, config.expert_width)
        out = suffix.project_output(expert_output)
        
        assert out.shape == (batch_size, config.action_horizon, config.action_dim)


class TestSuffixTTNNvsTorchPCC:
    """PCC tests comparing TTNN and PyTorch suffix implementations."""
    
    def get_device(self):
        """Get TTNN device (manual setup for non-pytest runs)."""
        if not TTNN_AVAILABLE:
            return None
        return ttnn.open_device(device_id=0)
    
    def close_device(self, device):
        """Close TTNN device."""
        if device is not None:
            ttnn.close_device(device)
    
    def test_action_embedding_ttnn(self):
        """Test TTNN action embedding matches PyTorch."""
        if not TTNN_AVAILABLE:
            print("Skipping TTNN test - TTNN not available")
            return
        
        from ...ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch, SuffixEmbeddingTTNN
        from ...ttnn_suffix import convert_suffix_weights_to_ttnn
        
        device = self.get_device()
        try:
            config = SuffixConfig(action_dim=32, action_horizon=32, expert_width=256)
            torch_weights = create_suffix_weights(config)
            ttnn_weights = convert_suffix_weights_to_ttnn(torch_weights, device)
            
            torch_suffix = SuffixEmbeddingTorch(config, torch_weights)
            ttnn_suffix = SuffixEmbeddingTTNN(config, ttnn_weights, device)
            
            actions = torch.randn(2, config.action_horizon, config.action_dim)
            
            # PyTorch
            torch_emb = torch_suffix.embed_actions(actions)
            
            # TTNN
            actions_ttnn = torch_to_ttnn(actions, device)
            ttnn_emb = ttnn_suffix.embed_actions(actions_ttnn)
            ttnn_emb_torch = ttnn_to_torch(ttnn_emb)
            
            assert check_pcc(
                torch_emb, ttnn_emb_torch,
                threshold=0.97,
                test_name="action_embedding_ttnn_vs_torch"
            )
        finally:
            self.close_device(device)


def run_pcc_suffix_tests():
    """Run all PCC tests for suffix module."""
    print("=" * 60)
    print("PCC Tests: ttnn_suffix.py")
    print("=" * 60)
    
    test_action = TestActionEmbeddingPCC()
    test_action.test_action_embedding_consistency()
    test_action.test_action_embedding_shape()
    
    test_state = TestStateEmbeddingPCC()
    test_state.test_state_embedding_consistency()
    test_state.test_state_embedding_pi05_none()
    
    test_time = TestTimeEmbeddingPCC()
    test_time.test_time_embedding_consistency()
    test_time.test_time_embedding_different_values()
    
    test_fusion = TestActionTimeFusionPCC()
    test_fusion.test_fusion_consistency()
    test_fusion.test_fusion_shape()
    
    test_full = TestFullSuffixEmbeddingPCC()
    test_full.test_embed_suffix_consistency()
    test_full.test_embed_suffix_shape()
    
    test_proj = TestOutputProjectionPCC()
    test_proj.test_project_output_consistency()
    test_proj.test_project_output_shape()
    
    print("\n✓ All PCC tests for ttnn_suffix.py passed!")


if __name__ == "__main__":
    run_pcc_suffix_tests()

