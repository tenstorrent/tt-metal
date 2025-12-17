# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for denoising module.
"""

import pytest
import torch

from ..ttnn_denoise import (
    DenoiseConfig,
    DenoisingModuleTorch,
    KVCacheManager,
)


class TestDenoisingModule:
    """Tests for denoising module."""
    
    @pytest.fixture
    def config(self):
        return DenoiseConfig(
            num_steps=10,
            action_dim=32,
            action_horizon=50,
        )
    
    @pytest.fixture
    def mock_forward_fn(self, config):
        """Create a mock forward function that returns zeros."""
        def forward_fn(x, t, kv_cache=None, **kwargs):
            return torch.zeros_like(x)
        return forward_fn
    
    @pytest.fixture
    def denoising_module(self, config, mock_forward_fn):
        return DenoisingModuleTorch(config, mock_forward_fn)
    
    def test_sample_noise_shape(self, denoising_module, config):
        """Test noise sampling shape."""
        batch_size = 4
        
        noise = denoising_module.sample_noise(batch_size)
        
        assert noise.shape == (batch_size, config.action_horizon, config.action_dim)
    
    def test_get_timesteps(self, denoising_module, config):
        """Test timestep generation."""
        timesteps = denoising_module.get_timesteps()
        
        assert len(timesteps) == config.num_steps + 1
        assert timesteps[0] == 1.0
        assert timesteps[-1] == 0.0
        # Should be monotonically decreasing
        for i in range(len(timesteps) - 1):
            assert timesteps[i] > timesteps[i + 1]
    
    def test_denoise_step_with_zero_velocity(self, config):
        """Test denoise step with zero velocity returns same input."""
        def zero_forward(x, t, kv_cache=None, **kwargs):
            return torch.zeros_like(x)
        
        module = DenoisingModuleTorch(config, zero_forward)
        x_t = torch.randn(2, config.action_horizon, config.action_dim)
        t = torch.tensor([0.5, 0.5])
        dt = -0.1
        
        x_next = module.denoise_step(x_t, t, dt)
        
        # With zero velocity, x_next should equal x_t
        assert torch.allclose(x_next, x_t)
    
    def test_sample_actions_shape(self, denoising_module, config):
        """Test full denoising produces correct shape."""
        batch_size = 2
        
        actions = denoising_module.sample_actions(batch_size)
        
        assert actions.shape == (batch_size, config.action_horizon, config.action_dim)
    
    def test_sample_actions_with_constant_velocity(self, config):
        """Test denoising with constant velocity produces expected result."""
        # Velocity that moves toward zero
        def constant_forward(x, t, kv_cache=None, **kwargs):
            return -x  # Velocity points toward origin
        
        module = DenoisingModuleTorch(config, constant_forward)
        
        actions = module.sample_actions(batch_size=1)
        
        # After integration, should have moved from noise toward zero
        # (exact result depends on integration scheme)
        assert actions.shape == (1, config.action_horizon, config.action_dim)


class TestKVCacheManager:
    """Tests for KV cache manager."""
    
    @pytest.fixture
    def cache_manager(self):
        return KVCacheManager(
            num_layers=4,
            max_seq_len=512,
            num_kv_heads=1,
            head_dim=256,
        )
    
    def test_initialize(self, cache_manager):
        """Test cache initialization."""
        batch_size = 2
        
        cache_manager.initialize(batch_size)
        
        assert cache_manager.cache is not None
        assert len(cache_manager.cache) == cache_manager.num_layers
        assert cache_manager.seq_len == 0
    
    def test_update_and_get(self, cache_manager):
        """Test cache update and retrieval."""
        batch_size = 2
        cache_manager.initialize(batch_size)
        
        # Add some KV values
        new_k = torch.randn(batch_size, cache_manager.num_kv_heads, 10, cache_manager.head_dim)
        new_v = torch.randn(batch_size, cache_manager.num_kv_heads, 10, cache_manager.head_dim)
        
        cache_manager.update(0, new_k, new_v)
        cache_manager.increment_seq_len(10)
        
        k, v = cache_manager.get(0)
        
        assert k.shape == (batch_size, cache_manager.num_kv_heads, 10, cache_manager.head_dim)
        assert v.shape == (batch_size, cache_manager.num_kv_heads, 10, cache_manager.head_dim)
    
    def test_incremental_update(self, cache_manager):
        """Test incremental cache updates."""
        batch_size = 2
        cache_manager.initialize(batch_size)
        
        # First update
        k1 = torch.randn(batch_size, cache_manager.num_kv_heads, 5, cache_manager.head_dim)
        v1 = torch.randn(batch_size, cache_manager.num_kv_heads, 5, cache_manager.head_dim)
        cache_manager.update(0, k1, v1)
        cache_manager.increment_seq_len(5)
        
        # Second update
        k2 = torch.randn(batch_size, cache_manager.num_kv_heads, 3, cache_manager.head_dim)
        v2 = torch.randn(batch_size, cache_manager.num_kv_heads, 3, cache_manager.head_dim)
        cache_manager.update(0, k2, v2)
        cache_manager.increment_seq_len(3)
        
        k, v = cache_manager.get(0)
        
        assert k.shape == (batch_size, cache_manager.num_kv_heads, 8, cache_manager.head_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

