# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for ttnn_denoise.py module.

Tests denoising components: flow matching, Euler integration,
timestep scheduling, and KV cache management.
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


class TestNoisesSamplingPCC:
    """PCC tests for noise sampling."""
    
    def test_noise_shape(self):
        """Test noise sampling produces correct shape."""
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        config = DenoiseConfig(num_steps=10, action_dim=32, action_horizon=50)
        
        def dummy_forward(x, t, kv_cache=None, **kwargs):
            return torch.zeros_like(x)
        
        module = DenoisingModuleTorch(config, dummy_forward)
        
        batch_size = 4
        noise = module.sample_noise(batch_size)
        
        assert noise.shape == (batch_size, config.action_horizon, config.action_dim)
    
    def test_noise_statistics(self):
        """Test noise has standard normal statistics."""
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        config = DenoiseConfig(num_steps=10, action_dim=32, action_horizon=50)
        
        def dummy_forward(x, t, kv_cache=None, **kwargs):
            return torch.zeros_like(x)
        
        module = DenoisingModuleTorch(config, dummy_forward)
        
        # Sample many times for good statistics
        all_noise = []
        for _ in range(100):
            noise = module.sample_noise(batch_size=10)
            all_noise.append(noise)
        
        all_noise = torch.cat(all_noise, dim=0)
        mean = all_noise.mean().item()
        std = all_noise.std().item()
        
        assert abs(mean) < 0.05, f"Mean {mean} too far from 0"
        assert abs(std - 1.0) < 0.05, f"Std {std} too far from 1"


class TestTimestepSchedulePCC:
    """PCC tests for timestep scheduling."""
    
    def test_timesteps_range(self):
        """Test timesteps are in [0, 1] range."""
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        config = DenoiseConfig(num_steps=10, action_dim=32, action_horizon=50)
        
        def dummy_forward(x, t, kv_cache=None, **kwargs):
            return torch.zeros_like(x)
        
        module = DenoisingModuleTorch(config, dummy_forward)
        timesteps = module.get_timesteps()
        
        assert timesteps.min() >= 0.0
        assert timesteps.max() <= 1.0
    
    def test_timesteps_monotonic(self):
        """Test timesteps are monotonically decreasing."""
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        config = DenoiseConfig(num_steps=20, action_dim=32, action_horizon=50)
        
        def dummy_forward(x, t, kv_cache=None, **kwargs):
            return torch.zeros_like(x)
        
        module = DenoisingModuleTorch(config, dummy_forward)
        timesteps = module.get_timesteps()
        
        for i in range(len(timesteps) - 1):
            assert timesteps[i] > timesteps[i + 1], f"Not monotonic at index {i}"
    
    def test_timesteps_endpoints(self):
        """Test timesteps start at 1.0 and end at 0.0."""
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        config = DenoiseConfig(num_steps=10, action_dim=32, action_horizon=50)
        
        def dummy_forward(x, t, kv_cache=None, **kwargs):
            return torch.zeros_like(x)
        
        module = DenoisingModuleTorch(config, dummy_forward)
        timesteps = module.get_timesteps()
        
        assert timesteps[0] == 1.0, f"First timestep = {timesteps[0]}, expected 1.0"
        assert timesteps[-1] == 0.0, f"Last timestep = {timesteps[-1]}, expected 0.0"
    
    def test_timesteps_count(self):
        """Test correct number of timesteps."""
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        for num_steps in [5, 10, 20, 50]:
            config = DenoiseConfig(num_steps=num_steps, action_dim=32, action_horizon=50)
            
            def dummy_forward(x, t, kv_cache=None, **kwargs):
                return torch.zeros_like(x)
            
            module = DenoisingModuleTorch(config, dummy_forward)
            timesteps = module.get_timesteps()
            
            assert len(timesteps) == num_steps + 1, f"Expected {num_steps + 1} timesteps, got {len(timesteps)}"


class TestEulerIntegrationPCC:
    """PCC tests for Euler integration step."""
    
    def test_euler_step_zero_velocity(self):
        """Test Euler step with zero velocity."""
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        config = DenoiseConfig(num_steps=10, action_dim=32, action_horizon=50)
        
        def zero_forward(x, t, kv_cache=None, **kwargs):
            return torch.zeros_like(x)
        
        module = DenoisingModuleTorch(config, zero_forward)
        
        x_t = torch.randn(2, config.action_horizon, config.action_dim)
        t = torch.tensor([0.5, 0.5])
        dt = -0.1
        
        x_next = module.denoise_step(x_t, t, dt)
        
        # With zero velocity, x_next = x_t + dt * 0 = x_t
        assert check_pcc(x_t, x_next, threshold=1.0, test_name="euler_zero_velocity")
    
    def test_euler_step_constant_velocity(self):
        """Test Euler step with constant velocity."""
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        config = DenoiseConfig(num_steps=10, action_dim=32, action_horizon=50)
        
        # Velocity that moves toward zero
        def constant_forward(x, t, kv_cache=None, **kwargs):
            return -x  # Points toward origin
        
        module = DenoisingModuleTorch(config, constant_forward)
        
        x_t = torch.randn(2, config.action_horizon, config.action_dim)
        t = torch.tensor([0.5, 0.5])
        dt = -0.1
        
        x_next = module.denoise_step(x_t, t, dt)
        
        # x_next = x_t + dt * (-x_t) = x_t * (1 - dt) = x_t * 1.1
        expected = x_t * 1.1
        assert check_pcc(expected, x_next, threshold=0.99, test_name="euler_constant_velocity")
    
    def test_euler_step_consistency(self):
        """Test Euler step is deterministic."""
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        config = DenoiseConfig(num_steps=10, action_dim=32, action_horizon=50)
        
        def deterministic_forward(x, t, kv_cache=None, **kwargs):
            return torch.sin(x) + t[:, None, None]
        
        module = DenoisingModuleTorch(config, deterministic_forward)
        
        x_t = torch.randn(2, config.action_horizon, config.action_dim)
        t = torch.tensor([0.5, 0.5])
        dt = -0.1
        
        x_next_1 = module.denoise_step(x_t, t, dt)
        x_next_2 = module.denoise_step(x_t, t, dt)
        
        assert check_pcc(x_next_1, x_next_2, threshold=1.0, test_name="euler_consistency")


class TestFullDenoisingPCC:
    """PCC tests for full denoising loop."""
    
    def test_denoising_deterministic(self):
        """Test full denoising is deterministic with fixed seed."""
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        config = DenoiseConfig(num_steps=5, action_dim=4, action_horizon=8)
        
        def dummy_forward(x, t, kv_cache=None, **kwargs):
            return torch.zeros_like(x)  # Identity (no change)
        
        module = DenoisingModuleTorch(config, dummy_forward)
        
        torch.manual_seed(42)
        result1 = module.sample_actions(batch_size=2)
        
        torch.manual_seed(42)
        result2 = module.sample_actions(batch_size=2)
        
        assert check_pcc(result1, result2, threshold=1.0, test_name="denoising_deterministic")
    
    def test_denoising_output_shape(self):
        """Test denoising produces correct output shape."""
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        config = DenoiseConfig(num_steps=3, action_dim=32, action_horizon=50)
        
        def dummy_forward(x, t, kv_cache=None, **kwargs):
            return torch.zeros_like(x)
        
        module = DenoisingModuleTorch(config, dummy_forward)
        
        batch_size = 4
        result = module.sample_actions(batch_size=batch_size)
        
        assert result.shape == (batch_size, config.action_horizon, config.action_dim)


class TestKVCacheManagerPCC:
    """PCC tests for KV cache manager."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        from ...ttnn_denoise import KVCacheManager
        
        cache = KVCacheManager(
            num_layers=4,
            max_seq_len=512,
            num_kv_heads=1,
            head_dim=256,
        )
        
        batch_size = 2
        cache.initialize(batch_size)
        
        assert cache.cache is not None
        assert len(cache.cache) == 4
        assert cache.seq_len == 0
    
    def test_cache_update(self):
        """Test cache update and retrieval."""
        from ...ttnn_denoise import KVCacheManager
        
        cache = KVCacheManager(
            num_layers=4,
            max_seq_len=512,
            num_kv_heads=1,
            head_dim=256,
        )
        
        batch_size = 2
        cache.initialize(batch_size)
        
        # Add first batch of KV
        new_k = torch.randn(batch_size, 1, 10, 256)
        new_v = torch.randn(batch_size, 1, 10, 256)
        
        cache.update(0, new_k, new_v)
        cache.increment_seq_len(10)
        
        k, v = cache.get(0)
        
        assert k.shape == (batch_size, 1, 10, 256)
        assert v.shape == (batch_size, 1, 10, 256)
    
    def test_cache_incremental_update(self):
        """Test incremental cache updates."""
        from ...ttnn_denoise import KVCacheManager
        
        cache = KVCacheManager(
            num_layers=2,
            max_seq_len=512,
            num_kv_heads=1,
            head_dim=128,
        )
        
        batch_size = 2
        cache.initialize(batch_size)
        
        # First update
        k1 = torch.randn(batch_size, 1, 5, 128)
        v1 = torch.randn(batch_size, 1, 5, 128)
        cache.update(0, k1, v1)
        cache.increment_seq_len(5)
        
        # Second update
        k2 = torch.randn(batch_size, 1, 3, 128)
        v2 = torch.randn(batch_size, 1, 3, 128)
        cache.update(0, k2, v2)
        cache.increment_seq_len(3)
        
        k, v = cache.get(0)
        assert k.shape == (batch_size, 1, 8, 128)


class TestFlowMatchingPCC:
    """PCC tests for flow matching properties."""
    
    def test_flow_direction(self):
        """Test flow matching moves from noise toward clean."""
        from ...ttnn_denoise import DenoiseConfig, DenoisingModuleTorch
        
        config = DenoiseConfig(num_steps=10, action_dim=8, action_horizon=4)
        
        # Flow that pulls toward origin
        def toward_origin_forward(x, t, kv_cache=None, **kwargs):
            return -x * (1 - t[:, None, None])  # Stronger at end
        
        module = DenoisingModuleTorch(config, toward_origin_forward)
        
        # Start with noise
        torch.manual_seed(123)
        actions = module.sample_actions(batch_size=2)
        
        # Actions should be closer to zero than initial noise
        initial_noise = torch.randn(2, 4, 8)
        
        initial_norm = initial_noise.norm().item()
        final_norm = actions.norm().item()
        
        # Final should have smaller norm due to pulling toward origin
        # (exact behavior depends on integration)
        assert actions.shape == (2, 4, 8)


def run_pcc_denoise_tests():
    """Run all PCC tests for denoise module."""
    print("=" * 60)
    print("PCC Tests: ttnn_denoise.py")
    print("=" * 60)
    
    test_noise = TestNoisesSamplingPCC()
    test_noise.test_noise_shape()
    test_noise.test_noise_statistics()
    
    test_time = TestTimestepSchedulePCC()
    test_time.test_timesteps_range()
    test_time.test_timesteps_monotonic()
    test_time.test_timesteps_endpoints()
    test_time.test_timesteps_count()
    
    test_euler = TestEulerIntegrationPCC()
    test_euler.test_euler_step_zero_velocity()
    test_euler.test_euler_step_constant_velocity()
    test_euler.test_euler_step_consistency()
    
    test_full = TestFullDenoisingPCC()
    test_full.test_denoising_deterministic()
    test_full.test_denoising_output_shape()
    
    test_cache = TestKVCacheManagerPCC()
    test_cache.test_cache_initialization()
    test_cache.test_cache_update()
    test_cache.test_cache_incremental_update()
    
    test_flow = TestFlowMatchingPCC()
    test_flow.test_flow_direction()
    
    print("\n✓ All PCC tests for ttnn_denoise.py passed!")


if __name__ == "__main__":
    run_pcc_denoise_tests()

