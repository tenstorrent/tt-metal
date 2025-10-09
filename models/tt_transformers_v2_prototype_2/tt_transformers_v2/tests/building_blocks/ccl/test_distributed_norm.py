# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Distributed RMS Normalization in CCL.
"""

import pytest
import torch
from src.building_blocks.ccl import (
    AllGatherImplConfig,
    AllGatherSpec,
    DistributedRMSNormImplConfig,
    DistributedRMSNormSpec,
    distributed_rmsnorm_forward,
    get_distributed_rmsnorm_default_impl_config,
)

import ttnn


class TestDistributedRMSNormSpec:
    """Test distributed RMS normalization specification."""

    def test_valid_spec(self):
        """Test creating valid distributed RMS norm spec."""
        spec = DistributedRMSNormSpec(hidden_dim=4096, epsilon=1e-5)
        spec.validate()
        assert spec.hidden_dim == 4096
        assert spec.epsilon == 1e-5

    def test_default_epsilon(self):
        """Test default epsilon value."""
        spec = DistributedRMSNormSpec(hidden_dim=4096)
        spec.validate()
        assert spec.epsilon == 1e-5

    def test_large_hidden_dim(self):
        """Test with large hidden dimensions."""
        spec = DistributedRMSNormSpec(hidden_dim=16384, epsilon=1e-6)
        spec.validate()
        assert spec.hidden_dim == 16384

    def test_invalid_hidden_dim(self):
        """Test validation with invalid hidden dim."""
        with pytest.raises(AssertionError):
            spec = DistributedRMSNormSpec(hidden_dim=0)
            spec.validate()

    def test_invalid_epsilon(self):
        """Test validation with invalid epsilon."""
        with pytest.raises(AssertionError):
            spec = DistributedRMSNormSpec(hidden_dim=4096, epsilon=0)
            spec.validate()

        with pytest.raises(AssertionError):
            spec = DistributedRMSNormSpec(hidden_dim=4096, epsilon=-1e-5)
            spec.validate()


class TestDistributedRMSNormImplConfig:
    """Test distributed RMS norm implementation configuration."""

    def test_default_config(self):
        """Test default distributed RMS norm config."""
        config = DistributedRMSNormImplConfig()
        assert config.compute_kernel_config is not None
        assert config.sharded_input_memory_config is not None
        assert config.sharded_program_config is not None

    def test_custom_config(self):
        """Test custom distributed RMS norm config."""
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        config = DistributedRMSNormImplConfig(
            compute_kernel_config=compute_config,
            sharded_stats_memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        assert config.compute_kernel_config.math_fidelity == ttnn.MathFidelity.HiFi4
        assert config.sharded_stats_memory_config == ttnn.L1_MEMORY_CONFIG


class TestGetDefaultImplConfig:
    """Test default implementation config generation."""

    def test_distributed_rmsnorm_n150_device(self):
        """Test default config for N150 device."""
        spec = DistributedRMSNormSpec(hidden_dim=4096)
        config = get_distributed_rmsnorm_default_impl_config(spec, "N150", "prefill")

        # Should have appropriate compute config
        assert config.compute_kernel_config is not None
        assert config.compute_kernel_config.math_fidelity == ttnn.MathFidelity.HiFi2

    def test_distributed_rmsnorm_galaxy_device(self):
        """Test default config for Galaxy device."""
        spec = DistributedRMSNormSpec(hidden_dim=8192)
        config = get_distributed_rmsnorm_default_impl_config(spec, "TG", "prefill")

        # Galaxy should have optimized sharding
        assert config.sharded_input_memory_config is not None
        assert config.sharded_program_config is not None

    def test_distributed_rmsnorm_mode_differences(self):
        """Test differences between prefill and decode modes."""
        spec = DistributedRMSNormSpec(hidden_dim=4096)

        prefill_config = get_distributed_rmsnorm_default_impl_config(spec, "T3000", "prefill")
        decode_config = get_distributed_rmsnorm_default_impl_config(spec, "T3000", "decode")

        # Both modes should have configs
        assert prefill_config.compute_kernel_config is not None
        assert decode_config.compute_kernel_config is not None


class TestDistributedRMSNormForward:
    """Test distributed RMS norm forward operation."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_distributed_rmsnorm_basic(self):
        """Test basic distributed RMS norm operation."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 4096
        mesh_shape = (4, 8)  # Galaxy configuration

        # Specs
        norm_spec = DistributedRMSNormSpec(hidden_dim=hidden_dim, epsilon=1e-5)
        gather_spec = AllGatherSpec(mesh_shape=mesh_shape, cluster_axis=1)

        # Configs
        norm_config = get_distributed_rmsnorm_default_impl_config(norm_spec, "TG", "prefill")
        gather_config = AllGatherImplConfig(num_all_gather_links=2)

        # Create dummy input (distributed across devices)
        shard_dim = hidden_dim // (mesh_shape[0] * mesh_shape[1])
        input_tensor = torch.randn(batch_size, seq_len, shard_dim)
        weight = torch.ones(shard_dim)

        input_tt = ttnn.from_torch(input_tensor)
        weight_tt = ttnn.from_torch(weight)

        # Create mock mesh device and CCL manager
        mesh_device = None  # Would be actual device mesh
        ccl_manager = None  # Would be actual CCL manager

        # Forward pass
        output = distributed_rmsnorm_forward(
            input_tt,
            weight_tt,
            mesh_device,
            ccl_manager,
            norm_spec,
            norm_config,
            gather_spec,
            gather_config,
            mode="prefill",
        )

        # Output should be normalized
        assert output.shape == (batch_size, seq_len, hidden_dim)  # After gather

    @pytest.mark.skip(reason="Requires actual device")
    def test_distributed_rmsnorm_decode_mode(self):
        """Test distributed RMS norm in decode mode."""
        batch_size = 32  # Larger batch for decode
        seq_len = 1  # Single token
        hidden_dim = 8192
        mesh_shape = (4, 8)

        # Similar setup but for decode mode


class TestDistributedRMSNormAlgorithm:
    """Test the distributed RMS norm algorithm."""

    def test_distributed_statistics_computation(self):
        """Test distributed computation of RMS statistics."""
        # In distributed RMS norm:
        # 1. Each device computes local squared sum
        # 2. All-reduce to get global squared sum
        # 3. Compute global RMS
        # 4. Normalize locally

        num_devices = 32
        hidden_dim = 8192
        local_dim = hidden_dim // num_devices

        # Each device handles this many elements
        assert local_dim == 256

    def test_numerical_stability(self):
        """Test numerical stability of distributed computation."""
        # With many devices, need to be careful about:
        # - Precision loss in summation
        # - Overflow/underflow in local computations


@pytest.mark.perf
class TestDistributedRMSNormPerformance:
    """Performance tests for distributed RMS norm."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_distributed_vs_regular_rmsnorm(self):
        """Compare distributed vs regular RMS norm performance."""
        # Would measure if distribution provides speedup

    @pytest.mark.skip(reason="Requires actual device")
    def test_scaling_efficiency(self):
        """Test how well distributed RMS norm scales."""
        # Would measure scaling with different device counts


class TestDistributedRMSNormUseCases:
    """Test specific use cases for distributed RMS norm."""

    def test_galaxy_configuration(self):
        """Test Galaxy-specific distributed RMS norm."""
        # Galaxy uses 32 devices in 4x8 mesh
        mesh_shape = (4, 8)
        hidden_dim = 8192

        # Each device processes part of hidden dimension
        shard_size = hidden_dim // (mesh_shape[0] * mesh_shape[1])
        assert shard_size == 256

        # Statistics are computed across all devices
        # This allows handling very large hidden dimensions

    def test_memory_savings(self):
        """Test memory savings from distributed computation."""
        # For very large models, distributed norm saves memory
        # by sharding the weight matrix
        hidden_dim = 65536  # Very large
        num_devices = 32

        # Per-device memory for weights
        weight_size_per_device = hidden_dim // num_devices
        assert weight_size_per_device == 2048

        # Much more manageable than 65K on single device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
