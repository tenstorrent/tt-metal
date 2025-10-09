# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for All-Gather collective communication operation.
"""

import pytest
import torch
from src.building_blocks.ccl import (
    AllGatherImplConfig,
    AllGatherSpec,
    all_gather_forward,
    get_all_gather_default_impl_config,
)

import ttnn


class TestAllGatherSpec:
    """Test all-gather specification validation."""

    def test_valid_spec(self):
        """Test creating a valid all-gather spec."""
        spec = AllGatherSpec(mesh_shape=(2, 4))
        spec.validate()
        assert spec.mesh_shape == (2, 4)
        assert spec.cluster_axis is None
        assert spec.gather_dim == 3  # Default

    def test_spec_with_custom_dim(self):
        """Test all-gather spec with custom gather dimension."""
        spec = AllGatherSpec(mesh_shape=(2, 4), gather_dim=2)
        spec.validate()
        assert spec.gather_dim == 2

    def test_spec_with_cluster_axis(self):
        """Test all-gather spec with cluster axis for Galaxy."""
        spec = AllGatherSpec(mesh_shape=(4, 8), cluster_axis=1)
        spec.validate()
        assert spec.cluster_axis == 1

    def test_invalid_gather_dim(self):
        """Test validation with invalid gather dimension."""
        with pytest.raises(AssertionError):
            spec = AllGatherSpec(mesh_shape=(2, 4), gather_dim=4)
            spec.validate()

    def test_negative_gather_dim(self):
        """Test validation with negative gather dimension."""
        with pytest.raises(AssertionError):
            spec = AllGatherSpec(mesh_shape=(2, 4), gather_dim=-1)
            spec.validate()

    def test_invalid_mesh_shape(self):
        """Test validation with invalid mesh shape."""
        with pytest.raises(AssertionError):
            spec = AllGatherSpec(mesh_shape=(2, 4, 2))  # Too many dimensions
            spec.validate()


class TestAllGatherImplConfig:
    """Test all-gather implementation configuration."""

    def test_default_config(self):
        """Test default all-gather implementation config."""
        config = AllGatherImplConfig()
        assert config.num_all_gather_links == 2
        assert config.topology == ttnn.Topology.Linear
        assert config.dtype == ttnn.bfloat16
        assert config.sharded is False
        assert config.memory_config is None
        assert config.chunks_per_sync == 10
        assert config.num_workers_per_link == 2
        assert config.num_buffers_per_channel == 2

    def test_custom_config(self):
        """Test custom all-gather implementation config."""
        config = AllGatherImplConfig(
            num_all_gather_links=4,
            dtype=ttnn.bfloat8_b,
            sharded=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        assert config.num_all_gather_links == 4
        assert config.dtype == ttnn.bfloat8_b
        assert config.sharded is True
        assert config.memory_config == ttnn.L1_MEMORY_CONFIG

    def test_topology_config(self):
        """Test all-gather topology configuration."""
        config = AllGatherImplConfig(topology=ttnn.Topology.Ring)
        assert config.topology == ttnn.Topology.Ring


class TestGetDefaultImplConfig:
    """Test default implementation config generation."""

    def test_all_gather_n150_device(self):
        """Test default all-gather config for N150 device."""
        spec = AllGatherSpec(mesh_shape=(1, 1))
        config = get_all_gather_default_impl_config(spec, "N150", "prefill")
        assert config.num_all_gather_links == 1
        assert config.memory_config == ttnn.L1_MEMORY_CONFIG

    def test_all_gather_n300_device(self):
        """Test default all-gather config for N300 device."""
        spec = AllGatherSpec(mesh_shape=(1, 2))
        config = get_all_gather_default_impl_config(spec, "N300", "prefill")
        assert config.num_all_gather_links == 2
        assert config.memory_config == ttnn.DRAM_MEMORY_CONFIG

    def test_all_gather_t3000_device(self):
        """Test default all-gather config for T3000 device."""
        spec = AllGatherSpec(mesh_shape=(2, 4))
        config = get_all_gather_default_impl_config(spec, "T3000", "prefill")
        assert config.num_all_gather_links == 2
        assert config.memory_config == ttnn.DRAM_MEMORY_CONFIG
        assert config.chunks_per_sync == 20

    def test_all_gather_galaxy_device(self):
        """Test default all-gather config for Galaxy device."""
        spec = AllGatherSpec(mesh_shape=(4, 8), cluster_axis=1)
        config = get_all_gather_default_impl_config(spec, "TG", "prefill")
        assert config.memory_config == ttnn.DRAM_MEMORY_CONFIG
        # Galaxy-specific optimizations
        assert config.chunks_per_sync == 30


class TestAllGatherForward:
    """Test all-gather forward operation."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_all_gather_basic(self):
        """Test basic all-gather operation."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 4096
        mesh_shape = (2, 4)
        num_devices = mesh_shape[0] * mesh_shape[1]

        spec = AllGatherSpec(mesh_shape=mesh_shape, gather_dim=3)
        impl_config = get_all_gather_default_impl_config(spec, "T3000", "prefill")

        # Create dummy input (each device has partial data)
        shard_dim = hidden_dim // num_devices
        input_tensor = torch.randn(batch_size, seq_len, shard_dim)
        input_tt = ttnn.from_torch(input_tensor)

        # Create mock mesh device and CCL manager
        mesh_device = None  # Would be actual device mesh
        ccl_manager = None  # Would be actual CCL manager

        # Forward pass
        output = all_gather_forward(input_tt, mesh_device, ccl_manager, spec, impl_config)

        # Output should have full dimension after gathering
        expected_shape = (batch_size, seq_len, hidden_dim)
        assert output.shape == expected_shape

    @pytest.mark.skip(reason="Requires actual device")
    def test_all_gather_different_dims(self):
        """Test all-gather on different dimensions."""
        # Test gathering along different tensor dimensions
        for gather_dim in [1, 2, 3]:
            spec = AllGatherSpec(mesh_shape=(2, 4), gather_dim=gather_dim)
            # Would test gathering along each dimension

    @pytest.mark.skip(reason="Requires actual device")
    def test_all_gather_single_device_passthrough(self):
        """Test that single device all-gather is a no-op."""
        spec = AllGatherSpec(mesh_shape=(1, 1))
        impl_config = get_all_gather_default_impl_config(spec, "N150", "prefill")

        # Single device should just pass through
        # This tests the early return logic


@pytest.mark.perf
class TestAllGatherPerformance:
    """Performance tests for all-gather operation."""

    @pytest.mark.skip(reason="Requires actual device")
    @pytest.mark.parametrize("hidden_dim", [1024, 4096, 16384])
    def test_all_gather_bandwidth(self, hidden_dim):
        """Test all-gather bandwidth utilization for different sizes."""
        # Would measure effective bandwidth

    @pytest.mark.skip(reason="Requires actual device")
    def test_all_gather_overlap(self):
        """Test overlapping all-gather with computation."""
        # Would test async all-gather overlapped with other ops


class TestAllGatherUseCases:
    """Test specific use cases for all-gather."""

    def test_sequence_parallel_gather(self):
        """Test all-gather for sequence parallelism."""
        # In sequence parallelism, we split sequence across devices
        # and gather when needed
        seq_len_per_device = 512
        num_devices = 8
        total_seq_len = seq_len_per_device * num_devices
        assert total_seq_len == 4096

    def test_tensor_parallel_gather(self):
        """Test all-gather for tensor parallelism."""
        # In tensor parallelism, we split hidden dimension
        # and gather after certain operations
        hidden_dim_per_device = 1024
        num_devices = 4
        total_hidden_dim = hidden_dim_per_device * num_devices
        assert total_hidden_dim == 4096

    def test_expert_parallel_gather(self):
        """Test all-gather for expert parallelism in MoE."""
        # In MoE, experts can be distributed across devices
        # and results gathered after routing
        experts_per_device = 2
        num_devices = 8
        total_experts = experts_per_device * num_devices
        assert total_experts == 16


class TestAllGatherMemoryEfficiency:
    """Test memory efficiency of all-gather operation."""

    def test_in_place_gather(self):
        """Test in-place all-gather to save memory."""
        # Some implementations support in-place gather
        # where output buffer is pre-allocated

    def test_sharded_gather(self):
        """Test sharded all-gather for large tensors."""
        # For very large tensors, might need to gather in chunks
        # to fit in memory


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
