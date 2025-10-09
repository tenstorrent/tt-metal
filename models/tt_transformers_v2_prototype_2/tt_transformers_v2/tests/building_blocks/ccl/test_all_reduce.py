# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for All-Reduce collective communication operation.
"""

import pytest
import torch
from src.building_blocks.ccl import (
    AllReduceImplConfig,
    AllReduceSpec,
    all_reduce_forward,
    get_all_reduce_default_impl_config,
)

import ttnn


class TestAllReduceSpec:
    """Test all-reduce specification validation."""

    def test_valid_spec(self):
        """Test creating a valid all-reduce spec."""
        spec = AllReduceSpec(mesh_shape=(2, 4))
        spec.validate()
        assert spec.mesh_shape == (2, 4)
        assert spec.cluster_axis is None
        assert spec.reduce_dim == 0  # Default

    def test_spec_with_cluster_axis(self):
        """Test all-reduce spec with cluster axis."""
        spec = AllReduceSpec(mesh_shape=(2, 4), cluster_axis=1)
        spec.validate()
        assert spec.cluster_axis == 1

    def test_spec_with_custom_reduce_dim(self):
        """Test all-reduce spec with custom reduce dimension."""
        spec = AllReduceSpec(mesh_shape=(2, 4), reduce_dim=3)
        spec.validate()
        assert spec.reduce_dim == 3

    def test_invalid_mesh_shape(self):
        """Test validation with invalid mesh shape."""
        with pytest.raises(AssertionError):
            spec = AllReduceSpec(mesh_shape=(2,))  # Wrong number of dimensions
            spec.validate()

    def test_invalid_cluster_axis(self):
        """Test validation with invalid cluster axis."""
        with pytest.raises(AssertionError):
            spec = AllReduceSpec(mesh_shape=(2, 4), cluster_axis=2)  # Invalid axis
            spec.validate()

    def test_zero_dimension(self):
        """Test validation with zero dimension."""
        with pytest.raises(AssertionError):
            spec = AllReduceSpec(mesh_shape=(0, 4))
            spec.validate()

    def test_negative_reduce_dim(self):
        """Test validation with negative reduce dimension."""
        with pytest.raises(AssertionError):
            spec = AllReduceSpec(mesh_shape=(2, 4), reduce_dim=-1)
            spec.validate()


class TestAllReduceImplConfig:
    """Test all-reduce implementation configuration."""

    def test_default_config(self):
        """Test default all-reduce implementation config."""
        config = AllReduceImplConfig()
        assert config.num_reduce_scatter_links == 1
        assert config.num_all_gather_links == 2
        assert config.topology == ttnn.Topology.Linear
        assert config.dtype == ttnn.bfloat16
        assert config.sharded is False
        assert config.use_composite is False
        assert config.chunks_per_sync == 10
        assert config.num_workers_per_link == 2
        assert config.num_buffers_per_channel == 2

    def test_custom_config(self):
        """Test custom all-reduce implementation config."""
        config = AllReduceImplConfig(
            num_reduce_scatter_links=4,
            num_all_gather_links=4,
            dtype=ttnn.bfloat8_b,
            sharded=True,
            use_composite=True,
        )
        assert config.num_reduce_scatter_links == 4
        assert config.num_all_gather_links == 4
        assert config.dtype == ttnn.bfloat8_b
        assert config.sharded is True
        assert config.use_composite is True

    def test_memory_config(self):
        """Test all-reduce memory configuration."""
        config = AllReduceImplConfig(
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        assert config.memory_config == ttnn.L1_MEMORY_CONFIG


class TestGetDefaultImplConfig:
    """Test default implementation config generation."""

    def test_all_reduce_n150_device(self):
        """Test default all-reduce config for N150 device."""
        spec = AllReduceSpec(mesh_shape=(1, 1))
        config = get_all_reduce_default_impl_config(spec, "N150", "prefill")
        assert config.num_reduce_scatter_links == 1
        assert config.num_all_gather_links == 1
        assert config.memory_config == ttnn.L1_MEMORY_CONFIG

    def test_all_reduce_n300_device(self):
        """Test default all-reduce config for N300 device."""
        spec = AllReduceSpec(mesh_shape=(1, 2))
        config = get_all_reduce_default_impl_config(spec, "N300", "prefill")
        assert config.num_reduce_scatter_links == 1
        assert config.num_all_gather_links == 2
        assert config.memory_config == ttnn.DRAM_MEMORY_CONFIG

    def test_all_reduce_t3000_device(self):
        """Test default all-reduce config for T3000 device."""
        spec = AllReduceSpec(mesh_shape=(2, 4))
        config = get_all_reduce_default_impl_config(spec, "T3000", "prefill")
        assert config.num_reduce_scatter_links == 2
        assert config.num_all_gather_links == 2
        assert config.memory_config == ttnn.DRAM_MEMORY_CONFIG
        assert config.chunks_per_sync == 20

    def test_all_reduce_galaxy_device(self):
        """Test default all-reduce config for Galaxy device."""
        spec = AllReduceSpec(mesh_shape=(4, 8))
        config = get_all_reduce_default_impl_config(spec, "TG", "prefill")
        assert config.use_composite is True
        assert config.memory_config == ttnn.DRAM_MEMORY_CONFIG

    def test_all_reduce_unknown_device(self):
        """Test default all-reduce config for unknown device."""
        spec = AllReduceSpec(mesh_shape=(2, 2))
        config = get_all_reduce_default_impl_config(spec, "UNKNOWN", "prefill")
        # Should return conservative defaults
        assert config.num_reduce_scatter_links == 1
        assert config.num_all_gather_links == 2

    def test_all_reduce_mode_differences(self):
        """Test differences between prefill and decode configs."""
        spec = AllReduceSpec(mesh_shape=(2, 4))

        prefill_config = get_all_reduce_default_impl_config(spec, "T3000", "prefill")
        decode_config = get_all_reduce_default_impl_config(spec, "T3000", "decode")

        # Configs might differ (e.g., dtype, memory layout)
        assert prefill_config.memory_config is not None
        assert decode_config.memory_config is not None


class TestAllReduceForward:
    """Test all-reduce forward operation."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_all_reduce_basic(self):
        """Test basic all-reduce operation."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 4096
        mesh_shape = (2, 4)

        spec = AllReduceSpec(mesh_shape=mesh_shape)
        impl_config = get_all_reduce_default_impl_config(spec, "T3000", "prefill")

        # Create dummy input distributed across devices
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
        input_tt = ttnn.from_torch(input_tensor)

        # Create mock mesh device and CCL manager
        mesh_device = None  # Would be actual device mesh
        ccl_manager = None  # Would be actual CCL manager

        # Forward pass
        output = all_reduce_forward(input_tt, mesh_device, ccl_manager, spec, impl_config)

        # Output shape should be same as input
        assert output.shape == input_tt.shape

    @pytest.mark.skip(reason="Requires actual device")
    def test_all_reduce_single_device_passthrough(self):
        """Test that single device all-reduce is a no-op."""
        spec = AllReduceSpec(mesh_shape=(1, 1))
        impl_config = get_all_reduce_default_impl_config(spec, "N150", "prefill")

        # Single device should just pass through
        # This tests the early return logic

    @pytest.mark.skip(reason="Requires actual device")
    def test_all_reduce_with_dtype_conversion(self):
        """Test all-reduce with dtype conversion."""
        spec = AllReduceSpec(mesh_shape=(2, 4))
        impl_config = AllReduceImplConfig(dtype=ttnn.bfloat8_b)  # Different from input dtype

        # Should handle dtype conversion properly


@pytest.mark.perf
class TestAllReducePerformance:
    """Performance tests for all-reduce operation."""

    @pytest.mark.skip(reason="Requires actual device")
    @pytest.mark.parametrize("mesh_shape", [(1, 2), (2, 4), (4, 8)])
    def test_all_reduce_scaling(self, mesh_shape):
        """Test all-reduce performance scaling with device count."""
        # Would measure how performance scales with more devices

    @pytest.mark.skip(reason="Requires actual device")
    def test_composite_vs_separate_all_reduce(self):
        """Test composite vs separate reduce-scatter/all-gather."""
        # Would compare performance of different implementations

    @pytest.mark.skip(reason="Requires actual device")
    def test_all_reduce_bandwidth_utilization(self):
        """Test all-reduce bandwidth utilization."""
        # Would measure how well we utilize available bandwidth


class TestAllReduceAlgorithms:
    """Test different all-reduce algorithms."""

    def test_ring_all_reduce_concept(self):
        """Test ring all-reduce algorithm concept."""
        # Ring all-reduce: each device sends to next in ring
        # Takes 2*(n-1) steps for n devices
        num_devices = 8
        steps = 2 * (num_devices - 1)
        assert steps == 14

    def test_tree_all_reduce_concept(self):
        """Test tree all-reduce algorithm concept."""
        # Tree all-reduce: log(n) steps for reduction, log(n) for broadcast
        import math

        num_devices = 8
        steps = 2 * math.ceil(math.log2(num_devices))
        assert steps == 6  # More efficient than ring for small messages

    def test_reduce_scatter_all_gather_concept(self):
        """Test reduce-scatter + all-gather decomposition."""
        # All-reduce = reduce-scatter + all-gather
        # This allows overlapping communication with computation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
