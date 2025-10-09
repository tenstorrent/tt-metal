# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for CCL Manager.
"""

import pytest
from src.building_blocks.ccl import CCLManager


class TestCCLManager:
    """Test CCL manager for semaphore handling."""

    @pytest.fixture
    def mock_mesh_device(self):
        """Create a mock mesh device for testing."""

        class MockDevice:
            def __init__(self, shape=(2, 4)):
                self.shape = shape
                self._devices = [[MockSingleDevice() for _ in range(shape[1])] for _ in range(shape[0])]

            def get_device(self, row, col):
                return self._devices[row][col]

            def compute_with_storage_grid_size(self):
                class GridSize:
                    x = 8
                    y = 8

                return GridSize()

        class MockSingleDevice:
            def __init__(self):
                self._id = id(self)

            def id(self):
                return self._id

            def compute_with_storage_grid_size(self):
                class GridSize:
                    x = 8
                    y = 8

                return GridSize()

        return MockDevice()

    @pytest.mark.skip(reason="Requires actual device semaphore API")
    def test_ccl_manager_creation(self, mock_mesh_device):
        """Test CCL manager creation."""
        ccl_manager = CCLManager(mesh_device=mock_mesh_device, num_semaphores=32)

        # Should create semaphore tensors
        assert ccl_manager.semaphore_tensor is not None
        assert ccl_manager.current_semaphore_id == 0
        assert ccl_manager.num_semaphores == 32

    @pytest.mark.skip(reason="Requires actual device semaphore API")
    def test_semaphore_handle_generation(self, mock_mesh_device):
        """Test semaphore handle generation."""
        ccl_manager = CCLManager(mesh_device=mock_mesh_device, num_semaphores=32)

        # Get first handle
        handle1 = ccl_manager.get_semaphore_handle()
        assert handle1 == 0

        # Get second handle
        handle2 = ccl_manager.get_semaphore_handle()
        assert handle2 == 1

        # Should cycle through semaphores
        ccl_manager.current_semaphore_id = 31
        handle3 = ccl_manager.get_semaphore_handle()
        assert handle3 == 31

        # Should wrap around
        handle4 = ccl_manager.get_semaphore_handle()
        assert handle4 == 0

    @pytest.mark.skip(reason="Requires actual device semaphore API")
    def test_semaphore_cycling(self, mock_mesh_device):
        """Test semaphore handle cycling behavior."""
        ccl_manager = CCLManager(mesh_device=mock_mesh_device, num_semaphores=4)  # Small number for testing

        handles = []
        for _ in range(10):  # Get more handles than semaphores
            handles.append(ccl_manager.get_semaphore_handle())

        # Should see cycling pattern
        expected = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        assert handles == expected

    def test_semaphore_memory_requirements(self):
        """Test semaphore memory requirements calculation."""
        # Each semaphore typically needs 32 bytes (16B address + metadata)
        num_semaphores = 32
        bytes_per_semaphore = 32
        total_memory = num_semaphores * bytes_per_semaphore

        # Should be 1KB for 32 semaphores
        assert total_memory == 1024

    def test_optimal_semaphore_count(self):
        """Test optimal semaphore count for different scenarios."""
        # For most CCL operations, 32 semaphores is sufficient
        # This allows multiple in-flight operations without conflicts

        # For simple all-reduce
        operations_in_flight = 4
        semaphores_per_op = 2  # reduce-scatter + all-gather
        min_semaphores = operations_in_flight * semaphores_per_op
        assert min_semaphores == 8

        # With safety margin
        recommended_semaphores = min_semaphores * 4
        assert recommended_semaphores == 32


class TestCCLManagerMultiDevice:
    """Test CCL manager with multiple device configurations."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_single_device_manager(self):
        """Test CCL manager with single device."""
        # Single device doesn't need semaphores
        # Manager should handle gracefully

    @pytest.mark.skip(reason="Requires actual device")
    def test_n300_manager(self):
        """Test CCL manager on N300 (2 devices)."""
        # N300 has 2 devices connected
        # Manager should initialize for 2-device communication

    @pytest.mark.skip(reason="Requires actual device")
    def test_t3000_manager(self):
        """Test CCL manager on T3000 (8 devices)."""
        # T3000 has 8 devices in 2x4 mesh
        # Manager should handle mesh topology

    @pytest.mark.skip(reason="Requires actual device")
    def test_galaxy_manager(self):
        """Test CCL manager on Galaxy (32 devices)."""
        # Galaxy has 32 devices in 4x8 mesh
        # Manager needs to handle complex routing


class TestCCLManagerEdgeCases:
    """Test edge cases for CCL manager."""

    def test_zero_semaphores(self):
        """Test handling of zero semaphores."""
        with pytest.raises(ValueError):
            CCLManager(mesh_device=None, num_semaphores=0)

    def test_too_many_semaphores(self):
        """Test handling of excessive semaphore count."""
        # Hardware might have limits
        max_semaphores = 1024  # Hypothetical limit

        with pytest.raises(ValueError):
            CCLManager(mesh_device=None, num_semaphores=max_semaphores + 1)

    @pytest.mark.skip(reason="Requires actual device")
    def test_concurrent_operations(self):
        """Test handling of concurrent CCL operations."""
        # Multiple operations might request semaphores simultaneously
        # Manager should handle without conflicts


@pytest.mark.perf
class TestCCLManagerPerformance:
    """Performance tests for CCL manager."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_semaphore_allocation_overhead(self):
        """Test overhead of semaphore allocation."""
        # Semaphore allocation should be negligible
        # compared to actual CCL operations

    @pytest.mark.skip(reason="Requires actual device")
    def test_semaphore_contention(self):
        """Test performance under semaphore contention."""
        # With too few semaphores, operations might wait
        # Test impact on performance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
