# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import sys
from pathlib import Path

# Add sweep_framework to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sweep_utils.memory_utils import capture_peak_memory, capture_peak_memory_with_cache_comparison


class MockTestModule:
    """Mock sweep module for testing"""

    @staticmethod
    def run(input_shape, dtype, device):
        """Simple test that creates a tensor and does an operation"""
        input_tensor = ttnn.from_torch(
            torch.rand(input_shape, dtype=torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.relu(input_tensor)
        return [True, 0]  # status, e2e_perf


class MockTestModuleBroadcast:
    """Mock sweep module with broadcast operation for higher memory usage"""

    @staticmethod
    def run(input_shape_a, input_shape_b, dtype, device):
        """Test with broadcast to ensure non-zero memory"""
        input_a = ttnn.from_torch(
            torch.rand(input_shape_a, dtype=torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        input_b = ttnn.from_torch(
            torch.rand(input_shape_b, dtype=torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.add(input_a, input_b)
        return [True, 0]


def test_capture_peak_memory_no_dispatch():
    """Test memory capture in NO_DISPATCH mode"""
    with ttnn.manage_device(device_id=0) as device:
        test_vector = {"input_shape": (1, 1, 64, 64), "dtype": ttnn.bfloat16}

        peak_memory = capture_peak_memory(MockTestModule, test_vector, device, use_no_dispatch=True)

        assert peak_memory is not None
        assert isinstance(peak_memory, int)
        assert peak_memory >= 0


def test_capture_peak_memory_normal_mode():
    """Test memory capture in NORMAL mode"""
    with ttnn.manage_device(device_id=0) as device:
        test_vector = {"input_shape": (1, 1, 32, 32), "dtype": ttnn.bfloat16}

        peak_memory = capture_peak_memory(MockTestModule, test_vector, device, use_no_dispatch=False)

        assert peak_memory is not None
        assert isinstance(peak_memory, int)
        assert peak_memory > 0  # NORMAL mode should show actual allocations


def test_capture_peak_memory_broadcast_nonzero():
    """Test memory capture with broadcast operation - guaranteed non-zero"""
    with ttnn.manage_device(device_id=0) as device:
        test_vector = {
            "input_shape_a": (4, 1, 32, 32),
            "input_shape_b": (1, 1, 32, 32),
            "dtype": ttnn.bfloat16,
        }

        peak_memory = capture_peak_memory(MockTestModuleBroadcast, test_vector, device, use_no_dispatch=True)

        assert peak_memory is not None
        assert isinstance(peak_memory, int)
        assert peak_memory > 0, f"Expected non-zero memory with broadcast, got {peak_memory}"
        # Should be at least 20KB for this configuration
        assert peak_memory > 20000, f"Expected >20KB for broadcast, got {peak_memory:,} bytes"


def test_capture_with_cache_comparison():
    """Test memory capture for cache comparison"""
    with ttnn.manage_device(device_id=0) as device:
        test_vector = {"input_shape": (1, 1, 32, 32), "dtype": ttnn.bfloat16}

        memory_dict = capture_peak_memory_with_cache_comparison(MockTestModule, test_vector, device)

        assert isinstance(memory_dict, dict)
        assert "uncached" in memory_dict
        assert "cached" in memory_dict
        # In NO_DISPATCH mode, both should be the same
        assert memory_dict["uncached"] == memory_dict["cached"]


def test_capture_handles_test_failure():
    """Test that memory capture works even if test fails"""

    class FailingTestModule:
        @staticmethod
        def run(**kwargs):
            raise ValueError("Test intentionally fails")

    with ttnn.manage_device(device_id=0) as device:
        test_vector = {"input_shape": (1, 1, 32, 32)}

        # Should not raise, should return None or a value
        peak_memory = capture_peak_memory(FailingTestModule, test_vector, device, use_no_dispatch=True)

        # Memory capture might still succeed even if test fails
        assert peak_memory is None or isinstance(peak_memory, int)


def test_memory_utils_import():
    """Test that memory_utils can be imported"""
    from sweep_utils.memory_utils import capture_peak_memory, capture_peak_memory_with_cache_comparison

    assert callable(capture_peak_memory)
    assert callable(capture_peak_memory_with_cache_comparison)
