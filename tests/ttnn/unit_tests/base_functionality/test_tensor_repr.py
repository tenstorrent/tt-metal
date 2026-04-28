# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Tests for tensor repr/to_string functionality.

These tests verify that repr() (which uses tensor_impl::to_string()) correctly
displays tensor values for various shapes, layouts, and memory configurations.
"""

import pytest
import torch
import ttnn


# Shapes to test - includes edge cases that trigger padding bugs
TEST_SHAPES = [
    [32, 32],  # Simple tile-aligned
    [8, 1, 32],  # Bug case: middle dim < tile height
    [1, 8, 32],  # Batch = 1, height < tile
    [1, 1, 32, 64],  # 4D with small batch/channel
    [64, 64],  # Multiple tiles
    [33, 33],  # Non-tile-aligned (will be padded)
]


class TestTensorRepr:
    """Tests for tensor repr/to_string functionality."""

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
    @pytest.mark.parametrize("on_device", [False, True])
    def test_repr_no_spurious_zeros(self, device, shape, layout, on_device):
        """
        Verify repr() doesn't show zeros when data contains none.

        This catches the bug where to_string reads wrong physical indices
        for tiled tensors with padding.
        """
        # Create data with no zeros: [1, 2, 3, ...]
        numel = 1
        for dim in shape:
            numel *= dim
        data = (torch.arange(numel, dtype=torch.bfloat16) + 1).reshape(shape)

        # Create tensor
        if on_device:
            tensor = ttnn.from_torch(
                data,
                dtype=ttnn.bfloat16,
                layout=layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            tensor = ttnn.from_torch(data, dtype=ttnn.bfloat16, layout=layout)

        # Get repr and to_torch
        repr_str = repr(tensor)
        result = ttnn.to_torch(tensor)

        # Verify to_torch is correct (sanity check)
        assert torch.allclose(
            result, data, rtol=0.01, atol=0.01
        ), f"to_torch mismatch for shape={shape}, layout={layout}, on_device={on_device}"

        # Key check: repr should not contain "0.0000" since our data has no zeros
        # (values start at 1.0)
        assert "0.0000" not in repr_str, (
            f"BUG: repr() shows zeros but data has none.\n"
            f"Shape={shape}, layout={layout}, on_device={on_device}\n"
            f"First few values from to_torch: {result.flatten()[:8].tolist()}\n"
            f"repr:\n{repr_str[:500]}"
        )

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
    def test_repr_different_dtypes(self, device, shape, dtype):
        """Test repr works for different data types."""
        torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
        numel = 1
        for dim in shape:
            numel *= dim
        data = (torch.arange(numel, dtype=torch_dtype) + 1).reshape(shape)

        tensor = ttnn.from_torch(
            data,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        repr_str = repr(tensor)
        result = ttnn.to_torch(tensor)

        assert torch.allclose(result, data, rtol=0.01, atol=0.01)
        assert "0.0000" not in repr_str

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
    def test_repr_cpu_tensor(self, shape, layout):
        """Test repr for CPU-only tensor (no device)."""
        numel = 1
        for dim in shape:
            numel *= dim
        data = (torch.arange(numel, dtype=torch.bfloat16) + 1).reshape(shape)
        tensor = ttnn.from_torch(data, dtype=ttnn.bfloat16, layout=layout)

        repr_str = repr(tensor)
        result = ttnn.to_torch(tensor)

        assert torch.allclose(result, data, rtol=0.01, atol=0.01)
        assert "0.0000" not in repr_str

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
    def test_repr_different_memory_configs(self, device, shape, memory_config):
        """Test repr works for different memory configurations."""
        numel = 1
        for dim in shape:
            numel *= dim
        data = (torch.arange(numel, dtype=torch.bfloat16) + 1).reshape(shape)

        tensor = ttnn.from_torch(
            data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        repr_str = repr(tensor)
        result = ttnn.to_torch(tensor)

        assert torch.allclose(result, data, rtol=0.01, atol=0.01)
        assert "0.0000" not in repr_str
