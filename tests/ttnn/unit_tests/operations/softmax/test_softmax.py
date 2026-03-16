# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the softmax operation (stub validation)."""

import pytest
import torch
import ttnn

from ttnn.operations.softmax import softmax


class TestSoftmaxValidation:
    """Test input validation logic (Python-side, no device needed for most)."""

    def test_wrong_dtype(self, device):
        """Input must be bfloat16."""
        torch_input = torch.randn(1, 1, 32, 32)
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        with pytest.raises(ValueError, match="bfloat16"):
            softmax(ttnn_input)

    def test_wrong_layout(self, device):
        """Input must be TILE_LAYOUT."""
        torch_input = torch.randn(1, 1, 32, 32)
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        with pytest.raises(ValueError, match="TILE_LAYOUT"):
            softmax(ttnn_input)

    def test_invalid_dim(self, device):
        """dim must be -1 or -2."""
        torch_input = torch.randn(1, 1, 32, 32)
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        with pytest.raises(ValueError, match="dim must be -1 or -2"):
            softmax(ttnn_input, dim=0)


class TestSoftmaxStubExecution:
    """Test that the operation infrastructure works with stub kernels.

    With empty stub kernels, output values will be garbage, but the
    infrastructure (allocation, program descriptor, generic_op call)
    should work without errors.
    """

    def test_basic_shape_runs(self, device):
        """Basic 1x1x32x32 shape executes without Python-side errors."""
        torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = softmax(ttnn_input)

        # Verify output shape and dtype are correct
        assert (
            ttnn_output.shape == ttnn_input.shape
        ), f"Shape mismatch: expected {ttnn_input.shape}, got {ttnn_output.shape}"
        assert ttnn_output.dtype == ttnn.bfloat16
        assert ttnn_output.layout == ttnn.TILE_LAYOUT

    def test_multi_tile_shape_runs(self, device):
        """Multi-tile shape executes without Python-side errors."""
        torch_input = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = softmax(ttnn_input)

        assert ttnn_output.shape == ttnn_input.shape

    def test_batched_shape_runs(self, device):
        """Batched shape executes without Python-side errors."""
        torch_input = torch.randn(4, 2, 64, 64, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = softmax(ttnn_input)

        assert ttnn_output.shape == ttnn_input.shape

    def test_dim_minus2_runs(self, device):
        """dim=-2 (height) executes without Python-side errors."""
        torch_input = torch.randn(1, 1, 128, 32, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = softmax(ttnn_input, dim=-2)

        assert ttnn_output.shape == ttnn_input.shape

    def test_numeric_stable_false_runs(self, device):
        """numeric_stable=False executes without Python-side errors."""
        torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = softmax(ttnn_input, numeric_stable=False)

        assert ttnn_output.shape == ttnn_input.shape
