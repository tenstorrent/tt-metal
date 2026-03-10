# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Input validation tests for softmax operation.

These tests verify that the operation correctly rejects invalid inputs
on the Python side (pre-device), raising ValueError or RuntimeError.
No device execution is expected for these tests.
"""

import pytest
import torch
import ttnn

from ttnn.operations.softmax import softmax


def _make_tensor(device, shape=(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Create a valid TTNN tensor with specified properties."""
    torch_tensor = torch.randn(shape, dtype=torch.bfloat16)
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ---------------------------------------------------------------------------
# Wrong dtype
# ---------------------------------------------------------------------------


@pytest.mark.validation
def test_rejects_float32_dtype(device):
    """Must reject float32 input tensor."""
    t = _make_tensor(device, dtype=ttnn.float32)
    with pytest.raises((ValueError, RuntimeError)):
        softmax(t)


@pytest.mark.validation
def test_rejects_uint16_dtype(device):
    """Must reject uint16 input tensor."""
    t = _make_tensor(device, dtype=ttnn.uint16)
    with pytest.raises((ValueError, RuntimeError)):
        softmax(t)


# ---------------------------------------------------------------------------
# Wrong layout
# ---------------------------------------------------------------------------


@pytest.mark.validation
def test_rejects_row_major_layout(device):
    """Must reject ROW_MAJOR_LAYOUT input tensor."""
    # ROW_MAJOR requires shape not necessarily tile-aligned, but we keep it simple
    torch_tensor = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    t = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises((ValueError, RuntimeError)):
        softmax(t)


# ---------------------------------------------------------------------------
# Wrong rank
# ---------------------------------------------------------------------------


@pytest.mark.validation
def test_rejects_1d_tensor(device):
    """Must reject 1D input tensor (rank < 2)."""
    torch_tensor = torch.randn(32, dtype=torch.bfloat16)
    t = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises((ValueError, RuntimeError)):
        softmax(t)


# ---------------------------------------------------------------------------
# Invalid dim
# ---------------------------------------------------------------------------


@pytest.mark.validation
def test_rejects_dim_0(device):
    """Must reject dim=0 (only -1 and -2 are supported)."""
    t = _make_tensor(device)
    with pytest.raises((ValueError, RuntimeError)):
        softmax(t, dim=0)


@pytest.mark.validation
def test_rejects_dim_1(device):
    """Must reject dim=1 (only -1 and -2 are supported)."""
    t = _make_tensor(device)
    with pytest.raises((ValueError, RuntimeError)):
        softmax(t, dim=1)


@pytest.mark.validation
def test_rejects_dim_positive_3(device):
    """Must reject dim=3 (only -1 and -2 are supported, not positive equivalents)."""
    t = _make_tensor(device)
    with pytest.raises((ValueError, RuntimeError)):
        softmax(t, dim=3)


@pytest.mark.validation
def test_rejects_dim_out_of_range(device):
    """Must reject dim=-5 (completely out of range)."""
    t = _make_tensor(device)
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        softmax(t, dim=-5)
