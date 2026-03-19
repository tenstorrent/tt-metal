# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Golden tests: Input validation.

Verifies the operation correctly rejects invalid inputs.
These tests do NOT run on device — they test Python-side validation.
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


def _make_tensor(device, shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
    """Helper to create a tensor on device with given properties."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    t = torch.randn(shape, dtype=torch_dtype)
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_rejects_float32_input(device):
    """Must reject non-bfloat16 input."""
    t = _make_tensor(device, (1, 1, 32, 32), dtype=ttnn.float32)
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(t)


def test_rejects_tile_layout(device):
    """Must reject tile-layout input."""
    t = _make_tensor(device, (1, 1, 32, 32), layout=ttnn.TILE_LAYOUT)
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(t)


def test_rejects_wrong_gamma_shape(device):
    """Gamma shape must be (1, 1, 1, W)."""
    inp = _make_tensor(device, (1, 1, 32, 64))
    # Wrong width
    bad_gamma = _make_tensor(device, (1, 1, 1, 32))
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(inp, bad_gamma)


def test_rejects_wrong_beta_shape(device):
    """Beta shape must be (1, 1, 1, W)."""
    inp = _make_tensor(device, (1, 1, 32, 64))
    gamma = _make_tensor(device, (1, 1, 1, 64))
    bad_beta = _make_tensor(device, (1, 1, 1, 32))
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(inp, gamma, bad_beta)
