# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Input validation tests for rms_norm (Python-side, pre-device)."""

import pytest
import torch
import ttnn

from eval.golden_tests.rms_norm.helpers import to_ttnn
from ttnn.operations.rms_norm import rms_norm


# ---------------------------------------------------------------------------
# Wrong rank: input tensor rank < 2
# ---------------------------------------------------------------------------


@pytest.mark.validation
def test_rejects_rank_1_input(device):
    """Must reject 1D input tensor (rank < 2)."""
    x_torch = torch.randn(64, dtype=torch.bfloat16)
    x_tt = to_ttnn(x_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0), device)
    # Create a 1D-equivalent tensor and attempt the op
    x_1d = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises((ValueError, RuntimeError)):
        rms_norm(x_1d)


@pytest.mark.validation
def test_rejects_rank_0_input(device):
    """Must reject scalar input tensor (rank 0)."""
    x_torch = torch.tensor(1.0, dtype=torch.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        x_tt = ttnn.from_torch(
            x_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rms_norm(x_tt)


# ---------------------------------------------------------------------------
# Wrong gamma shape: gamma width != input last dimension
# ---------------------------------------------------------------------------


@pytest.mark.validation
def test_rejects_gamma_wrong_width(device):
    """Must reject gamma whose last dimension != input's last dimension."""
    x_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    # gamma width 128 != input width 64
    gamma_torch = torch.randn(1, 1, 1, 128, dtype=torch.bfloat16)

    x_tt = to_ttnn(x_torch, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    gamma_tt = to_ttnn(gamma_torch, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises((ValueError, RuntimeError)):
        rms_norm(x_tt, gamma=gamma_tt)


@pytest.mark.validation
def test_rejects_gamma_wrong_width_row_major(device):
    """Must reject gamma shape mismatch in ROW_MAJOR layout."""
    x_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    gamma_torch = torch.randn(1, 1, 1, 32, dtype=torch.bfloat16)

    x_tt = to_ttnn(x_torch, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    gamma_tt = to_ttnn(gamma_torch, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises((ValueError, RuntimeError)):
        rms_norm(x_tt, gamma=gamma_tt)
