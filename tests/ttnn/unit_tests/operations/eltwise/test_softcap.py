# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device


def torch_softcap(x, cap):
    return cap * torch.tanh(x / cap)


@pytest.mark.parametrize("cap", [50.0, 1.0, 10.0])
def test_softcap_bfloat16(device, cap):
    """Test softcap with bfloat16: target <2 ULP."""
    torch.manual_seed(0)
    x_torch = torch.randn((32, 64), dtype=torch.bfloat16) * cap * 2
    y_expected = torch_softcap(x_torch.float(), cap).to(torch.bfloat16)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.softcap(x_tt, cap=cap)
    y_actual = ttnn.to_torch(y_tt).to(torch.bfloat16)

    assert_with_ulp(y_expected, y_actual, ulp_threshold=2)


@pytest.mark.parametrize("cap", [50.0, 1.0, 10.0])
def test_softcap_fp32(device, cap):
    """Test softcap with fp32: polynomial tanh gives ~0.5% relative accuracy."""
    torch.manual_seed(0)
    x_torch = torch.randn((32, 64), dtype=torch.float32) * cap * 2
    y_expected = torch_softcap(x_torch, cap)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.softcap(x_tt, cap=cap)
    y_actual = ttnn.to_torch(y_tt)

    assert torch.allclose(y_expected, y_actual, atol=cap * 0.01, rtol=0.01)


def test_softcap_edge_cases(device):
    """Test softcap with edge cases: zero, small, large, negative."""
    cap = 50.0
    x_torch = torch.tensor(
        [0.0, 1e-6, -1e-6, 0.001, -0.001, 1.0, -1.0, 50.0, -50.0, 500.0, -500.0, 1e6, -1e6],
        dtype=torch.float32,
    )
    # Pad to tile size (32x32)
    padded = torch.zeros(32, 32, dtype=torch.float32)
    padded[0, : x_torch.numel()] = x_torch
    y_expected = torch_softcap(padded, cap)

    x_tt = ttnn.from_torch(padded, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.softcap(x_tt, cap=cap)
    y_actual = ttnn.to_torch(y_tt)

    assert torch.allclose(y_expected, y_actual, atol=cap * 0.01, rtol=0.01)


def test_softcap_saturation(device):
    """Verify softcap saturates to ±cap for very large inputs."""
    cap = 10.0
    x_torch = torch.tensor(
        [1000.0, -1000.0, 1e6, -1e6],
        dtype=torch.bfloat16,
    )
    padded = torch.zeros(32, 32, dtype=torch.bfloat16)
    padded[0, : x_torch.numel()] = x_torch
    y_expected = torch_softcap(padded.float(), cap).to(torch.bfloat16)

    x_tt = ttnn.from_torch(padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.softcap(x_tt, cap=cap)
    y_actual = ttnn.to_torch(y_tt).to(torch.bfloat16)

    assert_with_ulp(y_expected, y_actual, ulp_threshold=2)


def test_softcap_default_cap(device):
    """Test softcap with default cap=50.0."""
    torch.manual_seed(42)
    x_torch = torch.randn((32, 32), dtype=torch.bfloat16) * 100
    y_expected = torch_softcap(x_torch.float(), 50.0).to(torch.bfloat16)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.softcap(x_tt)  # default cap=50.0
    y_actual = ttnn.to_torch(y_tt).to(torch.bfloat16)

    assert_with_ulp(y_expected, y_actual, ulp_threshold=2)
