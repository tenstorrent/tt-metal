# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Per-element accuracy tests for ttnn.erfinv with Newton-Raphson refinement.

The old kernel used only the Winitzki (2008) closed-form approximation with
~0.2% peak relative error near the domain edges.  The NR refinement step
should bring this to fp32 precision.

Resolves #54049.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_erfinv_fp32_domain(device, h, w):
    """fp32 sweep on (-0.999, 0.999) — the valid domain of erfinv."""
    torch.manual_seed(0)

    torch_input = torch.empty((h, w), dtype=torch.float32).uniform_(-0.999, 0.999)
    torch_expected = torch.erfinv(torch_input)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.to_torch(ttnn.erfinv(input_tensor))

    # Per-element relative error check: NR refinement should bring peak
    # relative error well below the old 0.2% (1.96e-3).
    # Target: < 0.001% (1e-5 relative), allowing margin for hardware FMA.
    finite_mask = torch.isfinite(torch_expected) & torch.isfinite(output_tensor)
    rel_err = torch.abs(output_tensor[finite_mask] - torch_expected[finite_mask]) / (
        torch.abs(torch_expected[finite_mask]) + 1e-10
    )
    max_rel_err = rel_err.max().item()
    assert max_rel_err < 1e-4, f"Max relative error {max_rel_err:.2e} exceeds 1e-4"


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_erfinv_fp32_near_edges(device, h, w):
    """Test near the domain edges where the old kernel was worst."""
    torch.manual_seed(42)

    # Sample heavily near +-1 where the old kernel had 0.2% error
    torch_input = torch.empty((h, w), dtype=torch.float32)
    # Half near +1, half near -1
    torch_input[:h // 2] = torch.empty((h // 2, w)).uniform_(0.99, 0.9999)
    torch_input[h // 2:] = torch.empty((h - h // 2, w)).uniform_(-0.9999, -0.99)
    torch_expected = torch.erfinv(torch_input)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.to_torch(ttnn.erfinv(input_tensor))

    finite_mask = torch.isfinite(torch_expected) & torch.isfinite(output_tensor)
    rel_err = torch.abs(output_tensor[finite_mask] - torch_expected[finite_mask]) / (
        torch.abs(torch_expected[finite_mask]) + 1e-10
    )
    max_rel_err = rel_err.max().item()
    assert max_rel_err < 1e-4, f"Near-edge max relative error {max_rel_err:.2e} exceeds 1e-4"


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_erfinv_bf16_no_regression(device, h, w):
    """Verify bf16 accuracy is not regressed by the NR addition."""
    torch.manual_seed(0)

    torch_input = torch.empty((h, w), dtype=torch.bfloat16).uniform_(-0.999, 0.999)
    golden_function = ttnn.get_golden_function(ttnn.erfinv)
    torch_expected = golden_function(torch_input)

    input_tensor = ttnn.from_torch(
        torch_input, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.to_torch(ttnn.erfinv(input_tensor))

    assert_with_pcc(torch_expected, output_tensor, 0.999)


def test_erfinv_specific_values(device):
    """Test the specific failing values from the issue."""
    test_values = torch.tensor([[0.999, 0.9999, -0.999, -0.9999, 0.5, -0.5, 0.0]],
                               dtype=torch.float32)
    torch_expected = torch.erfinv(test_values)

    input_tensor = ttnn.from_torch(
        test_values, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.to_torch(ttnn.erfinv(input_tensor))

    # The old kernel had 1.91e-3 relative error at x=0.999.
    # After NR refinement this should be well below 1e-4.
    finite_mask = torch.isfinite(torch_expected)
    rel_err = torch.abs(output_tensor[finite_mask] - torch_expected[finite_mask]) / (
        torch.abs(torch_expected[finite_mask]) + 1e-10
    )
    max_rel_err = rel_err.max().item()
    assert max_rel_err < 1e-4, f"Specific values max relative error {max_rel_err:.2e} exceeds 1e-4"
