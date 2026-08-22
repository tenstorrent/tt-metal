# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end fp32 and bf16 ULP regression tests for ttnn.erfc.

Acceptance criteria from #54053:
  - fp32 output within a small ULP bound across the full [-5, 5] domain
  - Regression test that exercises fp32 dtype and the [2.5, 5] region
  - BF16 accuracy (currently 118 ULP) not regressed
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import (
    assert_with_ulp,
    flush_subnormal_values_to_zero,
    generate_all_bfloat16_bitpatterns,
)

pytestmark = pytest.mark.use_module_device


# ---------------------------------------------------------------------------
# fp32 ULP test over [-5, 5]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_erfc_fp32(device, h, w):
    """Uniform fp32 sweep on [-5, 5].  Acceptance threshold: 4 ULP."""
    torch.manual_seed(0)

    torch_input_tensor = torch.empty((h, w), dtype=torch.float32).uniform_(-5.0, 5.0)
    torch_output_tensor = torch.erfc(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.to_torch(ttnn.erfc(input_tensor))

    assert_with_ulp(torch_output_tensor, output_tensor, 4)


# ---------------------------------------------------------------------------
# fp32 dense bf16-bitpattern sweep
# ---------------------------------------------------------------------------


def test_erfc_fp32_all_bfloat16_bitpatterns(device):
    """Dense sweep over every bf16 bit-pattern in fp32 mode.  Threshold: 4 ULP."""
    all_bf16_values = generate_all_bfloat16_bitpatterns(torch.float32)
    x_torch = flush_subnormal_values_to_zero(all_bf16_values)

    x_tt = ttnn.from_torch(
        x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )

    y_tt = ttnn.erfc(x_tt)
    y_torch = torch.erfc(x_torch)

    tt_out = ttnn.to_torch(y_tt)
    y_torch = flush_subnormal_values_to_zero(y_torch)

    assert_with_ulp(y_torch, tt_out, 4, allow_nonfinite=True)


# ---------------------------------------------------------------------------
# bf16 regression — ensure no accuracy regression from fp32 changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_erfc_bf16_no_regression(device, h, w):
    """bf16 sweep on [-5, 5].  Threshold: 120 ULP (current is 118, leave margin)."""
    torch.manual_seed(0)

    torch_input_tensor = torch.empty((h, w), dtype=torch.bfloat16).uniform_(-5.0, 5.0)
    golden_function = ttnn.get_golden_function(ttnn.erfc)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.to_torch(ttnn.erfc(input_tensor))

    assert_with_ulp(torch_output_tensor, output_tensor, 120)


# ---------------------------------------------------------------------------
# Targeted tail test: [2.5, 5.0] — the catastrophic region
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_erfc_tail_fp32(device, h, w):
    """Dense fp32 sweep on [2.5, 5.0] — the region with 9M ULP in the old kernel."""
    torch.manual_seed(42)

    torch_input_tensor = torch.empty((h, w), dtype=torch.float32).uniform_(2.5, 5.0)
    torch_output_tensor = torch.erfc(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.to_torch(ttnn.erfc(input_tensor))

    assert_with_ulp(torch_output_tensor, output_tensor, 4)


# ---------------------------------------------------------------------------
# Negative side: [-5, -2.5] — erfc(-x) = 2 - erfc(x) path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_erfc_negative_tail_fp32(device, h, w):
    """fp32 sweep on [-5, -2.5] — validates the erfc(-x) = 2 - erfc(x) symmetry."""
    torch.manual_seed(42)

    torch_input_tensor = torch.empty((h, w), dtype=torch.float32).uniform_(-5.0, -2.5)
    torch_output_tensor = torch.erfc(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.to_torch(ttnn.erfc(input_tensor))

    assert_with_ulp(torch_output_tensor, output_tensor, 4)
