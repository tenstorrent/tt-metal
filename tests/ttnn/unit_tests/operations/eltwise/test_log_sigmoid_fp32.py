# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end fp32 and bf16 ULP regression tests for ttnn.log_sigmoid.

Acceptance criteria from #53787:
  - fp32 output within 4 ULP of the registered PyTorch golden on [-30, 30]
  - no discontinuity at x = +/-4
  - fp32, bf16 and both tails covered by regression tests
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
# fp32 ULP test over [-30, 30]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log_sigmoid_fp32(device, h, w):
    """Uniform fp32 sweep on [-30, 30].  Acceptance threshold: 4 ULP."""
    torch.manual_seed(0)

    torch_input_tensor = torch.empty((h, w), dtype=torch.float32).uniform_(-30.0, 30.0)
    torch_output_tensor = torch.nn.functional.logsigmoid(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.to_torch(ttnn.log_sigmoid(input_tensor))

    assert_with_ulp(torch_output_tensor, output_tensor, 4)


# ---------------------------------------------------------------------------
# fp32 dense bf16-bitpattern sweep (all representable bf16 values cast to fp32)
# ---------------------------------------------------------------------------


def test_log_sigmoid_fp32_all_bfloat16_bitpatterns(device):
    """Dense sweep over every bf16 bit-pattern in fp32 mode.  Threshold: 4 ULP."""
    all_bf16_values = generate_all_bfloat16_bitpatterns(torch.float32)
    x_torch = flush_subnormal_values_to_zero(all_bf16_values)

    x_tt = ttnn.from_torch(
        x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )

    y_tt = ttnn.log_sigmoid(x_tt)
    y_torch = torch.nn.functional.logsigmoid(x_torch)

    tt_out = ttnn.to_torch(y_tt)
    y_torch = flush_subnormal_values_to_zero(y_torch)

    assert_with_ulp(y_torch, tt_out, 4, allow_nonfinite=True)


# ---------------------------------------------------------------------------
# bf16 ULP test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log_sigmoid_bf16(device, h, w):
    """bf16 input/output sweep on [-30, 30].  Threshold: 2 ULP."""
    torch.manual_seed(0)

    torch_input_tensor = torch.empty((h, w), dtype=torch.bfloat16).uniform_(-30.0, 30.0)
    golden_function = ttnn.get_golden_function(ttnn.log_sigmoid)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.to_torch(ttnn.log_sigmoid(input_tensor))

    assert_with_ulp(torch_output_tensor, output_tensor, 2)


# ---------------------------------------------------------------------------
# Discontinuity regression at x = +/-4
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "boundary",
    [-4.0, 4.0],
    ids=["neg4", "pos4"],
)
def test_log_sigmoid_no_discontinuity_at_boundary(device, boundary):
    """Verify no discontinuity at x = +/-4 by sampling a tight neighbourhood.

    The old kernel had a range split at +/-4 that produced a 0.452% jump at x = -4.
    This test samples 1024 points in [boundary - 0.1, boundary + 0.1] and asserts
    that the maximum ULP error stays within 4 ULP.
    """
    x_torch = torch.linspace(boundary - 0.1, boundary + 0.1, 1024, dtype=torch.float32)
    # Pad to tile-aligned shape
    x_torch = x_torch.reshape(1, 1024)
    y_torch = torch.nn.functional.logsigmoid(x_torch)

    x_tt = ttnn.from_torch(
        x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    y_tt = ttnn.to_torch(ttnn.log_sigmoid(x_tt))

    assert_with_ulp(y_torch, y_tt, 4)


# ---------------------------------------------------------------------------
# Tail coverage: large negative and large positive inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "low,high,name",
    [
        (-30.0, -10.0, "negative_tail"),
        (10.0, 30.0, "positive_tail"),
    ],
)
def test_log_sigmoid_tails_fp32(device, low, high, name):
    """Verify accuracy in the tail regions where the old kernel was most inaccurate.

    Negative tail (x << 0): logsigmoid(x) ~ x, the old kernel returned raw x
    and missed the log1p(exp(x)) residual.
    Positive tail (x >> 0): logsigmoid(x) ~ -exp(-x), the old kernel used a
    one-term series with 0.91% error.
    """
    torch.manual_seed(42)

    x_torch = torch.empty((32, 64), dtype=torch.float32).uniform_(low, high)
    y_torch = torch.nn.functional.logsigmoid(x_torch)

    x_tt = ttnn.from_torch(
        x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    y_tt = ttnn.to_torch(ttnn.log_sigmoid(x_tt))

    assert_with_ulp(y_torch, y_tt, 4)
