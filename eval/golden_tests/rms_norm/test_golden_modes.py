# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mode variation tests for rms_norm: optional params, scalar ranges, dtypes, distributions."""

import pytest
import torch
import ttnn

from eval.golden_tests.rms_norm.helpers import pytorch_rms_norm, to_ttnn, check_output
from ttnn.operations.rms_norm import rms_norm


# Representative subset of shapes for mode tests
MODE_SHAPES = [
    (1, 1, 32, 32),  # minimal
    (1, 1, 128, 256),  # medium
    (1, 1, 64, 1024),  # non-square wide
    (2, 3, 64, 128),  # multi-batch-channel
    (1, 1, 512, 512),  # large square
]

LAYOUTS = [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT]

# Standard tolerances
BF16_RTOL = 0.01
BF16_ATOL = 0.05
FP32_RTOL = 0.001
FP32_ATOL = 0.01


def _shape_id(shape):
    return f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}"


def _layout_id(layout):
    return "row_major" if layout == ttnn.ROW_MAJOR_LAYOUT else "tile"


# ---------------------------------------------------------------------------
# 1. Gamma omitted (gamma=None)
# ---------------------------------------------------------------------------


@pytest.mark.standard
@pytest.mark.parametrize("layout", LAYOUTS, ids=[_layout_id(l) for l in LAYOUTS])
@pytest.mark.parametrize("shape", MODE_SHAPES, ids=[_shape_id(s) for s in MODE_SHAPES])
def test_rms_norm_no_gamma(device, shape, layout):
    """RMSNorm without gamma — pure normalization only."""
    torch.manual_seed(0)
    N, C, H, W = shape
    x_torch = torch.randn(N, C, H, W, dtype=torch.bfloat16)

    expected = pytorch_rms_norm(x_torch, gamma=None, epsilon=1e-6)

    x_tt = to_ttnn(x_torch, device, dtype=ttnn.bfloat16, layout=layout)
    output = rms_norm(x_tt, epsilon=1e-6)

    check_output(output, expected, shape, BF16_RTOL, BF16_ATOL, expected_dtype=ttnn.bfloat16, expected_layout=layout)


# ---------------------------------------------------------------------------
# 2. Identity gamma (all ones — should equal no-gamma result)
# ---------------------------------------------------------------------------


@pytest.mark.standard
@pytest.mark.parametrize("layout", LAYOUTS, ids=[_layout_id(l) for l in LAYOUTS])
@pytest.mark.parametrize("shape", MODE_SHAPES, ids=[_shape_id(s) for s in MODE_SHAPES])
def test_rms_norm_identity_gamma(device, shape, layout):
    """RMSNorm with gamma=ones — result should match no-gamma case."""
    torch.manual_seed(0)
    N, C, H, W = shape
    x_torch = torch.randn(N, C, H, W, dtype=torch.bfloat16)
    gamma_torch = torch.ones(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_rms_norm(x_torch, gamma=gamma_torch, epsilon=1e-6)

    x_tt = to_ttnn(x_torch, device, dtype=ttnn.bfloat16, layout=layout)
    gamma_tt = to_ttnn(gamma_torch, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = rms_norm(x_tt, gamma=gamma_tt, epsilon=1e-6)
    check_output(output, expected, shape, BF16_RTOL, BF16_ATOL, expected_dtype=ttnn.bfloat16, expected_layout=layout)


# ---------------------------------------------------------------------------
# 3. Epsilon variations
# ---------------------------------------------------------------------------

EPSILON_VALUES = [1e-8, 1e-6, 1e-5, 1e-3, 1e-2]


@pytest.mark.standard
@pytest.mark.parametrize("epsilon", EPSILON_VALUES, ids=[f"eps_{e}" for e in EPSILON_VALUES])
@pytest.mark.parametrize("shape", MODE_SHAPES[:3], ids=[_shape_id(s) for s in MODE_SHAPES[:3]])
def test_rms_norm_epsilon_variations(device, shape, epsilon):
    """RMSNorm with various epsilon values."""
    torch.manual_seed(0)
    N, C, H, W = shape
    x_torch = torch.randn(N, C, H, W, dtype=torch.bfloat16)
    gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_rms_norm(x_torch, gamma=gamma_torch, epsilon=epsilon)

    x_tt = to_ttnn(x_torch, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    gamma_tt = to_ttnn(gamma_torch, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = rms_norm(x_tt, gamma=gamma_tt, epsilon=epsilon)
    check_output(
        output, expected, shape, BF16_RTOL, BF16_ATOL, expected_dtype=ttnn.bfloat16, expected_layout=ttnn.TILE_LAYOUT
    )


# ---------------------------------------------------------------------------
# 4. Data distribution variations
# ---------------------------------------------------------------------------


@pytest.mark.standard
@pytest.mark.parametrize("layout", LAYOUTS, ids=[_layout_id(l) for l in LAYOUTS])
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 64, 128), (2, 1, 128, 256)],
    ids=["1x1x64x128", "2x1x128x256"],
)
class TestRmsNormDistributions:
    """Test various input data distributions."""

    def test_uniform(self, device, shape, layout):
        """Uniform input in [0, 1]."""
        torch.manual_seed(42)
        N, C, H, W = shape
        x_torch = torch.rand(N, C, H, W, dtype=torch.bfloat16)
        gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

        expected = pytorch_rms_norm(x_torch, gamma=gamma_torch, epsilon=1e-6)
        x_tt = to_ttnn(x_torch, device, dtype=ttnn.bfloat16, layout=layout)
        gamma_tt = to_ttnn(gamma_torch, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        output = rms_norm(x_tt, gamma=gamma_tt, epsilon=1e-6)
        check_output(
            output, expected, shape, BF16_RTOL, BF16_ATOL, expected_dtype=ttnn.bfloat16, expected_layout=layout
        )

    def test_small_magnitude(self, device, shape, layout):
        """Near-zero inputs (×0.01) — slightly relaxed tolerance."""
        torch.manual_seed(42)
        N, C, H, W = shape
        x_torch = torch.randn(N, C, H, W, dtype=torch.bfloat16) * 0.01
        gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

        expected = pytorch_rms_norm(x_torch, gamma=gamma_torch, epsilon=1e-6)
        x_tt = to_ttnn(x_torch, device, dtype=ttnn.bfloat16, layout=layout)
        gamma_tt = to_ttnn(gamma_torch, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        output = rms_norm(x_tt, gamma=gamma_tt, epsilon=1e-6)
        check_output(
            output,
            expected,
            shape,
            BF16_RTOL * 2.0,
            BF16_ATOL * 2.0,
            expected_dtype=ttnn.bfloat16,
            expected_layout=layout,
        )

    def test_large_magnitude(self, device, shape, layout):
        """Large inputs (×10.0)."""
        torch.manual_seed(42)
        N, C, H, W = shape
        x_torch = torch.randn(N, C, H, W, dtype=torch.bfloat16) * 10.0
        gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

        expected = pytorch_rms_norm(x_torch, gamma=gamma_torch, epsilon=1e-6)
        x_tt = to_ttnn(x_torch, device, dtype=ttnn.bfloat16, layout=layout)
        gamma_tt = to_ttnn(gamma_torch, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        output = rms_norm(x_tt, gamma=gamma_tt, epsilon=1e-6)
        check_output(
            output, expected, shape, BF16_RTOL, BF16_ATOL, expected_dtype=ttnn.bfloat16, expected_layout=layout
        )

    def test_positive_only(self, device, shape, layout):
        """Positive-only inputs (rand + 0.5)."""
        torch.manual_seed(42)
        N, C, H, W = shape
        x_torch = torch.rand(N, C, H, W, dtype=torch.bfloat16) + 0.5
        gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

        expected = pytorch_rms_norm(x_torch, gamma=gamma_torch, epsilon=1e-6)
        x_tt = to_ttnn(x_torch, device, dtype=ttnn.bfloat16, layout=layout)
        gamma_tt = to_ttnn(gamma_torch, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        output = rms_norm(x_tt, gamma=gamma_tt, epsilon=1e-6)
        check_output(
            output, expected, shape, BF16_RTOL, BF16_ATOL, expected_dtype=ttnn.bfloat16, expected_layout=layout
        )

    def test_negative_only(self, device, shape, layout):
        """Negative-only inputs (-(rand + 0.5))."""
        torch.manual_seed(42)
        N, C, H, W = shape
        x_torch = -(torch.rand(N, C, H, W, dtype=torch.bfloat16) + 0.5)
        gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

        expected = pytorch_rms_norm(x_torch, gamma=gamma_torch, epsilon=1e-6)
        x_tt = to_ttnn(x_torch, device, dtype=ttnn.bfloat16, layout=layout)
        gamma_tt = to_ttnn(gamma_torch, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        output = rms_norm(x_tt, gamma=gamma_tt, epsilon=1e-6)
        check_output(
            output, expected, shape, BF16_RTOL, BF16_ATOL, expected_dtype=ttnn.bfloat16, expected_layout=layout
        )


# ---------------------------------------------------------------------------
# 5. Float32 dtype
# ---------------------------------------------------------------------------


@pytest.mark.standard
@pytest.mark.parametrize("layout", LAYOUTS, ids=[_layout_id(l) for l in LAYOUTS])
@pytest.mark.parametrize("shape", MODE_SHAPES, ids=[_shape_id(s) for s in MODE_SHAPES])
def test_rms_norm_float32(device, shape, layout):
    """RMSNorm with float32 input — fp32 accumulation should be used."""
    torch.manual_seed(0)
    N, C, H, W = shape
    x_torch = torch.randn(N, C, H, W, dtype=torch.float32)
    gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.float32)

    expected = pytorch_rms_norm(x_torch, gamma=gamma_torch, epsilon=1e-6)

    x_tt = to_ttnn(x_torch, device, dtype=ttnn.float32, layout=layout)
    gamma_tt = to_ttnn(gamma_torch, device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = rms_norm(x_tt, gamma=gamma_tt, epsilon=1e-6)
    check_output(output, expected, shape, FP32_RTOL, FP32_ATOL, expected_dtype=ttnn.float32, expected_layout=layout)


@pytest.mark.standard
@pytest.mark.parametrize("shape", MODE_SHAPES[:3], ids=[_shape_id(s) for s in MODE_SHAPES[:3]])
def test_rms_norm_float32_no_gamma(device, shape):
    """RMSNorm with float32 input and no gamma."""
    torch.manual_seed(0)
    N, C, H, W = shape
    x_torch = torch.randn(N, C, H, W, dtype=torch.float32)

    expected = pytorch_rms_norm(x_torch, gamma=None, epsilon=1e-6)

    x_tt = to_ttnn(x_torch, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    output = rms_norm(x_tt, epsilon=1e-6)

    check_output(
        output, expected, shape, FP32_RTOL, FP32_ATOL, expected_dtype=ttnn.float32, expected_layout=ttnn.TILE_LAYOUT
    )
