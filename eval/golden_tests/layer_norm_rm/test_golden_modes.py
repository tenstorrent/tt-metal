# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Golden tests: Operation modes and parameter variations.

Tests different ways to call layer_norm_rm:
- Without gamma/beta (pure normalization)
- With gamma only (scale, no shift) — depends on agent supporting this
- With identity gamma/beta (should match pure normalization)
- Different epsilon values
- Different data distributions (uniform, near-zero, large values)
"""

import pytest
import torch

from ttnn.operations.layer_norm_rm import layer_norm_rm
from .helpers import pytorch_layer_norm, to_ttnn, check_output


RTOL = 0.02
ATOL = 0.1

# Representative subset of shapes for mode tests
MODE_SHAPES = [
    pytest.param((1, 1, 32, 32), id="32x32"),
    pytest.param((1, 1, 64, 128), id="64x128"),
    pytest.param((1, 1, 128, 256), id="128x256"),
    pytest.param((4, 2, 64, 64), id="b4c2_64x64"),
    pytest.param((1, 1, 256, 512), id="256x512"),
]


# ---------------------------------------------------------------------------
# No gamma/beta (pure normalization, should produce zero mean unit variance)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", MODE_SHAPES)
def test_no_gamma_beta(device, shape):
    """Pure normalization without affine transform."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    expected = pytorch_layer_norm(torch_input, gamma=None, beta=None)

    ttnn_input = to_ttnn(torch_input, device)
    ttnn_output = layer_norm_rm(ttnn_input)

    check_output(ttnn_output, expected, shape, RTOL, ATOL)


# ---------------------------------------------------------------------------
# Identity gamma (ones) and zero beta — should match pure normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", MODE_SHAPES)
def test_identity_affine(device, shape):
    """gamma=1, beta=0 should produce same result as no gamma/beta."""
    torch.manual_seed(42)
    W = shape[-1]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.ones(1, 1, 1, W, dtype=torch.bfloat16)
    beta = torch.zeros(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_layer_norm(torch_input, gamma, beta)

    ttnn_input = to_ttnn(torch_input, device)
    ttnn_gamma = to_ttnn(gamma, device)
    ttnn_beta = to_ttnn(beta, device)

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta)
    check_output(ttnn_output, expected, shape, RTOL, ATOL)


# ---------------------------------------------------------------------------
# Epsilon variations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "epsilon",
    [
        pytest.param(1e-5, id="eps_1e-5_default"),
        pytest.param(1e-6, id="eps_1e-6_small"),
        pytest.param(1e-3, id="eps_1e-3_large"),
        pytest.param(1e-2, id="eps_1e-2_very_large"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 128), id="32x128"),
        pytest.param((1, 1, 128, 128), id="128x128"),
        pytest.param((2, 1, 64, 256), id="b2_64x256"),
    ],
)
def test_epsilon_values(device, shape, epsilon):
    """Different epsilon values for numerical stability."""
    torch.manual_seed(42)
    W = shape[-1]
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    beta = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_layer_norm(torch_input, gamma, beta, epsilon=epsilon)

    ttnn_input = to_ttnn(torch_input, device)
    ttnn_gamma = to_ttnn(gamma, device)
    ttnn_beta = to_ttnn(beta, device)

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta, epsilon=epsilon)
    check_output(ttnn_output, expected, shape, RTOL, ATOL)


# ---------------------------------------------------------------------------
# Data distribution variations
# ---------------------------------------------------------------------------


DISTRIBUTION_SHAPES = [
    pytest.param((1, 1, 64, 128), id="64x128"),
    pytest.param((2, 1, 128, 256), id="b2_128x256"),
]


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
def test_uniform_input(device, shape):
    """Uniformly distributed input [0, 1]."""
    torch.manual_seed(42)
    W = shape[-1]
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    gamma = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    beta = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_layer_norm(torch_input, gamma, beta)

    ttnn_input = to_ttnn(torch_input, device)
    ttnn_gamma = to_ttnn(gamma, device)
    ttnn_beta = to_ttnn(beta, device)

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta)
    check_output(ttnn_output, expected, shape, RTOL, ATOL)


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
def test_small_magnitude_input(device, shape):
    """Small magnitude inputs (tests numerical stability near zero)."""
    torch.manual_seed(42)
    W = shape[-1]
    torch_input = (torch.randn(shape) * 0.01).to(torch.bfloat16)
    gamma = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    beta = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_layer_norm(torch_input, gamma, beta)

    ttnn_input = to_ttnn(torch_input, device)
    ttnn_gamma = to_ttnn(gamma, device)
    ttnn_beta = to_ttnn(beta, device)

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta)
    # Slightly relaxed for near-zero inputs where relative error can be larger
    check_output(ttnn_output, expected, shape, rtol=0.05, atol=0.15)


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
def test_large_magnitude_input(device, shape):
    """Larger magnitude inputs (tests range handling)."""
    torch.manual_seed(42)
    W = shape[-1]
    torch_input = (torch.randn(shape) * 10.0).to(torch.bfloat16)
    gamma = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    beta = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_layer_norm(torch_input, gamma, beta)

    ttnn_input = to_ttnn(torch_input, device)
    ttnn_gamma = to_ttnn(gamma, device)
    ttnn_beta = to_ttnn(beta, device)

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta)
    check_output(ttnn_output, expected, shape, RTOL, ATOL)


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
def test_positive_only_input(device, shape):
    """All-positive input (no negative values)."""
    torch.manual_seed(42)
    W = shape[-1]
    torch_input = torch.rand(shape, dtype=torch.bfloat16) + 0.5
    gamma = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    beta = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_layer_norm(torch_input, gamma, beta)

    ttnn_input = to_ttnn(torch_input, device)
    ttnn_gamma = to_ttnn(gamma, device)
    ttnn_beta = to_ttnn(beta, device)

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta)
    check_output(ttnn_output, expected, shape, RTOL, ATOL)


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
def test_negative_only_input(device, shape):
    """All-negative input."""
    torch.manual_seed(42)
    W = shape[-1]
    torch_input = -(torch.rand(shape, dtype=torch.bfloat16) + 0.5)
    gamma = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    beta = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_layer_norm(torch_input, gamma, beta)

    ttnn_input = to_ttnn(torch_input, device)
    ttnn_gamma = to_ttnn(gamma, device)
    ttnn_beta = to_ttnn(beta, device)

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta)
    check_output(ttnn_output, expected, shape, RTOL, ATOL)
