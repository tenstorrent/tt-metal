# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data distribution cross-product tests for softmax operation.

Tests representative shapes × dim × data distributions as a cross product.
Distributions vary how input tensors are generated (uniform, small, large,
positive-only, negative-only).
"""

import pytest
import torch

from eval.golden_tests.softmax.helpers import check_output, pytorch_softmax, to_ttnn
from ttnn.operations.softmax import softmax

# ---------------------------------------------------------------------------
# Representative shape subset (kept small to avoid combinatorial explosion)
# ---------------------------------------------------------------------------

REPR_SHAPES = [
    (1, 1, 32, 32),  # minimal
    (1, 1, 128, 256),  # medium
    (1, 1, 64, 512),  # non-square wide
    (2, 3, 64, 64),  # multi-batch + channel
    (1, 1, 1024, 1024),  # large
]

DIMS = [-1, -2]

# ---------------------------------------------------------------------------
# Data distributions
# ---------------------------------------------------------------------------


def _randn(shape):
    """Standard normal distribution."""
    return torch.randn(shape, dtype=torch.bfloat16)


def _uniform(shape):
    """Uniform [0, 1]."""
    return torch.rand(shape, dtype=torch.bfloat16)


def _small(shape):
    """Small magnitude (near zero, ×0.01)."""
    return torch.randn(shape, dtype=torch.bfloat16) * 0.01


def _large(shape):
    """Large magnitude (×10.0)."""
    return torch.randn(shape, dtype=torch.bfloat16) * 10.0


def _positive(shape):
    """Positive-only (rand + 0.5)."""
    return torch.rand(shape, dtype=torch.bfloat16) + 0.5


def _negative(shape):
    """Negative-only (-(rand + 0.5))."""
    return -(torch.rand(shape, dtype=torch.bfloat16) + 0.5)


DISTRIBUTIONS = [
    pytest.param(_randn, id="randn"),
    pytest.param(_uniform, id="uniform"),
    pytest.param(_small, id="small"),
    pytest.param(_large, id="large"),
    pytest.param(_positive, id="positive"),
    pytest.param(_negative, id="negative"),
]

# Tolerances: slightly relaxed for small-magnitude inputs
RTOL = 0.02
ATOL = 0.1
RTOL_SMALL = 0.03
ATOL_SMALL = 0.15


# ---------------------------------------------------------------------------
# Cross-product test: shape × dim × distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_input", DISTRIBUTIONS)
@pytest.mark.parametrize("dim", DIMS, ids=[f"dim{d}" for d in DIMS])
@pytest.mark.parametrize(
    "shape",
    REPR_SHAPES,
    ids=[f"{s[0]}x{s[1]}x{s[2]}x{s[3]}" for s in REPR_SHAPES],
)
@pytest.mark.stress
def test_softmax_distributions(shape, dim, make_input, device):
    torch_input = make_input(shape)
    expected = pytorch_softmax(torch_input, dim=dim)

    ttnn_input = to_ttnn(torch_input, device)
    ttnn_output = softmax(ttnn_input, dim=dim)

    # Relax tolerances for small-magnitude inputs
    is_small = make_input.__name__ == "_small"
    rtol = RTOL_SMALL if is_small else RTOL
    atol = ATOL_SMALL if is_small else ATOL
    check_output(ttnn_output, expected, shape, rtol, atol)
