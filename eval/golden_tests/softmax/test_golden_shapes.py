# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cross-product shape × dim × numeric_stable tests for softmax operation.

All combinations of ~70 shapes, dim in {-1, -2}, and numeric_stable in
{True, False} are tested via pytest parametrize cross product.
All shapes are tile-aligned (H, W divisible by 32).
"""

import pytest
import torch

from eval.golden_tests.softmax.helpers import check_output, pytorch_softmax, to_ttnn
from ttnn.operations.softmax import softmax

# ---------------------------------------------------------------------------
# Shape definitions
# ---------------------------------------------------------------------------

MINIMAL_SHAPES = [(1, 1, 32, 32)]

WIDTH_SCALING_SHAPES = [(1, 1, 32, w) for w in [64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096]]

HEIGHT_SCALING_SHAPES = [(1, 1, h, 32) for h in [64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096]]

SQUARE_SHAPES = [(1, 1, s, s) for s in [64, 128, 256, 512, 1024]]

WIDE_SHAPES = [
    (1, 1, 32, 512),
    (1, 1, 32, 1024),
    (1, 1, 64, 512),
    (1, 1, 64, 1024),
    (1, 1, 64, 2048),
    (1, 1, 128, 1024),
    (1, 1, 128, 2048),
]

TALL_SHAPES = [
    (1, 1, 512, 32),
    (1, 1, 1024, 32),
    (1, 1, 512, 64),
    (1, 1, 1024, 64),
    (1, 1, 2048, 64),
    (1, 1, 1024, 128),
    (1, 1, 2048, 128),
]

BATCH_SHAPES = [
    (2, 1, 32, 32),
    (4, 1, 32, 32),
    (8, 1, 32, 32),
    (2, 1, 64, 128),
    (4, 1, 128, 256),
    (2, 1, 256, 512),
    (8, 1, 64, 64),
]

CHANNEL_SHAPES = [
    (1, 2, 32, 32),
    (1, 4, 32, 32),
    (1, 8, 64, 64),
    (1, 3, 128, 128),
    (1, 16, 32, 64),
]

BATCH_CHANNEL_SHAPES = [
    (2, 3, 32, 32),
    (4, 2, 64, 64),
    (2, 4, 128, 128),
    (3, 3, 64, 128),
    (2, 2, 256, 256),
    (8, 4, 32, 64),
]

LARGE_SHAPES = [
    (1, 1, 2048, 2048),
    (1, 1, 4096, 128),
    (1, 1, 128, 4096),
    (2, 1, 1024, 1024),
    (1, 1, 4096, 256),
    (4, 1, 512, 512),
]

ALL_SHAPES = (
    MINIMAL_SHAPES
    + WIDTH_SCALING_SHAPES
    + HEIGHT_SCALING_SHAPES
    + SQUARE_SHAPES
    + WIDE_SHAPES
    + TALL_SHAPES
    + BATCH_SHAPES
    + CHANNEL_SHAPES
    + BATCH_CHANNEL_SHAPES
    + LARGE_SHAPES
)

DIMS = [-1, -2]

NUMERIC_STABLE_OPTIONS = [True, False]

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

RTOL_STABLE = 0.02
ATOL_STABLE = 0.1
RTOL_UNSTABLE = 0.05
ATOL_UNSTABLE = 0.2


# ---------------------------------------------------------------------------
# Cross-product test: shape × dim × numeric_stable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("numeric_stable", NUMERIC_STABLE_OPTIONS, ids=["stable", "unstable"])
@pytest.mark.parametrize("dim", DIMS, ids=[f"dim{d}" for d in DIMS])
@pytest.mark.parametrize(
    "shape",
    ALL_SHAPES,
    ids=[f"{s[0]}x{s[1]}x{s[2]}x{s[3]}" for s in ALL_SHAPES],
)
def test_softmax(shape, dim, numeric_stable, device):
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    expected = pytorch_softmax(torch_input, dim=dim, numeric_stable=numeric_stable)

    ttnn_input = to_ttnn(torch_input, device)
    ttnn_output = softmax(ttnn_input, dim=dim, numeric_stable=numeric_stable)

    rtol = RTOL_STABLE if numeric_stable else RTOL_UNSTABLE
    atol = ATOL_STABLE if numeric_stable else ATOL_UNSTABLE
    check_output(ttnn_output, expected, shape, rtol, atol)
