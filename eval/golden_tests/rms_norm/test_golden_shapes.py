# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shape coverage tests for rms_norm. ~66 shapes × 2 layouts."""

import pytest
import torch
import ttnn

from eval.golden_tests.rms_norm.helpers import pytorch_rms_norm, to_ttnn, check_output
from ttnn.operations.rms_norm import rms_norm


# ---------------------------------------------------------------------------
# Shape definitions (all tile-aligned: H, W divisible by 32)
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

LAYOUTS = [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT]


def _shape_id(shape):
    return f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}"


def _layout_id(layout):
    return "row_major" if layout == ttnn.ROW_MAJOR_LAYOUT else "tile"


# ---------------------------------------------------------------------------
# Minimal (quick smoke check)
# ---------------------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.parametrize("layout", LAYOUTS, ids=[_layout_id(l) for l in LAYOUTS])
@pytest.mark.parametrize("shape", MINIMAL_SHAPES, ids=[_shape_id(s) for s in MINIMAL_SHAPES])
def test_rms_norm_minimal(device, shape, layout):
    torch.manual_seed(0)
    N, C, H, W = shape
    x_torch = torch.randn(N, C, H, W, dtype=torch.bfloat16)
    gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_rms_norm(x_torch, gamma=gamma_torch, epsilon=1e-6)

    x_tt = to_ttnn(x_torch, device, dtype=ttnn.bfloat16, layout=layout)
    gamma_tt = to_ttnn(gamma_torch, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = rms_norm(x_tt, gamma=gamma_tt, epsilon=1e-6)
    check_output(output, expected, shape, expected_dtype=ttnn.bfloat16, expected_layout=layout)


# ---------------------------------------------------------------------------
# Standard shape coverage
# ---------------------------------------------------------------------------

STANDARD_SHAPES = (
    WIDTH_SCALING_SHAPES
    + HEIGHT_SCALING_SHAPES
    + SQUARE_SHAPES
    + WIDE_SHAPES
    + TALL_SHAPES
    + BATCH_SHAPES
    + CHANNEL_SHAPES
    + BATCH_CHANNEL_SHAPES
)


@pytest.mark.standard
@pytest.mark.parametrize("layout", LAYOUTS, ids=[_layout_id(l) for l in LAYOUTS])
@pytest.mark.parametrize("shape", STANDARD_SHAPES, ids=[_shape_id(s) for s in STANDARD_SHAPES])
def test_rms_norm_standard(device, shape, layout):
    torch.manual_seed(0)
    N, C, H, W = shape
    x_torch = torch.randn(N, C, H, W, dtype=torch.bfloat16)
    gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_rms_norm(x_torch, gamma=gamma_torch, epsilon=1e-6)

    x_tt = to_ttnn(x_torch, device, dtype=ttnn.bfloat16, layout=layout)
    gamma_tt = to_ttnn(gamma_torch, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = rms_norm(x_tt, gamma=gamma_tt, epsilon=1e-6)
    check_output(output, expected, shape, expected_dtype=ttnn.bfloat16, expected_layout=layout)


# ---------------------------------------------------------------------------
# Large shapes (stress memory and multi-core)
# ---------------------------------------------------------------------------


@pytest.mark.large
@pytest.mark.parametrize("layout", LAYOUTS, ids=[_layout_id(l) for l in LAYOUTS])
@pytest.mark.parametrize("shape", LARGE_SHAPES, ids=[_shape_id(s) for s in LARGE_SHAPES])
def test_rms_norm_large(device, shape, layout):
    torch.manual_seed(0)
    N, C, H, W = shape
    x_torch = torch.randn(N, C, H, W, dtype=torch.bfloat16)
    gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_rms_norm(x_torch, gamma=gamma_torch, epsilon=1e-6)

    x_tt = to_ttnn(x_torch, device, dtype=ttnn.bfloat16, layout=layout)
    gamma_tt = to_ttnn(gamma_torch, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = rms_norm(x_tt, gamma=gamma_tt, epsilon=1e-6)
    check_output(output, expected, shape, expected_dtype=ttnn.bfloat16, expected_layout=layout)
