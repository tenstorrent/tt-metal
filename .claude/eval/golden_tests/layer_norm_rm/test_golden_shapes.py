# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Golden tests: Comprehensive shape coverage for layer_norm_rm.

~70 parametrized test cases covering:
- Minimal single-tile
- Width scaling (32 to 4096)
- Height scaling (32 to 4096)
- Square shapes
- Rectangular (wide and tall)
- Multi-batch and multi-channel
- Large tensors

All shapes are tile-aligned (H and W divisible by 32).
Tolerances are strict: rtol=0.02, atol=0.1 (tighter than agent's 0.05/0.2).
"""

import pytest
import torch

from ttnn.operations.layer_norm_rm import layer_norm_rm
from .helpers import pytorch_layer_norm, to_ttnn, check_output


RTOL = 0.02
ATOL = 0.1


# ---------------------------------------------------------------------------
# Shape definitions organized by category
# ---------------------------------------------------------------------------

MINIMAL_SHAPES = [
    pytest.param((1, 1, 32, 32), id="minimal_1x1x32x32"),
]

# Width scaling: fixed H=32, vary W
WIDTH_SCALING_SHAPES = [
    pytest.param((1, 1, 32, 64), id="w64"),
    pytest.param((1, 1, 32, 96), id="w96"),
    pytest.param((1, 1, 32, 128), id="w128"),
    pytest.param((1, 1, 32, 192), id="w192"),
    pytest.param((1, 1, 32, 256), id="w256"),
    pytest.param((1, 1, 32, 384), id="w384"),
    pytest.param((1, 1, 32, 512), id="w512"),
    pytest.param((1, 1, 32, 768), id="w768"),
    pytest.param((1, 1, 32, 1024), id="w1024"),
    pytest.param((1, 1, 32, 2048), id="w2048"),
    pytest.param((1, 1, 32, 4096), id="w4096"),
]

# Height scaling: fixed W=32, vary H
HEIGHT_SCALING_SHAPES = [
    pytest.param((1, 1, 64, 32), id="h64"),
    pytest.param((1, 1, 96, 32), id="h96"),
    pytest.param((1, 1, 128, 32), id="h128"),
    pytest.param((1, 1, 192, 32), id="h192"),
    pytest.param((1, 1, 256, 32), id="h256"),
    pytest.param((1, 1, 384, 32), id="h384"),
    pytest.param((1, 1, 512, 32), id="h512"),
    pytest.param((1, 1, 768, 32), id="h768"),
    pytest.param((1, 1, 1024, 32), id="h1024"),
    pytest.param((1, 1, 2048, 32), id="h2048"),
    pytest.param((1, 1, 4096, 32), id="h4096"),
]

# Square shapes
SQUARE_SHAPES = [
    pytest.param((1, 1, 64, 64), id="sq64"),
    pytest.param((1, 1, 128, 128), id="sq128"),
    pytest.param((1, 1, 256, 256), id="sq256"),
    pytest.param((1, 1, 512, 512), id="sq512"),
    pytest.param((1, 1, 1024, 1024), id="sq1024"),
]

# Rectangular: wide (W >> H)
WIDE_SHAPES = [
    pytest.param((1, 1, 32, 512), id="wide_32x512"),
    pytest.param((1, 1, 32, 1024), id="wide_32x1024"),
    pytest.param((1, 1, 64, 512), id="wide_64x512"),
    pytest.param((1, 1, 64, 1024), id="wide_64x1024"),
    pytest.param((1, 1, 64, 2048), id="wide_64x2048"),
    pytest.param((1, 1, 128, 1024), id="wide_128x1024"),
    pytest.param((1, 1, 128, 2048), id="wide_128x2048"),
]

# Rectangular: tall (H >> W)
TALL_SHAPES = [
    pytest.param((1, 1, 512, 32), id="tall_512x32"),
    pytest.param((1, 1, 1024, 32), id="tall_1024x32"),
    pytest.param((1, 1, 512, 64), id="tall_512x64"),
    pytest.param((1, 1, 1024, 64), id="tall_1024x64"),
    pytest.param((1, 1, 2048, 64), id="tall_2048x64"),
    pytest.param((1, 1, 1024, 128), id="tall_1024x128"),
    pytest.param((1, 1, 2048, 128), id="tall_2048x128"),
]

# Multi-batch (N > 1)
BATCH_SHAPES = [
    pytest.param((2, 1, 32, 32), id="batch2_32x32"),
    pytest.param((4, 1, 32, 32), id="batch4_32x32"),
    pytest.param((8, 1, 32, 32), id="batch8_32x32"),
    pytest.param((2, 1, 64, 128), id="batch2_64x128"),
    pytest.param((4, 1, 128, 256), id="batch4_128x256"),
    pytest.param((2, 1, 256, 512), id="batch2_256x512"),
    pytest.param((8, 1, 64, 64), id="batch8_64x64"),
]

# Multi-channel (C > 1)
CHANNEL_SHAPES = [
    pytest.param((1, 2, 32, 32), id="chan2_32x32"),
    pytest.param((1, 4, 32, 32), id="chan4_32x32"),
    pytest.param((1, 8, 64, 64), id="chan8_64x64"),
    pytest.param((1, 3, 128, 128), id="chan3_128x128"),
    pytest.param((1, 16, 32, 64), id="chan16_32x64"),
]

# Multi-batch AND multi-channel
BATCH_CHANNEL_SHAPES = [
    pytest.param((2, 3, 32, 32), id="b2c3_32x32"),
    pytest.param((4, 2, 64, 64), id="b4c2_64x64"),
    pytest.param((2, 4, 128, 128), id="b2c4_128x128"),
    pytest.param((3, 3, 64, 128), id="b3c3_64x128"),
    pytest.param((2, 2, 256, 256), id="b2c2_256x256"),
    pytest.param((8, 4, 32, 64), id="b8c4_32x64"),
]

# Large shapes (stress memory and multi-core distribution)
LARGE_SHAPES = [
    pytest.param((1, 1, 2048, 2048), id="large_2048x2048"),
    pytest.param((1, 1, 4096, 128), id="large_4096x128"),
    pytest.param((1, 1, 128, 4096), id="large_128x4096"),
    pytest.param((2, 1, 1024, 1024), id="large_b2_1024x1024"),
    pytest.param((1, 1, 4096, 256), id="large_4096x256"),
    pytest.param((4, 1, 512, 512), id="large_b4_512x512"),
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


# ---------------------------------------------------------------------------
# Tests: with gamma and beta (full layer norm)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_layer_norm_with_affine(device, shape):
    """Full layer norm: normalize + gamma * x + beta."""
    torch.manual_seed(42)
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    beta = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_layer_norm(torch_input, gamma, beta)

    ttnn_input = to_ttnn(torch_input, device)
    ttnn_gamma = to_ttnn(gamma, device)
    ttnn_beta = to_ttnn(beta, device)

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta)
    check_output(ttnn_output, expected, shape, RTOL, ATOL)
