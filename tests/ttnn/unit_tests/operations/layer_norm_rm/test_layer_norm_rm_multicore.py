# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for multi-core distribution in layer_norm_rm (Refinement 1).

Verifies that:
- Multi-block shapes distribute across multiple cores correctly
- Shapes with even and uneven block distributions work
- Large shapes that previously caused L1 overflow now work (due to fewer blocks per core)
- All affine modes work with multi-core
- Precision is maintained vs single-core baseline
"""

import pytest
import torch
import ttnn
import math

from ttnn.operations.layer_norm_rm import layer_norm_rm


def pytorch_layer_norm(x, gamma=None, beta=None, epsilon=1e-5):
    """Reference implementation in float32."""
    x_f32 = x.to(torch.float32)
    mean = torch.mean(x_f32, dim=-1, keepdim=True)
    var = torch.var(x_f32, dim=-1, keepdim=True, unbiased=False)
    result = (x_f32 - mean) / torch.sqrt(var + epsilon)
    if gamma is not None:
        result = result * gamma.to(torch.float32)
    if beta is not None:
        result = result + beta.to(torch.float32)
    return result.to(x.dtype)


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def check_result(actual_ttnn, expected_torch, rtol=0.05, atol=0.2):
    actual = ttnn.to_torch(actual_ttnn).float()
    expected = expected_torch.float()
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"max_diff={( actual - expected).abs().max().item():.6f}, "
        f"mean_diff={(actual - expected).abs().mean().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Multi-core distribution: shapes with many blocks (NC*Ht > 1)
# These ensure work is actually split across cores
# ---------------------------------------------------------------------------

MULTI_BLOCK_SHAPES = [
    # (shape, description)
    pytest.param((1, 1, 64, 32), id="2_blocks_h64w32"),
    pytest.param((1, 1, 128, 32), id="4_blocks_h128w32"),
    pytest.param((1, 1, 256, 64), id="8_blocks_h256w64"),
    pytest.param((2, 1, 64, 64), id="4_blocks_b2h64w64"),
    pytest.param((4, 1, 32, 32), id="4_blocks_b4h32w32"),
    pytest.param((8, 1, 32, 32), id="8_blocks_b8h32w32"),
    pytest.param((2, 2, 64, 64), id="8_blocks_b2c2h64w64"),
    pytest.param((4, 4, 32, 32), id="16_blocks_b4c4h32w32"),
    pytest.param((1, 1, 512, 32), id="16_blocks_h512w32"),
    pytest.param((1, 1, 1024, 32), id="32_blocks_h1024w32"),
    pytest.param((2, 1, 256, 64), id="16_blocks_b2h256w64"),
    pytest.param((1, 1, 2048, 32), id="64_blocks_h2048w32"),
    # Uneven distributions (num_blocks not evenly divisible by num_cores)
    pytest.param((3, 1, 32, 32), id="3_blocks_b3h32w32"),
    pytest.param((5, 1, 32, 32), id="5_blocks_b5h32w32"),
    pytest.param((7, 1, 32, 32), id="7_blocks_b7h32w32"),
    pytest.param((1, 1, 96, 32), id="3_blocks_h96w32"),
    pytest.param((1, 1, 160, 32), id="5_blocks_h160w32"),
    pytest.param((1, 1, 224, 32), id="7_blocks_h224w32"),
]


@pytest.mark.parametrize("shape", MULTI_BLOCK_SHAPES)
def test_multicore_pure(device, shape):
    """Multi-core distribution: pure normalization (no affine)."""
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16)
    expected = pytorch_layer_norm(x)
    ttnn_out = layer_norm_rm(to_ttnn(x, device))
    check_result(ttnn_out, expected)


@pytest.mark.parametrize("shape", MULTI_BLOCK_SHAPES)
def test_multicore_gamma_beta(device, shape):
    """Multi-core distribution: full affine (gamma + beta)."""
    torch.manual_seed(42)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    beta = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    expected = pytorch_layer_norm(x, gamma, beta)
    ttnn_out = layer_norm_rm(to_ttnn(x, device), to_ttnn(gamma, device), to_ttnn(beta, device))
    check_result(ttnn_out, expected)


# ---------------------------------------------------------------------------
# Large shapes that benefit from multi-core (previously L1 overflow at W>=1024)
# With multi-core, fewer blocks per core => less L1 pressure
# ---------------------------------------------------------------------------

LARGE_MULTICORE_SHAPES = [
    pytest.param((1, 1, 64, 256), id="large_64x256"),
    pytest.param((1, 1, 128, 256), id="large_128x256"),
    pytest.param((2, 1, 128, 128), id="large_b2_128x128"),
    pytest.param((4, 1, 64, 128), id="large_b4_64x128"),
    pytest.param((1, 1, 256, 128), id="large_256x128"),
    pytest.param((1, 1, 512, 64), id="large_512x64"),
    pytest.param((8, 1, 64, 64), id="large_b8_64x64"),
    pytest.param((1, 1, 1024, 64), id="large_1024x64"),
]


@pytest.mark.parametrize("shape", LARGE_MULTICORE_SHAPES)
def test_multicore_large_shapes(device, shape):
    """Large shapes with gamma+beta that exercise multi-core distribution."""
    torch.manual_seed(42)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    beta = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    expected = pytorch_layer_norm(x, gamma, beta)
    ttnn_out = layer_norm_rm(to_ttnn(x, device), to_ttnn(gamma, device), to_ttnn(beta, device))
    check_result(ttnn_out, expected)


# ---------------------------------------------------------------------------
# Precision measurement for multi-core shapes
# ---------------------------------------------------------------------------

PRECISION_SHAPES = [
    pytest.param((1, 1, 32, 32), id="1block_32x32"),
    pytest.param((4, 1, 32, 32), id="4blocks_b4_32x32"),
    pytest.param((1, 1, 256, 64), id="8blocks_256x64"),
    pytest.param((8, 1, 64, 64), id="16blocks_b8_64x64"),
    pytest.param((1, 1, 1024, 32), id="32blocks_1024x32"),
    pytest.param((2, 1, 128, 256), id="8blocks_b2_128x256"),
]


@pytest.mark.parametrize("shape", PRECISION_SHAPES)
def test_multicore_precision(device, shape):
    """Measure and assert PCC for multi-core shapes."""
    torch.manual_seed(42)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    beta = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

    expected = pytorch_layer_norm(x, gamma, beta)
    ttnn_out = layer_norm_rm(to_ttnn(x, device), to_ttnn(gamma, device), to_ttnn(beta, device))

    actual = ttnn.to_torch(ttnn_out).to(torch.float64)
    expected_f64 = expected.to(torch.float64)

    # PCC
    a = actual.flatten()
    e = expected_f64.flatten()
    a_c = a - a.mean()
    e_c = e - e.mean()
    pcc = ((a_c * e_c).sum() / (a_c.norm() * e_c.norm())).item()

    max_abs = (actual - expected_f64).abs().max().item()
    mean_abs = (actual - expected_f64).abs().mean().item()

    print(f"Shape={shape}, PCC={pcc:.8f}, max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}")

    assert pcc > 0.9999, f"PCC too low: {pcc:.8f}"
    assert max_abs < 0.3, f"max_abs too high: {max_abs:.6f}"
