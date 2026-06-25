# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for ttnn.operations.softmax.softmax.

This is the immutable acceptance test — the implementer must not modify it.
It is the specification: if this test passes, the op meets the Phase 0 contract.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def pytorch_softmax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """PyTorch reference using the numerically-stable form (torch.softmax is stable)."""
    return torch.softmax(input_tensor.float(), dim=dim)


# PCC thresholds keyed by dtype — same as the golden suite.
PCC_THRESHOLD = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile_non_square"),
        pytest.param((2, 4, 64, 64), id="multi_batch"),
        pytest.param((4, 8, 32, 256), id="large_batch_wide"),
        pytest.param((1, 1, 128, 512), id="large_hw"),
        pytest.param((2, 3, 96, 96), id="multi_batch_square"),
    ],
)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim_W", "dim_H"])
def test_softmax_basic(device, shape, dim):
    """Softmax against PyTorch reference, float32, TILE_LAYOUT."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=dim)

    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {list(ttnn_output.shape)} vs {list(shape)}"
    assert ttnn_output.dtype == ttnn.float32, f"dtype mismatch: {ttnn_output.dtype}"
    assert ttnn_output.layout == ttnn.TILE_LAYOUT, f"layout mismatch: {ttnn_output.layout}"

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD[ttnn.float32])


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((2, 4, 64, 64), id="multi_batch"),
    ],
)
def test_softmax_default_dim(device, shape):
    """Default dim=-1 (last dimension)."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=-1)

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD[ttnn.float32])


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
    ],
)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim_W", "dim_H"])
def test_softmax_positive_dim_alias(device, shape, dim):
    """Positive dim aliases must work: dim=3 ≡ -1, dim=2 ≡ -2 for rank-4."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    positive_dim = dim + 4  # -1 → 3, -2 → 2
    ttnn_output = ttnn.softmax(ttnn_input, dim=positive_dim)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=positive_dim)

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD[ttnn.float32])


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
    ],
)
def test_softmax_compute_kernel_config(device, shape):
    """Phase 0 corner: explicit compute_kernel_config with HiFi4 + fp32_dest_acc_en=True."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_kernel_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )

    ttnn_output = ttnn.softmax(ttnn_input, compute_kernel_config=compute_kernel_config)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=-1)

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD[ttnn.float32])


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((2, 4, 64, 64), id="multi_batch"),
    ],
)
def test_softmax_l1_memory(device, shape):
    """Softmax with L1 memory config (input and output in L1)."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=-1)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=-1)

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD[ttnn.float32])


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
    ],
)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim_W", "dim_H"])
def test_softmax_numerical_stability(device, shape, dim):
    """Verify numerical stability with large-magnitude inputs.

    Softmax must subtract the row max before exp — if it doesn't,
    large inputs cause overflow and NaN.
    """
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32) * 100.0

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=dim)

    # No NaN or Inf allowed — stability check
    assert not torch_output.isnan().any(), "Output contains NaN — stability failure"
    assert not torch_output.isinf().any(), "Output contains Inf — stability failure"

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD[ttnn.float32])


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((2, 4, 64, 64), id="multi_batch"),
    ],
)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim_W", "dim_H"])
def test_softmax_uniform_input(device, shape, dim):
    """Softmax with uniform input — all outputs should be ~1/N along reduce dim."""
    torch.manual_seed(42)
    n_elements = shape[dim] if dim >= 0 else shape[dim]
    torch_input = torch.ones(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=dim)

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD[ttnn.float32])

    # Each element along reduce dim should be ~1/n_elements
    expected_val = 1.0 / n_elements
    assert torch.allclose(
        torch_output.float(),
        torch.full_like(torch_output.float(), expected_val),
        rtol=1e-4,
        atol=1e-4,
    ), f"Uniform input: expected {expected_val}, got max diff {(torch_output.float() - expected_val).abs().max()}"


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((2, 4, 64, 64), id="multi_batch"),
    ],
)
def test_softmax_negative_values(device, shape):
    """Softmax with negative input values — all outputs must be positive and sum to 1."""
    torch.manual_seed(42)
    torch_input = -torch.abs(torch.randn(shape, dtype=torch.float32))

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=-1)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=-1)

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD[ttnn.float32])

    # All outputs must be positive
    assert (torch_output.float() >= 0).all(), "Softmax output contains negative values"


@pytest.mark.parametrize("dim", [0, 1], ids=["dim_0_invalid", "dim_1_invalid"])
def test_softmax_unsupported_dim_raises(device, dim):
    """Dims other than -1/-2 must raise NotImplementedError."""
    torch.manual_seed(42)
    torch_input = torch.randn((1, 1, 32, 32), dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises((NotImplementedError, ValueError, RuntimeError)):
        ttnn.softmax(ttnn_input, dim=dim)


def test_softmax_fp32_dest_acc_false_raises(device):
    """fp32 input with fp32_dest_acc_en=False must raise (lossy combination)."""
    torch.manual_seed(42)
    torch_input = torch.randn((1, 1, 32, 32), dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )

    with pytest.raises((NotImplementedError, ValueError, RuntimeError)):
        ttnn.softmax(ttnn_input, compute_kernel_config=config)
