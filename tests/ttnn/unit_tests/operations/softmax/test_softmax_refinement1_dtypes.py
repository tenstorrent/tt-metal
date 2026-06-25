# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 tests — Numerical configurability (dtypes + fp32-dest-only policy).

Exercises the newly added dtype support (bfloat16, bfloat8_b) directly:
  - bf16 and bf8b with both dim=-1 and dim=-2
  - dtype-specific PCC thresholds
  - verify fp32_dest_acc_en=False still rejected for all dtypes
  - verify output dtype matches input dtype
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def pytorch_softmax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(input_tensor.float(), dim=dim)


# PCC thresholds from golden helpers (dtype-driven)
PCC_THRESHOLD = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.99,
    ttnn.bfloat8_b: 0.99,
}

DTYPES = [
    pytest.param(ttnn.float32, id="fp32"),
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.bfloat8_b, id="bf8b"),
]

SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 32, 64), id="32x64"),
    pytest.param((1, 1, 64, 128), id="64x128"),
    pytest.param((2, 4, 64, 64), id="multi_batch"),
    pytest.param((4, 8, 32, 256), id="large_batch_wide"),
    pytest.param((1, 1, 128, 512), id="large_hw"),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim_W", "dim_H"])
def test_softmax_dtype_support(device, shape, dtype, dim):
    """Softmax with each supported dtype, both dims, against PyTorch reference."""
    torch.manual_seed(42)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
    torch_output = ttnn.to_torch(ttnn_output)
    expected = pytorch_softmax(torch_input, dim=dim)

    # Output dtype must match input dtype
    assert ttnn_output.dtype == dtype, f"dtype mismatch: {ttnn_output.dtype} vs {dtype}"
    assert ttnn_output.layout == ttnn.TILE_LAYOUT
    assert list(ttnn_output.shape) == list(shape)

    assert_with_pcc(expected, torch_output, pcc=PCC_THRESHOLD[dtype])

    # No NaN or Inf
    assert not torch_output.isnan().any(), "Output contains NaN"
    assert not torch_output.isinf().any(), "Output contains Inf"


@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_fp32_dest_acc_false_rejected(device, dtype):
    """fp32_dest_acc_en=False must be rejected for ALL dtypes (fp32-dest-only policy)."""
    torch.manual_seed(42)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.randn((1, 1, 32, 32), dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
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


@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_output_dtype_matches_input(device, dtype):
    """Output dtype must match input dtype for all supported dtypes."""
    torch.manual_seed(42)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.randn((1, 1, 64, 128), dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=-1)
    assert ttnn_output.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_numerical_stability_large_input(device, dtype):
    """Large-magnitude inputs must not produce NaN/Inf — max subtraction works for all dtypes."""
    torch.manual_seed(42)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.randn((1, 1, 32, 64), dtype=torch_dtype) * 100.0

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=-1)
    torch_output = ttnn.to_torch(ttnn_output)

    assert not torch_output.isnan().any(), "Output contains NaN — stability failure"
    assert not torch_output.isinf().any(), "Output contains Inf — stability failure"

    expected = pytorch_softmax(torch_input, dim=-1)
    assert_with_pcc(expected, torch_output, pcc=PCC_THRESHOLD[dtype])


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_softmax_l1_memory(device, dtype):
    """bf16 and bf8b with L1 memory config."""
    torch.manual_seed(42)
    torch_dtype = torch.bfloat16
    torch_input = torch.randn((1, 1, 64, 128), dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=-1)
    torch_output = ttnn.to_torch(ttnn_output)
    expected = pytorch_softmax(torch_input, dim=-1)

    assert_with_pcc(expected, torch_output, pcc=PCC_THRESHOLD[dtype])
