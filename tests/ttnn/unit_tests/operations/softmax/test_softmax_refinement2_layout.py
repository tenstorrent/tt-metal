# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 tests: ROW_MAJOR layout support + multi-core distribution.

Tests that ROW_MAJOR_LAYOUT inputs produce correct softmax output,
matching the TILE_LAYOUT path numerically. Also tests multi-core
distribution by running shapes with many slabs (N*C > 1).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def pytorch_softmax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(input_tensor.float(), dim=dim)


PCC_THRESHOLD = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
}


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat16, id="bf16"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((2, 4, 64, 64), id="multi_batch"),
        pytest.param((4, 8, 32, 256), id="large_batch"),
    ],
)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim_W", "dim_H"])
def test_softmax_rm_basic(device, shape, dtype, dim):
    """ROW_MAJOR softmax matches PyTorch reference."""
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16

    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=dim)

    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT, f"Output layout mismatch: {ttnn_output.layout}"
    assert ttnn_output.dtype == dtype

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD[dtype])


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="1_slab"),
        pytest.param((1, 4, 32, 32), id="4_slabs"),
        pytest.param((1, 16, 32, 32), id="16_slabs"),
        pytest.param((4, 8, 32, 32), id="32_slabs"),
        pytest.param((8, 8, 32, 32), id="64_slabs"),
    ],
)
def test_softmax_rm_multicore(device, shape):
    """RM layout with many slabs — exercises multi-core distribution."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=-1)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=-1)

    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT

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
def test_softmax_rm_vs_tile_equivalence(device, shape, dim):
    """RM and TILE layouts produce numerically equivalent results."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)

    # TILE path
    ttnn_input_tile = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output_tile = ttnn.softmax(ttnn_input_tile, dim=dim)
    torch_output_tile = ttnn.to_torch(ttnn_output_tile)

    # RM path
    ttnn_input_rm = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output_rm = ttnn.softmax(ttnn_input_rm, dim=dim)
    torch_output_rm = ttnn.to_torch(ttnn_output_rm)

    # Both must match the PyTorch reference
    torch_expected = pytorch_softmax(torch_input, dim=dim)
    assert_with_pcc(torch_expected, torch_output_tile, pcc=PCC_THRESHOLD[ttnn.float32])
    assert_with_pcc(torch_expected, torch_output_rm, pcc=PCC_THRESHOLD[ttnn.float32])

    # And match each other closely
    assert_with_pcc(torch_output_tile, torch_output_rm, pcc=0.999)


def test_softmax_rm_output_layout_matches_input(device):
    """Output tensor must preserve ROW_MAJOR layout."""
    torch.manual_seed(42)
    torch_input = torch.randn((1, 1, 32, 64), dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=-1)

    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT, f"Output layout should be ROW_MAJOR, got {ttnn_output.layout}"


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
    ],
)
def test_softmax_rm_numerical_stability(device, shape):
    """RM path must be numerically stable with large inputs."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32) * 100.0

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=-1)
    torch_output = ttnn.to_torch(ttnn_output)

    assert not torch_output.isnan().any(), "Output contains NaN — stability failure"
    assert not torch_output.isinf().any(), "Output contains Inf — stability failure"

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
def test_softmax_rm_l1_memory(device, shape):
    """RM layout with L1 memory config."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=-1)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=-1)

    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD[ttnn.float32])
