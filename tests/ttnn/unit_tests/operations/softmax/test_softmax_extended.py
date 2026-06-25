# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Extended tests for softmax — focused shape/parameter coverage.

Covers gaps not exercised by the acceptance or golden suites:
- Multi-tile shapes where Ht × Wt spans multiple tiles
- dim=-1 vs dim=-2 on non-square shapes
- L1 memory config (already in acceptance, included here for completeness)
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def pytorch_softmax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(input_tensor.float(), dim=dim)


PCC_THRESHOLD = 0.999


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 64), id="1x1x32x64"),
        pytest.param((1, 1, 64, 32), id="1x1x64x32"),
        pytest.param((1, 1, 128, 256), id="1x1x128x256"),
        pytest.param((2, 4, 64, 128), id="2x4x64x128"),
    ],
)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim_W", "dim_H"])
def test_softmax_multi_tile(device, shape, dim):
    """Multi-tile shapes with both reduction dims."""
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

    assert list(ttnn_output.shape) == list(shape)
    assert ttnn_output.dtype == ttnn.float32
    assert ttnn_output.layout == ttnn.TILE_LAYOUT

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((2, 4, 64, 64), id="multi_batch"),
    ],
)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim_W", "dim_H"])
def test_softmax_l1_memory_multi_tile(device, shape, dim):
    """L1 memory config with multiple tiles and both dims."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=dim)

    assert_with_pcc(torch_expected, torch_output, pcc=PCC_THRESHOLD)
