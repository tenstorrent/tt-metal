# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_softsign(x):
    """Golden reference: softsign(x) = x / (1 + |x|)."""
    return x / (1 + torch.abs(x))


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_softsign(device, shape, dtype):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    torch_output = torch_softsign(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softsign(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_softsign_output_range(device, shape):
    """Verify softsign output is always in (-1, 1)."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 10  # wide range

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softsign(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert tt_output_torch.min() >= -1.0, f"Output below -1: {tt_output_torch.min()}"
    assert tt_output_torch.max() <= 1.0, f"Output above 1: {tt_output_torch.max()}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_softsign_allclose(device, shape):
    """Test softsign with allclose tolerances (rtol=1.6e-2, atol=1e-2)."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    torch_output = torch_softsign(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softsign(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert torch.allclose(
        torch_output, tt_output_torch, rtol=1.6e-2, atol=1e-2
    ), f"allclose failed: max diff = {(torch_output - tt_output_torch).abs().max()}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_softsign_negative_inputs(device, shape):
    """Test softsign with all-negative inputs: softsign(-x) = -softsign(x)."""
    torch.manual_seed(42)
    torch_input = -torch.abs(torch.randn(shape, dtype=torch.bfloat16))
    torch_output = torch_softsign(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softsign(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
    # All outputs should be negative
    assert tt_output_torch.max() <= 0.0, f"Expected all negative outputs, got max: {tt_output_torch.max()}"
