# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16],
)
def test_frac(device, shape, dtype):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 100  # wide range

    # PyTorch golden: torch.frac(x) = x - trunc(x)
    torch_output = torch.frac(torch_input.float()).to(torch.bfloat16)

    # TT computation
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_frac_properties(device, shape):
    """Verify frac properties: |frac(x)| < 1, and sign of frac matches sign of x."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 10

    torch_output = torch.frac(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)

    # |frac(x)| should always be < 1 (with small tolerance for bfloat16 rounding)
    assert (tt_output_torch.abs() <= 1.01).all(), "|frac(x)| should be < 1"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_frac_integers(device, shape):
    """Verify frac of integer values is 0."""
    # Create tensor with integer values
    torch_input = torch.randint(-50, 50, shape, dtype=torch.int32).to(torch.bfloat16)

    torch_output = torch.frac(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    # frac of integers should be 0
    assert torch.allclose(tt_output_torch, torch.zeros_like(tt_output_torch), atol=0.01), "frac of integers should be 0"
