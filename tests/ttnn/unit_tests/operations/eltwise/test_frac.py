# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_allclose


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
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # PyTorch golden: torch.frac computes x - trunc(x)
    torch_output = torch.frac(torch_input.float()).to(torch.bfloat16)

    # TT computation
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
    assert_allclose(torch_output, tt_output_torch, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_frac_negative(device, shape):
    """Verify frac works correctly for negative inputs (sign-preserving)."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 10  # wide range

    torch_output = torch.frac(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
    # frac output should always be in (-1, 1)
    assert tt_output_torch.min() > -1.0, f"Output <= -1: {tt_output_torch.min()}"
    assert tt_output_torch.max() < 1.0, f"Output >= 1: {tt_output_torch.max()}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_frac_integers(device, shape):
    """Verify frac returns 0 for integer inputs."""
    # Create integer-valued tensor
    torch_input = torch.arange(-16, 16, dtype=torch.bfloat16).reshape(1, 1, 32, 1).expand(shape)

    torch_output = torch.frac(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
    # All fractional parts of integers should be 0
    assert torch.allclose(tt_output_torch, torch.zeros_like(tt_output_torch), atol=1e-2)
