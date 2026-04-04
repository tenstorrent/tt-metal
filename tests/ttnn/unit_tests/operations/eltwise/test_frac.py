# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_frac(device, input_shape, dtype):
    """Test frac(x) = x - floor(x) for random inputs."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.randn(input_shape, dtype=torch_dtype) * 10.0

    torch_output = torch.frac(torch_input.to(torch.float32))

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    if dtype == ttnn.bfloat16:
        assert_with_pcc(torch_output, tt_output, pcc=0.999)
    else:
        assert_with_pcc(torch_output, tt_output, pcc=0.9999)


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_frac_negative_inputs(device, input_shape):
    """Test frac for negative inputs: frac(-2.7) = -0.7."""
    torch_input = -torch.abs(torch.randn(input_shape, dtype=torch.bfloat16)) * 10.0
    torch_output = torch.frac(torch_input.to(torch.float32))

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, pcc=0.999)


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_frac_integer_inputs(device, input_shape):
    """Test frac for integer values: frac(N) = 0 for any integer N."""
    torch_input = torch.randint(-100, 100, input_shape, dtype=torch.float32).to(torch.bfloat16)
    torch_output = torch.frac(torch_input.to(torch.float32))

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    # For integer inputs, frac should return 0 exactly
    assert_with_pcc(torch_output, tt_output, pcc=0.999)


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_frac_special_values(device, input_shape):
    """Test frac with specific known values."""
    # frac(3.7) = 0.7, frac(-3.7) = -0.7, frac(0.0) = 0.0, frac(1.0) = 0.0
    torch_input = torch.tensor(
        [[[[3.7, -3.7, 0.0, 1.0, -1.0, 2.5, -2.5, 100.25] + [0.0] * 24] + [[0.0] * 32] * 31]],
        dtype=torch.bfloat16,
    )
    torch_output = torch.frac(torch_input.to(torch.float32))

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, pcc=0.999)


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_frac_large_values(device, input_shape):
    """Test frac with large values where float precision matters."""
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16) * 1000.0
    torch_output = torch.frac(torch_input.to(torch.float32))

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, pcc=0.999)
