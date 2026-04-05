# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_cbrt(x):
    """Golden reference: cube root that handles negative values correctly."""
    return torch.sgn(x) * torch.pow(torch.abs(x), 1.0 / 3.0)


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_cbrt(device, input_shape, dtype):
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)

    torch_output = torch_cbrt(torch_input.to(torch.float32))

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.cbrt(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    # Cube root has good precision for most values
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
def test_cbrt_negative_inputs(device, input_shape):
    """Test that cbrt correctly handles negative inputs: cbrt(-x) = -cbrt(x)."""
    torch_input = -torch.abs(torch.randn(input_shape, dtype=torch.bfloat16))
    torch_output = torch_cbrt(torch_input.to(torch.float32))

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.cbrt(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, pcc=0.999)


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_cbrt_special_values(device, input_shape):
    """Test cbrt with values near zero and perfect cubes."""
    # Create tensor with specific values: 0, 1, 8, 27, -1, -8, -27
    torch_input = torch.tensor(
        [[[[0.0, 1.0, 8.0, 27.0, -1.0, -8.0, -27.0, 64.0] + [0.0] * 24] + [[0.0] * 32] * 31]], dtype=torch.bfloat16
    )
    torch_output = torch_cbrt(torch_input.to(torch.float32))

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.cbrt(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, pcc=0.999)
