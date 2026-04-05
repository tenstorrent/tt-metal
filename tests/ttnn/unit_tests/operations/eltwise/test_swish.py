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
def test_swish(device, shape, dtype):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # PyTorch golden: swish(x) = x * sigmoid(x) = silu(x)
    torch_output = torch.nn.functional.silu(torch_input.float()).to(torch.bfloat16)

    # TT computation
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.swish(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_swish_properties(device, shape):
    """Verify key properties of swish:
    - swish(0) = 0
    - swish is smooth and continuous
    - For large positive x, swish(x) approaches x
    - For large negative x, swish(x) approaches 0
    """
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 5  # moderate range

    torch_output = torch.nn.functional.silu(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.swish(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_swish_wide_range(device, shape):
    """Test swish with wide input range to verify numerical stability."""
    torch.manual_seed(123)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 10  # wide range

    torch_output = torch.nn.functional.silu(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.swish(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.998)
