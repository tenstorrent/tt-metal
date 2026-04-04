# SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 1, 32, 32],
        [1, 1, 64, 64],
        [1, 3, 320, 384],
        [4, 1, 32, 32],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.float32],
)
def test_atanh(device, input_shapes, dtype):
    # atanh domain: (-1, 1)
    torch_input = torch.empty(input_shapes, dtype=torch.float32).uniform_(-0.9, 0.9)

    if dtype == ttnn.bfloat16:
        torch_input = torch_input.to(torch.bfloat16).to(torch.float32)

    torch_output = torch.atanh(torch_input)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.atanh(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    if dtype == ttnn.bfloat16:
        assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
    else:
        assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 1, 32, 32],
    ],
)
def test_atanh_range(device, input_shapes):
    """Test atanh with various input ranges."""
    # Test near zero (atanh(0) = 0)
    torch_input = torch.zeros(input_shapes, dtype=torch.float32)
    torch_output = torch.atanh(torch_input)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.atanh(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)

    # Test small values near zero
    torch_input = torch.empty(input_shapes, dtype=torch.float32).uniform_(-0.3, 0.3)
    torch_input = torch_input.to(torch.bfloat16).to(torch.float32)
    torch_output = torch.atanh(torch_input)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.atanh(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)

    # Test values closer to boundaries (but not too close to avoid inf issues)
    torch_input = torch.empty(input_shapes, dtype=torch.float32).uniform_(-0.95, 0.95)
    torch_input = torch_input.to(torch.bfloat16).to(torch.float32)
    torch_output = torch.atanh(torch_input)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.atanh(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
