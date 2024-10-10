# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 20, 31])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([2, 4, 320, 1024])),
    ),
)
@pytest.mark.parametrize("fill_value", [1, 0, 5.5, -2.235])
def test_fill(device, input_shapes, fill_value):
    torch_input_tensor = torch.randn((input_shapes), dtype=torch.bfloat16)
    torch_output_tensor = torch.full((input_shapes), fill_value, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.fill(input_tensor, fill_value)
    output_tensor = ttnn.to_torch(output)
    assert_equal(torch_output_tensor, output_tensor)
