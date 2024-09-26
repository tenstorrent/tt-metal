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
def test_zero(device, input_shapes):
    torch_input_tensor = torch.randn((input_shapes), dtype=torch.bfloat16)
    torch_output_tensor = torch.zeros((input_shapes), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.zero(input_tensor)
    output_tensor = ttnn.to_torch(output)
    assert_equal(torch_output_tensor, output_tensor)
