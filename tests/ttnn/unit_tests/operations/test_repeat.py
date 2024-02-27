# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_repeat(device):
    torch_input_tensor = torch.randn((1, 2, 4, 4), dtype=torch.bfloat16)
    repeat_shape = torch.randn((1, 2, 1, 1), dtype=torch.bfloat16)

    input_tensor1 = ttnn.from_torch(repeat_shape, layout=ttnn.TILE_LAYOUT)
    input_tensor1 = ttnn.to_device(input_tensor1, device)
    torch_result = torch_input_tensor.repeat(repeat_shape.shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.repeat(input_tensor, input_tensor1.shape)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_result, output, 0.9999)
