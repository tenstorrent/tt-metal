# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_repeat(device):
    torch_input_tensor = torch.randn((1, 2, 4, 4), dtype=torch.bfloat16)
    repeat_shape = (1, 2, 1, 1)

    torch_result = torch_input_tensor.repeat(repeat_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.repeat(input_tensor, ttnn.Shape(repeat_shape))
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_result, output, 0.9999)


def test_repeat_with_tile_padding(device):
    torch_input_tensor = torch.randn((1, 1, 1, 1), dtype=torch.bfloat16)
    repeat_shape = (2, 2, 2, 2)

    torch_result = torch_input_tensor.repeat(repeat_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.repeat(input_tensor, ttnn.Shape(repeat_shape))
    output = ttnn.to_torch(output)

    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

    assert_with_pcc(torch_result, output, 0.9999), f"Output {output} does not match torch output {torch_result}"
