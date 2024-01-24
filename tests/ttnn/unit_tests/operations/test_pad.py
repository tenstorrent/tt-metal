# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


@skip_for_wormhole_b0()
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 1), (0, 2)), (0, 2, 0, 1)), (((1, 1), (4, 2)), (4, 2, 1, 1))])
@pytest.mark.parametrize("value", [0])
def test_pad(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)

    assert output_tensor.shape == ttnn.Shape((h + padding[0][0] + padding[0][1], w + padding[1][0] + padding[1][1]))

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 1), (0, 2)), (0, 2, 0, 1)), (((1, 1), (4, 2)), (4, 2, 1, 1))])
@pytest.mark.parametrize("value", [0])
def test_pad_back_to_back(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)
    torch_output_tensor = torch.nn.functional.pad(torch_output_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
    output_tensor = ttnn.pad(output_tensor, padding=padding, value=value)

    assert output_tensor.shape == ttnn.Shape(
        (h + (padding[0][0] + padding[0][1]) * 2, w + (padding[1][0] + padding[1][1]) * 2)
    )

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
