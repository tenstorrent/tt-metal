# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [128])
def test_example(device, height, width):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.prim.example(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [128])
def test_composite_example(device, height, width):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.composite_example(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [128])
@pytest.mark.parametrize("return_outputs", [[False, True], [True, False], [True, True]])
def test_example_multiple_return(device, height, width, return_outputs):
    torch.manual_seed(0)

    return_output1, return_output2 = return_outputs

    # run torch
    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor

    # run TT
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output1, output2 = ttnn.prim.example_multiple_return(
        input_tensor, return_output1=return_output1, return_output2=return_output2
    )

    if return_output1:
        output_tensor1 = ttnn.to_torch(output1)
        assert_equal(torch_output_tensor, output_tensor1)
    else:
        assert output1 == None

    if return_output2:
        output_tensor2 = ttnn.to_torch(output2)
        assert_equal(torch_output_tensor, output_tensor2)
    else:
        assert output2 == None


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [128])
@pytest.mark.parametrize("return_outputs", [[False, True], [True, False], [True, True]])
def test_composite_example_multiple_return(device, height, width, return_outputs):
    torch.manual_seed(0)

    return_output1, return_output2 = return_outputs

    # run torch
    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor

    # run TT
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output1, output2 = ttnn.composite_example_multiple_return(
        input_tensor, return_output1=return_output1, return_output2=return_output2
    )

    if return_output1:
        output_tensor1 = ttnn.to_torch(output1)
        assert_equal(torch_output_tensor, output_tensor1)
    else:
        assert output1 == None

    if return_output2:
        output_tensor2 = ttnn.to_torch(output2)
        assert_equal(torch_output_tensor, output_tensor2)
    else:
        assert output2 == None
