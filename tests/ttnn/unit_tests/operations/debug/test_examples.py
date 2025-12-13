# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal

from loguru import logger


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
def test_composite_example_sub_devices(device, height, width):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    # Run w/o sub devices
    logger.info("Running composite example without sub devices")
    output_tensor = ttnn.composite_example(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor, output_tensor)

    # Setup sub devices
    sub_device_1 = ttnn.SubDevice([ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 2))])])
    sub_device_2 = ttnn.SubDevice([ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(3, 3), ttnn.CoreCoord(5, 5))])])
    sub_devices = [sub_device_1, sub_device_2]
    sub_device_manager = device.create_sub_device_manager(sub_devices, 3200)
    device.load_sub_device_manager(sub_device_manager)

    logger.info("Running composite example on sub device 0")
    with ttnn.sub_device(device, 0):
        output_tensor_0 = ttnn.composite_example(input_tensor)
        output_tensor_0 = ttnn.to_torch(output_tensor_0)
        assert_equal(torch_output_tensor, output_tensor_0)

    logger.info("Running composite example on sub device 1")
    with ttnn.sub_device(device, 1):
        output_tensor_1 = ttnn.composite_example(input_tensor)
        output_tensor_1 = ttnn.to_torch(output_tensor_1)
        assert_equal(torch_output_tensor, output_tensor_1)

    logger.info("Running composite example on sub device 1 with an inline call on sub_device=0")
    with ttnn.sub_device(device, 0):
        output_tensor_0 = ttnn.composite_example(input_tensor)
        output_tensor_0 = ttnn.to_torch(output_tensor_0)
        assert_equal(torch_output_tensor, output_tensor_0)

        output_tensor_1 = ttnn.composite_example(input_tensor, sub_device_id=1)
        output_tensor_1 = ttnn.to_torch(output_tensor_1)
        assert_equal(torch_output_tensor, output_tensor_1)


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
