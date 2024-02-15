# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_sizes", [(), (1,)])
@pytest.mark.parametrize("height", [32, 64])
@pytest.mark.parametrize("width", [32, 96])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("on_device", [True, False])
def test_getitem(device, batch_sizes, height, width, input_layout, on_device):
    torch_input_tensor = torch.rand((*batch_sizes, height, width), dtype=torch.bfloat16)

    if batch_sizes:
        torch_output_tensor = torch_input_tensor[..., :32, :32]
    else:
        torch_output_tensor = torch_input_tensor[:32, :32]

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=input_layout)

    if on_device:
        input_tensor = ttnn.to_device(input_tensor, device)
        assert ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE)

    if batch_sizes:
        output_tensor = input_tensor[:1, :32, :32]
    else:
        output_tensor = input_tensor[:32, :32]
    assert output_tensor.layout == input_layout

    if on_device:
        assert ttnn.has_storage_type_of(output_tensor, ttnn.DEVICE_STORAGE_TYPE)
        output_tensor = ttnn.from_device(output_tensor)
        assert not ttnn.has_storage_type_of(output_tensor, ttnn.DEVICE_STORAGE_TYPE)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("height", [32, 64])
@pytest.mark.parametrize("width", [32, 96])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("on_device", [True, False])
def test_getitem_2d(device, height, width, input_layout, on_device):
    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)

    torch_output_tensor = torch_input_tensor[:32]

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=input_layout)

    if on_device:
        input_tensor = ttnn.to_device(input_tensor, device)
        assert ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE)

    output_tensor = input_tensor[:32]
    assert output_tensor.layout == input_layout

    if on_device:
        assert ttnn.has_storage_type_of(output_tensor, ttnn.DEVICE_STORAGE_TYPE)
        output_tensor = ttnn.from_device(output_tensor)
        assert not ttnn.has_storage_type_of(output_tensor, ttnn.DEVICE_STORAGE_TYPE)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)
    assert torch.allclose(torch_output_tensor, output_tensor)
