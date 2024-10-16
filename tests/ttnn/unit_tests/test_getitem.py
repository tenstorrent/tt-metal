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
        assert ttnn.is_tensor_storage_on_device(input_tensor)

    if batch_sizes:
        output_tensor = input_tensor[:1, :32, :32]
    else:
        output_tensor = input_tensor[:32, :32]
    assert output_tensor.layout == input_layout

    if on_device:
        assert ttnn.is_tensor_storage_on_device(output_tensor)
        output_tensor = ttnn.from_device(output_tensor)
        assert not ttnn.is_tensor_storage_on_device(output_tensor)

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
        assert ttnn.is_tensor_storage_on_device(input_tensor)

    output_tensor = input_tensor[:32]
    assert output_tensor.layout == input_layout

    if on_device:
        assert ttnn.is_tensor_storage_on_device(output_tensor)
        output_tensor = ttnn.from_device(output_tensor)
        assert not ttnn.is_tensor_storage_on_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)
    assert torch.allclose(torch_output_tensor, output_tensor)


def test_getitem_scalar_output():
    torch_input_tensor = torch.rand((16, 32), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor)

    with pytest.raises(RuntimeError) as e:
        input_tensor[0, 0]
    assert "Host tensor slice cannot return a scalar or empty tensor" in str(e.value)


@pytest.mark.parametrize("batch_sizes", [(), (1, 1)])
@pytest.mark.parametrize("height", [32, 64])
@pytest.mark.parametrize("width", [32, 96])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT])
def test_getitem_non_tile_boundary(device, batch_sizes, height, width, input_layout):
    torch_input_tensor = torch.rand((*batch_sizes, height, width), dtype=torch.bfloat16)

    if len(torch_input_tensor.shape) == 4:
        torch_output_tensor = torch_input_tensor[:, :, :, :1]
    elif len(torch_input_tensor.shape) == 2:
        torch_output_tensor = torch_input_tensor[:, :1]
    else:
        raise RuntimeError("Invalid batch size")

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=input_layout, device=device)

    if len(torch_input_tensor.shape) == 4:
        output_tensor = input_tensor[:, :, :, :1]
    elif len(torch_input_tensor.shape) == 2:
        output_tensor = input_tensor[:, :1]
    else:
        raise RuntimeError("Invalid batch size")
    assert output_tensor.layout == input_layout
    assert output_tensor.shape[-1] == 1
    assert output_tensor.shape.with_tile_padding()[-1] == 32

    output_tensor = ttnn.to_torch(output_tensor)
    print(torch_output_tensor)
    print(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)
    assert torch.allclose(torch_output_tensor, output_tensor)
