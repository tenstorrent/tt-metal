# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h", [32, 30])
@pytest.mark.parametrize("w", [32, 62])
@pytest.mark.parametrize("on_device", [True, False])
@pytest.mark.parametrize("from_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("to_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("start_with_padding", [False, True])
def test_to_layout_2D(device, h, w, on_device, from_layout, to_layout, start_with_padding):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    if start_with_padding:
        pad_h = (ttnn.TILE_SIZE - h % ttnn.TILE_SIZE) % ttnn.TILE_SIZE
        pad_w = (ttnn.TILE_SIZE - w % ttnn.TILE_SIZE) % ttnn.TILE_SIZE
        torch_padded_input_tensor = torch.nn.functional.pad(
            torch_input_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0
        )
        input_tensor = ttnn.from_torch(torch_padded_input_tensor)
        input_tensor = ttnn.reshape(input_tensor, shape=ttnn.Shape([h, w], ((0, pad_h), (0, pad_w))))

        if from_layout == ttnn.ROW_MAJOR_LAYOUT and to_layout == ttnn.ROW_MAJOR_LAYOUT:
            torch_input_tensor = torch_padded_input_tensor
    else:
        input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    input_tensor = ttnn.to_layout(input_tensor, from_layout)
    assert input_tensor.layout == from_layout

    if on_device:
        input_tensor = ttnn.to_device(input_tensor, device)
        assert ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE)

    output_tensor = ttnn.to_layout(input_tensor, to_layout)
    assert output_tensor.layout == to_layout

    if on_device:
        output_tensor = ttnn.from_device(output_tensor)
        assert not ttnn.has_storage_type_of(output_tensor, ttnn.DEVICE_STORAGE_TYPE)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
    assert torch.allclose(torch_input_tensor, output_tensor)
