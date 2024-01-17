# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h", [31])
@pytest.mark.parametrize("w", [2 * 33])
@pytest.mark.parametrize("tile_size", [32])
@pytest.mark.parametrize("on_device", [True, False])
# @pytest.mark.parametrize("start_with_padding", [False, True])
@pytest.mark.parametrize(
    "layout_from_to",
    [
        (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    ],
)
def test_to_layout_2D(device, h, w, tile_size, on_device, layout_from_to):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    # if start_with_padding:
    #     pad_h = (tile_size - h % tile_size) % tile_size
    #     pad_w = (tile_size - w % tile_size) % tile_size
    #     padded_height = h + pad_h
    #     padded_width = w + pad_w
    #     input_tensor = ttnn.reshape(input_tensor, shape=ttnn.Shape([h, w], [padded_height, padded_width]))
    #     assert list(input_tensor.shape) != list(input_tensor.shape.padded())
    input_tensor = ttnn.to_layout(input_tensor, layout_from_to[0])
    if on_device:
        input_tensor = ttnn.to_device(input_tensor, device)
        assert ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE)
    output_tensor = ttnn.to_layout(input_tensor, layout_from_to[1])
    if on_device:
        output_tensor = ttnn.from_device(output_tensor)
        assert not ttnn.has_storage_type_of(output_tensor, ttnn.DEVICE_STORAGE_TYPE)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_input_tensor, output_tensor)
