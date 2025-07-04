# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
@pytest.mark.parametrize("from_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("to_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
def test_to_dtype(height, width, from_dtype, to_dtype):
    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    input_tensor = ttnn.to_dtype(input_tensor, from_dtype)
    assert input_tensor.dtype == from_dtype
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tuple(input_tensor.shape) == (height, width)

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)
    assert output_tensor.dtype == to_dtype
    if to_dtype == ttnn.bfloat8_b:
        assert output_tensor.layout == ttnn.TILE_LAYOUT
    else:
        assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tuple(output_tensor.shape) == (height, width)

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_input_tensor.dtype)
    assert_with_pcc(torch_input_tensor, output_tensor)


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("torch_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_dtype_conversion_on_device(device, height, width, ttnn_dtype, torch_dtype, ttnn_layout):
    # wherever possible `to_torch` will try to perform type conversion operations on device.
    # so the test must validate different input tensor origins
    ttnn_dtype_requires_tile = ttnn_dtype in [ttnn.bfloat8_b, ttnn.bfloat4_b]

    for store_input_on_device in [True, False]:
        for convert_with_device in [True, False]:
            ttnn_input_tensor = ttnn.rand(
                (height, width),
                dtype=ttnn_dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT if ttnn_dtype_requires_tile else ttnn_layout,
            )

            if not store_input_on_device:
                ttnn_input_tensor = ttnn.from_device(ttnn_input_tensor)

            torch_result_tensor = ttnn.to_torch(
                ttnn_input_tensor, dtype=torch_dtype, device=device if convert_with_device else None
            )
            assert (
                torch_result_tensor.dtype == torch_dtype
            ), f"Expected result {torch_dtype}, got result tensor {torch_result_tensor.dtype} when converting TTNN tensor {ttnn_input_tensor.dtype}"

    for convert_on_device in [True, False]:
        if not convert_on_device and ttnn_dtype_requires_tile:
            ttnn_dtype = ttnn.float32

        torch_input_tensor = torch.rand((height, width), dtype=torch_dtype)
        ttnn_result_tensor = ttnn.from_torch(
            torch_input_tensor,
            device=device if convert_on_device else None,
            dtype=ttnn_dtype,
            layout=ttnn.TILE_LAYOUT if ttnn_dtype_requires_tile else ttnn.ROW_MAJOR_LAYOUT,
        )

        assert (
            ttnn_result_tensor.dtype == ttnn_dtype
        ), f"Expected result {ttnn_dtype}, got result tensor {ttnn_result_tensor.dtype} when converting torch tensor {torch_input_tensor.dtype}"
