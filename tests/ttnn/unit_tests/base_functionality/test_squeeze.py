# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize(
    "layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((), 0),
        ((), -1),
        ((1), 0),
        ((19), 0),
        ((1, 19), 0),
        ((1, 32, 16), 0),
        ((1, 1, 24576), 0),
        ((1, 1, 1, 30), 2),
        ((1, 1, 1, 256), 2),
        ((1, 1, 1, 30), -1),
        ((1, 1, 1, 256), -1),
        ((1, 1, 480, 640), 1),
        ((3, 50, 1, 1, 768), -2),
        ((3, 197, 1, 1, 768), -2),
        ((3, 50, 1, 1, 1024), -2),
        ((3, 197, 1, 1, 1024), -2),
        ((3, 1370, 1, 1, 1280), -2),
    ],
)
def test_squeeze(device, input_shape, dim, layout):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_squeeze_tensor = torch.squeeze(torch_input_tensor, dim)
    tt_input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    tt_squeeze_tensor = ttnn.squeeze(tt_input_tensor, dim)
    ttnn_squeeze_tensor = ttnn.to_torch(tt_squeeze_tensor)
    assert ttnn_squeeze_tensor.shape == torch_squeeze_tensor.shape
    assert torch.allclose(ttnn_squeeze_tensor, torch_squeeze_tensor)


@pytest.mark.parametrize(
    "layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        (),
        (1,),
        (1, 2),
        (1, 32, 16),
        (1, 1, 1, 256),
        (1, 1, 480, 640),
        (3, 1, 1, 1, 1280),
    ],
)
def test_squeeze_default(device, input_shape, layout):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_squeeze_tensor = torch.squeeze(torch_input_tensor)
    tt_input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    ttnn_squeeze_tensor = ttnn.squeeze(tt_input_tensor)
    ttnn_squeeze_tensor = ttnn.to_torch(ttnn_squeeze_tensor)
    assert ttnn_squeeze_tensor.shape == torch_squeeze_tensor.shape
    assert torch.allclose(ttnn_squeeze_tensor, torch_squeeze_tensor)


@pytest.mark.parametrize(
    "layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_shape, dims",
    [
        ((), []),
        ((), [-1]),
        ((1,), [0]),
        ((1, 2), [0]),
        ((1, 1, 1, 256), []),
        ((1, 1, 1, 256), [0, 1]),
        ((1, 1, 1, 256), [-4, -3]),
        ((1, 1, 480, 640), [0, 1]),
        ((3, 1, 1, 1, 1280), [1, 2, 3]),
    ],
)
def test_squeeze_multiple_dims(device, input_shape, dims, layout):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_squeeze_tensor = torch.squeeze(torch_input_tensor, list(dims))
    tt_input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    tt_squeeze_tensor = ttnn.squeeze(tt_input_tensor, list(dims))
    tt_squeeze_tensor = ttnn.to_torch(tt_squeeze_tensor)
    assert tt_squeeze_tensor.shape == torch_squeeze_tensor.shape
    assert torch.allclose(tt_squeeze_tensor, torch_squeeze_tensor)


@pytest.mark.parametrize(
    "layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_shape, dims, expected_exception",
    [
        ((1, 1, 1, 256), [4], RuntimeError),  # Out of range positive index
        ((1, 1, 1, 256), [-5], RuntimeError),  # Out of range negative index
        ((1, 1, 1, 256), [0, 0], RuntimeError),  # Duplicate indices
        ((1, 1, 1, 256), [0, -4], RuntimeError),  # Duplicate indices (positive and negative)
    ],
)
def test_squeeze_error_cases(device, input_shape, dims, expected_exception, layout):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    with pytest.raises(expected_exception):
        ttnn.squeeze(input_tensor, dims)


# This test verifies that padded_shape of tensor after squeezing is properly updated
@pytest.mark.parametrize(
    "layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_shape, dims",
    [
        ((1, 1, 1, 256), [0, 1]),
        ((1, 1, 480, 640), [0]),
        ((3, 1, 1, 1, 1280), [1, 2, 3]),
    ],
)
def test_squeeze_padded_shape_integrity(device, input_shape, dims, layout):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    tt_input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)

    tt_squeeze_tensor = ttnn.squeeze(tt_input_tensor, dims)
    new_tensor = ttnn.empty(tt_squeeze_tensor.shape, layout=layout, device=device, dtype=ttnn.bfloat16)
    assert tt_squeeze_tensor.padded_shape == new_tensor.padded_shape
