# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((1, 1, 1, 256), 2),
        ((1, 1, 1, 256), -1),
        ((1, 1, 1, 30), 2),
        ((1, 1, 1, 30), -1),
        ((1, 32, 16), 0),
        ((1, 1, 24576), 0),
        ((1, 19), 0),
        ((19), 0),
        ((1), 0),
        ((), 0),
        ((), -1),
        ((1, 1, 480, 640), 1),
        ((3, 1370, 1, 1, 1280), -2),
        ((3, 197, 1, 1, 1024), -2),
        ((3, 197, 1, 1, 768), -2),
        ((3, 50, 1, 1, 1024), -2),
        ((3, 50, 1, 1, 768), -2),
    ],
)
def test_squeeze(device, input_shape, dim):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_squeeze_tensor = torch.squeeze(torch_input_tensor, dim)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.squeeze(input_tensor, dim)
    torch_output_tensor = ttnn.to_torch(ttnn_output)
    assert torch.allclose(torch_output_tensor, torch_squeeze_tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 1, 256),
        (1, 32, 16),
        (1, 1, 480, 640),
        (3, 1, 1, 1, 1280),
        (),
        (1,),
        (1, 2),
    ],
)
def test_squeeze_default(device, input_shape):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_squeeze_tensor = torch.squeeze(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.squeeze(input_tensor)
    torch_output_tensor = ttnn.to_torch(ttnn_output)
    assert torch.allclose(torch_output_tensor, torch_squeeze_tensor)


@pytest.mark.parametrize(
    "input_shape, dims",
    [
        ((1, 1, 1, 256), [0, 1]),
        ((1, 1, 480, 640), [0, 1]),
        ((3, 1, 1, 1, 1280), [1, 2, 3]),
        ((1, 1, 1, 256), [-4, -3]),
        ((1, 1, 1, 256), []),
        ((1,), [0]),
        ((1, 2), [0]),
        ((), []),
        ((), [-1]),
    ],
)
def test_squeeze_multiple_dims(device, input_shape, dims):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_squeeze_tensor = torch.squeeze(torch_input_tensor, list(dims))
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor = ttnn.squeeze(input_tensor, list(dims))
    torch_output_tensor = ttnn.to_torch(input_tensor)
    assert torch.allclose(torch_output_tensor, torch_squeeze_tensor)


@pytest.mark.parametrize(
    "input_shape, dims, expected_exception",
    [
        ((1, 1, 1, 256), [4], RuntimeError),  # Out of range positive index
        ((1, 1, 1, 256), [-5], RuntimeError),  # Out of range negative index
        ((1, 1, 1, 256), [0, 0], RuntimeError),  # Duplicate indices
        ((1, 1, 1, 256), [0, -4], RuntimeError),  # Duplicate indices (positive and negative)
    ],
)
def test_squeeze_error_cases(device, input_shape, dims, expected_exception):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with pytest.raises(expected_exception):
        ttnn.squeeze(input_tensor, dims)
