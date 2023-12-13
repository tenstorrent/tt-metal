# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.utility_functions import torch_random


@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_mean(device, batch_size, h, w, dim, input_layout):
    if dim != -1 and input_layout != ttnn.TILE_LAYOUT:
        pytest.skip("Not supported yet")

    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=True, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, input_layout)

    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=True)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    if dim == -1:
        assert output_tensor.shape == (batch_size, h, 1)
        assert output_tensor.shape.padded() == (batch_size, h, ttnn.tensor.TILE_SIZE)
    elif dim == -2:
        assert output_tensor.shape == (batch_size, 1, w)
        assert output_tensor.shape.padded() == (batch_size, ttnn.tensor.TILE_SIZE, w)
    else:
        raise RuntimeError("Unsupported dim")

    output_tensor = ttnn.to_torch(output_tensor)
    if dim == -1:
        output_tensor = output_tensor[..., :1]
    elif dim == -2:
        output_tensor = output_tensor[..., :1, :]
    else:
        raise RuntimeError("Unsupported dim")

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
