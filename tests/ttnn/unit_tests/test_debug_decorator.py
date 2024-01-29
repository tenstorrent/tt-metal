# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from ttnn.decorators import override_pearson_correlation_coefficient
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.utility_functions import torch_random


@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_softmax(device, ttnn_enable_debug_decorator, batch_size, h, w, dim, input_layout):
    if dim != -1 and input_layout != ttnn.TILE_LAYOUT:
        pytest.skip("Not supported yet")

    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.softmax(torch_input_tensor, dim=dim, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, input_layout)

    input_tensor = ttnn.to_device(input_tensor, device)
    with override_pearson_correlation_coefficient(0.99):
        output_tensor = ttnn.softmax(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_exp(device, ttnn_enable_debug_decorator, batch_size, h, w, input_layout):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.exp(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, input_layout)

    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.exp(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)
