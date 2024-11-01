# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shape",
    [
        # [1, 4],
        [10, 10],
        # [8]
        # [100, 100]
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        3,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.Layout.ROW_MAJOR],
)
def test_fill_rm(device, input_shape, fill_value, layout):
    torch_input_tensor = torch.rand((input_shape), dtype=torch.bfloat16)
    torch_output_tensor = torch.full_like(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.fill(input_tensor, fill_value=fill_value)
    ttnn.set_printoptions(profile="full")
    print("output_tensor", output_tensor)
    # assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [-5, 3, 15, 25],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.Layout.TILE],
)
def test_fill_tile(device, input_shape, fill_value, layout):
    torch_input_tensor = torch.rand((input_shape), dtype=torch.bfloat16)
    torch_output_tensor = torch.full_like(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.fill(input_tensor, fill_value=fill_value)
    # assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)
