# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@skip_for_wormhole_b0()
def test_focus_concat_tile(device, async_mode):
    device.enable_async(async_mode)
    torch_input_tensor_a = torch.rand((1, 1, 320 * 320, 3), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((1, 1, 320 * 320, 3), dtype=torch.bfloat16)
    torch_input_tensor_c = torch.rand((1, 1, 320 * 320, 3), dtype=torch.bfloat16)
    torch_input_tensor_d = torch.rand((1, 1, 320 * 320, 3), dtype=torch.bfloat16)
    torch_output_tensor = torch.concat(
        [torch_input_tensor_a, torch_input_tensor_b, torch_input_tensor_c, torch_input_tensor_d], dim=3
    )

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_c = ttnn.from_torch(torch_input_tensor_c, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_d = ttnn.from_torch(torch_input_tensor_d, layout=ttnn.TILE_LAYOUT, device=device)

    # if ttnn.has_tile_padding(input_tensor_a, dim=dim) or ttnn.has_tile_padding(input_tensor_b, dim=3):
    #     pytest.skip("Cannot concat tensors with tile padding")
    print(input_tensor_a.shape, input_tensor_b.shape, input_tensor_c.shape, input_tensor_d.shape)
    output = ttnn.concat([input_tensor_a, input_tensor_b, input_tensor_c, input_tensor_d], dim=3)
    print(output.shape)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@skip_for_wormhole_b0()
def test_focus_concat_row_major(device, async_mode):
    device.enable_async(async_mode)
    torch_input_tensor_a = torch.rand((1, 1, 320 * 320, 3), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((1, 1, 320 * 320, 3), dtype=torch.bfloat16)
    torch_input_tensor_c = torch.rand((1, 1, 320 * 320, 3), dtype=torch.bfloat16)
    torch_input_tensor_d = torch.rand((1, 1, 320 * 320, 3), dtype=torch.bfloat16)
    torch_output_tensor = torch.concat(
        [torch_input_tensor_a, torch_input_tensor_b, torch_input_tensor_c, torch_input_tensor_d], dim=3
    )

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor_c = ttnn.from_torch(torch_input_tensor_c, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor_d = ttnn.from_torch(torch_input_tensor_d, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # if ttnn.has_tile_padding(input_tensor_a, dim=dim) or ttnn.has_tile_padding(input_tensor_b, dim=3):
    #     pytest.skip("Cannot concat tensors with tile padding")

    output = ttnn.concat([input_tensor_a, input_tensor_b, input_tensor_c, input_tensor_d], dim=3)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
