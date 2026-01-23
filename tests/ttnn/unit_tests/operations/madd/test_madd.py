# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch


# @pytest.mark.parametrize("size", [64, 1, 0])
# def test_madd_random(device, size):
#     torch.manual_seed(0)

#     torch_input_tensor = torch.rand((size,), dtype=torch.bfloat16)
#     torch_output_tensor = torch_input_tensor + 1

#     input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
#     output_tensor = ttnn.madd(input_tensor, 1, 0)

#     assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
#     assert output_tensor.shape == (size,)


# @pytest.mark.parametrize("dim", [1024, 128, 64, 32])
# @pytest.mark.parametrize("dim", [64])
# def test_madd_tile(device, dim):
def test_madd_tile(device):
    device.disable_and_clear_program_cache()

    torch.manual_seed(0)

    dim1 = 32
    dim2 = 64
    shape = (dim1, dim2)

    # torch_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_a = torch.ones(shape, dtype=torch.bfloat16)
    torch_b = torch.zeros(shape, dtype=torch.bfloat16)
    # torch_b = torch.rand(shape, dtype=torch.bfloat16)
    # torch_c = torch.rand(shape, dtype=torch.bfloat16)
    torch_c = torch.zeros(shape, dtype=torch.bfloat16) + 2
    torch_output_tensor = torch_a * torch_b  # + torch_c, add not implemented yet

    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_c = ttnn.from_torch(torch_c, layout=ttnn.TILE_LAYOUT, device=device)

    print(ttnn_a)
    print(ttnn_b)
    print(ttnn_c)

    output_tensor = ttnn.madd(ttnn_a, ttnn_b, ttnn_c)
    print(output_tensor)

    assert output_tensor.shape == shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
