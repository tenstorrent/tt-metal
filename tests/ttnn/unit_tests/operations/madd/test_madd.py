# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# import pytest
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


# @pytest.mark.parametrize("size", [64, 1, 0])
def test_madd_tile(device):
    torch.manual_seed(0)

    dim = 32
    torch_a = torch.rand((dim, dim), dtype=torch.bfloat16)
    torch_b = torch.rand((dim, dim), dtype=torch.bfloat16)
    torch_c = torch.rand((dim, dim), dtype=torch.bfloat16)
    torch_output_tensor = torch_a * torch_b  # + torch_c, add not implemented yet

    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_c = ttnn.from_torch(torch_c, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.madd(ttnn_a, ttnn_b, ttnn_c)

    assert output_tensor.shape == (dim, dim)
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
