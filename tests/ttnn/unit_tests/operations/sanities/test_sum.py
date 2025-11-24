import pytest
import torch
import ttnn
import torch.nn as nn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [204])
@pytest.mark.parametrize("w", [768])
@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("input_gen_method", [1, 2])
def test_sum(device, batch_size, h, w, dim, input_gen_method):
    torch.manual_seed(0)

    if input_gen_method == 1:
        torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    elif input_gen_method == 2:
        torch_input_tensor = torch.rand(1, h, w)

    print("torch_input_tensor.dtype", torch_input_tensor.dtype)

    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=True)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=True)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    print("output_tensor", output_tensor.shape)

    # import numpy as np
    # torch.set_printoptions(linewidth=1000,edgeitems=10,precision =20)

    # print("torch_input_tensor",torch_input_tensor,"\n")

    # print("torch_output_tensor.shape",torch_output_tensor.shape)
    # print("output_tensor.shape",output_tensor.shape)

    # print("=========================================")
    # print("torch_output_tensor\n",torch_output_tensor)
    # print("\noutput_tensor\n",output_tensor)
    # print("=========================================")

    assert_with_pcc(torch_output_tensor, output_tensor)
