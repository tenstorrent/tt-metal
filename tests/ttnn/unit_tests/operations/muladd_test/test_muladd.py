import pytest

import torch

import ttnn
from ttnn.operations.muladd_test import muladd_test
from tests.ttnn.utils_for_testing import assert_with_ulp
from models.common.utility_functions import skip_for_slow_dispatch


@pytest.mark.parametrize("hw", [(1024, 1024)])
def test_add_2D_tensors(device, hw):
    torch_input_tensor_a = torch.rand(hw, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(hw, dtype=torch.bfloat16)
    torch_input_tensor_c = torch.rand(hw, dtype=torch.bfloat16)
    torch_intermediate_tensor = torch.mul(torch_input_tensor_a, torch_input_tensor_b)
    torch_output_tensor = torch.add(torch_intermediate_tensor, torch_input_tensor_c)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_c = ttnn.from_torch(torch_input_tensor_c, layout=ttnn.TILE_LAYOUT, device=device)
    output = muladd_test(input_tensor_a, input_tensor_b, input_tensor_c)
    output = ttnn.to_torch(output)

    assert_with_ulp(torch_output_tensor, output, ulp_threshold=1)
