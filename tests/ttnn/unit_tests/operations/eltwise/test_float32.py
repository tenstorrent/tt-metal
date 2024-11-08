import pytest

import torch

import ttnn
from models.utility_functions import is_grayskull

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random

from loguru import logger


# https://github.com/tenstorrent/tt-metal/issues/14862
def test_unary_float32(device):
    torch.manual_seed(0)

    torch.set_printoptions(precision=10)

    torch_input_tensor = torch.tensor([[0.00001]], dtype=torch.float32)
    torch_output_tensor = -torch_input_tensor

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    import os

    print(f"Debugging PID: {os.getpid()}")
    ttnn_output_tensor = ttnn.neg(ttnn_input_tensor)

    print(torch_input_tensor, ttnn.to_torch(ttnn_input_tensor))
    print(torch_output_tensor, ttnn.to_torch(ttnn_output_tensor))

    output_tensor = ttnn.to_torch(ttnn_output_tensor)

    assert True == torch.allclose(torch_output_tensor, output_tensor, atol=1e-8, rtol=1e-5, equal_nan=False)


# https://github.com/tenstorrent/tt-metal/issues/14825
def test_binary_float32(device):
    torch.manual_seed(0)

    torch.set_printoptions(precision=10)

    torch_input_tensor_a = torch.tensor([[1]], dtype=torch.float32)
    torch_input_tensor_b = torch.tensor([[0.00030171126]], dtype=torch.float32)
    torch_output_tensor = torch_input_tensor_a - torch_input_tensor_b
    ttnn_input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    expected_ttnn_output_tensor = ttnn.from_torch(
        torch_output_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    import os

    print(f"Debugging PID: {os.getpid()}")
    actual_ttnn_output_tensor = ttnn.subtract(ttnn_input_tensor_a, ttnn_input_tensor_b)
    actual_ttnn_output_tensor = ttnn.to_torch(actual_ttnn_output_tensor)
    print(torch_output_tensor, ttnn.to_torch(expected_ttnn_output_tensor), actual_ttnn_output_tensor)

    assert True == torch.allclose(torch_output_tensor, actual_ttnn_output_tensor, atol=1e-6, rtol=1e-5, equal_nan=False)
