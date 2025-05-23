import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_add_4D_tensors_oft(device):
    torch_input_tensor_a = torch.rand((1, 256, 3, 25281), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((1, 256, 3, 25281), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


def test_sub_4D_oft(device):
    torch_input_tensor_a = torch.rand((1, 256, 3, 25281), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((1, 256, 3, 25281), dtype=torch.bfloat16)
    torch_output_tensor = torch.sub(torch_input_tensor_b, torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.subtract(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


def test_div_4D_tensors_oft(device):
    torch_input_tensor_a = torch.rand((1, 256, 3, 25281), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((1, 1, 3, 25281), dtype=torch.bfloat16)
    torch_output_tensor = torch.div(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.div(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


def test_multiply_4D_tensors_oft(device):
    torch_input_tensor_a = torch.rand((1, 256, 3, 25281), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((1, 1, 3, 25281), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a * torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.multiply(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
