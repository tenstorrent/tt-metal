# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def gelu_backward(grad: torch.Tensor, self: torch.Tensor, approximate: str = "none"):
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    if approximate == "tanh":
        kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
        kKappa = 0.044715
        x_sq = self * self
        x_cube = x_sq * self
        inner = kBeta * (self + kKappa * x_cube)
        tanh_inner = torch.tanh(inner)

        left = 0.5 * self
        right = 1 + tanh_inner

        left_derivative = 0.5 * right

        tanh_derivative = 1 - tanh_inner * tanh_inner
        inner_derivative = kBeta * (1 + 3 * kKappa * x_sq)
        right_derivative = left * tanh_derivative * inner_derivative

        return tanh_inner
        # return grad * (left_derivative + right_derivative)


@pytest.mark.parametrize(
    "shapes",
    [
        [[4, 2, 96, 192], [4, 2, 96, 192]],
    ],
)
def test_case3(device, shapes):
    torch.manual_seed(4378657)
    torch.set_printoptions(linewidth=200, threshold=10000, precision=10, sci_mode=False, edgeitems=17)

    high = 100
    low = -100
    in_data = torch.rand(shapes[0], requires_grad=True, dtype=torch.bfloat16) * (high - low) + low
    grad_data = torch.rand(shapes[1], requires_grad=False, dtype=torch.bfloat16) * (high - low) + low

    input_tensor = ttnn.from_torch(
        in_data, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    grad_tensor = ttnn.from_torch(
        grad_data, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    scalar = torch.tensor(1, dtype=torch.bfloat16).uniform_(-100, 100).item()
    # scalar = -10
    in_data1 = in_data + scalar

    # use torch implentation from derivatives.yaml to get output
    torch_output_tensor = gelu_backward(grad_data, in_data1, approximate="tanh")

    # use golden fn to get output
    # golden_function = ttnn.get_golden_function(ttnn.bias_gelu_bw)
    # torch_golden = golden_function(grad_data, in_data, scalar, value="tanh")[0]

    # ttnn output
    output_tensor = ttnn.bias_gelu_bw(grad_tensor, input_tensor, scalar, approximate="tanh")
    print("scalar", scalar)
    # print("torch_golden", torch_golden[0], torch_golden.shape)
    # torch_output_tensor[torch_output_tensor == -0.0] = 0.0

    # print("torch_output_tensor", torch_output_tensor)
    # output_tensor_rm = output_tensor[0].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    # print("output_tensor", output_tensor_rm)

    # diff = torch_output_tensor - output_tensor_rm
    # print("diff ",  (diff == 0).all())
    # print(diff, diff.min(), diff.max())

    output_tensor = ttnn.to_torch(output_tensor[0])

    # for i in range(4):            # Batch size
    #     for j in range(2):        # Channels
    #         for k in range(96):   # Height
    #             for l in range(192):  # Width
    #                 print(f"{i}-{j}-{k}-{l} input: {in_data1[i][j][k][l]} \t tt: {output_tensor[i][j][k][l]} \t torch: {torch_output_tensor[i][j][k][l]} \n")

    # assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999
    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)


@pytest.mark.parametrize(
    "shapes",
    [
        [[97, 129], [97, 129]],
    ],
)
def test_case4(device, shapes):
    torch.manual_seed(7580522)

    high = 100
    low = -100
    in_data = torch.rand(shapes[0], requires_grad=True, dtype=torch.bfloat16) * (high - low) + low
    grad_data = torch.rand(shapes[1], requires_grad=False, dtype=torch.bfloat16) * (high - low) + low

    input_tensor = ttnn.from_torch(
        in_data, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    grad_tensor = ttnn.from_torch(
        grad_data, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    scalar = torch.tensor(1, dtype=torch.bfloat16).uniform_(-100, 100).item()  # scalar -97.5
    in_data1 = in_data + scalar

    # use torch implentation from derivatives.yaml to get output
    torch_output_tensor = gelu_backward(grad_data, in_data1, approximate="tanh")

    # use golden fn to get output
    golden_function = ttnn.get_golden_function(ttnn.bias_gelu_bw)
    torch_golden = golden_function(grad_data, in_data, scalar, value="tanh")[0]

    # ttnn output
    output_tensor = ttnn.bias_gelu_bw(grad_tensor, input_tensor, scalar, approximate="tanh")
    print("scalar", scalar)
    print("torch_golden", torch_golden[0], torch_golden.shape)
    print("torch_output_tensor", torch_output_tensor)
    output_tensor_rm = output_tensor[0].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    print("output_tensor", output_tensor_rm)
    diff = torch_golden - torch_output_tensor
    # print("diff ",  (diff == 0).all())
    # print(diff)
    output_tensor = ttnn.to_torch(output_tensor[0])

    # assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999
    assert_with_pcc(torch_output_tensor, output_tensor, 0.998)


@pytest.mark.parametrize(
    "shapes",
    [
        [[1, 1, 32, 32]],
    ],
)
def test_add_float(device, shapes):
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    torch_input_tensor_a = torch.ones(shapes[0], dtype=torch.bfloat16)
    torch_input_tensor_b = 1.41421356237309504880
    torch_output_tensor = torch.mul(torch_input_tensor_a, torch_input_tensor_b)
    print("torch_output_tensor", torch_output_tensor)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = torch_input_tensor_b
    output_tensor = ttnn.mul(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)
    print("output_tensor", output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


@pytest.mark.parametrize(
    "shapes",
    [
        [[1, 1, 32, 32]],
    ],
)
def test_add_float_fp32(device, shapes):
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    torch_input_tensor_a = torch.ones(shapes[0], dtype=torch.float32)
    torch_input_tensor_b = 1.41421356237309504880
    torch_output_tensor = torch.mul(torch_input_tensor_a, torch_input_tensor_b)
    print("torch_output_tensor", torch_output_tensor)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = torch_input_tensor_b
    output_tensor = ttnn.mul(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)
    print("output_tensor", output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
