# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shapes",
    [
        [[], [], []],
    ],
)
def test_div_pcc_fails_bcast(device, shapes):
    torch.manual_seed(0)
    torch_input_tensor_a = torch.tensor(shapes[0], dtype=torch.bfloat16)
    torch_input_tensor_b = torch.tensor(shapes[1], dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a / torch_input_tensor_b
    ttnn.set_printoptions(profile="full")
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    output_tensor = ttnn.divide(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor).reshape((-1))
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize(
    "shapes",
    [
        [[0, 1], 1.0],
    ],
)
def test_div_pcc_fails_bcast_scalar(device, shapes):
    torch.manual_seed(0)
    torch_input_tensor_a = torch.ones(shapes[0], dtype=torch.bfloat16)
    torch_input_tensor_b = torch.tensor(shapes[1])
    torch_output_tensor = torch_input_tensor_a / torch_input_tensor_b
    ttnn.set_printoptions(profile="full")
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = torch_input_tensor_b
    output_tensor = ttnn.divide(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


# @pytest.mark.parametrize(
#     "shapes",
#     [
#         # [[1, 23, 40, 1],  [128]],
#         # [[1, 23, 40],  [1, 1, 40]],
#         # [[1, 23, 40],  [1, 23, 1]],
#         # [[1, 512],  [1, 1]],
#         # [[2, 512],  [2, 1]],
#         # [[3, 320, 320],  [3, 1, 1]],
#         # [[3, 480, 640],  [3, 1, 1]],
#         # [[96, 80], [80]],
#         [[], [], []],
#     ],
# )
# def test_div_pcc_fails_bcast(device, shapes):
#     torch.manual_seed(0)

#     torch_input_tensor_a = torch.tensor(shapes[0], dtype=torch.bfloat16)
#     # print("\nTorch Input a: \n", torch_input_tensor_a)
#     torch_input_tensor_b = torch.tensor(shapes[1], dtype=torch.bfloat16)
#     # print("\nTorch Input b: \n", torch_input_tensor_b)
#     torch_output_tensor = torch_input_tensor_a / torch_input_tensor_b
#     ttnn.set_printoptions(profile="full")
#     input_tensor_a = ttnn.from_torch(
#         torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
#     )
#     # input_tensor_a = ttnn.from_torch(
#     #     torch_input_tensor_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
#     # )
#     # print("\nTT Input a: \n", input_tensor_a)
#     # print("\nTT Input a: \n",  ttnn.to_torch(input_tensor_a))
#     input_tensor_b = ttnn.from_torch(
#         torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
#     )
#     # input_tensor_b = ttnn.from_torch(
#     #     torch_input_tensor_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
#     # )
#     # print("\nTT Input b: \n", input_tensor_b)
#     # print("\nTT Input b: \n",  ttnn.to_torch(input_tensor_b))
#     output_tensor = ttnn.divide(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
#     output_tensor = ttnn.to_torch(output_tensor).reshape((-1))
#     assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


# @pytest.mark.parametrize(
#     "shapes",
#     [
#         [[0, 1], 1.0],
#     ],
# )
# def test_div_pcc_fails_bcast_scalar(device, shapes):
#     torch.manual_seed(0)

#     torch_input_tensor_a = torch.ones(shapes[0], dtype=torch.bfloat16)
#     # print("\nTorch Input a: \n", torch_input_tensor_a)
#     torch_input_tensor_b = torch.tensor(shapes[1])
#     # print("\nTorch Input b: \n", torch_input_tensor_b)
#     torch_output_tensor = torch_input_tensor_a / torch_input_tensor_b
#     ttnn.set_printoptions(profile="full")
#     input_tensor_a = ttnn.from_torch(
#         torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
#     )
#     # input_tensor_a = ttnn.from_torch(
#     #     torch_input_tensor_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
#     # )
#     # print("\nTT Input a: \n", input_tensor_a)
#     # print("\nTT Input a: \n",  ttnn.to_torch(input_tensor_a))
#     input_tensor_b = torch_input_tensor_b
#     # input_tensor_b = ttnn.from_torch(
#     #     torch_input_tensor_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
#     # )
#     # print("\nTT Input b: \n", input_tensor_b)
#     # print("\nTT Input b: \n",  ttnn.to_torch(input_tensor_b))
#     output_tensor = ttnn.divide(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
#     output_tensor = ttnn.to_torch(output_tensor)
#     assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize(
    "data",
    [
        [[], [], []],
    ],
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_01_volume_tensors_add(device, data, memory_config):
    (a, b, c_golden) = data
    a = torch.BFloat16Tensor(a)
    b = torch.BFloat16Tensor(b)
    assert torch.div(a, b).tolist() == c_golden

    ttnn_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_c = ttnn.divide(ttnn_a, ttnn_b)
    c = ttnn.to_torch(ttnn_c).reshape((-1))
    print(c)

    assert c.tolist() == c_golden


@pytest.mark.parametrize(
    "data",
    [
        [[1, 12, 7, 7], [], []],
        [[1, 16, 1, 6], [], []],
    ],
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_01_volume_tensors_add_ne_empty(device, data, memory_config):
    (a, b, c_golden) = data
    print("data[0] ", data[0])
    print("data[1] ", data[1])
    # a = torch.rand(data[0], dtype=torch.bfloat16)
    a = torch.BFloat16Tensor(a)
    b = torch.BFloat16Tensor(b)
    assert torch.div(a, b).tolist() == c_golden

    ttnn_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_c = ttnn.divide(ttnn_a, ttnn_b)
    c = ttnn.to_torch(ttnn_c).reshape((-1))

    assert c.tolist() == c_golden
