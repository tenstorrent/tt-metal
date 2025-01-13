# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import json
import numpy as np
import ttnn
from models.utility_functions import (
    pad_activation,
    pad_weight,
    tilize_to_list,
    untilize,
    nearest_32,
    print_diff_argmax,
    tt2torch,
    tt2torch_rm,
)


def calculate_shape(input_tensor_shape):
    if len(input_tensor_shape) == 4:
        s1 = input_tensor_shape[0]
        s2 = input_tensor_shape[1]
        s3 = input_tensor_shape[2]
        s4 = input_tensor_shape[3]
    if len(input_tensor_shape) == 3:
        s1 = 1
        s2 = input_tensor_shape[0]
        s3 = input_tensor_shape[1]
        s4 = input_tensor_shape[2]
    if len(input_tensor_shape) == 2:
        s1 = 1
        s2 = 1
        s3 = input_tensor_shape[0]
        s4 = input_tensor_shape[1]

    if s3 % 32 != 0:
        diff = s3 % 32
        s3 = s3 + (32 - diff)

    if s4 % 32 != 0:
        diff = s4 % 32
        s4 = s4 + (32 - diff)

    padded_shape = [s1, s2, s3, s4]
    return padded_shape


def create_padded_tensor(
    input_tensors_shape,
    input_tensor,
    output_tensor_shape,
    pad_value,
    device,
    input_tensor_start=[0, 0, 0, 0],
):
    while len(input_tensors_shape) < 4:
        input_tensors_shape.insert(0, 1)

    if isinstance(input_tensor, ttnn.Tensor):
        torch_tensor = input_tensor.to_torch()
    else:
        torch_tensor = input_tensor

    # Create tensor on host
    a = ttnn.Tensor(
        torch_tensor.reshape(-1).tolist(),
        input_tensors_shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    # Pad inputs on host
    a_pad = a.pad(output_tensor_shape, input_tensor_start, pad_value)
    a_dev = a_pad.to(ttnn.TILE_LAYOUT).to(device)

    return a_dev


def create_unpadded_tensor(ttm_tensor, input_tensors_shape, input_tensor_start=[0, 0, 0, 0]):
    output_tensor_start = input_tensor_start
    output_tensor_end = tuple(input_tensor_start[i] + input_tensors_shape[i] for i in range(len(input_tensors_shape)))
    ttm_tensor = ttm_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad(output_tensor_start, output_tensor_end)

    return ttm_tensor


def torch2tt_tensor(py_tensor: torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = (
        ttnn.Tensor(
            py_tensor.reshape(-1).tolist(),
            size,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(tt_device)
    )

    return tt_tensor


def tt2torch_tensor(tt_tensor):
    tt_output = tt_tensor.cpu()
    if tt_output.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        tt_output = tt_output.to(ttnn.ROW_MAJOR_LAYOUT)
    return tt_output.to_torch()


def tt_const_tensor(value, shape, device):
    pytorch_const = torch.full(shape, value)
    tt_const = torch2tt_tensor(pytorch_const, device)
    return tt_const


def create_padded_tensor(
    input_tensors_shape,
    input_tensor,
    output_tensor_shape,
    pad_value,
    device,
    input_tensor_start=[0, 0, 0, 0],
):
    while len(input_tensors_shape) < 4:
        input_tensors_shape.insert(0, 1)

    if isinstance(input_tensor, ttnn.Tensor):
        torch_tensor = input_tensor.to_torch()
    else:
        torch_tensor = input_tensor

    # Create tensor on host
    a = ttnn.Tensor(
        torch_tensor.reshape(-1).tolist(),
        input_tensors_shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )

    # Pad inputs on host
    a_pad = a.pad(output_tensor_shape, input_tensor_start, pad_value)

    a_dev = a_pad.to(ttnn.TILE_LAYOUT).to(device)

    return a_dev


def closestNumberDivisibleByTileSize(n):
    if n % 32 == 0:
        return n

    q = int(n / 32)

    num = 32 * (q + 1)
    return num


def tt_load_layer_weights(layer_name, state_dict, device):
    weights = state_dict[layer_name]
    input_shape = list(weights.shape)

    while len(input_shape) < 4:
        input_shape.insert(0, 1)

    d0 = input_shape[0]
    d1 = input_shape[1]
    d2 = closestNumberDivisibleByTileSize(input_shape[2])
    d3 = closestNumberDivisibleByTileSize(input_shape[3])

    # print(f"Weights shape {[d0, d1, d2, d3]}")

    weights = create_padded_tensor(input_shape, weights, [d0, d1, d2, d3], 0, device)
    return weights


def pt_load_layer_weights(layer_name, state_dict):
    weights = torch.nn.Parameter(torch.tensor(state_dict[layer_name]))
    return weights


def read_model_config(json_file):
    # read file
    with open(json_file, "r") as myfile:
        data = myfile.read()

    # parse file
    obj = json.loads(data)
    return obj


def print_corr_coef(x, y):
    x = torch.reshape(x, (-1,))
    y = torch.reshape(y, (-1,))

    input = torch.stack((x, y))

    corrval = torch.corrcoef(input)
    print(f"Corr coef:")
    print(f"{corrval}")
