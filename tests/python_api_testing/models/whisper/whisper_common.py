import math
from pathlib import Path
import sys
import numpy as np

import torch
import torch.nn as nn
from loguru import logger

from libs import tt_lib as ttm
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, untilize


def torch2tt_tensor(py_tensor: torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = ttm.tensor.Tensor(
        py_tensor.reshape(-1).tolist(),
        size,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.ROW_MAJOR,
    ).to(ttm.tensor.Layout.TILE).to(tt_device)

    return tt_tensor


def tt2torch_tensor(tt_tensor):
    host = ttm.device.GetHost()
    tt_output = tt_tensor.to(host).to(ttm.tensor.Layout.ROW_MAJOR)
    py_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())
    return py_output


def create_padded_tensor(input_tensors_shape, input_tensor, output_tensor_shape, pad_value, device, input_tensor_start=[0,0,0,0]):
    while len(input_tensors_shape) < 4:
        input_tensors_shape.insert(0, 1)

    if isinstance(input_tensor, ttm.tensor.Tensor):
        torch_tensor = torch.Tensor(input_tensor.data()).reshape(input_tensor.shape())
    else:
        torch_tensor = input_tensor

    # Create tensor on host
    a = ttm.tensor.Tensor(
        torch_tensor.reshape(-1).tolist(),
        input_tensors_shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.ROW_MAJOR,
    )
    # Pad inputs on host
    a_pad = a.pad(output_tensor_shape, input_tensor_start, pad_value)
    #a_pt = torch.Tensor(a_pad.data()).reshape(*output_tensor_shape)
    a_dev = a_pad.to(ttm.tensor.Layout.TILE).to(device)

    return a_dev


def create_unpadded_tensor(ttm_tensor, input_tensors_shape, input_tensor_start=[0,0,0,0]):
    output_tensor_start = input_tensor_start
    output_tensor_end = tuple(
        input_tensor_start[i] + input_tensors_shape[i] - 1
        for i in range(len(input_tensors_shape))
    )
    ttm_tensor = ttm_tensor.to(ttm.device.GetHost()).to(ttm.tensor.Layout.ROW_MAJOR).unpad(output_tensor_start, output_tensor_end)

    return ttm_tensor
