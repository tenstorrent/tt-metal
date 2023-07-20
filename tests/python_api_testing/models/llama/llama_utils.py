import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np
from loguru import logger
import json
import tt_lib


def torch2tt_tensor(py_tensor: torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    if size[-1] % 2 != 0:
        tt_device = tt_lib.device.GetHost()

    tt_tensor = tt_lib.tensor.Tensor(
        py_tensor.reshape(-1).tolist(),
        size,
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    ).to(tt_device)

    return tt_tensor


def tt2torch_tensor(tt_tensor):
    host = tt_lib.device.GetHost()
    tt_output = tt_tensor.to(host)
    if tt_output.layout() != tt_lib.tensor.Layout.ROW_MAJOR:
        tt_output = tt_output.to(tt_lib.tensor.Layout.ROW_MAJOR)
    dtype = {
        tt_lib.tensor.DataType.FLOAT32:   torch.float,
        tt_lib.tensor.DataType.BFLOAT16:  torch.bfloat16,
        tt_lib.tensor.DataType.BFLOAT8_B: torch.float,
    }[tt_tensor.dtype()]

    py_output = torch.frombuffer(tt_output.data(), dtype=dtype).reshape(tt_output.shape())
    return py_output


def linear(x, weight, bias=None):
    weight = tt_lib.tensor.transpose(weight)
    x = tt_lib.tensor.matmul(x, weight)
    if bias is not None:
        x = tt_lib.tensor.bcast(
            x, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
        )
    return x


def read_model_config(json_file):
    # read file
    with open(json_file, "r") as myfile:
        data = myfile.read()

    # parse file
    obj = json.loads(data)
    return obj


def print_corr_coef(x: torch.Tensor, y: torch.Tensor):
    x = torch.reshape(x, (-1,))
    y = torch.reshape(y, (-1,))

    input = torch.stack((x, y))

    corrval = torch.corrcoef(input)
    print(f"Corr coef:")
    print(f"{corrval}")
