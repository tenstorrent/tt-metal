import math
from pathlib import Path
import sys
import numpy as np

import torch
import torch.nn as nn

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


def ttm_to_torch_tensor(tt_tensor: ttm.tensor.Tensor) -> torch.Tensor:
    host = ttm.device.GetHost()

    # move TT Tensor output from TT accelerator device to host
    # and then on host, change memory layout of TT Tensor to ROW_MAJOR
    tt_output = tt_tensor.to(host).to(ttm.tensor.Layout.ROW_MAJOR)

    # create a 1D PyTorch tensor from values in TT Tensor obtained with data() member function
    # and then reshape PyTorch tensor to shape of TT Tensor
    py_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())

    return py_output

def torch_to_ttm_tensor(torch_tensor: torch.Tensor, shape:list, device: ttm.device.Device) -> ttm.tensor.Tensor:

    tt_tensor = (
        ttm.tensor.Tensor(
            torch_tensor.reshape(-1).tolist(), # PyTorch tensor flatten into a list of floats
            shape,                             # shape of TT Tensor that will be created
            ttm.tensor.DataType.BFLOAT16,      # data type that will be used in created TT Tensor
            ttm.tensor.Layout.ROW_MAJOR,       # memory layout that will be used in created TT Tensor
        )
        .to(ttm.tensor.Layout.TILE)            # change memory layout of TT Tensor to TILE (as operation that will use it expects TILE layout)
        .to(device)                            # move TT Tensor from host to TT accelerator device (device is of type tt_lib.device.Device)
    )
    return tt_tensor

def np_compare_tensors(torch_tensor: torch.Tensor, ttm_tensor: ttm.tensor.Tensor, squeeze:bool = True, rtol:float=0.1, atol:float=0.1):
    print("Torch tensor output size")
    print(torch_tensor.size())

    print("TT Metal tensor output size")
    print(ttm_tensor.shape())

    tt_out_to_torch = ttm_to_torch_tensor(ttm_tensor)

    print("TT Metal to Torch size")
    print(tt_out_to_torch.size())

    if squeeze:
        tt_out_to_torch = torch.squeeze(tt_out_to_torch, 0)
        print("TT Metal to Torch tensor after torch.squeeze size")
        print(tt_out_to_torch.size())

    print("Torch sample data")
    print(torch_tensor.detach().numpy()[0])
    print("TTMetal sample data")
    print(tt_out_to_torch.numpy()[0])

    assert np.allclose(torch_tensor.detach().numpy(), tt_out_to_torch.numpy(), rtol, atol)

def print_corr_coef(x: torch.Tensor, y: torch.Tensor):
    x = torch.reshape(x, (-1, ))
    y = torch.reshape(y, (-1, ))

    input = torch.stack((x, y))

    corrval = torch.corrcoef(input)
    print(f"Corr coef:")
    print(f"{corrval}")
