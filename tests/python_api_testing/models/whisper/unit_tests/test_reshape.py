import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Optional, Tuple, Union

from pymetal import ttlib as ttm
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, untilize


def ttm_to_torch_tensor(tt_tensor: ttm.tensor.Tensor, host: ttm.device.Host) -> torch.Tensor:
    shape = tt_tensor.shape()
    tt_tensor_out = tt_tensor.to(host)
    tt_out_tilized = torch.Tensor(tt_tensor_out.data())
    out_tensor_torch = untilize(tt_out_tilized.reshape(shape))

    return out_tensor_torch

def torch_to_ttm_tensor(torch_tensor: torch.Tensor, shape:list, device: ttm.device.Device ) -> ttm.tensor.Tensor:
    tt_tensor = tilize_to_list(pad_activation(torch_tensor))
    tt_tensor = ttm.tensor.Tensor(tt_tensor, shape, ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)
    return tt_tensor


def test_reshape():
    a = 1
    b = 1
    c = 64
    d = 128

    initial_shape = [a, b, c, d]
    shape_reshape = [b, a, d, c]

    torch_tensor = torch.randn(initial_shape)

    ttm_tensor = torch_to_ttm_tensor(torch_tensor, initial_shape, device)

    print("Before reshape")
    print(torch_tensor.size())
    print(ttm_tensor.shape())

    torch_reshaped_tensor = torch.reshape(torch_tensor, shape_reshape)
    ttm_reshaped_tensor = ttm.tensor.reshape(ttm_tensor, *shape_reshape)

    print("After reshape")
    print(ttm_reshaped_tensor.shape())
    print(torch_reshaped_tensor.size())

    ttm_to_torch = ttm_to_torch_tensor(ttm_tensor, host)

    assert ttm_to_torch.size() == torch_reshaped_tensor.size()
    print("TTM SAMPLES")
    print(ttm_to_torch[0][0])
    print("TORCH SAMPLES")
    print(torch_reshaped_tensor[0][0])
    rtol = 0.1
    atol = 0.1

    assert np.allclose(torch_reshaped_tensor.detach().numpy(), ttm_to_torch.numpy(), rtol, atol)


if __name__ == "__main__":
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    test_reshape()
    ttm.device.CloseDevice(device)
