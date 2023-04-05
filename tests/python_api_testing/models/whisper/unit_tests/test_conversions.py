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


def test_conversions():
    X = 32
    Y = 32
    Z = 128
    torch_tensor = torch.randn(1, X, Y, Z)
    ttm_tensor = torch_to_ttm_tensor(torch_tensor, [1, X, Y, Z], device)
    ttm_to_torch = ttm_to_torch_tensor(ttm_tensor, host)
    rtol = 0.01
    atol = 0.01
    print(ttm_tensor.shape())
    print(ttm_to_torch.size())
    print(torch_tensor.size())
    # fails for
    #rtol=1e-03 atol=1e-03
    assert np.allclose(torch_tensor.detach().numpy(), ttm_to_torch.numpy(), rtol, atol)


if __name__ == "__main__":
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    test_conversions()
    ttm.device.CloseDevice(device)
