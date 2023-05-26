import math
from pathlib import Path
import sys
import numpy as np

import torch
import torch.nn as nn
from loguru import logger

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
    py_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())
    return py_output
