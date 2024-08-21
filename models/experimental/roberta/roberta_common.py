# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
import numpy as np

import torch
import torch.nn as nn
from loguru import logger

import ttnn


def torch2tt_tensor(py_tensor: torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = ttnn.Tensor(
        py_tensor.reshape(-1).tolist(),
        size,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if size[-1] % 2 == 0:
        tt_tensor = tt_tensor.to(tt_device)

    return tt_tensor


def tt2torch_tensor(tt_tensor):
    tt_output = tt_tensor.cpu()
    if tt_output.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        tt_output = tt_output.to(ttnn.ROW_MAJOR_LAYOUT)
    return tt_output.to_torch()
