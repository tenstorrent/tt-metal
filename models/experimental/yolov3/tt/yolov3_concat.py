# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch

from models.experimental.yolov3.reference.models.common import autopad
import ttnn
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class TtConcat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, device, state_dict, base_address, dimension=1):
        super().__init__()
        self.device = device
        self.base_address = base_address

        self.d = dimension

    def forward(self, x):
        return ttnn.concat(x, self.d)
