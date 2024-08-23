# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch

from models.experimental.yolov3.reference.models.common import autopad
from models.experimental.yolov3.tt.yolov3_conv import TtConv

import ttnn
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class TtBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        c1,
        c2,
        shortcut=True,
        g=1,
        e=0.5,
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        self.device = device
        self.base_address = base_address

        c_ = int(c2 * e)  # hidden channels

        self.cv1 = TtConv(
            base_address=base_address + ".cv1",
            state_dict=state_dict,
            device=device,
            c1=c1,
            c2=c_,
            k=1,
            s=1,
        )
        self.cv2 = TtConv(
            base_address=base_address + ".cv2",
            state_dict=state_dict,
            device=device,
            c1=c_,
            c2=c2,
            k=3,
            s=1,
            g=g,
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        output = self.cv2(self.cv1(x))
        if self.add:
            output = ttnn.add(x, output)

        return output
