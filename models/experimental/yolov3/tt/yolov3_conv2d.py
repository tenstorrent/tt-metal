# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch

from models.experimental.yolov3.reference.models.common import autopad
from models.experimental.yolov3.reference.models.yolo import Conv, Model
import ttnn
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)
from models.utility_functions import (
    run_conv_on_device_wrapper,
    is_conv_supported_on_device,
)


class TtConv2D(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
    ):
        super().__init__()
        self.device = device
        self.conv_on_device = False

        # Get conv2d parameters
        self.conv_weight = state_dict[f"{base_address}.weight"]
        bias_key = f"{base_address}.bias"

        if bias_key in state_dict:
            self.conv_bias = state_dict[bias_key]
        else:
            self.conv_bias = None

        if p is None:
            p = autopad(k, p, d)

        # conv_params = [out_channels, in_channels, kernel_size, kernel_size, stride, stride, padding, padding, dilation, groups]
        self.conv_params = [c2, c1, k, k, s, s, p, p, d, g]

        if self.conv_on_device and is_conv_supported_on_device(self.conv_params):
            self.conv_bias = self.conv_bias.unsqueeze(-1).unsqueeze(-1)

            self.conv = run_conv_on_device_wrapper(
                self.conv_weight.reshape(-1).tolist(),
                self.conv_params,
                self.device,
                conv_bias=None,
            )

        else:
            self.conv_on_device = False

            self.conv_weight = torch_to_tt_tensor_rm(self.conv_weight, self.device, put_on_device=False)
            self.conv_bias = torch_to_tt_tensor_rm(self.conv_bias, self.device) if self.conv_bias is not None else None

            self.conv = fallback_ops.Conv2d(
                weights=self.conv_weight,
                biases=self.conv_bias,
                in_channels=c1,
                out_channels=c2,
                kernel_size=k,
                stride=s,
                padding=p,
                groups=g,
                dilation=d,
                bias=self.conv_bias is not None,
            )

    def forward(self, x):
        if self.conv_on_device:
            x = tt2torch_tensor(x)
            x = self.conv(x)
            x = x + self.conv_bias
            x = torch2tt_tensor(x, self.device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            x = self.conv(x)

        return x
