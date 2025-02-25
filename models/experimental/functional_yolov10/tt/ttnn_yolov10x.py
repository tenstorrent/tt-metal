# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.experimental.functional_yolov10.tt.common import Conv
from models.experimental.functional_yolov10.reference.yolov10 import Conv as Torch_conv
import math


class ttnn_SCDown:
    def __init__(self, device=None, parameters=None, conv_pt=None, torch_conv=False):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )

        if torch_conv:
            self.cv2 = nn.Conv2d(
                in_channels=640,
                out_channels=640,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=640,
                bias=False,
            )
            self.cv2.weight = torch.nn.Parameter(ttnn.to_torch(ttnn.from_device(self.conv_pt.cv2.conv.weight)))
            self.cv2.bias = torch.nn.Parameter(
                ttnn.to_torch(ttnn.from_device(self.conv_pt.cv2.conv.bias)).squeeze(0).squeeze(0).squeeze(0)
            )

        else:
            self.cv2 = Conv(
                device,
                parameters.cv2,
                self.conv_pt.cv2,
                enable_identity=True,
                use_1d_systolic_array=False,
                config_override={"act_block_h": 512},
            )

    def __call__(self, x):
        x = self.cv1(x)
        x = self.cv2(x)  # Statically allocated circular buffers
        return x
