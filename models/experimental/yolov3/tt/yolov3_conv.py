# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from loguru import logger
import ttnn

from models.experimental.yolov3.tt.yolov3_conv2d import TtConv2D


class TtConv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

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
        act=True,
    ):
        super().__init__()

        self.device = device
        self.base_address = base_address

        self.conv = TtConv2D(
            base_address=f"{base_address}.conv",
            state_dict=state_dict,
            device=device,
            c1=c1,
            c2=c2,
            k=k,
            s=s,
            p=p,
            g=g,
            d=d,
        )

        self.act = act
        if self.act != True:
            logger.warning(
                f"Configuration for activation function {self.act} not supported. Using ttnn.SiLU act function"
            )
            raise NotImplementedError

    def forward(self, x):
        return ttnn.silu(self.conv(x))
