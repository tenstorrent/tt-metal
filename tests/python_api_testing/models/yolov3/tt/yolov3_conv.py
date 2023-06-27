import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch

from python_api_testing.models.yolov3.reference.models.common import autopad
from python_api_testing.models.yolov3.reference.models.yolo import Conv, Model
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)


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

        self.conv_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.conv.weight"], self.device, put_on_device=False
        )
        if f"{base_address}.conv.bias" in state_dict:
            self.conv_bias = torch_to_tt_tensor_rm(
                state_dict[f"{base_address}.conv.bias"], self.device
            )
        else:
            self.conv_bias = None

        self.conv = fallback_ops.Conv2d(
            weights=self.conv_weight,
            biases=self.conv_bias,
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            groups=g,
            dilation=d,
        )

        self.act = act
        if self.act != True:
            logger.warning(
                f"Configuration for activation function {self.act} not supported. Using fallback.SiLU act function"
            )
            raise NotImplementedError

    def forward(self, x):
        return fallback_ops.silu(self.conv(x))
