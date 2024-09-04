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
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)


class TtDetect(nn.Module):
    # YOLOv3 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, device, state_dict, base_address, nc=80, anchors=(), ch=(), inplace=True):
        # detection layer
        super().__init__()

        self.device = device
        self.base_address = base_address

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor

        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid

        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

        conv_list = []
        for i, x in enumerate(ch):
            self.conv_weight = torch_to_tt_tensor_rm(
                state_dict[f"{base_address}.m.{i}.weight"],
                self.device,
                put_on_device=False,
            )
            if f"{base_address}.m.{i}.bias" in state_dict:
                self.conv_bias = torch_to_tt_tensor_rm(
                    state_dict[f"{base_address}.m.{i}.bias"],
                    self.device,
                    put_on_device=False,
                )
            else:
                self.conv_bias = None

            conv_list.append(
                fallback_ops.Conv2d(
                    weights=self.conv_weight,
                    biases=self.conv_bias,
                    in_channels=x,
                    out_channels=self.no * self.na,
                    kernel_size=1,
                )
            )

        self.m = nn.ModuleList(conv_list)  # output conv

        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            x[i] = tt2torch_tensor(x[i])

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            # Cannot be ported further until 5d tensors are supported for ttnn ops
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # Detect (boxes only)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)

                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
