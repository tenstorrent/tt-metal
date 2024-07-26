# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from models.experimental.yolov4.reference.downsample1 import DownSample1
from models.experimental.yolov4.reference.downsample2 import DownSample2
from models.experimental.yolov4.reference.downsample3 import DownSample3
from models.experimental.yolov4.reference.downsample4 import DownSample4
from models.experimental.yolov4.reference.downsample5 import DownSample5
from models.experimental.yolov4.reference.neck import Neck
from models.experimental.yolov4.reference.head import Head

import torch
import torch.nn as nn


class Yolov4(nn.Module):
    def __init__(self):
        super(Yolov4, self).__init__()
        self.downsample1 = DownSample1()
        self.downsample2 = DownSample2()
        self.downsample3 = DownSample3()
        self.downsample4 = DownSample4()
        self.downsample5 = DownSample5()
        self.neck = Neck()
        self.head = Head()

    def forward(self, input: torch.Tensor):
        d1 = self.downsample1(input)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)
        d5 = self.downsample5(d4)
        x20, x13, x6 = self.neck(d5, d4, d3)
        x4, x5, x6 = self.head(x20, x13, x6)

        return x4, x5, x6
