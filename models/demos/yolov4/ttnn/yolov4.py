# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Any
import ttnn
import torch

ttnn.enable_fast_runtime_mode = False
ttnn.enable_logging = True
ttnn.report_name = "yolo_fail"
ttnn.enable_graph_report = False
ttnn.enable_detailed_buffer_report = True
ttnn.enable_detailed_tensor_report = True
ttnn.enable_comparison_mode = False

from models.demos.yolov4.ttnn.downsample1 import Down1
from models.demos.yolov4.ttnn.downsample2 import Down2
from models.demos.yolov4.ttnn.downsample3 import Down3
from models.demos.yolov4.ttnn.downsample4 import Down4
from models.demos.yolov4.ttnn.downsample5 import Down5
from models.demos.yolov4.ttnn.neck import TtNeck
from models.demos.yolov4.ttnn.head import TtHead


class TtYOLOv4:
    def __init__(self, device, path) -> None:
        if type(path) is str:
            self.torch_model = torch.load(path)
        else:
            self.torch_model = path
        self.torch_keys = self.torch_model.keys()
        self.down1 = Down1(device, self)
        self.down2 = Down2(device, self)
        self.down3 = Down3(device, self)
        self.down4 = Down4(device, self)
        self.down5 = Down5(device, self)

        self.neck = TtNeck(device, self)
        self.head = TtHead(device, self)

        self.downs = []  # [self.down1]

    def __call__(self, input_tensor):
        d1 = self.down1(input_tensor)
        d2 = self.down2(d1)
        ttnn.deallocate(d1)
        d3 = self.down3(d2)
        ttnn.deallocate(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        x20, x13, x6 = self.neck([d5, d4, d3])
        x4, x5, x6 = self.head([x20, x13, x6])

        return x4, x5, x6

    def __str__(self) -> str:
        this_str = ""
        for down in self.downs:
            this_str += str(down)
            this_str += " \n"
        return this_str
