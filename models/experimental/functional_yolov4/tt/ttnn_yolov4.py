# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from models.experimental.functional_yolov4.tt.ttnn_downsample1 import TtDownSample1
from models.experimental.functional_yolov4.tt.ttnn_downsample2 import TtDownSample2
from models.experimental.functional_yolov4.tt.ttnn_downsample3 import TtDownSample3
from models.experimental.functional_yolov4.tt.ttnn_downsample4 import TtDownSample4
from models.experimental.functional_yolov4.tt.ttnn_downsample5 import TtDownSample5
from models.experimental.functional_yolov4.tt.ttnn_neck import TtNeck
from models.experimental.functional_yolov4.tt.ttnn_head import TtHead
import ttnn


class TtYolov4:
    def __init__(self, device, parameters) -> None:
        self.downsample1 = TtDownSample1(parameters["downsample1"])
        self.downsample2 = TtDownSample2(parameters["downsample2"])
        self.downsample3 = TtDownSample3(parameters["downsample3"])
        self.downsample4 = TtDownSample4(parameters["downsample4"])
        self.downsample5 = TtDownSample5(parameters["downsample5"])
        self.neck = TtNeck(device, parameters["neck"])
        self.head = TtHead(device, parameters["head"])

    def __call__(self, device, input_tensor):
        d1 = self.downsample1(device, input_tensor)
        d2 = self.downsample2(device, d1)
        d3 = self.downsample3(device, d2)
        d4 = self.downsample4(device, d3)
        d5 = self.downsample5(device, d4)
        x20, x13, x6 = self.neck(device, [d5, d4, d3])
        x4, x5, x6 = self.head(device, [x20, x13, x6])
        return x4, x5, x6
