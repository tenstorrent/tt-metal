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
from models.demos.yolov4.ttnn.genboxes import TtGenBoxes


class TtYOLOv4:
    def __init__(self, path, device) -> None:
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

        self.boxes_confs_0 = TtGenBoxes(device)
        self.boxes_confs_1 = TtGenBoxes(device)
        self.boxes_confs_2 = TtGenBoxes(device)

        self.downs = []  # [self.down1]
        self.device = device

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

        x4_boxes_confs = self.boxes_confs_0(self.device, x4)
        x5_boxes_confs = self.boxes_confs_1(self.device, x5)
        x6_boxes_confs = self.boxes_confs_2(self.device, x6)

        confs_1 = ttnn.to_layout(x4_boxes_confs[1], ttnn.ROW_MAJOR_LAYOUT)
        confs_2 = ttnn.to_layout(x5_boxes_confs[1], ttnn.ROW_MAJOR_LAYOUT)
        confs_3 = ttnn.to_layout(x6_boxes_confs[1], ttnn.ROW_MAJOR_LAYOUT)
        confs = ttnn.concat([confs_1, confs_2, confs_3], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

        boxes_1 = ttnn.to_layout(x4_boxes_confs[0], ttnn.ROW_MAJOR_LAYOUT)
        boxes_2 = ttnn.to_layout(x5_boxes_confs[0], ttnn.ROW_MAJOR_LAYOUT)
        boxes_3 = ttnn.to_layout(x6_boxes_confs[0], ttnn.ROW_MAJOR_LAYOUT)
        boxes_1 = ttnn.reshape(boxes_1, (1, 4, 1, 4800))
        boxes_2 = ttnn.reshape(boxes_2, (1, 4, 1, 1200))
        boxes_3 = ttnn.pad(boxes_3, ((0, 0), (0, 0), (0, 0), (0, 28)), 0)
        boxes_3 = ttnn.reshape(boxes_3, (1, 4, 1, 384))
        boxes_1 = ttnn.permute(boxes_1, (0, 2, 3, 1))
        boxes_2 = ttnn.permute(boxes_2, (0, 2, 3, 1))
        boxes_3 = ttnn.permute(boxes_3, (0, 2, 3, 1))
        boxes = ttnn.concat([boxes_1, boxes_2, boxes_3], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)

        return boxes, confs

    def __str__(self) -> str:
        this_str = ""
        for down in self.downs:
            this_str += str(down)
            this_str += " \n"
        return this_str
