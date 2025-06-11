# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.yolov4.common import get_model_result
from models.demos.yolov4.tt.yolov4 import TtYOLOv4


class YOLOv4Runner:
    def __init__(self, device, parameters, resolution):
        self.model = TtYOLOv4(parameters, device)
        self.resolution = resolution

    def run(self, input):
        ttnn_output_tensor = self.model(input)
        return get_model_result(ttnn_output_tensor, self.resolution)
