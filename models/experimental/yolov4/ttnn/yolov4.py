# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Any
import ttnn
import torch
import pytest

ttnn.enable_fast_runtime_mode = False
ttnn.enable_logging = True
ttnn.report_name = "yolo_fail"
ttnn.enable_graph_report = False
ttnn.enable_detailed_buffer_report = True
ttnn.enable_detailed_tensor_report = True
ttnn.enable_comparison_mode = False

from models.experimental.yolov4.reference.yolov4 import Yolov4
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.yolov4.ttnn.downsample1 import Down1
from models.experimental.yolov4.ttnn.downsample2 import Down2
from models.experimental.yolov4.ttnn.downsample3 import Down3
from models.experimental.yolov4.ttnn.downsample4 import Down4
from models.experimental.yolov4.ttnn.downsample5 import Down5
from models.experimental.yolov4.ttnn.neck import TtNeck
from models.experimental.yolov4.ttnn.head import TtHead


class TtYOLOv4:
    def __init__(self, path) -> None:
        self.torch_model = torch.load(path)
        self.torch_keys = self.torch_model.keys()
        self.down1 = Down1(self)
        self.down2 = Down2(self)
        self.down3 = Down3(self)
        self.down4 = Down4(self)
        self.down5 = Down5(self)

        self.neck = TtNeck(self)
        self.head = TtHead(self)

        self.downs = []  # [self.down1]

    def __call__(self, device, input_tensor, model):
        d1 = self.down1(device, input_tensor)
        d2 = self.down2(device, d1)
        ttnn.deallocate(d1)
        d3 = self.down3(device, d2)
        ttnn.deallocate(d2)
        d4 = self.down4(device, d3)
        d5 = self.down5(device, d4)
        x20, x13, x6 = self.neck(device, [d5, d4, d3])
        x4, x5, x6 = self.head(device, [x20, x13, x6], model.head)

        return x4, x5, x6

    def __str__(self) -> str:
        this_str = ""
        for down in self.downs:
            this_str += str(down)
            this_str += " \n"
        return this_str


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov4(device, reset_seeds):
    ttnn_model = TtYOLOv4("tests/ttnn/integration_tests/yolov4/yolov4.pth")
    print(ttnn_model)

    torch_input = torch.randn((1, 320, 320, 3), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    torch_input = torch_input.permute(0, 3, 1, 2).float()
    torch_model = Yolov4()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    ds_state_dict = {k: v for k, v in ttnn_model.torch_model.items()}

    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    print(keys)
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    result_1, result_2, result_3 = ttnn_model(device, ttnn_input, torch_model)
    result_1 = ttnn.to_torch(result_1)
    result_2 = ttnn.to_torch(result_2)
    result_3 = ttnn.to_torch(result_3)

    ref1, ref2, ref3 = torch_model(torch_input)

    result_1 = result_1.reshape(1, ref1.shape[2], ref1.shape[3], 256)
    result_1 = result_1.permute(0, 3, 1, 2)

    result_2 = result_2.reshape(1, ref2.shape[2], ref2.shape[3], 256)
    result_2 = result_2.permute(0, 3, 1, 2)

    result_3 = result_3.reshape(1, ref3.shape[2], ref3.shape[3], 256)
    result_3 = result_3.permute(0, 3, 1, 2)

    # Output is sliced because ttnn.conv returns 256 channels instead of 255.
    result_1 = result_1[:, :255, :, :]
    result_2 = result_2[:, :255, :, :]
    result_3 = result_3[:, :255, :, :]

    assert_with_pcc(result_1, ref1, 0.95)  # PCC = 0.95
    assert_with_pcc(result_2, ref2, 0.96)  # PCC = 0.96
    assert_with_pcc(result_3, ref3, 0.98)  # PCC = 0.98
