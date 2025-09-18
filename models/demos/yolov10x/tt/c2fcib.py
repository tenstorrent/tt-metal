# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov10x.tt.cib import TtnnCIB
from models.demos.yolov10x.tt.common import Conv, deallocate_tensors
from models.experimental.yolo_common.yolo_utils import concat

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TtnnC2fCIB:
    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None, path=""):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.shortcut = shortcut

        parameters.cv1.conv.out_channels //= 2
        self.cv1_a = Conv(
            device,
            parameters.cv1,
            self.conv_pt[f"{path}.cv1_a"],
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        self.cv1_b = Conv(
            device,
            parameters.cv1,
            self.conv_pt[f"{path}.cv1_b"],
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt[f"{path}.cv2"],
            use_1d_systolic_array=False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )

        self.m = [
            TtnnCIB(
                shortcut=self.shortcut,
                device=self.device,
                parameters=self.parameters[2],
                conv_pt=self.conv_pt,
                path=f"{path}.m.{_}",
            )
            for _ in range(n)
        ]

    def __call__(self, input_tensor):
        if use_signpost:
            signpost(header="TtnnC2fCIB Start")

        x1 = self.cv1_a(input_tensor)
        x2 = self.cv1_b(input_tensor)
        x1 = ttnn.to_memory_config(x1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x2 = ttnn.to_memory_config(x2, memory_config=ttnn.L1_MEMORY_CONFIG)

        y = [x1, x2]

        for m in self.m:
            out = m(y[-1])

            y.append(ttnn.to_layout(out, ttnn.TILE_LAYOUT))

        out = concat(-1, False, *y)

        output = self.cv2(out)
        deallocate_tensors(*y)
        if use_signpost:
            signpost(header="TtnnC2fCIB End")
        return output
