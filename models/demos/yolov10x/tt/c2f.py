# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov10x.tt.bottleneck import TtnnBottleNeck
from models.demos.yolov10x.tt.common import Conv, deallocate_tensors

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TtnnC2f:
    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None, path=""):
        self.shortcut = shortcut
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
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
            deallocate_activation=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt[f"{path}.cv2"],
            deallocate_activation=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )

        self.m = [
            TtnnBottleNeck(
                self.shortcut,
                device=self.device,
                parameters=self.parameters[_],
                conv_pt=self.conv_pt,
                path=f"{path}.m.{_}",
            )
            for _ in range(n)
        ]

    def __call__(self, input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG):
        if use_signpost:
            signpost(header="TtnnC2f Start")

        x1 = self.cv1_a(input_tensor)
        x2 = self.cv1_b(input_tensor)
        x1 = ttnn.to_memory_config(x1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x2 = ttnn.to_memory_config(x2, memory_config=ttnn.L1_MEMORY_CONFIG)

        y = [x1, x2]

        for m in self.m:
            out = m(y[-1])
            y.append(out)
        for i in range(2, len(y)):
            y[i] = ttnn.to_memory_config(y[i], memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.concat(y, -1, memory_config=memory_config)
        deallocate_tensors(x1, x2, *y)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        output = self.cv2(out)
        ttnn.deallocate(out)
        if use_signpost:
            signpost(header="TtnnC2f End")
        return output
