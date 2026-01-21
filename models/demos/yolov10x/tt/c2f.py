# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov10x.tt.bottleneck import TtnnBottleNeck
from models.demos.yolov10x.tt.common import Conv, deallocate_tensors


class TtnnC2f:
    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None):
        self.shortcut = shortcut
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
            deallocate_activation=True,
        )
        self.cv2 = Conv(device, parameters.cv2, self.conv_pt.cv2, deallocate_activation=True)

        self.m = [
            TtnnBottleNeck(self.shortcut, device=self.device, parameters=self.parameters[_], conv_pt=self.conv_pt.m[_])
            for _ in range(n)
        ]

    def __call__(self, input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG):
        cv1 = self.cv1(input_tensor)
        cv1 = ttnn.to_memory_config(cv1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x1 = cv1[:, :, :, : cv1.shape[-1] // 2]
        x2 = cv1[:, :, :, cv1.shape[-1] // 2 : cv1.shape[-1]]
        y = [x1, x2]

        for m in self.m:
            out = m(y[-1])
            out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)
            y.append(out)

        out = ttnn.concat(y, -1, memory_config=memory_config)
        deallocate_tensors(x1, x2, *y, cv1)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        output = self.cv2(out)

        return output
