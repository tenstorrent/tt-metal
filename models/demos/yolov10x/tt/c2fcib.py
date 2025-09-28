# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov10x.tt.cib import TtnnCIB
from models.demos.yolov10x.tt.common import Conv, deallocate_tensors
from models.experimental.yolo_common.yolo_utils import concat


class TtnnC2fCIB:
    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.shortcut = shortcut

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )
        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
            use_1d_systolic_array=False,
        )

        self.m = [
            TtnnCIB(
                shortcut=self.shortcut,
                device=self.device,
                parameters=self.parameters[2],
                conv_pt=self.conv_pt.m[_],
            )
            for _ in range(n)
        ]

    def __call__(self, input_tensor):
        cv1 = self.cv1(input_tensor)
        if cv1.is_sharded():
            cv1 = ttnn.sharded_to_interleaved(cv1, ttnn.L1_MEMORY_CONFIG)
        x1 = cv1[:, :, :, : cv1.shape[-1] // 2]
        x2 = cv1[:, :, :, cv1.shape[-1] // 2 : cv1.shape[-1]]

        y = [x1, x2]

        for m in self.m:
            out = m(y[-1])

            y.append(ttnn.to_layout(out, ttnn.TILE_LAYOUT))

        out = concat(-1, False, *y)

        output = self.cv2(out)
        deallocate_tensors(*y)

        return output
