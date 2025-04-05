# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_yolov10.tt.common import Conv


class TtnnSPPF:
    def __init__(self, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )

        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
            auto_shard=True,
        )

    def __call__(self, x):
        y = [self.cv1(x)]
        y[0] = ttnn.sharded_to_interleaved(y[0], ttnn.L1_MEMORY_CONFIG)
        y[0] = ttnn.to_layout(y[0], ttnn.ROW_MAJOR_LAYOUT)

        for i in range(3):
            tt_max = y[-1]
            tt_out = ttnn.max_pool2d(
                input_tensor=y[-1],
                batch_size=x.shape[0],
                input_h=20,
                input_w=20,
                channels=320,
                kernel_size=[5, 5],
                stride=[1, 1],
                padding=[2, 2],
                dilation=[1, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            y.append(tt_out)

        out = ttnn.concat(y, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        out = self.cv2(out)
        return out
