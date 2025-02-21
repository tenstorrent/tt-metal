# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov10.tt.common import Conv, deallocate_tensors
from models.experimental.yolo_common.yolo_utils import concat


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
        cv1 = self.cv1(x)
        cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT)
        y = [cv1]

        TILE_WIDTH = 32
        in_c = self.parameters.cv2.conv.in_channels
        in_c_padded = in_c
        if in_c % TILE_WIDTH != 0 and in_c != 16:
            in_c_padded = in_c + (TILE_WIDTH - in_c % TILE_WIDTH)

        for i in range(3):
            if y[-1].is_sharded():
                y[-1] = ttnn.sharded_to_interleaved(y[-1])
            tt_out = ttnn.max_pool2d(
                input_tensor=y[-1],
                batch_size=x.shape[0],
                input_h=20,
                input_w=20,
                channels=in_c_padded,
                kernel_size=[5, 5],
                stride=[1, 1],
                padding=[2, 2],
                dilation=[1, 1],
                applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            )
            y.append(tt_out)

        out = concat(-1, True, *y)

        deallocate_tensors(*y)

        out = ttnn.sharded_to_interleaved(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        out = self.cv2(out)

        return out
