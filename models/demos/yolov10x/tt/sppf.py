# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov10x.tt.common import Conv, deallocate_tensors
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
            use_1d_systolic_array=False,
        )

    def __call__(self, x):
        cv1 = self.cv1(x)
        if cv1.get_layout() == ttnn.TILE_LAYOUT:
            cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT)
        y = [cv1]

        if y[-1].is_sharded():
            y[-1] = ttnn.sharded_to_interleaved(y[-1])

        for i in range(3):
            if not y[-1].is_sharded():
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
                    applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                )
            else:
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
                )
            y.append(tt_out)

        out = concat(-1, True, *y)

        deallocate_tensors(*y)

        out = ttnn.sharded_to_interleaved(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        out = self.cv2(out)

        return out
