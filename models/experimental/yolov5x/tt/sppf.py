# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.yolov5x.tt.common import TtYOLOv5xConv2D, deallocate_tensors
from models.experimental.yolo_common.yolo_utils import concat


class TtnnSPPF:
    def __init__(self, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = TtYOLOv5xConv2D(
            device,
            parameters.cv1.conv,
            self.conv_pt.cv1.conv,
            activation="silu",
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        self.cv2 = TtYOLOv5xConv2D(
            device,
            parameters.cv2.conv,
            self.conv_pt.cv2.conv,
            activation="silu",
            use_1d_systolic_array=True,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        )

    def __call__(self, x):
        cv1 = self.cv1(x)
        cv1 = ttnn.sharded_to_interleaved(cv1, memory_config=ttnn.L1_MEMORY_CONFIG)
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

        out = self.cv2(out)

        return out
