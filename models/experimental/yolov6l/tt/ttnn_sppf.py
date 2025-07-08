# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov6l.tt.common import Yolov6l_Conv2D


class TtSppf:
    def __init__(self, device, parameters, model_params):
        self.parameters = parameters
        self.model_params = model_params
        self.cv1 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv1.block.conv,
            conv_pth=parameters.cv1.block.conv,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            # auto_shard=True,
            activation="silu",
            is_nhwc=True,
        )
        self.cv2 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv2.block.conv,
            conv_pth=parameters.cv2.block.conv,
            shard_layout=None,
            auto_shard=True,
            activation="silu",
            is_nhwc=True,
        )

    def __call__(self, x):
        conv1 = self.cv1(x)
        conv1 = ttnn.to_memory_config(conv1, ttnn.L1_MEMORY_CONFIG)
        conv1 = ttnn.sharded_to_interleaved(conv1, ttnn.L1_MEMORY_CONFIG)
        m1 = ttnn.max_pool2d(
            conv1,
            batch_size=1,
            input_h=20,
            input_w=15,
            channels=512,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        m2 = ttnn.max_pool2d(
            m1,
            batch_size=1,
            input_h=20,
            input_w=15,
            channels=512,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        m3 = ttnn.max_pool2d(
            m2,
            batch_size=1,
            input_h=20,
            input_w=15,
            channels=512,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )

        conv1 = ttnn.to_layout(conv1, ttnn.ROW_MAJOR_LAYOUT)
        m1 = ttnn.sharded_to_interleaved(m1, ttnn.L1_MEMORY_CONFIG)
        m2 = ttnn.sharded_to_interleaved(m2, ttnn.L1_MEMORY_CONFIG)
        m3 = ttnn.sharded_to_interleaved(m3, ttnn.L1_MEMORY_CONFIG)
        concat_output = ttnn.concat([conv1, m1, m2, m3], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        conv2 = self.cv2(concat_output)
        conv2 = ttnn.reshape(conv2, (1, 20, 15, 1024), memory_config=ttnn.L1_MEMORY_CONFIG)
        return conv2
