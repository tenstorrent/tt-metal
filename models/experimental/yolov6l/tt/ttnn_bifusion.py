# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov6l.tt.common import Yolov6l_Conv2D, Yolov6x_Conv_T_2D


class TtBiFusion:
    def __init__(self, device, parameters, model_params):
        self.parameters = parameters
        self.model_params = model_params
        self.cv1 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv1.block.conv,
            conv_pth=parameters.cv1.block.conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhwc=True,
            reshape=True,
        )
        self.cv2 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv2.block.conv,
            conv_pth=parameters.cv2.block.conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhwc=True,
            reshape=True,
        )
        self.cv3 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv3.block.conv,
            conv_pth=parameters.cv3.block.conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhwc=True,
            reshape=True,
        )
        self.upsample = Yolov6x_Conv_T_2D(
            model_params.upsample.upsample_transpose,
            parameters.upsample.upsample_transpose,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            device=device,
            reshape=True,
        )
        self.downsample = Yolov6l_Conv2D(
            device=device,
            conv=model_params.downsample.block.conv,
            conv_pth=parameters.downsample.block.conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhwc=True,
            reshape=True,
        )

    def __call__(self, x):
        conv_t = self.upsample(x[0])
        conv1 = self.cv1(x[1])
        conv2 = self.cv2(x[2])
        downsample = self.downsample(conv2)
        output = ttnn.concat([conv_t, conv1, downsample], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        output = self.cv3(output)
        return output
