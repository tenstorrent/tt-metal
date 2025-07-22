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
            activation="relu",
        )
        self.cv2 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv2.block.conv,
            conv_pth=parameters.cv2.block.conv,
            activation="relu",
        )
        self.cv3 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv3.block.conv,
            conv_pth=parameters.cv3.block.conv,
            activation="relu",
            deallocate_activation=True,
            return_height_width=True,
        )
        self.upsample = Yolov6x_Conv_T_2D(
            model_params.upsample.upsample_transpose,
            parameters.upsample.upsample_transpose,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            device=device,
        )
        self.downsample = Yolov6l_Conv2D(
            device=device,
            conv=model_params.downsample.block.conv,
            conv_pth=parameters.downsample.block.conv,
            activation="relu",
            deallocate_activation=True,
        )

    def __call__(self, x):
        conv_t = self.upsample(x[0])
        conv1 = self.cv1(x[1])
        conv2 = self.cv2(x[2])
        downsample = self.downsample(conv2)
        if conv_t.memory_config() != conv1.memory_config():
            conv1 = ttnn.to_memory_config(conv1, conv_t.memory_config())
        if conv_t.memory_config() != downsample.memory_config():
            downsample = ttnn.to_memory_config(downsample, conv_t.memory_config())
        output_sharded_memory_config = ttnn.create_sharded_memory_config(
            [
                conv_t.memory_config().shard_spec.shape[0],
                3 * conv_t.memory_config().shard_spec.shape[1],
            ],
            core_grid=conv_t.memory_config().shard_spec.grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        output = ttnn.concat([conv_t, conv1, downsample], dim=-1, memory_config=output_sharded_memory_config)
        output, out_h, out_w = self.cv3(output)
        return output, out_h, out_w
