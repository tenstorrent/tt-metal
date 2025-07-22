# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov6l.tt.common import Yolov6l_Conv2D


class TtBottleRep:
    def __init__(self, device, parameters, model_params, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED):
        self.parameters = parameters
        self.model_params = model_params
        self.cv1 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.conv1.block.conv,
            conv_pth=parameters.conv1.block.conv,
            shard_layout=shard_layout,
            activation="silu",
            activation_dtype=ttnn.bfloat16,
        )
        self.cv2 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.conv2.block.conv,
            conv_pth=parameters.conv2.block.conv,
            shard_layout=shard_layout,
            act_block_h=True,
            act_blocks=32,
            activation="silu",
            return_height_width=True,
            deallocate_activation=True,
        )

    def __call__(self, x):
        x_conv1 = self.cv1(x)
        x_conv2, out_h, out_w = self.cv2(x_conv1)

        self.parameters.alpha = ttnn.to_layout(self.parameters.alpha, ttnn.TILE_LAYOUT)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = self.parameters.alpha * x
        ttnn.deallocate(x_conv1)

        x_conv2 = ttnn.to_layout(x_conv2, ttnn.TILE_LAYOUT)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        if x_conv2.memory_config() != x.memory_config():
            x = ttnn.to_memory_config(x, x_conv2.memory_config())

        output = x_conv2 + x

        ttnn.deallocate(x_conv2)
        ttnn.deallocate(x)
        return output, out_h, out_w
