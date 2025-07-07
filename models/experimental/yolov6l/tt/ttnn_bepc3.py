# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov6l.tt.ttnn_repblock import TtRepBlock
from models.experimental.yolov6l.tt.common import Yolov6l_Conv2D


class TtBepC3:
    def __init__(
        self,
        device,
        parameters,
        model_params,
        n=6,
        shard_layout=None,
        shard_layout_cv2=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        shard_layout_rep_block_first_two=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ):
        self.parameters = parameters
        self.model_params = model_params
        self.cv1 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv1.block.conv,
            conv_pth=parameters.cv1.block.conv,
            shard_layout=shard_layout_cv2 if shard_layout == None else shard_layout,
            activation="silu",
        )
        self.cv2 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv2.block.conv,
            conv_pth=parameters.cv2.block.conv,
            shard_layout=shard_layout_cv2 if shard_layout == None else shard_layout,
            activation="silu",
        )
        self.cv3 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv3.block.conv,
            conv_pth=parameters.cv3.block.conv,
            activation="silu",
            shard_layout=shard_layout_cv2 if shard_layout == None else shard_layout,
            reshape=True,
        )
        self.repblock = TtRepBlock(
            device,
            parameters.m,
            model_params.m,
            n=n,
            shard_layout_rep_block_first_two=shard_layout_rep_block_first_two if shard_layout == None else shard_layout,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED if shard_layout == None else shard_layout,
        )

    def __call__(self, x):
        conv1 = self.cv1(x)
        rep, _, _ = self.repblock(conv1)
        conv2 = self.cv2(x)
        conv2 = ttnn.to_memory_config(conv2, memory_config=ttnn.L1_MEMORY_CONFIG)
        conv2 = ttnn.to_layout(conv2, layout=ttnn.TILE_LAYOUT)
        rep = ttnn.to_memory_config(rep, memory_config=ttnn.L1_MEMORY_CONFIG)

        concat_output = ttnn.concat([rep, conv2], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        conv3 = self.cv3(concat_output)
        return conv3
