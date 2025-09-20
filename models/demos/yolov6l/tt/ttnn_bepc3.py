# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov6l.tt.common import Yolov6l_Conv2D
from models.demos.yolov6l.tt.ttnn_repblock import TtRepBlock

try:
    from tracy import signpost

    use_signpost = True

except ModuleNotFoundError:
    use_signpost = False


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
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        )
        self.cv2 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv2.block.conv,
            conv_pth=parameters.cv2.block.conv,
            shard_layout=shard_layout_cv2 if shard_layout == None else shard_layout,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            deallocate_activation=True,
        )
        self.cv3 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv3.block.conv,
            conv_pth=parameters.cv3.block.conv,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            shard_layout=shard_layout_cv2 if shard_layout == None else shard_layout,
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
        if use_signpost:
            signpost(header="TtBepC3 Start")
        conv1 = self.cv1(x)
        rep, _, _ = self.repblock(conv1)
        conv2 = self.cv2(x)

        if conv2.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            strategy_layout = ttnn.ShardStrategy.HEIGHT
            output_sharded_memory_config = ttnn.create_sharded_memory_config(
                [
                    rep.memory_config().shard_spec.shape[0],
                    2 * rep.memory_config().shard_spec.shape[1],
                ],
                core_grid=rep.memory_config().shard_spec.grid,
                strategy=strategy_layout,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            conv2 = ttnn.to_memory_config(conv2, memory_config=ttnn.L1_MEMORY_CONFIG)
            conv2 = ttnn.to_layout(conv2, layout=ttnn.TILE_LAYOUT)
            rep = ttnn.to_memory_config(rep, memory_config=ttnn.L1_MEMORY_CONFIG)
            output_sharded_memory_config = ttnn.L1_MEMORY_CONFIG

        concat_output = ttnn.concat([rep, conv2], dim=-1, memory_config=output_sharded_memory_config)
        ttnn.deallocate(rep)
        ttnn.deallocate(conv2)
        conv3 = self.cv3(concat_output)
        if use_signpost:
            signpost(header="TtBepC3 End")
        return conv3
