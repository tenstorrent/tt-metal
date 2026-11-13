# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.vadv1.tt.common import TtConv2D
from models.experimental.vadv1.tt.tt_bottleneck import TtBottleneck


class TtResnet50:
    def __init__(self, conv_args, conv_pth, device):
        self.maxpool_args = conv_args.maxpool
        self.device = device

        self.conv1 = TtConv2D(
            conv_args.conv1,
            conv_pth.conv1,
            device=self.device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h=32,
        )

        # Layer 1, at the highest resolution: its circular buffers with float32 accumulation do not fit L1.
        # Each stage's first block strides and projects the shortcut; the remaining blocks repeat, and
        # their count follows the reference model's.
        stage_options = (
            (conv_args.layer1, {"fp32_dest_acc": False}, {"fp32_dest_acc": False}),
            (conv_args.layer2, {"blk_sharded": True, "activation_dtype": ttnn.bfloat8_b}, {}),
            (conv_args.layer3, {"blk_sharded": True, "activation_dtype": ttnn.bfloat8_b}, {}),
            (
                conv_args.layer4,
                {"blk_sharded": True, "activation_dtype": ttnn.bfloat8_b, "conv3_blk_sharded": True},
                {"conv3_blk_sharded": True},
            ),
        )
        self.stages = []
        for stage, (blocks_args, first_options, repeat_options) in enumerate(stage_options, start=1):
            blocks_pth = [conv_pth[name] for name in conv_pth if name.startswith(f"layer{stage}_")]
            self.stages.append(
                [
                    TtBottleneck(
                        blocks_args[block],
                        block_pth,
                        device=self.device,
                        **({"is_downsample": True, **first_options} if block == 0 else repeat_options),
                    )
                    for block, block_pth in enumerate(blocks_pth)
                ]
            )

    def __call__(self, x, batch_size=1):
        x, out_ht, out_wdth = self.conv1(x)

        x = ttnn.sharded_to_interleaved(x)

        if self.maxpool_args.batch_size > 1:
            current_batch_size = self.maxpool_args.batch_size
            split_point = current_batch_size // 2
            x0 = ttnn.slice(
                x,
                [0, 0, 0, 0],
                [1, 1, split_point * self.maxpool_args.input_height * self.maxpool_args.input_width, x.shape[3]],
            )
            x1 = ttnn.slice(
                x,
                [0, 0, split_point * self.maxpool_args.input_height * self.maxpool_args.input_width, 0],
                [1, 1, current_batch_size * self.maxpool_args.input_height * self.maxpool_args.input_width, x.shape[3]],
            )
            x0 = ttnn.max_pool2d(
                input_tensor=x0,
                batch_size=split_point,
                input_h=self.maxpool_args.input_height,
                input_w=self.maxpool_args.input_width,
                channels=x.shape[3],
                kernel_size=[3, 3],
                stride=[2, 2],
                padding=[1, 1],
                dilation=[1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ceil_mode=False,
            )
            x1 = ttnn.max_pool2d(
                input_tensor=x1,
                batch_size=current_batch_size - split_point,
                input_h=self.maxpool_args.input_height,
                input_w=self.maxpool_args.input_width,
                channels=x.shape[3],
                kernel_size=[3, 3],
                stride=[2, 2],
                padding=[1, 1],
                dilation=[1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ceil_mode=False,
            )
            # On a small grid max_pool2d returns NHWC rather than flattened rows, and joining that on
            # dim 2 would interleave the two halves' images row by row.
            x0, x1 = (ttnn.reshape(half, (1, 1, -1, half.shape[-1])) for half in (x0, x1))
            x = ttnn.concat((x0, x1), dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            x = ttnn.max_pool2d(
                input_tensor=x,
                batch_size=self.maxpool_args.batch_size,
                input_h=self.maxpool_args.input_height,
                input_w=self.maxpool_args.input_width,
                channels=x.shape[3],
                kernel_size=[3, 3],
                stride=[2, 2],
                padding=[1, 1],
                dilation=[1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ceil_mode=False,
            )
        outputs = []
        for stage, blocks in enumerate(self.stages, start=1):
            for block, bottleneck in enumerate(blocks):
                x = bottleneck(x)
                if stage == 1 and block == 0:
                    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        outputs.append(x)

        return outputs
