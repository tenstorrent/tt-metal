# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


from typing import List

import ttnn

from models.tt_cnn.tt.builder import TtConv2d, Conv2dConfiguration, AutoShardedStrategyConfiguration
from models.experimental.MapTR.tt.ttnn_bottleneck import TtBottleneck


def create_conv_config_from_args(conv_args, conv_pth, activation=None):
    return Conv2dConfiguration.from_model_args(
        conv2d_args=conv_args,
        weights=conv_pth.weight,
        bias=conv_pth.bias if hasattr(conv_pth, "bias") else None,
        activation=activation,
        sharding_strategy=AutoShardedStrategyConfiguration(),
    )


class TtResNet50:
    def __init__(
        self,
        conv_args,
        conv_pth,
        device: ttnn.Device,
    ):
        self.device = device
        self.maxpool_args = conv_args.maxpool

        conv1_config = create_conv_config_from_args(
            conv_args.conv1,
            conv_pth.conv1,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        )
        self.conv1 = TtConv2d(conv1_config, device)

        self.layer1_0 = TtBottleneck(
            conv_args.layer1[0],
            conv_pth.layer1_0,
            device=self.device,
            is_downsample=True,
        )
        self.layer1_1 = TtBottleneck(conv_args.layer1[1], conv_pth.layer1_1, device=self.device)
        self.layer1_2 = TtBottleneck(conv_args.layer1[2], conv_pth.layer1_2, device=self.device)

        self.layer2_0 = TtBottleneck(
            conv_args.layer2[0],
            conv_pth.layer2_0,
            device=self.device,
            is_downsample=True,
            blk_sharded=True,
            activation_dtype=ttnn.bfloat8_b,
        )
        self.layer2_1 = TtBottleneck(conv_args.layer2[1], conv_pth.layer2_1, device=self.device)
        self.layer2_2 = TtBottleneck(conv_args.layer2[2], conv_pth.layer2_2, device=self.device)
        self.layer2_3 = TtBottleneck(conv_args.layer2[3], conv_pth.layer2_3, device=self.device)

        self.layer3_0 = TtBottleneck(
            conv_args.layer3[0],
            conv_pth.layer3_0,
            device=self.device,
            is_downsample=True,
            blk_sharded=True,
            activation_dtype=ttnn.bfloat8_b,
        )
        self.layer3_1 = TtBottleneck(conv_args.layer3[1], conv_pth.layer3_1, device=self.device)
        self.layer3_2 = TtBottleneck(conv_args.layer3[2], conv_pth.layer3_2, device=self.device)
        self.layer3_3 = TtBottleneck(conv_args.layer3[3], conv_pth.layer3_3, device=self.device)
        self.layer3_4 = TtBottleneck(conv_args.layer3[4], conv_pth.layer3_4, device=self.device)
        self.layer3_5 = TtBottleneck(conv_args.layer3[5], conv_pth.layer3_5, device=self.device)

        self.layer4_0 = TtBottleneck(
            conv_args.layer4[0],
            conv_pth.layer4_0,
            device=self.device,
            is_downsample=True,
            blk_sharded=True,
            activation_dtype=ttnn.bfloat8_b,
            conv3_blk_sharded=True,
        )
        self.layer4_1 = TtBottleneck(
            conv_args.layer4[1],
            conv_pth.layer4_1,
            device=self.device,
            conv3_blk_sharded=True,
        )
        self.layer4_2 = TtBottleneck(
            conv_args.layer4[2],
            conv_pth.layer4_2,
            device=self.device,
            conv3_blk_sharded=True,
        )

    def __call__(self, x: ttnn.Tensor, batch_size: int = 1) -> List[ttnn.Tensor]:
        x = self.conv1(x)
        x = ttnn.sharded_to_interleaved(x)

        if self.maxpool_args.batch_size > 1:
            x = self._split_maxpool(x)
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

        x = self.layer1_0(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.layer1_1(x)
        x = self.layer1_2(x)

        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)

        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x = self.layer3_5(x)

        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)

        return [x]

    def _split_maxpool(self, x: ttnn.Tensor) -> ttnn.Tensor:
        config = self.maxpool_args
        split_point = config.batch_size // 2
        spatial_size = config.input_height * config.input_width
        channels = x.shape[3]

        x0 = ttnn.slice(x, [0, 0, 0, 0], [1, 1, split_point * spatial_size, channels])
        x1 = ttnn.slice(
            x,
            [0, 0, split_point * spatial_size, 0],
            [1, 1, config.batch_size * spatial_size, channels],
        )

        x0 = ttnn.max_pool2d(
            input_tensor=x0,
            batch_size=split_point,
            input_h=config.input_height,
            input_w=config.input_width,
            channels=channels,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ceil_mode=False,
        )
        x1 = ttnn.max_pool2d(
            input_tensor=x1,
            batch_size=config.batch_size - split_point,
            input_h=config.input_height,
            input_w=config.input_width,
            channels=channels,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ceil_mode=False,
        )

        return ttnn.concat((x0, x1), dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
