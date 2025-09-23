# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.vadv2.tt.common import TtConv2D
from models.experimental.vadv2.tt.tt_bottleneck import TtBottleneck


class TtResnet50:
    def __init__(self, conv_args, conv_pth, device):
        self.maxpool_args = conv_args.maxpool
        self.device = device

        self.conv1 = TtConv2D(conv_args.conv1, conv_pth.conv1, device=self.device, activation="relu", act_block_h=32)

        # Layer 1
        self.layer1_0 = TtBottleneck(
            conv_args.layer1[0],
            conv_pth.layer1_0,
            device=self.device,
            is_downsample=True,
        )
        self.layer1_1 = TtBottleneck(conv_args.layer1[1], conv_pth.layer1_1, device=self.device)
        self.layer1_2 = TtBottleneck(conv_args.layer1[2], conv_pth.layer1_2, device=self.device)

        # Layer 2
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

        # Layer 3
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

        # Layer 4
        self.layer4_0 = TtBottleneck(
            conv_args.layer4[0],
            conv_pth.layer4_0,
            device=self.device,
            is_downsample=True,
            blk_sharded=True,
            activation_dtype=ttnn.bfloat8_b,
            conv3_blk_sharded=True,
        )
        self.layer4_1 = TtBottleneck(conv_args.layer4[1], conv_pth.layer4_1, device=self.device, conv3_blk_sharded=True)
        self.layer4_2 = TtBottleneck(conv_args.layer4[2], conv_pth.layer4_2, device=self.device, conv3_blk_sharded=True)

    def __call__(self, x, batch_size=1):
        x, out_ht, out_wdth = self.conv1(x)

        x = ttnn.sharded_to_interleaved(x)

        # Note: Using in_place_halo is not performant, as discussed in Issue https://github.com/tenstorrent/tt-metal/issues/23184
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
            in_place_halo=True,
        )
        outputs = []
        # Layer 1
        x = self.layer1_0(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.layer1_1(x)
        x = self.layer1_2(x)

        # Layer 2
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        # Layer 3
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x = self.layer3_5(x)

        # Layer 4
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)
        outputs.append(x)

        return outputs
