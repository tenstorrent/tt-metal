# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.experimental.vadv2.tt.common import TtnnConv2D
from models.experimental.vadv2.tt.ttnn_bottleneck import TtnnBottleneck


class TtnnResnet50:
    def __init__(self, conv_args, conv_pth, device):
        self.maxpool_args = conv_args.maxpool
        self.device = device
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = TtnnConv2D(conv_args.conv1, conv_pth.conv1, device=self.device, activation="relu", act_block_h=32)

        # Layer 1
        self.layer1_0 = TtnnBottleneck(
            conv_args.layer1[0],
            conv_pth.layer1_0,
            device=self.device,
            is_downsample=True,
        )
        self.layer1_1 = TtnnBottleneck(conv_args.layer1[1], conv_pth.layer1_1, device=self.device)
        self.layer1_2 = TtnnBottleneck(conv_args.layer1[2], conv_pth.layer1_2, device=self.device)

        # Layer 2
        self.layer2_0 = TtnnBottleneck(
            conv_args.layer2[0],
            conv_pth.layer2_0,
            device=self.device,
            is_downsample=True,
            blk_sharded=True,
            activation_dtype=ttnn.bfloat8_b,
        )
        self.layer2_1 = TtnnBottleneck(conv_args.layer2[1], conv_pth.layer2_1, device=self.device)
        self.layer2_2 = TtnnBottleneck(conv_args.layer2[2], conv_pth.layer2_2, device=self.device)
        self.layer2_3 = TtnnBottleneck(conv_args.layer2[3], conv_pth.layer2_3, device=self.device)

        # Layer 3
        self.layer3_0 = TtnnBottleneck(
            conv_args.layer3[0],
            conv_pth.layer3_0,
            device=self.device,
            is_downsample=True,
            blk_sharded=True,
            activation_dtype=ttnn.bfloat8_b,
        )
        self.layer3_1 = TtnnBottleneck(conv_args.layer3[1], conv_pth.layer3_1, device=self.device)
        self.layer3_2 = TtnnBottleneck(conv_args.layer3[2], conv_pth.layer3_2, device=self.device)
        self.layer3_3 = TtnnBottleneck(conv_args.layer3[3], conv_pth.layer3_3, device=self.device)
        self.layer3_4 = TtnnBottleneck(conv_args.layer3[4], conv_pth.layer3_4, device=self.device)
        self.layer3_5 = TtnnBottleneck(conv_args.layer3[5], conv_pth.layer3_5, device=self.device)

        # Layer 4
        self.layer4_0 = TtnnBottleneck(
            conv_args.layer4[0],
            conv_pth.layer4_0,
            device=self.device,
            is_downsample=True,
            blk_sharded=True,
            activation_dtype=ttnn.bfloat8_b,
            conv3_blk_sharded=True,
        )
        self.layer4_1 = TtnnBottleneck(
            conv_args.layer4[1], conv_pth.layer4_1, device=self.device, conv3_blk_sharded=True
        )
        self.layer4_2 = TtnnBottleneck(
            conv_args.layer4[2], conv_pth.layer4_2, device=self.device, conv3_blk_sharded=True
        )

    def __call__(self, x, batch_size=1):
        x, out_ht, out_wdth = self.conv1(x)

        x = ttnn.to_torch(x)
        x = x.reshape(6, 192, 320, 64).to(torch.float32)
        x = x.permute(0, 3, 1, 2)

        x = self.maxpool(x)

        x = x.permute(0, 2, 3, 1).to(torch.float32)
        x = x.reshape(1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3])
        x = ttnn.from_torch(x, device=self.device, dtype=ttnn.bfloat16)

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
        outputs.append(x)
        # Layer 3
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x = self.layer3_5(x)
        outputs.append(x)

        # Layer 4
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)
        outputs.append(x)

        return outputs
