# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
from models.demos.ufld_v2.ttnn.common import TtnnUFLDV2Conv2D
from models.demos.ufld_v2.ttnn.ttnn_basic_block import TtnnBasicBlock

class BasicBlock(nn.Module):
    """
    Original BasicBlock from the started file. 
    Keeping it for reference or if we need a torch-fallback block within this file.
    But for TTNN implementation, we primarily use TtnnBasicBlock from ufld_v2.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = ttnn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = ttnn.relu(out)
        return out


class TtnnResNet34:
    def __init__(self, conv_args, conv_pth, device):
        self.maxpool_args = conv_args.maxpool
        self.device = device
        
        # Initial Conv
        self.conv1 = TtnnUFLDV2Conv2D(
            conv_args.conv1,
            conv_pth.conv1,
            device=self.device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            dealloc_act=True,
            activation_dtype=ttnn.bfloat8_b,
        )

        # Layer 1
        self.layer1_0 = TtnnBasicBlock(conv_args.layer1[0], conv_pth.layer1_0, device=self.device, is_downsample=False, precision=ttnn.bfloat8_b)
        self.layer1_1 = TtnnBasicBlock(conv_args.layer1[1], conv_pth.layer1_1, device=self.device, is_downsample=False, precision=ttnn.bfloat8_b)
        self.layer1_2 = TtnnBasicBlock(conv_args.layer1[2], conv_pth.layer1_2, device=self.device, is_downsample=False, precision=ttnn.bfloat8_b)

        # Layer 2
        self.layer2_0 = TtnnBasicBlock(conv_args.layer2[0], conv_pth.layer2_0, device=self.device, is_downsample=True, precision=ttnn.bfloat8_b)
        self.layer2_1 = TtnnBasicBlock(conv_args.layer2[1], conv_pth.layer2_1, device=self.device, is_downsample=False, precision=ttnn.bfloat8_b)
        self.layer2_2 = TtnnBasicBlock(conv_args.layer2[2], conv_pth.layer2_2, device=self.device, is_downsample=False, precision=ttnn.bfloat8_b)
        self.layer2_3 = TtnnBasicBlock(conv_args.layer2[3], conv_pth.layer2_3, device=self.device, is_downsample=False, precision=ttnn.bfloat8_b)

        # Layer 3
        self.layer3_0 = TtnnBasicBlock(conv_args.layer3[0], conv_pth.layer3_0, device=self.device, is_downsample=True, blk_sharded=True)
        self.layer3_1 = TtnnBasicBlock(conv_args.layer3[1], conv_pth.layer3_1, device=self.device, is_downsample=False, blk_sharded=True)
        self.layer3_2 = TtnnBasicBlock(conv_args.layer3[2], conv_pth.layer3_2, device=self.device, is_downsample=False, blk_sharded=True)
        self.layer3_3 = TtnnBasicBlock(conv_args.layer3[3], conv_pth.layer3_3, device=self.device, is_downsample=False, blk_sharded=True)
        self.layer3_4 = TtnnBasicBlock(conv_args.layer3[4], conv_pth.layer3_4, device=self.device, is_downsample=False, blk_sharded=True)
        self.layer3_5 = TtnnBasicBlock(conv_args.layer3[5], conv_pth.layer3_5, device=self.device, is_downsample=False, blk_sharded=True)

        # Layer 4
        self.layer4_0 = TtnnBasicBlock(conv_args.layer4[0], conv_pth.layer4_0, device=self.device, is_downsample=True, blk_sharded=True)
        self.layer4_1 = TtnnBasicBlock(conv_args.layer4[1], conv_pth.layer4_1, device=self.device, is_downsample=False, blk_sharded=True)
        self.layer4_2 = TtnnBasicBlock(conv_args.layer4[2], conv_pth.layer4_2, device=self.device, is_downsample=False, blk_sharded=True)

    def __call__(self, input, batch_size=1, min_channels=16, shard_height_for_maxcores=16128):
        n, c, h, w = input.shape
        # Pad input to min_channels if needed (input is usually 3 channels)
        channel_padding_needed = min_channels - c
        x = ttnn.pad(input, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
        ttnn.deallocate(input)
        x = ttnn.permute(x, (0, 2, 3, 1))
        x = ttnn.reshape(x, (1, 1, n * h * w, min_channels))
        
        # Conv1
        x1, out_ht, out_wdth = self.conv1(x)
        ttnn.deallocate(x)
        
        # MaxPool
        x1 = ttnn.max_pool2d(
            x1,
            batch_size=batch_size,
            input_h=out_ht,
            input_w=out_wdth,
            channels=x1.shape[-1],
            kernel_size=[self.maxpool_args.kernel_size, self.maxpool_args.kernel_size],
            stride=[self.maxpool_args.stride, self.maxpool_args.stride],
            padding=[self.maxpool_args.padding, self.maxpool_args.padding],
            dilation=[self.maxpool_args.dilation, self.maxpool_args.dilation],
        )

        # Shard config for subsequent layers
        mem_config = ttnn.create_sharded_memory_config_(
            ttnn.Shape([shard_height_for_maxcores, x1.shape[-1]]),
            x1.memory_config().shard_spec.grid,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
        )
        x = ttnn.to_memory_config(x1, mem_config)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Layer 1
        x = self.layer1_0(x)
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

        return x
