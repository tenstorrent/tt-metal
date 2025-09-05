# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import Dict

from models.experimental.panoptic_deeplab.reference.pytorch_stem import StemBlock
from models.experimental.panoptic_deeplab.reference.pytorch_bottleneck import BottleneckBlock


class ResNet(nn.Module):
    """
    PyTorch implementation of ResNet backbone for Panoptic DeepLab.

    Based on the model structure, the ResNet contains:
    - stem: DeepLabStem (3 conv layers)
    - res2: Sequential with 3 BottleneckBlocks (first has shortcut)
    - res3: Sequential with 4 BottleneckBlocks (first has shortcut with stride=2)
    - res4: Sequential with 6 BottleneckBlocks (first has shortcut with stride=2)
    - res5: Sequential with 3 BottleneckBlocks (first has shortcut, uses dilated convolutions)
    """

    def __init__(self):
        super().__init__()

        # Initialize stem
        self.stem = StemBlock()

        # Initialize res2 (3 blocks, first has shortcut)
        # Input: 128 channels from stem, Output: 256 channels
        self.res2 = nn.Sequential(
            # Block 0: has shortcut, 128->256 channels
            BottleneckBlock(
                in_channels=128,
                bottleneck_channels=64,
                out_channels=256,
                stride=1,
                dilation=1,
                has_shortcut=True,
                shortcut_stride=1,
            ),
            # Block 1: no shortcut, 256->256 channels
            BottleneckBlock(
                in_channels=256, bottleneck_channels=64, out_channels=256, stride=1, dilation=1, has_shortcut=False
            ),
            # Block 2: no shortcut, 256->256 channels
            BottleneckBlock(
                in_channels=256, bottleneck_channels=64, out_channels=256, stride=1, dilation=1, has_shortcut=False
            ),
        )

        # Initialize res3 (4 blocks, first has shortcut with stride=2)
        # Input: 256 channels, Output: 512 channels
        self.res3 = nn.Sequential(
            # Block 0: has shortcut with stride=2, 256->512 channels
            BottleneckBlock(
                in_channels=256,
                bottleneck_channels=128,
                out_channels=512,
                stride=2,
                dilation=1,
                has_shortcut=True,
                shortcut_stride=2,
            ),
            # Block 1: no shortcut, 512->512 channels
            BottleneckBlock(
                in_channels=512, bottleneck_channels=128, out_channels=512, stride=1, dilation=1, has_shortcut=False
            ),
            # Block 2: no shortcut, 512->512 channels
            BottleneckBlock(
                in_channels=512, bottleneck_channels=128, out_channels=512, stride=1, dilation=1, has_shortcut=False
            ),
            # Block 3: no shortcut, 512->512 channels
            BottleneckBlock(
                in_channels=512, bottleneck_channels=128, out_channels=512, stride=1, dilation=1, has_shortcut=False
            ),
        )

        # Initialize res4 (6 blocks, first has shortcut with stride=2)
        # Input: 512 channels, Output: 1024 channels
        self.res4 = nn.Sequential(
            # Block 0: has shortcut with stride=2, 512->1024 channels
            BottleneckBlock(
                in_channels=512,
                bottleneck_channels=256,
                out_channels=1024,
                stride=2,
                dilation=1,
                has_shortcut=True,
                shortcut_stride=2,
            ),
            # Block 1: no shortcut, 1024->1024 channels
            BottleneckBlock(
                in_channels=1024, bottleneck_channels=256, out_channels=1024, stride=1, dilation=1, has_shortcut=False
            ),
            # Block 2: no shortcut, 1024->1024 channels
            BottleneckBlock(
                in_channels=1024, bottleneck_channels=256, out_channels=1024, stride=1, dilation=1, has_shortcut=False
            ),
            # Block 3: no shortcut, 1024->1024 channels
            BottleneckBlock(
                in_channels=1024, bottleneck_channels=256, out_channels=1024, stride=1, dilation=1, has_shortcut=False
            ),
            # Block 4: no shortcut, 1024->1024 channels
            BottleneckBlock(
                in_channels=1024, bottleneck_channels=256, out_channels=1024, stride=1, dilation=1, has_shortcut=False
            ),
            # Block 5: no shortcut, 1024->1024 channels
            BottleneckBlock(
                in_channels=1024, bottleneck_channels=256, out_channels=1024, stride=1, dilation=1, has_shortcut=False
            ),
        )

        # Initialize res5 (3 blocks, first has shortcut, uses dilated convolutions)
        # Input: 1024 channels, Output: 2048 channels
        # Note: Based on model structure, res5 uses dilated convolutions (dilation=2,4,8)
        self.res5 = nn.Sequential(
            # Block 0: has shortcut, 1024->2048 channels, dilation=2
            BottleneckBlock(
                in_channels=1024,
                bottleneck_channels=512,
                out_channels=2048,
                stride=1,
                dilation=2,
                has_shortcut=True,
                shortcut_stride=1,
            ),
            # Block 1: no shortcut, 2048->2048 channels, dilation=4
            BottleneckBlock(
                in_channels=2048, bottleneck_channels=512, out_channels=2048, stride=1, dilation=4, has_shortcut=False
            ),
            # Block 2: no shortcut, 2048->2048 channels, dilation=8
            BottleneckBlock(
                in_channels=2048, bottleneck_channels=512, out_channels=2048, stride=1, dilation=8, has_shortcut=False
            ),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ResNet backbone.

        Args:
            x: Input tensor of shape [batch_size, 3, height, width]

        Returns:
            Dictionary containing intermediate feature maps:
            - "res2": Output from res2 layer (256 channels)
            - "res3": Output from res3 layer (512 channels)
            - "res4": Output from res4 layer (1024 channels)
            - "res5": Output from res5 layer (2048 channels)
        """
        # Stem processing
        x = self.stem(x)

        # res2
        x = self.res2(x)
        res2_out = x

        # res3
        x = self.res3(x)
        res3_out = x

        # res4
        x = self.res4(x)
        res4_out = x

        # res5
        x = self.res5(x)
        res5_out = x

        return {
            "res2": res2_out,
            "res3": res3_out,
            "res4": res4_out,
            "res5": res5_out,
        }

    def forward_single_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only the final output (res5).

        Args:
            x: Input tensor of shape [batch_size, 3, height, width]

        Returns:
            Final feature map from res5 layer (2048 channels)
        """
        # Stem processing
        x = self.stem(x)

        # res2
        x = self.res2(x)

        # res3
        x = self.res3(x)

        # res4
        x = self.res4(x)

        # res5
        x = self.res5(x)

        return x
