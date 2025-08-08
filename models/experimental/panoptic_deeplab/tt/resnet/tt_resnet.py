# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import ttnn

from .tt_stem import TtStem
from .tt_bottleneck import TtBottleneck


class TtResNet(nn.Module):
    """
    TTNN implementation of ResNet backbone for Panoptic DeepLab.

    Based on the model structure, the ResNet contains:
    - stem: DeepLabStem (3 conv layers)
    - res2: Sequential with 3 BottleneckBlocks (first has shortcut)
    - res3: Sequential with 4 BottleneckBlocks (first has shortcut with stride=2)
    - res4: Sequential with 6 BottleneckBlocks (first has shortcut with stride=2)
    - res5: Sequential with 3 BottleneckBlocks (first has shortcut, uses dilated convolutions)
    """

    def __init__(
        self,
        device: ttnn.MeshDevice,
        state_dict: dict[str, torch.Tensor],
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device

        # Initialize stem
        stem_state = {k.replace("stem.", ""): v for k, v in state_dict.items() if k.startswith("stem.")}
        self.stem = TtStem(device, stem_state, dtype)

        # Initialize res2 (3 blocks, first has shortcut)
        self.res2 = nn.ModuleList()
        for i in range(3):
            block_state = {k.replace(f"res2.{i}.", ""): v for k, v in state_dict.items() if k.startswith(f"res2.{i}.")}
            has_shortcut = i == 0  # First block has shortcut
            self.res2.append(TtBottleneck(device, block_state, dtype, has_shortcut))

        # Initialize res3 (4 blocks, first has shortcut with stride=2)
        self.res3 = nn.ModuleList()
        for i in range(4):
            block_state = {k.replace(f"res3.{i}.", ""): v for k, v in state_dict.items() if k.startswith(f"res3.{i}.")}
            has_shortcut = i == 0  # First block has shortcut
            self.res3.append(TtBottleneck(device, block_state, dtype, has_shortcut))

        # Initialize res4 (6 blocks, first has shortcut with stride=2)
        self.res4 = nn.ModuleList()
        for i in range(6):
            block_state = {k.replace(f"res4.{i}.", ""): v for k, v in state_dict.items() if k.startswith(f"res4.{i}.")}
            has_shortcut = i == 0  # First block has shortcut
            self.res4.append(TtBottleneck(device, block_state, dtype, has_shortcut))

        # Initialize res5 (3 blocks, first has shortcut, uses dilated convolutions)
        self.res5 = nn.ModuleList()
        for i in range(3):
            block_state = {k.replace(f"res5.{i}.", ""): v for k, v in state_dict.items() if k.startswith(f"res5.{i}.")}
            has_shortcut = i == 0  # First block has shortcut
            self.res5.append(TtBottleneck(device, block_state, dtype, has_shortcut))

    def forward(self, x: ttnn.Tensor) -> dict[str, ttnn.Tensor]:
        """
        Forward pass through ResNet backbone.

        Args:
            x: Input tensor of shape [batch_size, height, width, 3]

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
        for block in self.res2:
            x = block(x)
        res2_out = x

        # res3
        for block in self.res3:
            x = block(x)
        res3_out = x

        # res4
        for block in self.res4:
            x = block(x)
        res4_out = x

        # res5
        for block in self.res5:
            x = block(x)
        res5_out = x

        return {
            "res2": res2_out,
            "res3": res3_out,
            "res4": res4_out,
            "res5": res5_out,
        }

    def forward_single_output(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass returning only the final output (res5).

        Args:
            x: Input tensor of shape [batch_size, height, width, 3]

        Returns:
            Final feature map from res5 layer (2048 channels)
        """
        # Stem processing
        x = self.stem(x)

        # res2
        for block in self.res2:
            x = block(x)

        # res3
        for block in self.res3:
            x = block(x)

        # res4
        for block in self.res4:
            x = block(x)

        # res5
        for block in self.res5:
            x = block(x)

        return x
