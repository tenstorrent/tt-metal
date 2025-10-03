# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Modified from the Panoptic-DeepLab implementation in Detectron2 library
# https://github.com/facebookresearch/detectron2/tree/main/projects/Panoptic-DeepLab
# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F


class StemBlock(nn.Module):
    """
    PyTorch implementation of StemBlock for ResNet.

    Based on the model structure, the stem contains:
    - conv1: Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    - conv2: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    - conv3: Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    Each with SyncBatchNorm and ReLU activation.
    """

    def __init__(self):
        super().__init__()

        # Create conv1 with norm as a submodule to match state dict structure
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv1.norm = nn.SyncBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Create conv2 with norm as a submodule to match state dict structure
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2.norm = nn.SyncBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Create conv3 with norm as a submodule to match state dict structure
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3.norm = nn.SyncBatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1 + BatchNorm + ReLU
        x = self.conv1(x)
        x = self.conv1.norm(x)
        x = F.relu(x)

        # Conv2 + BatchNorm + ReLU
        x = self.conv2(x)
        x = self.conv2.norm(x)
        x = F.relu(x)

        # Conv3 + BatchNorm + ReLU
        x = self.conv3(x)
        x = self.conv3.norm(x)
        x = F.relu(x)

        # Max pooling with kernel_size=3, stride=2, padding=1
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        return x
