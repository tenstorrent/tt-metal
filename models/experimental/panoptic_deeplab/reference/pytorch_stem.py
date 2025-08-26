# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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
        print(f"[STEM] Starting stem processing with input shape: {x.shape}")

        # Conv1 + BatchNorm + ReLU
        print("[STEM] Processing conv1...")
        x = self.conv1(x)
        x = self.conv1.norm(x)
        x = F.relu(x)
        print(f"[STEM] Conv1 complete, shape: {x.shape}")

        # Conv2 + BatchNorm + ReLU
        print("[STEM] Processing conv2...")
        x = self.conv2(x)
        x = self.conv2.norm(x)
        x = F.relu(x)
        print(f"[STEM] Conv2 complete, shape: {x.shape}")

        # Conv3 + BatchNorm + ReLU
        print("[STEM] Processing conv3...")
        x = self.conv3(x)
        x = self.conv3.norm(x)
        x = F.relu(x)
        print(f"[STEM] Conv3 complete, shape: {x.shape}")

        # Max pooling with kernel_size=3, stride=2, padding=1
        print("[STEM] Processing max pooling...")
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        print(f"[STEM] Stem processing complete, final shape: {x.shape}")

        return x
