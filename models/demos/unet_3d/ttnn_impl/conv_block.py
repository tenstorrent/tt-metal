# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import ttnn

from .batch_norm3d import BatchNorm3D
from .conv3d import Conv3D


class ConvBlock:
    def __init__(self, device, in_channels: int, out_channels: int, kernel_size: int = 3):
        self.conv = Conv3D(device, in_channels, out_channels, kernel_size)
        self.bn = BatchNorm3D(device, out_channels)

    def init_params(self, device, params_dict: dict[str, torch.Tensor], module_prefix: Optional[str] = None):
        conv_prefix = f"{module_prefix}.conv" if module_prefix else "conv"
        bn_prefix = f"{module_prefix}.bn" if module_prefix else "bn"
        self.conv.init_params(device, params_dict, conv_prefix)
        self.bn.init_params(device, params_dict, bn_prefix)

    def __call__(self, x0) -> ttnn.Tensor:
        x1 = self.conv(x0)
        ttnn.deallocate(x0)
        # permute N D H W C to N C D H W for batchnorm
        x2 = ttnn.permute(x1, (0, 4, 1, 2, 3))
        x3 = self.bn(x2)
        ttnn.deallocate(x2)
        # permute back to N D H W C
        x4 = ttnn.permute(x3, (0, 2, 3, 4, 1))
        x5 = ttnn.relu(x4)
        ttnn.deallocate(x4)
        return x5
