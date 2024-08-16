# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

import ttnn
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch_to_tt_tensor_rm
from models.experimental.hrnet.hrnet_utils import create_batchnorm


class TtBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, state_dict, base_address, device, stride=1):
        super(TtBasicBlock, self).__init__()
        self.stride = stride
        self.device = device

        self.conv1_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.conv1.weight"], self.device, put_on_device=False
        )
        self.conv1 = fallback_ops.Conv2d(
            self.conv1_weights,
            biases=None,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
        )

        self.bn1 = create_batchnorm(out_ch, state_dict, f"{base_address}.bn1", device)

        self.conv2_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.conv2.weight"], self.device, put_on_device=False
        )
        self.conv2 = fallback_ops.Conv2d(
            self.conv2_weights,
            biases=None,
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
        )

        self.bn2 = create_batchnorm(out_ch, state_dict, f"{base_address}.bn2", device)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = ttnn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = ttnn.add(out, residual)
        out = ttnn.relu(out)

        return out
