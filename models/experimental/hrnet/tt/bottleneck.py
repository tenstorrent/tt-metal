# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

import ttnn
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch_to_tt_tensor_rm
from models.experimental.hrnet.hrnet_utils import create_batchnorm


class TtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, state_dict, base_address, device, stride=1):
        super(TtBottleneck, self).__init__()

        self.stride = stride
        self.device = device
        self.num_inchannels = in_ch
        self.num_channels = out_ch

        self.conv1_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.conv1.weight"], self.device, put_on_device=False
        )
        self.conv1 = fallback_ops.Conv2d(
            self.conv1_weights,
            biases=None,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=stride,
            bias=False,
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
            stride=stride,
            bias=False,
        )

        self.bn2 = create_batchnorm(out_ch, state_dict, f"{base_address}.bn2", device)

        self.conv3_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.conv3.weight"], self.device, put_on_device=False
        )
        self.conv3 = fallback_ops.Conv2d(
            self.conv3_weights,
            biases=None,
            in_channels=out_ch,
            out_channels=out_ch * self.expansion,
            kernel_size=1,
            stride=stride,
            bias=False,
        )

        self.bn3 = create_batchnorm(out_ch * self.expansion, state_dict, f"{base_address}.bn3", device)

        self.conv_ds_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.downsample.0.weight"],
            self.device,
            put_on_device=False,
        )
        self.conv_ds = fallback_ops.Conv2d(
            self.conv_ds_weights,
            biases=None,
            in_channels=in_ch,
            out_channels=out_ch * self.expansion,
            kernel_size=1,
            stride=stride,
            bias=False,
        )

        self.bn_ds = create_batchnorm(out_ch * self.expansion, state_dict, f"{base_address}.downsample.1", device)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = ttnn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = ttnn.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Downsample
        if self.stride != 1 or self.num_inchannels != self.num_channels * self.expansion:
            residual = self.conv_ds(x)
            residual = self.bn_ds(residual)

        out = ttnn.add(out, residual)
        out = ttnn.relu(out)

        return out
