# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import logging

import ttnn
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch_to_tt_tensor_rm
from models.experimental.hrnet.hrnet_utils import create_batchnorm

logger = logging.getLogger(__name__)


class TtInterpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(TtInterpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: ttnn.Tensor):
        out = fallback_ops.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return out


class TtHighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        block,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method,
        state_dict,
        base_address,
        device,
        multi_scale_output=True,
    ):
        super(TtHighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.state_dict = state_dict
        self.base_address = base_address
        self.device = device

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.block = block

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()

    def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                self.state_dict,
                f"{self.base_address}.branches.{branch_index}.{0}",
                self.device,
                stride,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    self.state_dict,
                    f"{self.base_address}.branches.{branch_index}.{i}",
                    self.device,
                    stride,
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return branches

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    self.conv_weight = torch_to_tt_tensor_rm(
                        self.state_dict[f"{self.base_address}.fuse_layers.{i}.{j}.0.weight"],
                        self.device,
                        put_on_device=False,
                    )
                    self.bn = create_batchnorm(
                        num_inchannels[i],
                        self.state_dict,
                        f"{self.base_address}.fuse_layers.{i}.{j}.1",
                        self.device,
                    )

                    fuse_layer.append(
                        [
                            fallback_ops.Conv2d(
                                self.conv_weight,
                                biases=None,
                                in_channels=num_inchannels[j],
                                out_channels=num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            self.bn,
                            TtInterpolate(scale_factor=2 ** (j - i), mode="nearest"),
                        ]
                    )

                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            self.conv_weight = torch_to_tt_tensor_rm(
                                self.state_dict[f"{self.base_address}.fuse_layers.{i}.{j}.{k}.0.weight"],
                                self.device,
                                put_on_device=False,
                            )
                            self.bn = create_batchnorm(
                                num_inchannels[i],
                                self.state_dict,
                                f"{self.base_address}.fuse_layers.{i}.{j}.{k}.1",
                                self.device,
                            )
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                [
                                    fallback_ops.Conv2d(
                                        self.conv_weight,
                                        biases=None,
                                        in_channels=num_inchannels[j],
                                        out_channels=num_outchannels_conv3x3,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False,
                                    ),
                                    self.bn,
                                ]
                            )
                        else:
                            self.conv_weight = torch_to_tt_tensor_rm(
                                self.state_dict[f"{self.base_address}.fuse_layers.{i}.{j}.{k}.0.weight"],
                                self.device,
                                put_on_device=False,
                            )
                            self.bn = create_batchnorm(
                                num_inchannels[j],
                                self.state_dict,
                                f"{self.base_address}.fuse_layers.{i}.{j}.{k}.1",
                                self.device,
                            )
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                [
                                    fallback_ops.Conv2d(
                                        self.conv_weight,
                                        biases=None,
                                        in_channels=num_inchannels[j],
                                        out_channels=num_outchannels_conv3x3,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False,
                                    ),
                                    self.bn,
                                    ttnn.relu,
                                ]
                            )
                    merged_list = [item for sublist in conv3x3s for item in sublist]
                    if not fuse_layer:
                        merged_list = [merged_list]
                    fuse_layer.append(merged_list)
            fuse_layers.append(fuse_layer)

        return fuse_layers

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x: ttnn.Tensor):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0]
            if i:
                module_list = self.fuse_layers[i][0][0]
                y = module_list[0](y) if module_list[0] else y
                for k in range(1, len(module_list)):
                    y = module_list[k](y) if module_list[k] is not None else y

            for j in range(1, self.num_branches):
                if i == j:
                    y = ttnn.add(y, x[j])
                else:
                    res = x[j]
                    module_list = self.fuse_layers[i][j]
                    res = module_list[0](res) if module_list[0] else res
                    for k in range(1, len(module_list)):
                        res = module_list[k](res) if module_list[k] is not None else res
                    y = ttnn.add(y, res)
            x_fuse.append(ttnn.relu(y))

        return x_fuse
