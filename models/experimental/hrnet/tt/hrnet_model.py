# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import logging
import torch.nn as nn
import timm

import ttnn
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch_to_tt_tensor_rm

from models.helper_funcs import Linear as TtLinear

from models.experimental.hrnet.tt.basicblock import TtBasicBlock
from models.experimental.hrnet.tt.bottleneck import TtBottleneck
from models.experimental.hrnet.tt.high_resolution_module import TtHighResolutionModule


from models.experimental.hrnet.hrnet_utils import create_batchnorm


logger = logging.getLogger(__name__)

config = dict(
    stem_width=64,
    stage1=dict(
        num_modules=1,
        num_branches=1,
        block_type="BOTTLENECK",
        num_blocks=(1,),
        num_channels=(32,),
        fuse_method="SUM",
    ),
    stage2=dict(
        num_modules=1,
        num_branches=2,
        block_type="BASIC",
        num_blocks=(2, 2),
        num_channels=(16, 32),
        fuse_method="SUM",
    ),
    stage3=dict(
        num_modules=1,
        num_branches=3,
        block_type="BASIC",
        num_blocks=(2, 2, 2),
        num_channels=(16, 32, 64),
        fuse_method="SUM",
    ),
    stage4=dict(
        num_modules=1,
        num_branches=4,
        block_type="BASIC",
        num_blocks=(2, 2, 2, 2),
        num_channels=(16, 32, 64, 128),
        fuse_method="SUM",
    ),
)

blocks_dict = {"BASIC": TtBasicBlock, "BOTTLENECK": TtBottleneck}


class TtHighResolutionNet(nn.Module):
    def __init__(self, state_dict, device, multi_scale_output=True):
        super(TtHighResolutionNet, self).__init__()
        self.cfg = config
        self.base_address = ""
        self.device = device
        self.state_dict = state_dict

        self.conv1_weight = torch_to_tt_tensor_rm(
            self.state_dict["conv1.weight"],
            self.device,
            put_on_device=False,
        )
        self.bn1 = create_batchnorm(
            64,
            self.state_dict,
            "bn1",
            self.device,
        )
        self.conv1 = fallback_ops.Conv2d(
            self.conv1_weight,
            biases=None,
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv2_weight = torch_to_tt_tensor_rm(
            self.state_dict["conv2.weight"],
            self.device,
            put_on_device=False,
        )
        self.bn2 = create_batchnorm(
            64,
            self.state_dict,
            "bn2",
            self.device,
        )
        self.conv2 = fallback_ops.Conv2d(
            self.conv2_weight,
            biases=None,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.relu = ttnn.relu

        self.stage1_cfg = self.cfg["stage1"]
        num_channels = self.stage1_cfg["num_channels"][0]
        block = blocks_dict[self.stage1_cfg["block_type"]]
        num_blocks = self.stage1_cfg["num_blocks"][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks, base_address="layer1.0")
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = self.cfg["stage2"]
        num_channels = self.stage2_cfg["num_channels"]
        block = blocks_dict[self.stage2_cfg["block_type"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = self.cfg["stage3"]
        num_channels = self.stage3_cfg["num_channels"]
        block = blocks_dict[self.stage3_cfg["block_type"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = self.cfg["stage4"]
        num_channels = self.stage4_cfg["num_channels"]
        block = blocks_dict[self.stage4_cfg["block_type"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        # Classification Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(pre_stage_channels)
        self.avg_pool2d = fallback_ops.AdaptiveAvgPool2d(1)
        self.linear_weight = torch_to_tt_tensor_rm(
            self.state_dict[f"classifier.weight"],
            self.device,
            put_on_device=False,
        )
        self.linear_bias = torch_to_tt_tensor_rm(
            self.state_dict[f"classifier.bias"],
            self.device,
            put_on_device=False,
        )
        self.classifier = TtLinear(2048, 1000, self.linear_weight, self.linear_bias)

    def _make_head(self, pre_stage_channels):
        head_block = TtBottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(
                head_block,
                channels,
                head_channels[i],
                1,
                stride=1,
                base_address=f"incre_modules.{i}.0",
            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            self.ds_conv_weight = torch_to_tt_tensor_rm(
                self.state_dict[f"downsamp_modules.{i}.0.weight"],
                self.device,
                put_on_device=False,
            )
            self.ds_bn = create_batchnorm(out_channels, self.state_dict, f"downsamp_modules.{i}.1", self.device)
            downsamp_module = [
                fallback_ops.Conv2d(
                    self.ds_conv_weight,
                    biases=None,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                self.ds_bn,
                ttnn.relu,
            ]

            downsamp_modules.append(downsamp_module)

        self.final_conv_weight = torch_to_tt_tensor_rm(
            self.state_dict[f"final_layer.0.weight"],
            self.device,
            put_on_device=False,
        )
        self.final_conv_bias = torch_to_tt_tensor_rm(
            self.state_dict[f"final_layer.0.bias"],
            self.device,
            put_on_device=False,
        )
        self.final_bn = create_batchnorm(2048, self.state_dict, f"final_layer.1", self.device)
        final_layer = [
            fallback_ops.Conv2d(
                self.final_conv_weight,
                self.final_conv_bias,
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            self.final_bn,
            ttnn.relu,
        ]

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    self.conv_weight = torch_to_tt_tensor_rm(
                        self.state_dict[f"transition{num_branches_pre}.{i}.0.weight"],
                        self.device,
                        put_on_device=False,
                    )
                    self.bn = create_batchnorm(
                        num_channels_cur_layer[i],
                        self.state_dict,
                        f"transition{num_branches_pre}.{i}.1",
                        self.device,
                    )

                    transition_layers.append(
                        [
                            fallback_ops.Conv2d(
                                self.conv_weight,
                                biases=None,
                                in_channels=num_channels_pre_layer[i],
                                out_channels=num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                            self.bn,
                            ttnn.relu,
                        ]
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    self.conv_weight = torch_to_tt_tensor_rm(
                        self.state_dict[f"transition{num_branches_pre}.{i}.{j}.0.weight"],
                        self.device,
                        put_on_device=False,
                    )
                    self.bn = create_batchnorm(
                        outchannels,
                        self.state_dict,
                        f"transition{num_branches_pre}.{i}.{j}.1",
                        self.device,
                    )
                    conv3x3s.append(
                        [
                            fallback_ops.Conv2d(
                                self.conv_weight,
                                biases=None,
                                in_channels=inchannels,
                                out_channels=outchannels,
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
                if not transition_layers:
                    merged_list = [merged_list]
                transition_layers.append(merged_list)
        return transition_layers

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, base_address=""):
        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                self.state_dict,
                base_address,
                self.device,
                stride=1,
            )
        )

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["num_modules"]
        num_branches = layer_config["num_branches"]
        num_blocks = layer_config["num_blocks"]
        num_channels = layer_config["num_channels"]
        block = blocks_dict[layer_config["block_type"]]
        fuse_method = layer_config["fuse_method"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                TtHighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    self.state_dict,
                    self.base_address + f"stage{num_branches}.{0}",
                    self.device,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x: ttnn.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["num_branches"]):
            if self.transition1[i] is not None:
                t1_list = self.transition1[i]
                y = t1_list[0](x)
                for j in range(1, len(t1_list)):
                    y = t1_list[j](y)
                x_list.append(y)
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["num_branches"]):
            if self.transition2[i] is not None:
                t2_list = self.transition2[i]
                y = t2_list[0](y_list[-1])
                for j in range(1, len(t2_list)):
                    y = t2_list[j](y)
                x_list.append(y)
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["num_branches"]):
            if self.transition3[i] is not None:
                t3_list = self.transition3[i]
                y = t3_list[0](y_list[-1])
                for j in range(1, len(t3_list)):
                    y = t3_list[j](y)
                x_list.append(y)
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Classification Head
        y = self.incre_modules[0](y_list[0])

        for i in range(len(self.downsamp_modules)):
            for module in self.downsamp_modules[i]:
                y = module(y)
            y = ttnn.add(self.incre_modules[i + 1](y_list[i + 1]), y)

        for module in self.final_layer:
            y = module(y)

        y = self.avg_pool2d(y)
        y = ttnn.permute(y, (0, 3, 2, 1))
        y = self.classifier(y)

        return y


def _hrnet_for_image_classification(device, state_dict, base_address="") -> TtHighResolutionNet:
    return TtHighResolutionNet(state_dict, device, multi_scale_output=True)


def hrnet_w18_small(device, multi_scale_output=True) -> TtHighResolutionNet:
    torch_model = timm.create_model("hrnet_w18_small", pretrained=True)
    state_dict = torch_model.state_dict()
    tt_model = _hrnet_for_image_classification(device, state_dict)

    return tt_model
