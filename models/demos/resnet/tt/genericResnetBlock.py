# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Type, Union, Optional, List, Callable

import ttnn
import torch
import torch.nn as nn
import math
from models.demos.resnet.utils import fold_bn_to_conv_weights_bias
from models.utility_functions import (
    tt2torch_tensor,
    is_conv_supported_on_device,
)
from tt_lib.utils import pad_weight

from tt_lib.fused_ops.average_pool import run_avg_pool_on_device_wrapper as TtAvgPool
from tt_lib.fused_ops.linear import Linear as TtLinear
from tt_lib.fused_ops.conv import conv as TtConv
from tt_lib.fallback_ops import fallback_ops


def _nearest_y(x, y):
    return math.ceil(x / y) * y


# Local copy of unpad_from_zero to always set output to
def unpad_from_zero(x, desired_shape):
    if x.get_legacy_shape()[-1] == desired_shape[-1] and x.get_legacy_shape()[-2] == desired_shape[-2]:
        x = tt2torch_tensor(x)
    else:
        x = x.cpu()
        if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            x = x.to(ttnn.ROW_MAJOR_LAYOUT)
        x = x.unpad(
            (0, 0, 0, 0),
            (
                desired_shape[0] - 1,
                desired_shape[1] - 1,
                desired_shape[2] - 1,
                desired_shape[3] - 1,
            ),
        )
        x = x.to_torch().to(torch.float)
    return x


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device=None,
        state_dict=None,
        base_address=None,
        fold_batchnorm=False,
        downsample_conv_on_tt=None,
        norm_layer_after_downsample_conv_on_tt=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.state_dict = state_dict
        self.base_address = base_address
        self.fold_batchnorm = fold_batchnorm
        self.downsample_conv_on_tt = downsample_conv_on_tt
        self.norm_layer_after_downsample_conv_on_tt = norm_layer_after_downsample_conv_on_tt
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        conv1_weight = state_dict[f"{base_address}.conv1.weight"]
        conv1_bias = None

        self.bn1 = norm_layer(width)
        self.bn1.weight = nn.Parameter(state_dict[f"{self.base_address}.bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"{self.base_address}.bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address}.bn1.num_batches_tracked"],
            requires_grad=False,
        )
        self.bn1.eval()

        conv2_weight = state_dict[f"{base_address}.conv2.weight"]
        conv2_bias = None

        self.bn2 = norm_layer(width)
        self.bn2.weight = nn.Parameter(state_dict[f"{self.base_address}.bn2.weight"])
        self.bn2.bias = nn.Parameter(state_dict[f"{self.base_address}.bn2.bias"])
        self.bn2.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_mean"])
        self.bn2.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_var"])
        self.bn2.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address}.bn2.num_batches_tracked"],
            requires_grad=False,
        )
        self.bn2.eval()

        conv3_weight = state_dict[f"{base_address}.conv3.weight"]
        conv3_bias = None

        self.bn3 = norm_layer(planes * self.expansion)
        self.bn3.weight = nn.Parameter(state_dict[f"{self.base_address}.bn3.weight"])
        self.bn3.bias = nn.Parameter(state_dict[f"{self.base_address}.bn3.bias"])
        self.bn3.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn3.running_mean"])
        self.bn3.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn3.running_var"])
        self.bn3.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address}.bn3.num_batches_tracked"],
            requires_grad=False,
        )
        self.bn3.eval()

        self.relu = ttnn.relu
        self.downsample = downsample
        self.stride = stride

        if self.fold_batchnorm:
            conv1_weight, conv1_bias = fold_bn_to_conv_weights_bias(conv1_weight, self.bn1)
            conv2_weight, conv2_bias = fold_bn_to_conv_weights_bias(conv2_weight, self.bn2)
            conv3_weight, conv3_bias = fold_bn_to_conv_weights_bias(conv3_weight, self.bn3)
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        self.conv1_params = [width, inplanes, 1, 1, 1, 1, 0, 0, dilation, groups]
        if is_conv_supported_on_device(self.conv1_params):
            self.conv1 = TtConv(
                conv1_weight.reshape(-1).tolist(),
                self.conv1_params,
                self.device,
                conv1_bias.tolist() if conv1_bias is not None else None,
            )
        else:
            self.conv1 = fallback_ops.Conv2d(
                conv1_weight,
                conv1_bias,
                inplanes,
                width,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        self.conv2_params = [width, width, 3, 3, stride, stride, 1, 1, dilation, groups]
        if is_conv_supported_on_device(self.conv2_params):
            self.conv2 = TtConv(
                conv2_weight.reshape(-1).tolist(),
                self.conv2_params,
                self.device,
                conv2_bias.tolist() if conv2_bias is not None else None,
            )
        else:
            self.conv2 = fallback_ops.Conv2d(
                conv2_weight,
                conv2_bias,
                width,
                width,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.conv3_params = [
            planes * self.expansion,
            width,
            1,
            1,
            1,
            1,
            0,
            0,
            dilation,
            groups,
        ]
        if is_conv_supported_on_device(self.conv3_params):
            self.conv3 = TtConv(
                conv3_weight.reshape(-1).tolist(),
                self.conv3_params,
                self.device,
                conv3_bias.tolist() if conv3_bias is not None else None,
            )
        else:
            self.conv3 = fallback_ops.Conv2d(
                conv3_weight,
                conv3_bias,
                width,
                planes * self.expansion,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        if not self.fold_batchnorm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if not self.fold_batchnorm:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if not self.fold_batchnorm:
            out = self.bn3(out)

        if self.downsample_conv_on_tt is not None:
            identity = self.downsample_conv_on_tt(x)
            assert self.norm_layer_after_downsample_conv_on_tt is not None
            if not self.fold_batchnorm:
                identity = self.norm_layer_after_downsample_conv_on_tt(identity)
        elif self.downsample is not None:
            identity = self.downsample(x)

        out = ttnn.add(out, identity)

        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device=None,
        state_dict=None,
        base_address=None,
        fold_batchnorm=False,
        downsample_conv_on_tt=None,
        norm_layer_after_downsample_conv_on_tt=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.base_address = base_address
        self.fold_batchnorm = fold_batchnorm
        self.downsample_conv_on_tt = downsample_conv_on_tt
        self.norm_layer_after_downsample_conv_on_tt = norm_layer_after_downsample_conv_on_tt
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        conv1_weight = state_dict[f"{base_address}.conv1.weight"]
        conv1_bias = None

        self.bn1 = norm_layer(planes)
        self.bn1.weight = nn.Parameter(state_dict[f"{self.base_address}.bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"{self.base_address}.bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address}.bn1.num_batches_tracked"],
            requires_grad=False,
        )
        self.bn1.eval()

        self.relu = ttnn.relu

        conv2_weight = state_dict[f"{base_address}.conv2.weight"]
        conv2_bias = None

        self.bn2 = norm_layer(planes)

        self.bn2.weight = nn.Parameter(state_dict[f"{self.base_address}.bn2.weight"])
        self.bn2.bias = nn.Parameter(state_dict[f"{self.base_address}.bn2.bias"])
        self.bn2.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_mean"])
        self.bn2.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_var"])
        self.bn2.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address}.bn2.num_batches_tracked"],
            requires_grad=False,
        )
        self.bn2.eval()

        if self.fold_batchnorm:
            conv1_weight, conv1_bias = fold_bn_to_conv_weights_bias(conv1_weight, self.bn1)
            conv2_weight, conv2_bias = fold_bn_to_conv_weights_bias(conv2_weight, self.bn2)
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        self.conv1_params = [
            planes,
            inplanes,
            3,
            3,
            stride,
            stride,
            1,
            1,
            dilation,
            groups,
        ]
        if is_conv_supported_on_device(self.conv1_params):
            self.conv1 = TtConv(
                conv1_weight.reshape(-1).tolist(),
                self.conv1_params,
                self.device,
                conv1_bias.tolist() if conv1_bias is not None else None,
            )
        else:
            self.conv1 = fallback_ops.Conv2d(
                conv1_weight,
                conv1_bias,
                inplanes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.conv2_params = [planes, planes, 3, 3, 1, 1, 1, 1, dilation, groups]
        if is_conv_supported_on_device(self.conv2_params):
            self.conv2 = TtConv(
                conv2_weight.reshape(-1).tolist(),
                self.conv2_params,
                self.device,
                conv2_bias.tolist() if conv2_bias is not None else None,
            )
        else:
            self.conv2 = fallback_ops.Conv2d(
                conv2_weight,
                conv2_bias,
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        if not self.fold_batchnorm:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        if not self.fold_batchnorm:
            out = self.bn2(out)

        if self.downsample_conv_on_tt is not None:
            identity = self.downsample_conv_on_tt(x)
            assert self.norm_layer_after_downsample_conv_on_tt is not None
            if not self.fold_batchnorm:
                identity = self.norm_layer_after_downsample_conv_on_tt(identity)
        elif self.downsample is not None:
            identity = self.downsample(x)

        out = ttnn.add(out, identity)

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device=None,
        state_dict=None,
        base_address=None,
        fold_batchnorm=False,
    ) -> None:
        super().__init__()
        self.device = device
        self.base_address_with_dot = base_address  # this is root layer, no dot is needed
        self.state_dict = state_dict
        self.fold_batchnorm = fold_batchnorm

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        conv1_weight = state_dict[f"{self.base_address_with_dot}conv1.weight"]
        conv1_bias = None

        self.bn1 = norm_layer(self.inplanes)  # batch norm
        self.bn1.weight = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address_with_dot}bn1.num_batches_tracked"],
            requires_grad=False,
        )
        self.bn1.eval()

        if self.fold_batchnorm:
            conv1_weight, conv1_bias = fold_bn_to_conv_weights_bias(conv1_weight, self.bn1)
            self.bn1 = nn.Identity()

        self.conv1_params = [self.inplanes, 3, 7, 7, 2, 2, 3, 3, 1, groups]
        if is_conv_supported_on_device(self.conv1_params):
            self.conv1 = TtConv(
                conv1_weight.reshape(-1).tolist(),
                self.conv1_params,
                self.device,
                conv1_bias.tolist() if conv1_bias is not None else None,
            )
        else:
            self.conv1 = fallback_ops.Conv2d(
                conv1_weight,
                conv1_bias,
                3,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
            )

        self.relu = ttnn.relu
        self.maxpool = fallback_ops.MaxPool2d(kernel_size=3, stride=2, padding=1, channels_last=True)
        self.layer1 = self._make_layer(block, 64, layers[0], name="layer1", state_dict=state_dict)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            name="layer2",
            state_dict=state_dict,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            name="layer3",
            state_dict=state_dict,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            name="layer4",
            state_dict=state_dict,
        )
        self.avgpool = TtAvgPool(self.device)

        fc_weight = pad_weight(state_dict[f"{self.base_address_with_dot}fc.weight"])
        fc_weight = ttnn.Tensor(
            fc_weight.reshape(-1).tolist(),
            fc_weight.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        ).to(ttnn.TILE_LAYOUT)
        fc_bias = pad_weight(state_dict[f"{self.base_address_with_dot}fc.bias"])
        fc_bias = ttnn.Tensor(
            fc_bias.reshape(-1).tolist(),
            fc_bias.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        ).to(ttnn.TILE_LAYOUT)

        self.fc = TtLinear(512 * block.expansion, 1024, fc_weight, fc_bias, self.device)  # num_classes = 1000
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        name: str = None,
        state_dict=None,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        self.downsample_conv_on_tt = None
        self.norm_layer_after_downsample_conv_on_tt = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            nl = norm_layer(planes * block.expansion)
            nl.weight = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.weight"])
            nl.bias = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.bias"])
            nl.running_mean = nn.Parameter(
                state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.running_mean"]
            )
            nl.running_var = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.running_var"])
            nl.num_batches_tracked = nn.Parameter(
                state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.num_batches_tracked"],
                requires_grad=False,
            )
            nl.eval()
            downsample_conv_weight = state_dict[f"{self.base_address_with_dot}{name}.0.downsample.0.weight"]
            downsample_conv_bias = None

            if self.fold_batchnorm:
                (
                    downsample_conv_weight,
                    downsample_conv_bias,
                ) = fold_bn_to_conv_weights_bias(downsample_conv_weight, nl)
                nl = nn.Identity()

            self.downsample_params = [
                planes * block.expansion,
                self.inplanes,
                1,
                1,
                stride,
                stride,
                0,
                0,
                self.dilation,
                1,
            ]
            if is_conv_supported_on_device(self.downsample_params):
                self.downsample_conv_on_tt = TtConv(
                    downsample_conv_weight.reshape(-1).tolist(),
                    self.downsample_params,
                    self.device,
                    downsample_conv_bias.tolist() if downsample_conv_bias is not None else None,
                )
                self.norm_layer_after_downsample_conv_on_tt = nl
            else:
                downsample_conv = fallback_ops.Conv2d(
                    downsample_conv_weight,
                    downsample_conv_bias,
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                )
                downsample = nn.Sequential(
                    downsample_conv,
                    nl,
                )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                device=self.device,
                state_dict=self.state_dict,
                base_address=f"{self.base_address_with_dot}{name}.0",
                fold_batchnorm=self.fold_batchnorm,
                downsample_conv_on_tt=self.downsample_conv_on_tt,
                norm_layer_after_downsample_conv_on_tt=self.norm_layer_after_downsample_conv_on_tt,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    device=self.device,
                    state_dict=self.state_dict,
                    base_address=f"{self.base_address_with_dot}{name}.{_}",
                    fold_batchnorm=self.fold_batchnorm,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.permute(x, (0, 2, 3, 1))
        x = ttnn.Tensor(x, ttnn.bfloat16)
        x = x.pad(
            (
                x.get_legacy_shape()[0],
                x.get_legacy_shape()[1],
                x.get_legacy_shape()[2],
                _nearest_y(x.get_legacy_shape()[3], 16),
            ),
            (0, 0, 0, 0),
            0,
        )
        x = x.to(self.device)
        x = self.conv1(x)
        if not self.fold_batchnorm:
            x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # x = fallback_ops.reshape(x, 1, 1, 1, x.get_legacy_shape()[3])
        # x = torch.flatten(x, 1).unsqueeze(1).unsqueeze(1)

        x = self.fc(x)
        desired_shape = [x.get_legacy_shape()[0], x.get_legacy_shape()[1], 1, 1000]
        x = unpad_from_zero(x, desired_shape)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)
