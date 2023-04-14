import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")

from typing import Type, Union, Optional, List, Callable

import torch
import torch.nn as nn

from utils import conv3x3, conv1x1
from BasicBlock import BasicBlock
from Bottleneck import Bottleneck
from utility_functions import tt2torch_tensor, torch2tt_tensor
from libs.tt_lib.utils import pad_activation, pad_weight, print_diff_argmax
from libs import tt_lib as ttl
from python_api_testing.fused_ops.linear import Linear as TtLinear
from python_api_testing.fused_ops.softmax import softmax as TtSoftmax
from utils import pad_by_zero, unpad_from_zero

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
        device = None,
        host = None,
        state_dict = None,
        base_address = None
    ) -> None:
        super().__init__()
        self.device = device
        self.host = host
        self.base_address_with_dot = base_address # this is root layer, no dot is needed
        self.state_dict = state_dict

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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(state_dict[f"{self.base_address_with_dot}conv1.weight"])
        self.bn1 = norm_layer(self.inplanes) # batch norm
        self.bn1.weight = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.num_batches_tracked"], requires_grad=False)
        self.bn1.eval()

        self.relu = ttl.tensor.relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], name="layer1", state_dict=state_dict)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], name="layer2", state_dict=state_dict)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], name="layer3", state_dict=state_dict)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], name="layer4", state_dict=state_dict)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        fc_weight = pad_weight(state_dict[f"{self.base_address_with_dot}fc.weight"])
        fc_weight = ttl.tensor.Tensor(fc_weight.reshape(-1).tolist(), fc_weight.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()
        fc_bias = pad_weight(state_dict[f"{self.base_address_with_dot}fc.bias"])
        fc_bias = ttl.tensor.Tensor(fc_bias.reshape(-1).tolist(), fc_bias.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        self.fc = TtLinear(512 * block.expansion, 1024, fc_weight, fc_bias, self.device) # num_classes = 1000
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules(): #Note: Commented out since we are loading weights

        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            assert False, "we should never get here!"
            # for m in self.modules():
            #     if isinstance(m, Bottleneck) and m.bn3.weight is not None:
            #         nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
            #     elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
            #         nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        name: str = None,
        state_dict = None
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            nl = norm_layer(planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, state_dict=self.state_dict, base_address=f"{self.base_address_with_dot}{name}.0.downsample.0"),
                nl,

            )
            nl.weight = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.weight"])
            nl.bias = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.bias"])
            nl.running_mean = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.running_mean"])
            nl.running_var = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.running_var"])
            nl.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.num_batches_tracked"], requires_grad=False)
            nl.eval()

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
                host=self.host,
                state_dict=self.state_dict,
                base_address=f"{self.base_address_with_dot}{name}.0"
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
                    host=self.host,
                    state_dict=self.state_dict,
                    base_address=f"{self.base_address_with_dot}{name}.{_}",
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)

        x = torch2tt_tensor(x, self.device)
        x = self.relu(x)
        x = tt2torch_tensor(x, self.host)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1).unsqueeze(1).unsqueeze(1)

        x, x_initial_shape = pad_by_zero(x, self.device)
        x = self.fc(x)
        desired_shape = [x.shape()[0], x.shape()[1], 1, 1000]
        x = unpad_from_zero(x, desired_shape, self.host)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)
