import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Union, Optional, Callable, List


class Bottleneck(nn.Module):
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
        state_dict=None,
        base_address=""
    ) -> None:
        super().__init__()
        self.base_address = base_address
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, state_dict=state_dict, base_address=f"{base_address}.conv1")
        self.bn1 = norm_layer(width)
        self.bn1.weight = nn.Parameter(state_dict[f"{self.base_address}.bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"{self.base_address}.bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address}.bn1.num_batches_tracked"], requires_grad=False)
        self.bn1.eval()
        self.relu1 = nn.ReLU(inplace=True)


        self.conv2 = conv3x3(width, width, stride, groups, dilation, state_dict=state_dict, base_address=f"{self.base_address}.conv2")
        self.bn2 = norm_layer(width)
        self.bn2.weight = nn.Parameter(state_dict[f"{self.base_address}.bn2.weight"])
        self.bn2.bias = nn.Parameter(state_dict[f"{self.base_address}.bn2.bias"])
        self.bn2.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_mean"])
        self.bn2.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_var"])
        self.bn2.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address}.bn2.num_batches_tracked"], requires_grad=False)
        self.bn2.eval()
        self.relu2 = nn.ReLU(inplace=True)


        self.conv3 = conv1x1(width, planes * self.expansion, state_dict=state_dict, base_address=f"{base_address}.conv3")
        self.bn3 = norm_layer(planes * self.expansion)
        self.bn3.weight = nn.Parameter(state_dict[f"{self.base_address}.bn3.weight"])
        self.bn3.bias = nn.Parameter(state_dict[f"{self.base_address}.bn3.bias"])
        self.bn3.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn3.running_mean"])
        self.bn3.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn3.running_var"])
        self.bn3.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address}.bn3.num_batches_tracked"], requires_grad=False)
        self.bn3.eval()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
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
        state_dict = None,
        base_address = None
    ) -> None:
        super().__init__()
        self.base_address = base_address
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, state_dict=state_dict, base_address=f"{base_address}.conv1")

        self.bn1 = norm_layer(planes)
        self.bn1.weight = nn.Parameter(state_dict[f"{self.base_address}.bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"{self.base_address}.bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address}.bn1.num_batches_tracked"], requires_grad=False)
        self.bn1.eval()

        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, state_dict=state_dict, base_address=f"{base_address}.conv2")

        self.bn2 = norm_layer(planes)
        self.bn2.weight = nn.Parameter(state_dict[f"{self.base_address}.bn2.weight"])
        self.bn2.bias = nn.Parameter(state_dict[f"{self.base_address}.bn2.bias"])
        self.bn2.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_mean"])
        self.bn2.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_var"])
        self.bn2.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address}.bn2.num_batches_tracked"], requires_grad=False)
        self.bn2.eval()
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, state_dict=None, base_address=None) -> nn.Conv2d:
    """3x3 convolution with padding"""
    conv =  nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
    conv.weight = nn.Parameter(state_dict[f"{base_address}.weight"])
    return conv


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, state_dict=None, base_address=None) -> nn.Conv2d:
    """1x1 convolution"""
    conv =  nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    conv.weight = nn.Parameter(state_dict[f"{base_address}.weight"])
    return conv


def _make_layer(
    block: Type[Union[BasicBlock, Bottleneck]],
    planes: int,
    blocks: int,
    stride: int = 1,
    dilate: bool = False,
    name: str = None,
    state_dict=None
    ) -> nn.Sequential:
    # norm_layer = self._norm_layer
    norm_layer = nn.BatchNorm2d
    inplanes = 64
    downsample = None
    dilation = 1
    groups = 1
    previous_dilation = dilation
    if dilate:
        dilation *= stride
        stride = 1
    if stride != 1 or inplanes != planes * block.expansion:
        nl = norm_layer(planes * block.expansion)
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride, state_dict=state_dict, base_address=f"{name}.0.downsample.0"),
            nl,
        )
        nl.weight = nn.Parameter(state_dict[f"{name}.0.downsample.1.weight"])
        nl.bias = nn.Parameter(state_dict[f"{name}.0.downsample.1.bias"])
        nl.running_mean = nn.Parameter(state_dict[f"{name}.0.downsample.1.running_mean"])
        nl.running_var = nn.Parameter(state_dict[f"{name}.0.downsample.1.running_var"])
        nl.num_batches_tracked = nn.Parameter(state_dict[f"{name}.0.downsample.1.num_batches_tracked"], requires_grad=False)
        nl.eval()

    base_width = 64
    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            downsample,
            groups,
            base_width,
            previous_dilation,
            norm_layer,
            state_dict=state_dict,
            base_address=f"{name}.0"
        )
    )
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                groups=groups,
                base_width=base_width,
                dilation=dilation,
                norm_layer=norm_layer,
                state_dict=state_dict,
                base_address=f"{name}.{_}"
            )
        )

    return nn.Sequential(*layers)

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
        state_dict=None,
        base_address=""
    ) -> None:
        super().__init__()
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
        self.conv1.weight = nn.Parameter(state_dict["conv1.weight"])

        self.bn1 = norm_layer(self.inplanes)
        self.bn1.weight = nn.Parameter(state_dict[f"bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(state_dict[f"bn1.num_batches_tracked"], requires_grad=False)
        self.bn1.eval()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _make_layer(block, 64, layers[0], state_dict=state_dict, name="layer1")
        self.layer2 = _make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], state_dict=state_dict, name="layer2")
        self.layer3 = _make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], state_dict=state_dict, name="layer3")
        self.layer4 = _make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], state_dict=state_dict, name="layer4")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc.weight = nn.Parameter(state_dict["fc.weight"])
        self.fc.bias = nn.Parameter(state_dict["fc.bias"])


    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
