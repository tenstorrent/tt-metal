# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from loguru import logger

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
}


def conv3x3(in_planes, out_planes, stride=1, dtype=torch.float32):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, dtype=dtype)


def conv1x1(in_planes, out_planes, stride=1, dtype=torch.float32):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, dtype=dtype)


def printdebug_tensor_shape(tensor, name):
    if tensor is not None:
        logger.debug(f"{name} shape: {tensor.shape=}")
    else:
        logger.debug(f"{name} is None")


class SequentialWithIntermediates(nn.Sequential):
    """A sequential container that optionally returns intermediate activations."""

    def __init__(self, *args, return_intermediates=False):
        super(SequentialWithIntermediates, self).__init__(*args)
        self.return_intermediates = return_intermediates

    def forward(self, input):
        intermediates = []
        x = input

        if self.return_intermediates:
            intermediates.append(x.clone())

        for module in self:
            x = module(x)
            if self.return_intermediates:
                intermediates.append(x.clone())

        if self.return_intermediates:
            return x, intermediates
        else:
            return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dtype=torch.float32):
        super(BasicBlock, self).__init__()

        # Update conv functions to pass dtype
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, dtype=dtype)
        self.bn1 = nn.GroupNorm(16, planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype)
        self.bn2 = nn.GroupNorm(16, planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False, dtype=dtype),
                nn.GroupNorm(16, planes),
            )
        else:
            self.downsample = None

        self.dtype = dtype

        # Convert all parameters to the specified dtype using PyTorch's to() method
        self.to(dtype)

    def forward(self, x):
        identity = x
        # printdebug_tensor_shape(identity, "identity")
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # printdebug_tensor_shape(out, "out")
        out = self.bn2(self.conv2(out))
        # return out
        if self.downsample is not None:
            # printdebug_tensor_shape(identity, "identity")
            identity = self.downsample(x)

        # printdebug_tensor_shape(out, "out")
        # printdebug_tensor_shape(identity, "identity")
        out += identity
        # printdebug_tensor_shape(out, "out")
        out = F.relu(out, inplace=True)

        return out


class ResNetFeatures(nn.Module):
    def __init__(self, block, layers, dtype, num_classes=1000, zero_init_residual=False, return_intermediates=False):
        super(ResNetFeatures, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, dtype=dtype)
        self.bn1 = nn.GroupNorm(16, 64)  # GroupNorm doesn't have dtype parameter
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dtype = dtype
        self.return_intermediates = return_intermediates

        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=1, dtype=dtype, return_intermediates=return_intermediates
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dtype=dtype, return_intermediates=return_intermediates
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dtype=dtype, return_intermediates=return_intermediates
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dtype=dtype, return_intermediates=return_intermediates
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.return_intermediates = return_intermediates

        # Convert all parameters and buffers to the specified dtype using PyTorch's to() method
        self.to(dtype)

    def _make_layer(self, block, planes, blocks, stride=1, dtype=torch.float32, return_intermediates=False):
        layers = []
        layers.append(block(self.inplanes, planes, stride, dtype=dtype))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dtype=dtype))

        return SequentialWithIntermediates(*layers, return_intermediates=return_intermediates)

    def forward(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        ref_x = x.clone()
        conv1f = self.conv1(x)
        ref_conv1f = conv1f.clone()
        gn = self.bn1(conv1f)
        ref_gn = gn.clone()
        conv1 = F.relu(gn, inplace=True)
        ref_conv1 = conv1.clone()
        conv1 = self.maxpool(conv1)
        ref_conv1_maxpool = conv1.clone()

        if self.return_intermediates:
            feats4, i4 = self.layer1(conv1)
            feats8, i8 = self.layer2(feats4)
            feats16, i16 = self.layer3(feats8)
            feats32, i32 = self.layer4(feats16)
            return [ref_x, i4, i8, i16, i32, ref_conv1f, ref_gn, ref_conv1, ref_conv1_maxpool], feats8, feats16, feats32
        else:
            feats4 = self.layer1(conv1)
            feats8 = self.layer2(feats4)
            feats16 = self.layer3(feats8)
            feats32 = self.layer4(feats16)
            return feats8, feats16, feats32


def resnet18(pretrained=False, dtype=torch.float32, **kwargs):
    model = ResNetFeatures(BasicBlock, [2, 2, 2, 2], dtype=dtype, **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, dtype=torch.float32, **kwargs):
    model = ResNetFeatures(BasicBlock, [3, 4, 6, 3], dtype=dtype, **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls["resnet34"]))
    return model


def _load_pretrained(model, pretrained):
    model_dict = model.state_dict()
    # Convert pretrained weights to the model's dtype
    dtype = next(model.parameters()).dtype
    pretrained = {k: v.to(dtype) if v.is_floating_point() else v for k, v in pretrained.items() if k in model_dict}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)


@pytest.mark.parametrize(
    "inplanes, planes, stride, input_shape",
    [
        (64, 64, 1, (1, 64, 56, 56)),  # identity path
        (64, 128, 2, (1, 64, 56, 56)),  # downsample path
        (128, 256, 2, (1, 128, 64, 80)),
    ],
)
def test_basicblock_forward(inplanes, planes, stride, input_shape):
    torch.manual_seed(0)
    block = BasicBlock(inplanes, planes, stride)
    x = torch.randn(*input_shape)
    out = block.forward(x)
    # Output shape should match expected ResNet block output
    expected_h = input_shape[2] // stride
    expected_w = input_shape[3] // stride
    assert out.shape == (input_shape[0], planes, expected_h, expected_w)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "input_shape, layers",
    [
        ((1, 3, 224, 224), [2, 2, 2, 2]),  # ResNet-18
        ((2, 3, 128, 128), [2, 2, 2, 2]),  # batch size 2
    ],
)
def test_resnetfeatures_forward(input_shape, layers):
    torch.manual_seed(0)
    logger.debug(f"Testing ResNetFeatures with input shape: {input_shape} and layers: {layers}")
    model = ResNetFeatures(BasicBlock, layers)
    x = torch.randn(*input_shape)
    feats8, feats16, feats32 = model(x)
    # Output shapes after each stage (approximate for stride=2 blocks)
    b, c, h, w = input_shape
    # After conv1 + maxpool: h//4, w//4
    h4, w4 = h // 4, w // 4
    # After layer2: h//8, w//8
    h8, w8 = h // 8, w // 8
    # After layer3: h//16, w//16
    h16, w16 = h // 16, w // 16
    # After layer4: h//32, w//32
    h32, w32 = h // 32, w // 32
    assert feats8.shape[2] == h8 and feats8.shape[3] == w8
    assert feats16.shape[2] == h16 and feats16.shape[3] == w16
    assert feats32.shape[2] == h32 and feats32.shape[3] == w32
    assert feats8.shape[0] == b and feats16.shape[0] == b and feats32.shape[0] == b
    logger.debug("ResNetFeatures forward test passed.")
