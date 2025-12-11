import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet as vrn
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)


def register(f):
    all = sys.modules[f.__module__].__dict__.setdefault("__all__", [])
    if f.__name__ in all:
        raise RuntimeError("{} already exist!".format(f.__name__))
    all.append(f.__name__)
    return f


class ResNet(vrn.ResNet):
    "Deep Residual Network - https://arxiv.org/abs/1512.03385"

    def __init__(
        self,
        layers=[3, 4, 6, 3],
        bottleneck=vrn.Bottleneck,
        outputs=[5],
        groups=1,
        width_per_group=64,
        url=None,
    ):
        self.stride = 128
        self.bottleneck = bottleneck
        self.outputs = outputs
        self.url = url

        kwargs = {
            "block": bottleneck,
            "layers": layers,
            "groups": groups,
            "width_per_group": width_per_group,
        }
        super().__init__(**kwargs)
        self.unused_modules = ["fc"]

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outputs = []
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            level = i + 2
            if level > max(self.outputs):
                break
            x = layer(x)
            if level in self.outputs:
                outputs.append(x)

        return outputs


class FPN(nn.Module):
    "Feature Pyramid Network - https://arxiv.org/abs/1612.03144"

    def __init__(self, features):
        super().__init__()

        self.stride = 128
        self.features = features

        if isinstance(features, ResNet):
            is_light = features.bottleneck == vrn.BasicBlock
            channels = [128, 256, 512] if is_light else [512, 1024, 2048]

        self.lateral3 = nn.Conv2d(channels[0], 256, 1)
        self.lateral4 = nn.Conv2d(channels[1], 256, 1)
        self.lateral5 = nn.Conv2d(channels[2], 256, 1)
        self.pyramid6 = nn.Conv2d(channels[2], 256, 3, stride=2, padding=1)
        self.pyramid7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

        self.apply(init_layer)

        self.features.initialize()

    def forward(self, x):
        c3, c4, c5 = self.features(x)

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p4 = F.interpolate(p5, scale_factor=2) + p4
        p3 = self.lateral3(c3)
        p3 = F.interpolate(p4, scale_factor=2) + p3

        p6 = self.pyramid6(c5)
        p7 = self.pyramid7(F.relu(p6))

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)

        return [p3, p4, p5, p6, p7]


@register
def ResNet18FPN():
    return FPN(
        ResNet(
            layers=[2, 2, 2, 2],
            bottleneck=vrn.BasicBlock,
            outputs=[3, 4, 5],
            url=torch.utils.model_zoo.load_url(ResNet18_Weights.IMAGENET1K_V1.url),
        )
    )


@register
def ResNet34FPN():
    return FPN(
        ResNet(
            layers=[3, 4, 6, 3],
            bottleneck=vrn.BasicBlock,
            outputs=[3, 4, 5],
            url=torch.utils.model_zoo.load_url(ResNet34_Weights.IMAGENET1K_V1.url),
        )
    )


@register
def ResNet50FPN():
    return FPN(
        ResNet(
            layers=[3, 4, 6, 3],
            bottleneck=vrn.Bottleneck,
            outputs=[3, 4, 5],
            url=torch.utils.model_zoo.load_url(ResNet50_Weights.IMAGENET1K_V2.url),
        )
    )


@register
def ResNet101FPN():
    return FPN(
        ResNet(
            layers=[3, 4, 23, 3],
            bottleneck=vrn.Bottleneck,
            outputs=[3, 4, 5],
            url=torch.utils.model_zoo.load_url(ResNet101_Weights.IMAGENET1K_V2.url),
        )
    )


@register
def ResNet152FPN():
    return FPN(
        ResNet(
            layers=[3, 8, 36, 3],
            bottleneck=vrn.Bottleneck,
            outputs=[3, 4, 5],
            url=torch.utils.model_zoo.load_url(ResNet152_Weights.IMAGENET1K_V2.url),
        )
    )
