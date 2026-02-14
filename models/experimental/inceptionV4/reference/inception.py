# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

""" Pytorch Inception-V4 implementation
Sourced from https://github.com/Cadene/tensorflow-model-zoo.torch (MIT License) which is
based upon Google's Tensorflow implementation and pretrained weights (Apache 2.0 License)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.inceptionV4.reference._utils import create_classifier
from models.experimental.inceptionV4.reference._utils import (
    assign_weight_seq,
    assign_weight_basic_conv,
    assign_weight_linear,
)
from models.experimental.inceptionV4.reference.basicconv import BasicConv2d

IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
# taken from timm.data.constants
__all__ = ["InceptionV4"]

default_cfgs = {
    "inception_v4": {
        "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/inceptionv4-8e4777a0.pth",
        "num_classes": 1000,
        "input_size": (3, 299, 299),
        "pool_size": (8, 8),
        "crop_pct": 0.875,
        "interpolation": "bicubic",
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "features.0.conv",
        "classifier": "last_linear",
        "label_offset": 1,  # 1001 classes in pretrained weights
    }
}


class Mixed3a(nn.Module):
    def __init__(self, state_dict, base_address):
        super(Mixed3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)
        assign_weight_basic_conv(self.conv, state_dict, f"{base_address}.conv")

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed4a(nn.Module):
    def __init__(self, state_dict, base_address):
        super(Mixed4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1),
        )
        assign_weight_seq(self.branch0, state_dict, f"{base_address}.branch0")

        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=1),
        )

        assign_weight_seq(self.branch1, state_dict, f"{base_address}.branch1")

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed5a(nn.Module):
    def __init__(self, state_dict, base_address):
        super(Mixed5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        assign_weight_basic_conv(self.conv, state_dict, f"{base_address}.conv")
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class InceptionA(nn.Module):
    def __init__(self, state_dict, base_address):
        super(InceptionA, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1),
        )

        assign_weight_basic_conv(self.branch0, state_dict, f"{base_address}.branch0")
        assign_weight_seq(self.branch1, state_dict, f"{base_address}.branch1")
        assign_weight_seq(self.branch2, state_dict, f"{base_address}.branch2")
        assign_weight_seq(self.branch3, state_dict, f"{base_address}.branch3")

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class ReductionA(nn.Module):
    def __init__(self, state_dict, base_address=""):
        super(ReductionA, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2),
        )

        assign_weight_basic_conv(self.branch0, state_dict, f"{base_address}.branch0")
        assign_weight_seq(self.branch1, state_dict, f"{base_address}.branch1")

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class InceptionB(nn.Module):
    def __init__(self, state_dict, base_address):
        super(InceptionB, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1),
        )

        assign_weight_basic_conv(self.branch0, state_dict, f"{base_address}.branch0")
        assign_weight_seq(self.branch1, state_dict, f"{base_address}.branch1")
        assign_weight_seq(self.branch2, state_dict, f"{base_address}.branch2")
        assign_weight_seq(self.branch3, state_dict, f"{base_address}.branch3")

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class ReductionB(nn.Module):
    def __init__(self, state_dict, base_address):
        super(ReductionB, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2),
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2),
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

        assign_weight_seq(self.branch0, state_dict, f"{base_address}.branch0")
        assign_weight_seq(self.branch1, state_dict, f"{base_address}.branch1")

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class InceptionC(nn.Module):
    def __init__(self, state_dict, base_address):
        super(InceptionC, self).__init__()

        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1),
        )

        assign_weight_basic_conv(self.branch0, state_dict, f"{base_address}.branch0")

        assign_weight_basic_conv(self.branch1_0, state_dict, f"{base_address}.branch1_0")
        assign_weight_basic_conv(self.branch1_1a, state_dict, f"{base_address}.branch1_1a")
        assign_weight_basic_conv(self.branch1_1b, state_dict, f"{base_address}.branch1_1b")

        assign_weight_basic_conv(self.branch2_0, state_dict, f"{base_address}.branch2_0")
        assign_weight_basic_conv(self.branch2_1, state_dict, f"{base_address}.branch2_1")
        assign_weight_basic_conv(self.branch2_2, state_dict, f"{base_address}.branch2_2")
        assign_weight_basic_conv(self.branch2_3a, state_dict, f"{base_address}.branch2_3a")
        assign_weight_basic_conv(self.branch2_3b, state_dict, f"{base_address}.branch2_3b")

        assign_weight_basic_conv(self.branch3[1], state_dict, f"{base_address}.branch3.1")

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        in_chans=3,
        output_stride=32,
        drop_rate=0.0,
        global_pool="avg",
        state_dict=None,
        base_address="",
    ):
        super(InceptionV4, self).__init__()
        assert output_stride == 32
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = 1536

        self.features = nn.Sequential(
            BasicConv2d(in_chans, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed3a(state_dict, "features.3"),
            Mixed4a(state_dict, "features.4"),
            Mixed5a(state_dict, "features.5"),
            InceptionA(state_dict, "features.6"),
            InceptionA(state_dict, "features.7"),
            InceptionA(state_dict, "features.8"),
            InceptionA(state_dict, "features.9"),
            ReductionA(state_dict, "features.10"),  # Mixed6a
            InceptionB(state_dict, "features.11"),
            InceptionB(state_dict, "features.12"),
            InceptionB(state_dict, "features.13"),
            InceptionB(state_dict, "features.14"),
            InceptionB(state_dict, "features.15"),
            InceptionB(state_dict, "features.16"),
            InceptionB(state_dict, "features.17"),
            ReductionB(state_dict, "features.18"),  # Mixed7a
            InceptionC(state_dict, "features.19"),
            InceptionC(state_dict, "features.20"),
            InceptionC(state_dict, "features.21"),
        )

        assign_weight_basic_conv(self.features[0], state_dict, "features.0")
        assign_weight_basic_conv(self.features[1], state_dict, "features.1")
        assign_weight_basic_conv(self.features[2], state_dict, "features.2")

        self.feature_info = [
            dict(num_chs=64, reduction=2, module="features.2"),
            dict(num_chs=160, reduction=4, module="features.3"),
            dict(num_chs=384, reduction=8, module="features.9"),
            dict(num_chs=1024, reduction=16, module="features.17"),
            dict(num_chs=1536, reduction=32, module="features.21"),
        ]
        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool
        )

        assign_weight_linear(self.last_linear, state_dict, "last_linear")

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem=r"^features\.[012]\.", blocks=r"^features\.(\d+)")

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, "gradient checkpointing not supported"

    @torch.jit.ignore
    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool
        )

    def forward_features(self, x):
        return self.features(x)

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.last_linear(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
