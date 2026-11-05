# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Standalone PyTorch PP-OCRv6_small_det, parameter names match
# PaddlePaddle/PP-OCRv6_small_det_safetensors.

import torch
import torch.nn as nn
import torch.nn.functional as F

STAGE_CONFIGS = [
    [[3, 48, 48, 1, True], [3, 48, 48, 1, False]],
    [[3, 48, 96, 2, False], [3, 96, 96, 1, True], [3, 96, 96, 1, False]],
    [
        [3, 96, 192, 2, False],
        [3, 192, 192, 1, True],
        [3, 192, 192, 1, False],
        [3, 192, 192, 1, True],
        [3, 192, 192, 1, False],
    ],
    [[3, 192, 384, 2, False], [3, 384, 384, 1, True], [3, 384, 384, 1, False]],
]
STEM_CHANNELS = (3, 24, 48)
NECK_IN_CHANNELS = (48, 96, 192, 384)
NECK_CHANNELS = 96
REDUCTION = 4


class ConvBN(nn.Module):
    def __init__(self, cin, cout, k, stride=1, groups=1, act=True):
        super().__init__()
        self.convolution = nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, groups=groups, bias=False)
        self.normalization = nn.BatchNorm2d(cout)
        self.act = act

    def forward(self, x):
        x = self.normalization(self.convolution(x))
        return F.relu(x) if self.act else x


class BackboneSE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.convolutions = nn.ModuleList(
            [nn.Conv2d(c, c // REDUCTION, 1), nn.ReLU(), nn.Conv2d(c // REDUCTION, c, 1), nn.Hardsigmoid()]
        )

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        for layer in self.convolutions:
            s = layer(s)
        return x * s


class LCNetV4Layer(nn.Module):
    def __init__(self, k, cin, cout, stride, use_se):
        super().__init__()
        self.has_residual = cin == cout and stride == 1
        if self.has_residual:
            self.token_conv = nn.Conv2d(cin, cin, k, stride, k // 2, groups=cin)
        else:
            self.token_conv = ConvBN(cin, cin, k, stride, groups=cin, act=False)
        self.token_squeeze_excitation = BackboneSE(cin) if use_se else nn.Identity()
        self.channel_conv1 = ConvBN(cin, cin * 2, 1, act=False)
        self.channel_conv2 = ConvBN(cin * 2, cout, 1, act=False)

    def forward(self, x):
        x = self.token_squeeze_excitation(self.token_conv(x))
        y = self.channel_conv2(F.gelu(self.channel_conv1(x)))
        return x + y if self.has_residual else y


class Stage(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([LCNetV4Layer(*c) for c in cfg])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class LargeStem(nn.Module):
    def __init__(self, stem_channels=STEM_CHANNELS):
        super().__init__()
        c0, c1, c2 = stem_channels
        self.stem1 = ConvBN(c0, c1, 3, 2)
        self.stem2a = ConvBN(c1, c1 // 2, 2)
        self.stem2b = ConvBN(c1 // 2, c1, 2)
        self.stem3 = ConvBN(c1 * 2, c1, 3, 2)
        self.stem4 = ConvBN(c1, c2, 1)

    def forward(self, x):
        x = F.pad(self.stem1(x), (0, 1, 0, 1))
        a = self.stem2b(F.pad(self.stem2a(x), (0, 1, 0, 1)))
        p = F.max_pool2d(x, 2, 1, ceil_mode=True)
        return self.stem4(self.stem3(torch.cat([p, a], 1)))


class Encoder(nn.Module):
    def __init__(self, stem_channels=STEM_CHANNELS, stage_configs=STAGE_CONFIGS):
        super().__init__()
        self.convolution = LargeStem(stem_channels)
        self.blocks = nn.ModuleList([Stage(c) for c in stage_configs])

    def forward(self, x):
        x = self.convolution(x)
        feats = []
        for stage in self.blocks:
            x = stage(x)
            feats.append(x)
        return feats


class Backbone(nn.Module):
    def __init__(self, stem_channels=STEM_CHANNELS, stage_configs=STAGE_CONFIGS):
        super().__init__()
        self.encoder = Encoder(stem_channels, stage_configs)

    def forward(self, x):
        return self.encoder(x)


class NeckSE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c // REDUCTION, 1)
        self.conv2 = nn.Conv2d(c // REDUCTION, c, 1)

    def forward(self, x):
        s = self.conv2(F.relu(self.conv1(x.mean((2, 3), keepdim=True))))
        return x * torch.clamp(0.2 * s + 0.5, 0.0, 1.0)


class InsertConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.in_conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.squeeze_excitation_block = NeckSE(cout)

    def forward(self, x):
        x = self.in_conv(x)
        return x + self.squeeze_excitation_block(x)


class InputConv(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.depthwise_convolution = nn.Conv2d(c, c, k, 1, k // 2, groups=c)
        self.pointwise_convolution = nn.Conv2d(c, c // 4, 1, bias=False)
        self.squeeze_excitation_module = NeckSE(c // 4)

    def forward(self, x):
        x = self.pointwise_convolution(self.depthwise_convolution(x))
        return x + self.squeeze_excitation_module(x)


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.insert_conv = nn.ModuleList([InsertConv(c, NECK_CHANNELS) for c in NECK_IN_CHANNELS])
        self.input_conv = nn.ModuleList([InputConv(NECK_CHANNELS, 7) for _ in NECK_IN_CHANNELS])

    def forward(self, feats):
        fused = [conv(f) for conv, f in zip(self.insert_conv, feats)]
        for i in range(2, -1, -1):
            fused[i] = fused[i] + F.interpolate(fused[i + 1], scale_factor=2, mode="nearest")
        outs = []
        for i, (conv, f) in enumerate(zip(self.input_conv, fused)):
            y = conv(f)
            outs.append(F.interpolate(y, scale_factor=2**i, mode="nearest") if i else y)
        return torch.cat(outs[::-1], 1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck()

    def forward(self, x):
        return self.neck(self.backbone(x))


class HeadConvBN(nn.Module):
    def __init__(self, cin, cout, k, transpose=False):
        super().__init__()
        if transpose:
            self.convolution = nn.ConvTranspose2d(cin, cout, k, 2)
        else:
            self.convolution = nn.Conv2d(cin, cout, k, 1, k // 2, bias=False)
        self.norm = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.norm(self.convolution(x)))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        c = NECK_CHANNELS
        self.conv_down = HeadConvBN(c, c // 4, 3)
        self.conv_up = HeadConvBN(c // 4, c // 4, 2, transpose=True)
        self.conv_final = nn.ConvTranspose2d(c // 4, 1, 2, 2)

    def forward(self, x):
        return torch.sigmoid(self.conv_final(self.conv_up(self.conv_down(x))))


class PPOCRv6SmallDet(nn.Module):
    """Input: BGR-normalized [B, 3, H, W], H and W multiples of 32. Output: text probability [B, 1, H, W]."""

    def __init__(self):
        super().__init__()
        self.model = Model()
        self.head = Head()

    def forward(self, x):
        return self.head(self.model(x))
