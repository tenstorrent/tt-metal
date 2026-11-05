# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Standalone PyTorch PP-OCRv6_small_rec, parameter names match
# PaddlePaddle/PP-OCRv6_small_rec_safetensors.

import torch.nn as nn
import torch.nn.functional as F

from models.demos.blackhole.pp_ocrv6.reference.det import Backbone

STAGE_CONFIGS = [
    [[3, 96, 96, 1, True]],
    [[3, 96, 96, 1, False], [3, 96, 96, 1, False]],
    [
        [3, 96, 192, (2, 1), False],
        [3, 192, 192, 1, True],
        [3, 192, 192, 1, False],
        [3, 192, 192, 1, True],
        [3, 192, 192, 1, False],
        [3, 192, 192, 1, True],
        [3, 192, 192, 1, False],
    ],
    [[3, 192, 384, (2, 1), False], [3, 384, 384, 1, True], [3, 384, 384, 1, False]],
]
STEM_CHANNELS = (3, 48, 96)
BACKBONE_CHANNELS = 384
HIDDEN = 120
HEADS = 8
MLP_RATIO = 2.0
DEPTH = 2
NUM_CLASSES = 18710
LOCAL_KERNEL = (1, 7)
LN_EPS = 1e-6


class ConvBNSiLU(nn.Module):
    def __init__(self, cin, cout, k, groups=1):
        super().__init__()
        self.convolution = nn.Conv2d(cin, cout, k, padding=(k[0] // 2, k[1] // 2), groups=groups, bias=False)
        self.normalization = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.silu(self.normalization(self.convolution(x)))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(HIDDEN, 3 * HIDDEN)
        self.projection = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, HEADS, c // HEADS).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-1, -2)) * (c // HEADS) ** -0.5
        out = (attn.softmax(-1) @ v).transpose(1, 2).reshape(b, n, c)
        return self.projection(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN, int(HIDDEN * MLP_RATIO))
        self.fc2 = nn.Linear(int(HIDDEN * MLP_RATIO), HIDDEN)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(HIDDEN, eps=LN_EPS)
        self.self_attn = Attention()
        self.layer_norm2 = nn.LayerNorm(HIDDEN, eps=LN_EPS)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.ModuleList(
            [
                ConvBNSiLU(BACKBONE_CHANNELS, HIDDEN, (1, 1)),
                ConvBNSiLU(BACKBONE_CHANNELS, HIDDEN, (1, 1)),
                ConvBNSiLU(HIDDEN, HIDDEN, LOCAL_KERNEL, groups=HIDDEN),
            ]
        )
        self.svtr_block = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(HIDDEN, eps=LN_EPS)

    def forward(self, x):
        skip = self.conv_block[0](x)
        x = self.conv_block[1](x)
        x = x + self.conv_block[2](x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        for blk in self.svtr_block:
            x = blk(x)
        x = self.norm(x).reshape(b, h, w, c).permute(0, 3, 1, 2) + skip
        return x.squeeze(2).transpose(1, 2)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(HIDDEN, NUM_CLASSES)

    def forward(self, x):
        return self.head(self.encoder(x)).softmax(-1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone(STEM_CHANNELS, STAGE_CONFIGS)

    def forward(self, x):
        return F.avg_pool2d(self.backbone(x)[-1], (3, 2))


class PPOCRv6SmallRec(nn.Module):
    """Input: [B, 3, 48, W] in [-1, 1]. Output: per-step character probabilities [B, W/8, NUM_CLASSES]."""

    def __init__(self):
        super().__init__()
        self.model = Model()
        self.head = Head()

    def forward(self, x):
        return self.head(self.model(x))
