# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import nn

from models.experimental.xtts.config import (  # noqa: F401
    SPK_ASP_DIM as ASP_DIM,
    SPK_INPUT_DIM as INPUT_DIM,
    SPK_LAYERS as LAYERS,
    SPK_LOG_INPUT as LOG_INPUT,
    SPK_NUM_FILTERS as NUM_FILTERS,
    SPK_OUTMAP_SIZE as OUTMAP_SIZE,
    SPK_PROJ_DIM as PROJ_DIM,
    SPK_REDUCTION as REDUCTION,
)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=REDUCTION):
        """Build squeeze-and-excitation channel attention."""
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Rescale channels by pooled squeeze-excitation weights."""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=REDUCTION):
        """Build a residual SE conv block."""
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Apply SE residual block with optional downsample."""
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetSpeakerEncoder(nn.Module):
    def __init__(self):
        """Build ResNet-SE speaker encoder with attentive statistics pooling."""
        super().__init__()
        self.inplanes = NUM_FILTERS[0]
        self.conv1 = nn.Conv2d(1, NUM_FILTERS[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(NUM_FILTERS[0])
        self.instancenorm = nn.InstanceNorm1d(INPUT_DIM)

        self.layer1 = self._make_layer(NUM_FILTERS[0], LAYERS[0])
        self.layer2 = self._make_layer(NUM_FILTERS[1], LAYERS[1], stride=(2, 2))
        self.layer3 = self._make_layer(NUM_FILTERS[2], LAYERS[2], stride=(2, 2))
        self.layer4 = self._make_layer(NUM_FILTERS[3], LAYERS[3], stride=(2, 2))

        self.attention = nn.Sequential(
            nn.Conv1d(ASP_DIM, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, ASP_DIM, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.fc = nn.Linear(ASP_DIM * 2, PROJ_DIM)

    def _make_layer(self, planes, blocks, stride=1):
        """Create a stack of SEBasicBlocks with optional downsample."""
        downsample = None
        if stride != 1 or self.inplanes != planes * SEBasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * SEBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * SEBasicBlock.expansion),
            )
        layers = [SEBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * SEBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(SEBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, mel, l2_norm=True):
        """Encode mel features into an optional L2-normalized speaker emb."""
        x = mel
        if LOG_INPUT:
            x = (x + 1e-6).log()
        x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size(0), -1, x.size(-1))
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, sg), dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if l2_norm:
            x = F.normalize(x, p=2, dim=1)
        return x


def build_reference_speaker_encoder(state_dict):
    """Load speaker encoder weights from a checkpoint state dict."""
    prefix = "hifigan_decoder.speaker_encoder."
    slice_sd = {
        k[len(prefix) :]: v
        for k, v in state_dict.items()
        if k.startswith(prefix) and not k.startswith(prefix + "torch_spec.")
    }
    module = ResNetSpeakerEncoder()
    module.load_state_dict(slice_sd, strict=True)
    return module.eval()
