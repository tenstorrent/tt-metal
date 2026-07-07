# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN XTTS-v2 speaker encoder (``ResNetSpeakerEncoder``): log-mel -> 512-d ``g``.

Mirrors ``reference/xtts_speaker_encoder.py``. The SE-ResNet-34 body runs entirely
channels-last ``[N, H, W, C]`` (H=freq, W=time) so ttnn.conv2d needs no transposes.
BatchNorm is applied as a precomputed per-channel affine (``x*scale + shift`` with
``scale = gamma/sqrt(var+eps)``, ``shift = beta - mean*scale``) — this avoids
``ttnn.batch_norm`` (which wants NCHW) and needs no relayout, and is exact at
inference. Only the attentive-statistics-pooling reshape needs one permute, into a
``[C=2048, T']`` column layout where the attention softmax and the ASP reductions
are over the last dim.

Weights are read from the folded/eval reference module.
"""

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.tt.xtts_conv import TtConv2d

BN_EPS = 1e-5
INSTANCENORM_EPS = 1e-5
ASP_EPS = 1e-5


def _to_tile(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)


def _bn_scale_shift(bn, eps=BN_EPS):
    """Fold a BatchNorm into an inference-time per-channel affine (scale, shift)."""
    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + eps)
    shift = bn.bias.detach() - bn.running_mean.detach() * scale
    return scale, shift


class _BnAffine(LightweightModule):
    """Per-channel affine ``x*scale + shift`` broadcast to ``bcast_shape`` (the
    channel axis holds C, all others are 1)."""

    def __init__(self, device, bn, bcast_shape):
        super().__init__()
        scale, shift = _bn_scale_shift(bn)
        self.scale = _to_tile(scale.reshape(bcast_shape), device)
        self.shift = _to_tile(shift.reshape(bcast_shape), device)

    def forward(self, x):
        return ttnn.add(ttnn.mul(x, self.scale), self.shift)


class TtSELayer(LightweightModule):
    """Squeeze-excite: global avg-pool -> Linear(C->C/8) -> relu -> Linear -> sigmoid -> scale."""

    def __init__(self, device, se):
        super().__init__()
        # torch Linear weight is [out, in]; ttnn.linear wants [in, out].
        self.w1 = _to_tile(se.fc[0].weight.t(), device)
        self.b1 = _to_tile(se.fc[0].bias.reshape(1, -1), device)
        self.w2 = _to_tile(se.fc[2].weight.t(), device)
        self.b2 = _to_tile(se.fc[2].bias.reshape(1, -1), device)

    def forward(self, x):  # x: [N, H, W, C]
        n, _, _, c = x.shape
        y = ttnn.global_avg_pool2d(x)  # [N, 1, 1, C]
        y = ttnn.reshape(y, [n, c])
        y = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
        y = ttnn.relu(ttnn.linear(y, self.w1, bias=self.b1))
        y = ttnn.sigmoid(ttnn.linear(y, self.w2, bias=self.b2))  # [N, C]
        y = ttnn.reshape(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), [n, 1, 1, c])
        return ttnn.mul(x, y)


class TtSEBasicBlock(LightweightModule):
    """conv1 -> relu -> bn1 -> conv2 -> bn2 -> SE -> (+downsample) -> relu."""

    def __init__(self, device, block):
        super().__init__()
        stride = block.stride[0] if isinstance(block.stride, tuple) else block.stride
        self.conv1 = TtConv2d(device, block.conv1.weight.detach(), None, stride=stride, padding=1)
        self.bn1 = _BnAffine(device, block.bn1, [1, 1, 1, -1])
        self.conv2 = TtConv2d(device, block.conv2.weight.detach(), None, stride=1, padding=1)
        self.bn2 = _BnAffine(device, block.bn2, [1, 1, 1, -1])
        self.se = TtSELayer(device, block.se)
        self.downsample_conv = None
        if block.downsample is not None:
            self.downsample_conv = TtConv2d(device, block.downsample[0].weight.detach(), None, stride=stride, padding=0)
            self.downsample_bn = _BnAffine(device, block.downsample[1], [1, 1, 1, -1])

    def forward(self, x):
        out = self.bn1(ttnn.relu(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        residual = x if self.downsample_conv is None else self.downsample_bn(self.downsample_conv(x))
        return ttnn.relu(ttnn.add(out, residual))


class TtResNetSpeakerEncoder(LightweightModule):
    """log-mel ``[1, 64, T]`` -> speaker embedding ``[1, 512]`` (L2-normalized)."""

    def __init__(self, device, ref):
        super().__init__()
        self.device = device
        self.conv1 = TtConv2d(device, ref.conv1.weight.detach(), ref.conv1.bias.detach(), stride=1, padding=1)
        self.bn1 = _BnAffine(device, ref.bn1, [1, 1, 1, -1])
        self.layers = [
            [TtSEBasicBlock(device, blk) for blk in layer] for layer in (ref.layer1, ref.layer2, ref.layer3, ref.layer4)
        ]

        # Attention (ASP) in [C, T'] column layout: y = W @ x + b.
        att = ref.attention
        self.att_w1 = _to_tile(att[0].weight.detach().squeeze(-1), device)  # [128, 2048]
        self.att_b1 = _to_tile(att[0].bias.detach().reshape(-1, 1), device)  # [128, 1]
        self.att_bn = _BnAffine(device, att[2], [-1, 1])  # BatchNorm1d(128) -> [128,1]
        self.att_w2 = _to_tile(att[3].weight.detach().squeeze(-1), device)  # [2048, 128]
        self.att_b2 = _to_tile(att[3].bias.detach().reshape(-1, 1), device)  # [2048, 1]

        self.fc_w = _to_tile(ref.fc.weight.detach(), device)  # [512, 4096]
        self.fc_b = _to_tile(ref.fc.bias.detach().reshape(-1, 1), device)  # [512, 1]

    def _instance_norm_log(self, mel):
        # mel: [1, 64, T] TILE. log(+1e-6) then InstanceNorm1d over time (per freq).
        x = ttnn.log(ttnn.add(mel, 1e-6))
        mean = ttnn.mean(x, dim=2, keepdim=True)  # [1, 64, 1]
        xc = ttnn.sub(x, mean)
        var = ttnn.mean(ttnn.mul(xc, xc), dim=2, keepdim=True)
        return ttnn.mul(xc, ttnn.rsqrt(ttnn.add(var, INSTANCENORM_EPS)))

    def forward(self, mel):  # mel: ttnn [1, 64, T] TILE
        _, freq, time = mel.shape
        x = self._instance_norm_log(mel)  # [1, 64, T]
        # -> channels-last conv input [N=1, H=freq=64, W=time=T, C=1]
        x = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [1, freq, time, 1])

        x = self.bn1(ttnn.relu(self.conv1(x)))
        for layer in self.layers:
            for block in layer:
                x = block(x)  # -> [1, 8, T', 256]

        # ASP reshape: [N, H, W, C] -> [N, C, H, W] -> [C*H, W] = [2048, T'].
        _, h, w, c = x.shape
        x = ttnn.permute(x, (0, 3, 1, 2))  # [1, 256, 8, T']
        x = ttnn.reshape(x, [c * h, w])  # [2048, T']
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Attention weights over time.
        a = ttnn.relu(ttnn.add(ttnn.matmul(self.att_w1, x), self.att_b1))  # [128, T']
        a = self.att_bn(a)
        a = ttnn.add(ttnn.matmul(self.att_w2, a), self.att_b2)  # [2048, T']
        wgt = ttnn.softmax(a, dim=-1)  # over time

        # Attentive statistics pooling.
        mu = ttnn.sum(ttnn.mul(x, wgt), dim=-1, keepdim=True)  # [2048, 1]
        e2 = ttnn.sum(ttnn.mul(ttnn.mul(x, x), wgt), dim=-1, keepdim=True)
        var = ttnn.sub(e2, ttnn.mul(mu, mu))
        sg = ttnn.sqrt(ttnn.clamp(var, min=ASP_EPS))
        feat = ttnn.concat([mu, sg], dim=0)  # [4096, 1]

        g = ttnn.add(ttnn.matmul(self.fc_w, feat), self.fc_b)  # [512, 1]
        # L2 normalize over the 512 dim.
        norm = ttnn.sqrt(ttnn.sum(ttnn.mul(g, g), dim=0, keepdim=True))
        g = ttnn.div(g, norm)
        return ttnn.reshape(g, [1, 512])
