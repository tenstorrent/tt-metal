# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""WeSpeaker ResNet34 speaker embedding (pyannote/wespeaker-voxceleb-resnet34-LM) in TTNN.

pyannote embeds every (chunk, speaker) pair with the chunk's whole waveform and that speaker's
activity as pooling weights, so the ResNet trunk runs once per chunk here and only the pooling and the
linear head run per speaker."""

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.pyannote.reference.wespeaker import NUM_MEL_BINS, ResNet34

BN_EPS = 1e-5
RELU = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
# Chunks per trunk call. Measured on N150 for 21 chunks of 10 s: 1-4 per call take 270-300 ms,
# 21 per call 490 ms.
TRUNK_BATCH = 3


def _fold_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d):
    """Conv weight and bias with the batch norm folded in, transposed to (time, freq) kernels.

    The device tensors are time-major (height = frames, width = mel bins), the reference is
    (freq, time)."""
    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + BN_EPS)
    shift = bn.bias.detach() - bn.running_mean.detach() * scale
    weight = conv.weight.detach() * scale.reshape(-1, 1, 1, 1)
    return weight.transpose(-1, -2).contiguous(), shift


class TtConv2d(LightweightModule):
    """Conv2d + folded batch norm on flattened NHWC activations [1, 1, batch * h * w, channels]."""

    def __init__(self, device, conv, bn, *, padding, activation=None):
        super().__init__()
        weight, bias = _fold_bn(conv, bn)
        self.device = device
        self.out_channels, self.in_channels = weight.shape[:2]
        self.kernel_size = tuple(weight.shape[2:])
        self.stride = conv.stride
        self.padding = (padding, padding)
        self.weight = ttnn.from_torch(weight, ttnn.bfloat16)
        self.bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1), ttnn.bfloat16)
        # Prepared weights are specific to the input shape; ttnn does not detect a stale one.
        self.prepared = {}
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=activation,
            output_layout=ttnn.TILE_LAYOUT,
            # 36 convs, each prepared per input shape, overflow L1_SMALL with their config tensors.
            config_tensors_in_dram=True,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def forward(self, x, batch, h, w):
        key = (batch, h, w)
        weight, bias = self.prepared.get(key, (self.weight, self.bias))
        out, (oh, ow), [weight, bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=batch,
            input_height=h,
            input_width=w,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        self.prepared[key] = (weight, bias)
        return out, oh, ow


class TtBasicBlock(LightweightModule):
    def __init__(self, device, block):
        super().__init__()
        self.conv1 = TtConv2d(device, block.conv1, block.bn1, padding=1, activation=RELU)
        self.conv2 = TtConv2d(device, block.conv2, block.bn2, padding=1)
        self.shortcut = None
        if len(block.shortcut):
            self.shortcut = TtConv2d(device, block.shortcut[0], block.shortcut[1], padding=0)

    def forward(self, x, batch, h, w):
        out, oh, ow = self.conv1(x, batch, h, w)
        out, _, _ = self.conv2(out, batch, oh, ow)
        residual = x if self.shortcut is None else self.shortcut(x, batch, h, w)[0]
        if residual.memory_config() != out.memory_config():
            residual = ttnn.to_memory_config(residual, out.memory_config())
        return ttnn.add(out, residual, activations=[RELU]), oh, ow


class TtWeSpeakerResNet34(LightweightModule):
    def __init__(self, device, ref: ResNet34):
        super().__init__()
        self.device = device
        self.conv1 = TtConv2d(device, ref.conv1, ref.bn1, padding=1, activation=RELU)
        self.blocks = [
            TtBasicBlock(device, b) for layer in (ref.layer1, ref.layer2, ref.layer3, ref.layer4) for b in layer
        ]
        self.channels = ref.layer4[-1].conv2.out_channels

        # The device trunk orders pooled features (freq, channel); the reference orders (channel, freq).
        freq = NUM_MEL_BINS // 8
        perm = torch.tensor([c * freq + f for f in range(freq) for c in range(self.channels)])
        weight = ref.seg_1.weight.detach()
        weight = torch.cat([weight[:, perm], weight[:, perm + perm.numel()]], dim=1)
        self.head_w = ttnn.from_torch(weight.t().contiguous(), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.head_b = ttnn.from_torch(
            ref.seg_1.bias.detach().reshape(1, -1), ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def trunk(self, fbank: torch.Tensor) -> ttnn.Tensor:
        """fbank [chunks, frames, 80] (host) -> device features [chunks, frames / 8, 10 * 256]."""
        batch, frames, freq = fbank.shape
        x = ttnn.from_torch(
            fbank.reshape(1, 1, batch * frames * freq, 1),
            ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        x, h, w = self.conv1(x, batch, frames, freq)
        for block in self.blocks:
            x, h, w = block(x, batch, h, w)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, [batch, h, w * self.channels])
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def pool_and_embed(self, features: ttnn.Tensor, weights: torch.Tensor) -> ttnn.Tensor:
        """features [chunks, T, D] (device), weights [chunks, speakers, mask frames] (host) ->
        embeddings [chunks, speakers, 256] (device)."""
        frames = features.shape[1]
        if weights.shape[-1] != frames:
            weights = F.interpolate(weights.float(), size=frames, mode="nearest")
        # Weighted mean and unbiased weighted variance (pyannote's StatsPool) as two matmuls against the
        # normalised weights: mean = sum(w x) / v1, var = (sum(w x^2) - v1 mean^2) / (v1 - v2 / v1).
        v1 = weights.sum(dim=-1, keepdim=True) + 1e-8
        v2 = weights.square().sum(dim=-1, keepdim=True)
        scale = v1 / (v1 - v2 / v1 + 1e-8)
        wn = ttnn.from_torch(weights / v1, ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        k = ttnn.from_torch(scale, ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)

        x = ttnn.typecast(features, ttnn.float32)
        mm = {"compute_kernel_config": self.compute_config, "dtype": ttnn.float32}
        mean = ttnn.matmul(wn, x, **mm)
        sq = ttnn.matmul(wn, ttnn.mul(x, x), **mm)
        var = ttnn.mul(ttnn.sub(sq, ttnn.mul(mean, mean)), k)
        std = ttnn.sqrt(ttnn.relu(var))
        stats = ttnn.concat([mean, std], dim=-1)
        return ttnn.linear(stats, self.head_w, bias=self.head_b, **mm)

    def forward(self, fbank: torch.Tensor, weights: torch.Tensor) -> ttnn.Tensor:
        """fbank [chunks, frames, 80], weights [chunks, speakers, mask frames] -> [chunks, speakers, 256]."""
        outs = [
            self.pool_and_embed(self.trunk(fbank[i : i + TRUNK_BATCH]), weights[i : i + TRUNK_BATCH])
            for i in range(0, fbank.shape[0], TRUNK_BATCH)
        ]
        return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=0)
