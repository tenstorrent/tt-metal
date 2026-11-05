# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Shared ttnn blocks for PP-OCRv6. Activations are flattened NHWC: [1, 1, B*H*W, C].

import torch
import torch.nn as nn

import ttnn

RELU = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
SILU = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)


def fold_bn(weight, bias, bn: nn.BatchNorm2d):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = weight * scale.reshape(-1, *([1] * (weight.dim() - 1)))
    bias = torch.zeros_like(bn.running_mean) if bias is None else bias
    return weight, (bias - bn.running_mean) * scale + bn.bias


def compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def to_row_major(x):
    return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


class Conv:
    def __init__(self, device, weight, bias, stride=1, padding=None, groups=1, activation=None):
        self.device = device
        self.cout, cin_g, self.kh, self.kw = weight.shape
        self.cin = cin_g * groups
        self.stride = _pair(stride)
        self.padding = padding if padding is not None else ((self.kh - 1) // 2, (self.kw - 1) // 2)
        self.groups = groups
        self.host_weight = ttnn.from_torch(weight.detach().float(), dtype=ttnn.float32)
        self.host_bias = ttnn.from_torch(bias.detach().float().reshape(1, 1, 1, -1), dtype=ttnn.float32)
        # Device-prepared weights depend on the input shape (sharding), so keep one copy per shape.
        self.prepared = {}
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=activation,
            output_layout=ttnn.TILE_LAYOUT,
        )
        self.compute_config = compute_config(device)

    def __call__(self, x, batch, h, w, memory_config=None):
        key = (batch, h, w, memory_config is None)
        weight, bias = self.prepared.get(key, (self.host_weight, self.host_bias))
        y, (oh, ow), self.prepared[key] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            bias_tensor=bias,
            device=self.device,
            in_channels=self.cin,
            out_channels=self.cout,
            batch_size=batch,
            input_height=h,
            input_width=w,
            kernel_size=(self.kh, self.kw),
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            memory_config=memory_config,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return y, oh, ow


class Linear:
    """1x1 convolution (or nn.Linear weight [out, in]) as a matmul over the last dim."""

    def __init__(self, device, weight, bias=None, activation=None):
        w = weight.detach().float().reshape(weight.shape[0], -1).t().contiguous()
        self.weight = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.bias = (
            None
            if bias is None
            else ttnn.from_torch(
                bias.detach().float().reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
        )
        self.activation = activation
        self.compute_config = compute_config(device)

    def __call__(self, x, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.linear(
            x,
            self.weight,
            bias=self.bias,
            activation=self.activation,
            memory_config=memory_config,
            compute_kernel_config=self.compute_config,
        )


class SqueezeExcite:
    """x * clamp(a * W2 relu(W1 mean_hw(x)) + c, 0, 1), optionally plus x."""

    def __init__(self, device, conv1: nn.Conv2d, conv2: nn.Conv2d, gate_scale, gate_shift, add_identity):
        self.fc1 = Linear(device, conv1.weight, conv1.bias, activation="relu")
        self.fc2 = Linear(device, conv2.weight * gate_scale, conv2.bias * gate_scale + gate_shift)
        self.add_identity = add_identity

    def __call__(self, x, batch=1):
        shape = x.shape
        if batch > 1:
            x = ttnn.reshape(x, (batch, 1, shape[2] // batch, shape[3]))
        s = ttnn.mean(x, dim=2, keepdim=True)
        s = self.fc2(self.fc1(s, ttnn.L1_MEMORY_CONFIG), ttnn.L1_MEMORY_CONFIG)
        s = ttnn.clamp(s, 0.0, 1.0)
        if self.add_identity:
            s = ttnn.add(s, 1.0)
        x = ttnn.multiply(x, s, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(x, shape) if batch > 1 else x


class LCNetV4Layer:
    def __init__(self, device, layer, cfg):
        _, cin, cout, stride, use_se = cfg
        self.has_residual = layer.has_residual
        if self.has_residual:
            tw, tb = layer.token_conv.weight, layer.token_conv.bias
        else:
            tw, tb = fold_bn(layer.token_conv.convolution.weight, None, layer.token_conv.normalization)
        self.token_conv = Conv(device, tw, tb, stride=stride, groups=cin)
        self.se = None
        if use_se:
            c = layer.token_squeeze_excitation.convolutions
            self.se = SqueezeExcite(device, c[0], c[2], 1 / 6, 0.5, add_identity=False)
        self.fc1 = Linear(
            device, *fold_bn(layer.channel_conv1.convolution.weight, None, layer.channel_conv1.normalization)
        )
        self.fc2 = Linear(
            device, *fold_bn(layer.channel_conv2.convolution.weight, None, layer.channel_conv2.normalization)
        )

    def __call__(self, x, batch, h, w):
        x, h, w = self.token_conv(x, batch, h, w)
        if self.se is not None:
            x = self.se(x, batch)
        y = ttnn.gelu(self.fc1(x), fast_and_approximate_mode=False)
        y = self.fc2(y)
        if self.has_residual:
            y = ttnn.add(x, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return y, h, w


class LargeStem:
    def __init__(self, device, stem):
        def cbn(m, **kw):
            return Conv(device, *fold_bn(m.convolution.weight, None, m.normalization), **kw)

        self.stem1 = cbn(stem.stem1, stride=2, activation=RELU)
        self.stem2a = cbn(stem.stem2a, padding=(0, 1, 0, 1), activation=RELU)
        self.stem2b = cbn(stem.stem2b, padding=(0, 1, 0, 1), activation=RELU)
        self.stem3 = cbn(stem.stem3, stride=2, activation=RELU)
        self.stem4 = Linear(device, *fold_bn(stem.stem4.convolution.weight, None, stem.stem4.normalization), "relu")

    def __call__(self, x, batch, h, w):
        s1, h1, w1 = self.stem1(x, batch, h, w, memory_config=ttnn.L1_MEMORY_CONFIG)
        a, _, _ = self.stem2a(s1, batch, h1, w1)
        a, _, _ = self.stem2b(a, batch, h1, w1)
        # Zero padding before the pool equals -inf padding here because s1 is post-ReLU.
        p = ttnn.max_pool2d(
            to_row_major(s1),
            batch_size=batch,
            input_h=h1,
            input_w=w1,
            channels=s1.shape[-1],
            kernel_size=[2, 2],
            stride=[1, 1],
            padding=[0, 1, 0, 1],
            dilation=[1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        x = ttnn.concat([to_row_major(p), to_row_major(a)], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x, h, w = self.stem3(x, batch, h1, w1)
        return self.stem4(x), h, w


class Backbone:
    """Returns the output of every stage as (tensor, h, w)."""

    def __init__(self, device, torch_backbone, stage_configs):
        enc = torch_backbone.encoder
        self.stem = LargeStem(device, enc.convolution)
        self.stages = [
            [LCNetV4Layer(device, layer, cfg) for layer, cfg in zip(stage.blocks, cfgs)]
            for stage, cfgs in zip(enc.blocks, stage_configs)
        ]

    def __call__(self, x, batch, h, w):
        x, h, w = self.stem(x, batch, h, w)
        feats = []
        for stage in self.stages:
            for layer in stage:
                x, h, w = layer(x, batch, h, w)
            feats.append((x, h, w))
        return feats
