# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `s_e_basic_block` of coqui/XTTS-v2.

Reference submodule: `hifigan_decoder.speaker_encoder.layer1.0`, a
`TTS.encoder.models.resnet.SEBasicBlock` — the squeeze-excitation residual
block that makes up the speaker encoder's ResNet stack:

    residual = x
    out = bn1(relu(conv1(x)))          # conv1: 3x3, bias=False; relu BEFORE bn1
    out = bn2(conv2(out))              # conv2: 3x3, bias=False
    out = se(out)                      # squeeze-excitation channel gating
    if downsample: residual = downsample(x)   # 1x1 stride-2 conv + bn
    out = relu(out + residual)

Native ttnn:
  * conv/bn -> `ttnn.conv2d` with BatchNorm (eval) FOLDED into the immediately
    preceding conv weights where no nonlinearity intervenes (conv2->bn2,
    downsample conv->bn); applied as a per-channel affine after `relu` for
    conv1->relu->bn1.
  * SE      -> global-avg-pool (`ttnn.mean`) + two `ttnn.matmul` + `ttnn.relu`
    + `ttnn.sigmoid`, broadcast-scaling the activation over the spatial dims.
  * residual add + relu -> elementwise ttnn.

The block runs bf16 conv activations with HiFi4 + fp32 accumulation; the SE
gating math runs in float32. Input/output are NCHW to match the torch module.
"""

from __future__ import annotations

import ttnn
from models.demos.kokoro_82m._stubs.s_e_layer import build as _b_se_layer

HF_MODEL_ID = "coqui/XTTS-v2"


def build(device, torch_module):
    """Bind trained weights (BatchNorm folded) and return a native ttnn forward closure."""
    import torch

    blk = torch_module

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, shard_layout=None, deallocate_activation=False)
    ACT = ttnn.bfloat16

    def bn_scale_shift(bn):
        a = (bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)).float()
        b = (bn.bias.detach() - bn.running_mean.detach() * a).float()
        return a, b

    def fold_conv_bn(conv, bn):
        a, b = bn_scale_shift(bn)
        W = conv.weight.detach().float() * a.reshape(-1, 1, 1, 1)
        cb = conv.bias.detach().float() if conv.bias is not None else torch.zeros_like(a)
        return W, cb * a + b

    def conv_w(W):
        return ttnn.as_tensor(
            W.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def conv_b(b):
        return ttnn.as_tensor(
            b.reshape(1, 1, 1, -1).contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def affine_t(a):
        return ttnn.as_tensor(
            a.reshape(1, 1, 1, -1).contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def f32(t):
        return ttnn.as_tensor(
            t.contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    in_c = int(blk.conv1.in_channels)
    planes = int(blk.conv1.out_channels)
    stride = int(blk.conv1.stride[0])

    w1 = conv_w(blk.conv1.weight.detach().float())  # conv1 bias=False, relu before bn1
    bn1_a, bn1_b = bn_scale_shift(blk.bn1)
    bn1_at, bn1_bt = affine_t(bn1_a), affine_t(bn1_b)
    W2, bias2 = fold_conv_bn(blk.conv2, blk.bn2)
    w2, b2 = conv_w(W2), conv_b(bias2)

    # squeeze-excitation channel gate: graduated leaf stub (s_e_layer)
    se_gate = _b_se_layer(device, blk.se)

    ds = None
    if blk.downsample is not None:
        dW, db = fold_conv_bn(blk.downsample[0], blk.downsample[1])
        ds = dict(
            w=conv_w(dW),
            b=conv_b(db),
            k=1,
            s=int(blk.downsample[0].stride[0]),
            p=0,
            ic=int(blk.downsample[0].in_channels),
            oc=int(blk.downsample[0].out_channels),
        )

    def run_conv(x_nhwc, w, b, ic, oc, k, s, p, H, W):
        out, [oh, ow] = ttnn.conv2d(
            input_tensor=x_nhwc,
            weight_tensor=w,
            in_channels=ic,
            out_channels=oc,
            device=device,
            bias_tensor=b,
            kernel_size=(k, k),
            stride=(s, s),
            padding=(p, p),
            dilation=(1, 1),
            batch_size=1,
            input_height=H,
            input_width=W,
            conv_config=conv_config,
            compute_config=compute_config,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=ACT,
        )
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        out = ttnn.reshape(out, (1, oh, ow, oc))
        return out, oh, ow

    def to_conv_in(x):
        return ttnn.to_layout(ttnn.typecast(x, ACT), ttnn.ROW_MAJOR_LAYOUT)

    def forward(x, *args, **kwargs):
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.as_tensor(
                x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        # input NCHW [1, C, H, W] -> NHWC [1, H, W, C]
        x = ttnn.typecast(x, ttnn.float32)
        H, W = int(x.shape[2]), int(x.shape[3])
        xnhwc = ttnn.permute(x, (0, 2, 3, 1))
        residual = xnhwc

        # conv1 -> relu -> bn1(affine)
        out, oh, ow = run_conv(to_conv_in(xnhwc), w1, None, in_c, planes, 3, stride, 1, H, W)
        out = ttnn.typecast(out, ttnn.float32)
        out = ttnn.relu(out)
        out = ttnn.add(ttnn.multiply(out, bn1_at), bn1_bt)
        # conv2 (bn2 folded)
        out, oh, ow = run_conv(to_conv_in(out), w2, b2, planes, planes, 3, 1, 1, oh, ow)
        out = ttnn.typecast(out, ttnn.float32)

        # SE gating (graduated leaf: s_e_layer) — stub is NCHW; out here is NHWC
        out = ttnn.permute(out, (0, 3, 1, 2))  # NHWC -> NCHW
        out = se_gate(out)  # per-channel SE rescale
        out = ttnn.permute(out, (0, 2, 3, 1))  # NCHW -> NHWC

        # residual
        if ds is not None:
            res, _, _ = run_conv(
                to_conv_in(residual), ds["w"], ds["b"], ds["ic"], ds["oc"], ds["k"], ds["s"], ds["p"], H, W
            )
            res = ttnn.typecast(res, ttnn.float32)
        else:
            res = residual
        out = ttnn.relu(ttnn.add(out, res))

        # NHWC [1, oh, ow, planes] -> NCHW [1, planes, oh, ow]
        return ttnn.permute(out, (0, 3, 1, 2))

    return forward


def s_e_basic_block(*args, **kwargs):
    raise RuntimeError(
        "s_e_basic_block requires build(device, torch_module) to bind trained conv/bn/SE "
        "weights; the bare callable has no parameters."
    )
