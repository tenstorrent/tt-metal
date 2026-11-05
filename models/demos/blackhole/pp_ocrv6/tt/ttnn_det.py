# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# PP-OCRv6_small_det on ttnn. Activations are flattened NHWC: [1, 1, B*H*W, C].

import torch

import ttnn
from models.demos.blackhole.pp_ocrv6.reference.det import STAGE_CONFIGS
from models.demos.blackhole.pp_ocrv6.tt.common import RELU, Backbone, Conv, Linear, SqueezeExcite, fold_bn, to_row_major


def _upsample(x, batch, h, w, scale):
    c = x.shape[-1]
    x = ttnn.reshape(to_row_major(x), (batch, h, w, c))
    x = ttnn.upsample(x, scale, mode="nearest", memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(x, (1, 1, batch * h * scale * w * scale, c))


class TtPPOCRv6SmallDet:
    """Returns the 4x4 sub-pixel probability map [1, 1, B*(H/4)*(W/4), 16]; see `unshuffle_probability`."""

    def __init__(self, device, torch_model):
        neck = torch_model.model.neck
        head = torch_model.head
        self.device = device
        self.backbone = Backbone(device, torch_model.model.backbone, STAGE_CONFIGS)

        self.insert = []
        for m in neck.insert_conv:
            se = m.squeeze_excitation_block
            self.insert.append(
                (Linear(device, m.in_conv.weight), SqueezeExcite(device, se.conv1, se.conv2, 0.2, 0.5, True))
            )
        self.inp = []
        for m in neck.input_conv:
            se = m.squeeze_excitation_module
            dw = m.depthwise_convolution
            self.inp.append(
                (
                    Conv(device, dw.weight, dw.bias, groups=dw.in_channels),
                    Linear(device, m.pointwise_convolution.weight),
                    SqueezeExcite(device, se.conv1, se.conv2, 0.2, 0.5, True),
                )
            )

        self.conv_down = Conv(
            device, *fold_bn(head.conv_down.convolution.weight, None, head.conv_down.norm), activation=RELU
        )

        # ConvTranspose2d(k=2, s=2) == per-pixel matmul to 4 sub-pixels.
        up = head.conv_up.convolution
        w_up = up.weight.permute(0, 2, 3, 1)  # [cin, a, b, cout]
        scale = head.conv_up.norm.weight / torch.sqrt(head.conv_up.norm.running_var + head.conv_up.norm.eps)
        shift = (up.bias - head.conv_up.norm.running_mean) * scale + head.conv_up.norm.bias
        w_up = (w_up * scale).reshape(w_up.shape[0], -1)
        self.up = Linear(device, w_up.t().reshape(-1, w_up.shape[0], 1, 1), shift.repeat(4), activation="relu")

        fin = head.conv_final
        c = fin.weight.shape[0]
        w_fin = torch.zeros(4 * c, 16)
        for g in range(4):
            w_fin[g * c : (g + 1) * c, g * 4 : (g + 1) * 4] = fin.weight[:, 0].reshape(c, 4)
        self.final = Linear(device, w_fin.t().reshape(16, 4 * c, 1, 1), fin.bias.repeat(16))

    def __call__(self, x, batch, h, w):
        # x: host or device NHWC [B, H, W, 3]
        feats = self.backbone(x, batch, h, w)

        fused = [se(lin(f), batch) for (lin, se), (f, _, _) in zip(self.insert, feats)]
        for i in range(2, -1, -1):
            _, fh, fw = feats[i + 1]
            up = ttnn.to_layout(_upsample(fused[i + 1], batch, fh, fw, 2), ttnn.TILE_LAYOUT)
            fused[i] = ttnn.add(fused[i], up, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        outs = []
        for i, ((dw, pw, se), f, (_, fh, fw)) in enumerate(zip(self.inp, fused, feats)):
            y, _, _ = dw(f, batch, fh, fw)
            y = se(pw(y), batch)
            outs.append(_upsample(y, batch, fh, fw, 2**i) if i else to_row_major(y))
        x = ttnn.concat(outs[::-1], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        _, h2, w2 = feats[0]
        x, _, _ = self.conv_down(x, batch, h2, w2)
        x = self.up(x)
        x = ttnn.sigmoid(self.final(x))
        return x, h2, w2


def unshuffle_probability(sub_pixel, batch, h4, w4):
    """[1, 1, B*h4*w4, 16] (or padded) -> [B, 1, 4*h4, 4*w4]."""
    t = sub_pixel.reshape(-1, sub_pixel.shape[-1])[: batch * h4 * w4, :16]
    t = t.reshape(batch, h4, w4, 2, 2, 2, 2)  # (i, j, a, b, a2, b2)
    t = t.permute(0, 1, 3, 5, 2, 4, 6)  # (i, a, a2, j, b, b2)
    return t.reshape(batch, 1, 4 * h4, 4 * w4)
