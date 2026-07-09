# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the XTTS-v2 HiFi-GAN generator (``waveform_decoder``).

Mirrors ``reference/xtts_hifigan.py`` exactly:

    o = conv_pre(x) + cond_layer(g)
    for each of 4 upsample stages i:
        o = leaky_relu(o, 0.1); o = ups[i](o); o = o + conds[i](g)
        o = mean_j resblocks[i*3 + j](o)         # multi-receptive-field fusion
    o = leaky_relu(o, 0.01); o = conv_post(o); o = tanh(o)

Everything stays channels-last ``[N, L, C]`` ROW_MAJOR — the conv primitives and
all needed eltwise ops (leaky_relu, broadcast add, scalar mul, tanh) run in that
layout, so no per-op relayout is required. Weights come from the folded (weight-
norm removed) reference ``state_dict``.
"""

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_hifigan import (
    LRELU_SLOPE,
    RESBLOCK_DILATION_SIZES,
    RESBLOCK_KERNEL_SIZES,
    UPSAMPLE_RATES,
    get_padding,
)
from models.experimental.xtts.tt.xtts_conv import TtConv1d, TtConvTranspose1d

FINAL_LRELU_SLOPE = 0.01  # coqui's pre-conv_post activation uses F.leaky_relu default


class TtResBlock1(LightweightModule):
    """HiFi-GAN ResBlock "1": 3 (dilated conv -> plain conv) residual pairs."""

    def __init__(self, device, state_dict, prefix, kernel_size, dilation):
        super().__init__()
        self.convs1 = [
            TtConv1d(
                device,
                state_dict[f"{prefix}convs1.{j}.weight"],
                state_dict[f"{prefix}convs1.{j}.bias"],
                padding=get_padding(kernel_size, d),
                dilation=d,
            )
            for j, d in enumerate(dilation)
        ]
        self.convs2 = [
            TtConv1d(
                device,
                state_dict[f"{prefix}convs2.{j}.weight"],
                state_dict[f"{prefix}convs2.{j}.bias"],
                padding=get_padding(kernel_size, 1),
                dilation=1,
            )
            for j in range(len(dilation))
        ]

    def forward(self, x):
        # Free each conv/activation temporary as soon as it is consumed. The block's
        # input ``x`` is preserved on the first iteration (the caller reuses it for the
        # other MRF resblocks); later residuals are internal and freed.
        for idx, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            a = ttnn.leaky_relu(x, negative_slope=LRELU_SLOPE)
            b = c1(a)
            ttnn.deallocate(a)
            c = ttnn.leaky_relu(b, negative_slope=LRELU_SLOPE)
            ttnn.deallocate(b)
            d = c2(c)
            ttnn.deallocate(c)
            nxt = ttnn.add(d, x)
            ttnn.deallocate(d)
            if idx > 0:
                ttnn.deallocate(x)
            x = nxt
        return x


class TtHifiganGenerator(LightweightModule):
    """XTTS-v2 ``waveform_decoder``: GPT latent ``x`` (+ speaker embedding ``g``) -> waveform.

    Inputs are channels-last ROW_MAJOR: ``x`` is ``[N, T, 1024]``, ``g`` is
    ``[N, 1, 512]``. Output is ``[N, T*256, 1]``.
    """

    def __init__(self, device, state_dict):
        super().__init__()
        self.device = device
        self.num_kernels = len(RESBLOCK_KERNEL_SIZES)  # 3
        self.num_upsamples = len(UPSAMPLE_RATES)  # 4
        self.inv_num_kernels = 1.0 / self.num_kernels

        self.conv_pre = TtConv1d(device, state_dict["conv_pre.weight"], state_dict["conv_pre.bias"], padding=3)
        self.cond_layer = TtConv1d(device, state_dict["cond_layer.weight"], state_dict["cond_layer.bias"])

        self.ups = [
            TtConvTranspose1d(
                device,
                state_dict[f"ups.{i}.weight"],
                state_dict[f"ups.{i}.bias"],
                stride=UPSAMPLE_RATES[i],
            )
            for i in range(self.num_upsamples)
        ]
        self.conds = [
            TtConv1d(device, state_dict[f"conds.{i}.weight"], state_dict[f"conds.{i}.bias"])
            for i in range(self.num_upsamples)
        ]

        self.resblocks = []
        for i in range(self.num_upsamples):
            for j, (k, d) in enumerate(zip(RESBLOCK_KERNEL_SIZES, RESBLOCK_DILATION_SIZES)):
                self.resblocks.append(TtResBlock1(device, state_dict, f"resblocks.{i * self.num_kernels + j}.", k, d))

        # conv_post has no bias in XTTS.
        self.conv_post = TtConv1d(device, state_dict["conv_post.weight"], None, padding=3)

    def forward(self, x, g):
        # Deep conv chain — the vocoder's memory-dominant path, whose activation
        # footprint grows with output length. Each temporary is freed the moment the
        # next op consumes it; ``g`` is never freed (reused by every cond layer).
        cond_global = self.cond_layer(g)  # [N, 1, 512], broadcasts over T
        pre = self.conv_pre(x)
        ttnn.deallocate(x)  # upsampler output, not reused after conv_pre
        o = ttnn.add(pre, cond_global)
        ttnn.deallocate(pre)
        ttnn.deallocate(cond_global)

        for i in range(self.num_upsamples):
            a = ttnn.leaky_relu(o, negative_slope=LRELU_SLOPE)
            ttnn.deallocate(o)
            u = self.ups[i](a)
            ttnn.deallocate(a)
            cg = self.conds[i](g)
            o = ttnn.add(u, cg)
            ttnn.deallocate(u)
            ttnn.deallocate(cg)
            z_sum = None
            for j in range(self.num_kernels):
                res = self.resblocks[i * self.num_kernels + j](o)  # does not free o
                if z_sum is None:
                    z_sum = res
                else:
                    z_new = ttnn.add(z_sum, res)
                    ttnn.deallocate(z_sum)
                    ttnn.deallocate(res)
                    z_sum = z_new
            o_new = ttnn.mul(z_sum, self.inv_num_kernels)
            ttnn.deallocate(z_sum)
            ttnn.deallocate(o)  # free the resblock input once the MRF sum is done
            o = o_new

        a = ttnn.leaky_relu(o, negative_slope=FINAL_LRELU_SLOPE)
        ttnn.deallocate(o)
        p = self.conv_post(a)
        ttnn.deallocate(a)
        out = ttnn.tanh(p)
        ttnn.deallocate(p)
        return out
