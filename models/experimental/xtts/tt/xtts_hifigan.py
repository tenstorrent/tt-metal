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

    def __init__(self, device, state_dict, prefix, kernel_size, dilation, activations_dtype=ttnn.float32):
        super().__init__()
        # The leaky_relu that sits between convs1 and convs2 is fused onto the
        # convs1 output (post-bias) — convs1's output feeds only that activation, so
        # ``leaky_relu(convs1(a))`` folds into the conv exactly. The pre-activation
        # before convs1 cannot fuse (its input is reused raw by the residual add).
        mid_act = ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, LRELU_SLOPE)
        self.convs1 = [
            TtConv1d(
                device,
                state_dict[f"{prefix}convs1.{j}.weight"],
                state_dict[f"{prefix}convs1.{j}.bias"],
                padding=get_padding(kernel_size, d),
                dilation=d,
                activation=mid_act,
                activations_dtype=activations_dtype,
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
                activations_dtype=activations_dtype,
            )
            for j in range(len(dilation))
        ]

    def forward(self, x):
        # Free each conv/activation temporary as soon as it is consumed. The block's
        # input ``x`` is preserved on the first iteration (the caller reuses it for the
        # other MRF resblocks); later residuals are internal and freed.
        for idx, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            a = ttnn.leaky_relu(x, negative_slope=LRELU_SLOPE)
            b = c1(a)  # leaky_relu(0.1) is fused onto this conv's output
            ttnn.deallocate(a)
            d = c2(b)
            ttnn.deallocate(b)
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

        # Mixed precision: the last upsample stage runs its ups/conds/resblocks in
        # bf16, the rest in fp32. The late stages carry the largest activations (most
        # DRAM eltwise traffic) and the least remaining accumulation depth, so bf16
        # there buys the most device time for the least PCC drift. Measured: stage 3
        # alone holds PCC >=0.99 and cuts ~7.5% device time; adding stage 2 drops PCC
        # to ~0.984 (below threshold).
        bf16_stages = {self.num_upsamples - 1}

        def act_dtype(i):
            return ttnn.bfloat16 if i in bf16_stages else ttnn.float32

        self.conv_pre = TtConv1d(device, state_dict["conv_pre.weight"], state_dict["conv_pre.bias"], padding=3)
        self.cond_layer = TtConv1d(device, state_dict["cond_layer.weight"], state_dict["cond_layer.bias"])

        self.ups = [
            TtConvTranspose1d(
                device,
                state_dict[f"ups.{i}.weight"],
                state_dict[f"ups.{i}.bias"],
                stride=UPSAMPLE_RATES[i],
                activations_dtype=act_dtype(i),
            )
            for i in range(self.num_upsamples)
        ]
        self.conds = [
            TtConv1d(
                device,
                state_dict[f"conds.{i}.weight"],
                state_dict[f"conds.{i}.bias"],
                activations_dtype=act_dtype(i),
            )
            for i in range(self.num_upsamples)
        ]

        self.resblocks = []
        for i in range(self.num_upsamples):
            for j, (k, d) in enumerate(zip(RESBLOCK_KERNEL_SIZES, RESBLOCK_DILATION_SIZES)):
                self.resblocks.append(
                    TtResBlock1(
                        device,
                        state_dict,
                        f"resblocks.{i * self.num_kernels + j}.",
                        k,
                        d,
                        activations_dtype=act_dtype(i),
                    )
                )

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
