# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from models.experimental.xtts.config import (  # noqa: F401
    COND_CHANNELS,
    DECODER_INPUT_DIM,
    LRELU_SLOPE,
    OUT_CHANNELS,
    RESBLOCK_DILATION_SIZES,
    RESBLOCK_KERNEL_SIZES,
    UPSAMPLE_INITIAL_CHANNEL,
    UPSAMPLE_KERNEL_SIZES,
    UPSAMPLE_RATES,
)


def get_padding(k: int, d: int) -> int:
    """Compute same-padding for a dilated convolution."""
    return int((k * d - d) / 2)


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        """Build dilated residual conv pairs with weight norm."""
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
                for _ in dilation
            ]
        )

    def forward(self, x):
        """Apply residual dilated convolutions with leaky ReLU."""
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """Strip weight-norm parametrizations from residual convs."""
        for layer in self.convs1:
            remove_parametrizations(layer, "weight")
        for layer in self.convs2:
            remove_parametrizations(layer, "weight")


class XttsHifiganGenerator(nn.Module):
    def __init__(self):
        """Build HiFi-GAN upsampler, residual blocks, and speaker cond layers."""
        super().__init__()
        self.num_kernels = len(RESBLOCK_KERNEL_SIZES)
        self.num_upsamples = len(UPSAMPLE_RATES)

        # XTTS: no weight-norm on conv_pre/conv_post; conv_post has no bias.
        self.conv_pre = Conv1d(DECODER_INPUT_DIM, UPSAMPLE_INITIAL_CHANNEL, 7, 1, padding=3)

        self.ups = nn.ModuleList(
            [
                weight_norm(
                    ConvTranspose1d(
                        UPSAMPLE_INITIAL_CHANNEL // (2**i),
                        UPSAMPLE_INITIAL_CHANNEL // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
                for i, (u, k) in enumerate(zip(UPSAMPLE_RATES, UPSAMPLE_KERNEL_SIZES))
            ]
        )

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = UPSAMPLE_INITIAL_CHANNEL // (2 ** (i + 1))
            for k, d in zip(RESBLOCK_KERNEL_SIZES, RESBLOCK_DILATION_SIZES):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = Conv1d(ch, OUT_CHANNELS, 7, 1, padding=3, bias=False)

        self.cond_layer = Conv1d(COND_CHANNELS, UPSAMPLE_INITIAL_CHANNEL, 1)
        self.conds = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = UPSAMPLE_INITIAL_CHANNEL // (2 ** (i + 1))
            self.conds.append(Conv1d(COND_CHANNELS, ch, 1))

    def forward(self, x, g):
        """Generate waveform from latents conditioned on speaker embedding."""
        o = self.conv_pre(x)
        o = o + self.cond_layer(g)
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            o = o + self.conds[i](g)
            z_sum = None
            for j in range(self.num_kernels):
                res = self.resblocks[i * self.num_kernels + j](o)
                z_sum = res if z_sum is None else z_sum + res
            o = z_sum / self.num_kernels
        o = F.leaky_relu(o)  # default slope 0.01 here, not LRELU_SLOPE
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        """Strip weight-norm from upsamplers and residual blocks."""
        for layer in self.ups:
            remove_parametrizations(layer, "weight")
        for block in self.resblocks:
            block.remove_weight_norm()


def build_reference_waveform_decoder(state_dict):
    """Load HiFi-GAN generator weights and return eval-mode module."""
    prefix = "hifigan_decoder.waveform_decoder."
    slice_sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

    module = XttsHifiganGenerator()
    module.load_state_dict(slice_sd, strict=True)
    module.remove_weight_norm()
    return module.eval()
