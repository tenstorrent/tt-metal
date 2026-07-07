# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference (pure-PyTorch) implementation of the XTTS-v2 HiFi-GAN generator.

This is coqui's *XTTS-specific* ``HifiganGenerator`` (from
``TTS/tts/layers/xtts/hifigan_decoder.py``) — the ``waveform_decoder`` inside
``HifiDecoder``. It differs from the stock vocoder ``HifiganGenerator`` by the
speaker-conditioning path (``cond_layer`` + per-upsample ``conds``), and by
XTTS's config: ``conv_pre``/``conv_post`` carry no weight-norm and
``conv_post`` has no bias.

We reimplement it verbatim (rather than depend on ``coqui-tts``) so the real
checkpoint weights load directly. Weights are stored weight-normed in the
checkpoint (``parametrizations.weight.original0/original1``); building with the
same ``weight_norm`` parametrization lets ``load_state_dict`` match, and
``remove_weight_norm()`` then folds them exactly as coqui does at inference.

Verified module shapes against the checkpoint ``hifigan_decoder.waveform_decoder.*``.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

# XTTS-v2 waveform_decoder hyper-parameters (coqui/XTTS-v2 config, model_args).
DECODER_INPUT_DIM = 1024  # GPT latent dim fed to conv_pre
UPSAMPLE_INITIAL_CHANNEL = 512
UPSAMPLE_RATES = [8, 8, 2, 2]  # product = 256 = output_hop_length
UPSAMPLE_KERNEL_SIZES = [16, 16, 4, 4]
RESBLOCK_KERNEL_SIZES = [3, 7, 11]
RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
COND_CHANNELS = 512  # d_vector_dim (speaker embedding)
OUT_CHANNELS = 1
LRELU_SLOPE = 0.1


def get_padding(k: int, d: int) -> int:
    """ "same"-length padding for a dilated conv (coqui's helper)."""
    return int((k * d - d) / 2)


class ResBlock1(nn.Module):
    """HiFi-GAN ResBlock type "1": 3 dilated (convs1) + 3 plain (convs2) pairs
    with residual adds and leaky-relu(0.1)."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
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
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_parametrizations(layer, "weight")
        for layer in self.convs2:
            remove_parametrizations(layer, "weight")


class XttsHifiganGenerator(nn.Module):
    """XTTS-v2 ``waveform_decoder``: GPT-latent (+ speaker embedding) -> waveform."""

    def __init__(self):
        super().__init__()
        self.num_kernels = len(RESBLOCK_KERNEL_SIZES)
        self.num_upsamples = len(UPSAMPLE_RATES)

        # conv_pre/conv_post carry NO weight-norm in XTTS; conv_post has no bias.
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

        # Speaker conditioning: global (cond_layer) + per-upsample (conds).
        self.cond_layer = Conv1d(COND_CHANNELS, UPSAMPLE_INITIAL_CHANNEL, 1)
        self.conds = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = UPSAMPLE_INITIAL_CHANNEL // (2 ** (i + 1))
            self.conds.append(Conv1d(COND_CHANNELS, ch, 1))

    def forward(self, x, g):
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
        o = F.leaky_relu(o)  # NOTE: default slope 0.01 here, not LRELU_SLOPE
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_parametrizations(layer, "weight")
        for block in self.resblocks:
            block.remove_weight_norm()


def build_reference_waveform_decoder(state_dict):
    """Instantiate the XTTS waveform_decoder with real weights, weight-norm folded.

    Args:
        state_dict: full checkpoint state dict from ``load_xtts_state_dict``.
    Returns:
        ``XttsHifiganGenerator`` in eval mode; every conv exposes a folded ``.weight``.
    """
    prefix = "hifigan_decoder.waveform_decoder."
    slice_sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

    module = XttsHifiganGenerator()
    module.load_state_dict(slice_sd, strict=True)
    module.remove_weight_norm()
    return module.eval()
