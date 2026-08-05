# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HiFTGenerator: mel + f0 -> waveform, assembled from the modules around it.

    f0 = f0_predictor(mel)                      [B, 1, T_mel] -> [B, T_mel]
    s  = m_source(upsample(f0, 256))            harmonic excitation, audio rate
    s_stft = stft(s)                            [B, 18, T_frames]
    x  = conv_pre(mel)                          80 -> 512
    for stage i in 0, 1:
        x = ups[i](leaky_relu(x))               x8 each
        if last: x = reflection_pad(x)          (1, 0)  <- see LENGTHS below
        x = x + source_resblocks[i](source_downs[i](s_stft))
        x = mean_j resblocks[i*3 + j](x)
    x = conv_post(leaky_relu(x))                -> 18 channels (n_fft + 2)
    mag, phase = exp(x[:, :9]), sin(x[:, 9:])
    wav = clamp(istft(mag, phase), +-0.99)

LENGTHS -- the part that is easy to get wrong
---------------------------------------------
Traced from the captured reference (282 mel frames, 3.3 s of audio):

    mel            282
    conv_pre       282            80 -> 512
    ups[0]         282 -> 2256    512 -> 256   (x8)
    resblocks      2256           (length unchanged, "same" padding)
    ups[1]         2256 -> 18048  256 -> 128   (x8)
    reflection_pad 18048 -> 18049               <-- +1 sample
    conv_post      18049          128 -> 18
    istft          18049 -> 72192               (x4, minus the center trim)

and 72192 / 282 = 256 = the mel hop_size, so the round trip is exact.

That `+1` is NOT cosmetic. The source branch runs `source_downs[1] = Conv1d(18,
128, k=1, s=1)` over the excitation spectrogram, which has **18049** frames --
so `x = x + si` only aligns because of the reflection pad. Get it wrong and the
failure surfaces as a shape mismatch at a residual add, forty-odd convolutions
deep. `shape_trace()` below computes the whole chain without a device precisely
so that class of bug is caught before any silicon time is spent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import ttnn

from .conv import TtConv1d
from .istft import TtIStft
from .resblock import TtResBlock
from .stft import TtStft
from .upsample import TtConvTranspose1d


@dataclass
class Stage:
    """One upsample stage's shapes, as computed by shape_trace()."""

    index: int
    in_channels: int
    out_channels: int
    in_length: int
    up_length: int  # after ConvTranspose1d
    padded_length: int  # after reflection_pad, if this is the last stage
    source_length: int  # what source_downs[i] produces from s_stft


def shape_trace(
    mel_frames: int,
    base_channels: int = 512,
    upsample_rates=(8, 8),
    upsample_kernel_sizes=(16, 16),
    n_fft: int = 16,
    hop_len: int = 4,
    in_channels: int = 80,
) -> dict:
    """Walk the whole vocoder graph in pure Python, returning every length.

    Device-free and dependency-free, so it can assert against the captured
    reference shapes in a test that runs anywhere.
    """
    total_up = int(np.prod(upsample_rates)) * hop_len
    audio_length = mel_frames * total_up

    # The excitation is transformed at audio rate with center=True.
    stft_frames = (audio_length + n_fft - n_fft) // hop_len + 1

    # source_downs mirrors the upsample rates in reverse and cumulatively:
    # downsample_rates = [1] + upsample_rates[::-1][:-1]; then cumprod, reversed.
    downsample_rates = [1] + list(upsample_rates[::-1])[:-1]
    cum = np.cumprod(downsample_rates)[::-1]

    stages, length = [], mel_frames
    n = len(upsample_rates)
    for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
        pad = (k - u) // 2
        up_len = (length - 1) * u - 2 * pad + (k - 1) + 1
        padded = up_len + (1 if i == n - 1 else 0)  # ReflectionPad1d((1, 0))

        d = int(cum[i])
        if d == 1:
            src_len = stft_frames  # Conv1d(18, C, 1, 1)
        else:
            # Conv1d(18, C, kernel=d*2, stride=d, padding=d//2)
            src_len = (stft_frames + 2 * (d // 2) - d * 2) // d + 1

        stages.append(
            Stage(
                index=i,
                in_channels=base_channels // (2**i),
                out_channels=base_channels // (2 ** (i + 1)),
                in_length=length,
                up_length=up_len,
                padded_length=padded,
                source_length=src_len,
            )
        )
        length = padded

    # istft: center=True trims n_fft//2 from each end
    wav_length = (length - 1) * hop_len + n_fft - 2 * (n_fft // 2)

    return {
        "mel_frames": mel_frames,
        "mel_channels": in_channels,
        "total_upsample": total_up,
        "audio_length": audio_length,
        "stft_frames": stft_frames,
        "stages": stages,
        "conv_post_length": length,
        "waveform_length": wav_length,
    }


class TtHiFTGenerator:
    """The vocoder, on device. Tensors are channels-last `[B, L, C]` internally."""

    def __init__(self, device, module: torch.nn.Module, dtype=ttnn.bfloat16):
        """Built from a live cosyvoice.hifigan.generator.HiFTGenerator, so every
        weight comes from the checkpoint rather than being re-derived."""
        self.device = device
        self.dtype = dtype
        self.n_fft = module.istft_params["n_fft"]
        self.hop_len = module.istft_params["hop_len"]
        self.lrelu_slope = module.lrelu_slope
        self.audio_limit = module.audio_limit
        self.num_kernels = module.num_kernels
        self.num_upsamples = module.num_upsamples
        self.bins = self.n_fft // 2 + 1

        self.conv_pre = TtConv1d.from_module(device, module.conv_pre, dtype=dtype)
        self.ups = [TtConvTranspose1d.from_module(device, m, dtype=dtype) for m in module.ups]
        self.source_downs = [TtConv1d.from_module(device, m, dtype=dtype) for m in module.source_downs]
        self.source_resblocks = [TtResBlock.from_module(device, m, dtype=dtype) for m in module.source_resblocks]
        self.resblocks = [TtResBlock.from_module(device, m, dtype=dtype) for m in module.resblocks]
        self.conv_post = TtConv1d.from_module(device, module.conv_post, dtype=dtype)

        window = module.stft_window.detach().cpu().numpy()
        self.stft = TtStft(device, self.n_fft, self.hop_len, window=window, dtype=dtype)
        self.istft = TtIStft(device, self.n_fft, self.hop_len, window=window, dtype=dtype)

    def decode(self, mel, s, mel_frames: int, batch_size: int = 1):
        """mel: ttnn [B, T_mel, 80]; s: ttnn [B, T_audio, 1] -> [B, L, 1] waveform."""
        trace = shape_trace(mel_frames, n_fft=self.n_fft, hop_len=self.hop_len)

        s_stft, _ = self.stft(s, trace["audio_length"], batch_size)  # [B, 18, T]
        s_stft = ttnn.permute(s_stft, (0, 2, 1))  # [B, T, 18]

        x, _ = self.conv_pre(mel, mel_frames, batch_size)

        for st in trace["stages"]:
            x = ttnn.leaky_relu(x, self.lrelu_slope)
            x, _ = self.ups[st.index](x, st.in_length, batch_size)

            if st.index == self.num_upsamples - 1:
                # ReflectionPad1d((1, 0)): prepend x[:, 1] -- one sample, so the
                # "reflection" is a single slice, no exchange matmul needed.
                head = ttnn.slice(x, [0, 1, 0], [batch_size, 2, st.out_channels])
                x = ttnn.concat([head, x], dim=1)
                ttnn.deallocate(head)

            si, _ = self.source_downs[st.index](s_stft, trace["stft_frames"], batch_size)
            si = self.source_resblocks[st.index](si, st.source_length, batch_size)
            nx = ttnn.add(x, si)
            ttnn.deallocate(si)
            ttnn.deallocate(x)
            x = nx

            acc = None
            for j in range(self.num_kernels):
                out = self.resblocks[st.index * self.num_kernels + j](x, st.padded_length, batch_size)
                acc = out if acc is None else ttnn.add(acc, out)
                if acc is not out:
                    ttnn.deallocate(out)
            ttnn.deallocate(x)
            x = ttnn.multiply(acc, 1.0 / self.num_kernels)
            ttnn.deallocate(acc)

        x = ttnn.leaky_relu(x, 0.01)  # F.leaky_relu default slope, as the reference
        x, _ = self.conv_post(x, trace["conv_post_length"], batch_size)

        # [B, T, 18] -> magnitude/phase, then back to [B, 9, T] for the iSTFT.
        x = ttnn.permute(x, (0, 2, 1))
        mag = ttnn.exp(ttnn.slice(x, [0, 0, 0], [batch_size, self.bins, trace["conv_post_length"]]))
        pha = ttnn.sin(ttnn.slice(x, [0, self.bins, 0], [batch_size, 2 * self.bins, trace["conv_post_length"]]))
        ttnn.deallocate(x)
        mag = ttnn.clamp(mag, None, 1e2)  # the reference clips magnitude at 1e2

        real = ttnn.multiply(mag, ttnn.cos(pha))
        imag = ttnn.multiply(mag, ttnn.sin(pha))
        ttnn.deallocate(mag)
        ttnn.deallocate(pha)

        wav = self.istft(real, imag)
        ttnn.deallocate(real)
        ttnn.deallocate(imag)
        return ttnn.clamp(wav, -self.audio_limit, self.audio_limit)
