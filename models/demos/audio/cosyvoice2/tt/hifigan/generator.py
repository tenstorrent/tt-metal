# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""HiFT vocoder decode: mel + excitation -> waveform, for CosyVoice2's 3-stage HiFT.

Scope of this module: `decode(mel, s, ...)` only -- mel plus an already-computed
excitation `s` in, waveform out. Computing `s` (f0 prediction -> SineGen2 ->
SourceModuleHnNSF) is deferred to a follow-up: CosyVoice2's `HiFTGenerator`
selects `SineGen2` at its actual 24 kHz sampling rate (confirmed against upstream
source: `sinegen_type='1' if self.sampling_rate == 22050 else '2'`), a
structurally different module from the `SineGen` (type 1) that the CosyVoice1
TTNN port (tenstorrent/tt-metal#52540) implements -- SineGen2 cumsum-integrates
phase at mel-frame rate, then linearly interpolates the phase (not f0) back up
to audio rate, rather than cumsum-integrating directly over the full audio-rate
signal the way SineGen does. That is new derivation work, not a port, and it is
being scoped separately rather than blocking this piece on it. `decode()` takes
`s` as an explicit input for exactly this reason -- it is the real seam upstream
uses too (`HiFTGenerator.decode(x, s)` in cosyvoice/hifigan/generator.py).

    x = conv_pre(mel)                             80 -> base_channels
    for stage i in range(num_upsamples):
        x = ups[i](leaky_relu(x))                 upsample_rates[i]x
        if last stage: x = reflection_pad(x)      (1, 0)
        x = x + source_resblocks[i](source_downs[i](s_stft))
        x = mean_j resblocks[i*num_kernels + j](x)
    x = conv_post(leaky_relu(x))                   -> n_fft + 2 channels
    mag, phase = exp(x[:, :bins]), sin(x[:, bins:])
    wav = clamp(istft(mag*cos(phase), mag*sin(phase)), +-audio_limit)

verbatim from `cosyvoice.hifigan.generator.HiFTGenerator.decode`, confirmed
against upstream source directly (not summarized) rather than assumed unchanged
from the CosyVoice1 port. CosyVoice2's actual topology, from
examples/libritts/cosyvoice2/conf/cosyvoice2.yaml in QwenAudio/CosyVoice:

    upsample_rates              [8, 5, 3]           (CosyVoice1: [8, 8])
    upsample_kernel_sizes       [16, 11, 7]         (CosyVoice1: [16, 16])
    resblock_kernel_sizes       [3, 7, 11]          (same both)
    source_resblock_kernel_sizes [7, 7, 11]         (CosyVoice1: [7, 11])
    n_fft / hop_len              16 / 4             (same both)
    sampling_rate                24000              (CosyVoice1: 22050)

STAGE-COUNT GENERALITY, checked rather than assumed
----------------------------------------------------
Both upstream's `HiFTGenerator.__init__`/`decode` and the CosyVoice1 TTNN port's
`TtHiFTGenerator` build every per-stage list (`ups`, `source_downs`,
`source_resblocks`, `resblocks`) via a loop over `upsample_rates`, and process
stages via `for i in range(num_upsamples)` / `for st in trace["stages"]` -- so 2
stages versus 3 is genuinely a config/checkpoint difference, not a rewrite, in
both places.

The one bug found in the CosyVoice1 port that *would* have broken silently at 3
stages: its `_decode_impl` calls `shape_trace(mel_frames, n_fft=.., hop_len=..)`
without passing `upsample_rates`/`upsample_kernel_sizes`, so `shape_trace`'s own
2-stage function defaults would have silently won, producing a `trace["stages"]`
list of length 2 against a 3-stage checkpoint's weights -- the third
ups/source_downs/resblock triplet would exist but never run, with no error, just
wrong-shaped, wrong-length, plausible audio. Fixed here by always threading the
generator's own `upsample_rates`/`upsample_kernel_sizes` through explicitly (see
`TtHiFTDecoder.decode`) -- there is no function-default path left to fall onto.

The underlying per-stage length math (`shape_trace` below) was hand-verified to
generalize correctly to CosyVoice2's [8, 5, 3]: the cumulative-downsample factors
that size each stage's `source_downs` conv come out to [15, 3, 1] (versus
CosyVoice1's [8, 1]), consistent with "how far behind full audio-rate resolution
stage i's excitation branch is" at each point in the upsample chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm

import ttnn

from .conv import TtConv1d
from .istft import TtIStft, periodic_hann
from .resblock import TtResBlock, get_padding
from .snake import TtSnake
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
    source_downsample: int  # cumulative downsample factor feeding source_downs[i]


def shape_trace(
    mel_frames: int,
    base_channels: int,
    upsample_rates,
    upsample_kernel_sizes,
    n_fft: int,
    hop_len: int,
    in_channels: int = 80,
) -> dict:
    """Walk the whole decode() graph in pure Python, returning every length.

    Device-free and dependency-free, so it is unit-testable on its own and so a
    shape bug is caught before any silicon time is spent. Every argument that
    determines topology is required (no defaults to silently fall back to) --
    see the module docstring for why that matters at 3 stages.
    """
    assert len(upsample_rates) == len(upsample_kernel_sizes)
    n = len(upsample_rates)
    total_up = int(np.prod(upsample_rates)) * hop_len
    audio_length = mel_frames * total_up

    # torch.stft(..., center=True) frame count at even n_fft: see TtStft.n_frames.
    stft_frames = audio_length // hop_len + 1

    # downsample_rates mirrors upsample_rates in reverse, cumulatively: the
    # excitation is at full audio rate, and stage i's activation has only been
    # upsampled by prod(upsample_rates[:i+1]) so far, so source_downs[i] must
    # downsample the excitation spectrogram by total_up / (that partial product
    # * hop_len) to match. Verbatim structure of upstream's
    # `downsample_cum_rates = np.cumprod([1] + upsample_rates[::-1][:-1])`.
    downsample_rates = [1] + list(upsample_rates[::-1])[:-1]
    cum = np.cumprod(downsample_rates)[::-1]

    stages, length = [], mel_frames
    for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
        pad = (k - u) // 2
        up_len = (length - 1) * u - 2 * pad + (k - 1) + 1
        padded = up_len + (1 if i == n - 1 else 0)  # ReflectionPad1d((1, 0)) on the last stage only

        d = int(cum[i])
        if d == 1:
            src_len = stft_frames  # Conv1d(n_fft+2, C, 1, 1)
        else:
            # Conv1d(n_fft+2, C, kernel=d*2, stride=d, padding=d//2)
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
                source_downsample=d,
            )
        )
        length = padded

    # istft: center=True trims n_fft//2 from each end.
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


class TorchHiFTDecodeRef(torch.nn.Module):
    """Untouched-PyTorch reference for `TtHiFTDecoder.decode`.

    Built entirely from `torch.nn.Conv1d`/`ConvTranspose1d`/weight_norm/
    `torch.stft`/`torch.istft` and the plain Snake formula -- no TTNN, no
    `cosyvoice` package dependency (not installed here), so device output can be
    compared against it with no shared derivation on either side, the same bar
    `tests/pcc/test_istft.py::test_device_istft_matches_real_torch_istft` set.
    Random-initialised (there is no CosyVoice2 checkpoint in this environment
    yet); `TtHiFTDecoder.from_torch_ref` builds the device module from the exact
    same weights, including folding the weight_norm wrapped here.
    """

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        upsample_rates=(8, 5, 3),
        upsample_kernel_sizes=(16, 11, 7),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        source_resblock_kernel_sizes=(7, 7, 11),
        source_resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        n_fft: int = 16,
        hop_len: int = 4,
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        seed: int = 0,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def init_(m, std=0.02):
            for p in m.parameters():
                with torch.no_grad():
                    p.copy_(torch.empty_like(p).normal_(0, std, generator=g))
            return m

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.upsample_rates = tuple(upsample_rates)
        self.n_fft, self.hop_len = n_fft, hop_len
        self.bins = n_fft // 2 + 1
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.conv_pre = init_(weight_norm(Conv1d(in_channels, base_channels, 7, 1, padding=3)))

        self.ups = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                init_(
                    weight_norm(
                        ConvTranspose1d(
                            base_channels // (2**i), base_channels // (2 ** (i + 1)), k, u, padding=(k - u) // 2
                        )
                    )
                )
            )

        self.source_downs = torch.nn.ModuleList()
        self.source_resblocks = torch.nn.ModuleList()
        downsample_rates = [1] + list(upsample_rates[::-1])[:-1]
        cum = np.cumprod(downsample_rates)[::-1]
        for i, (d, k, dl) in enumerate(zip(cum, source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            d = int(d)
            ch = base_channels // (2 ** (i + 1))
            if d == 1:
                self.source_downs.append(init_(Conv1d(n_fft + 2, ch, 1, 1)))
            else:
                self.source_downs.append(init_(Conv1d(n_fft + 2, ch, d * 2, d, padding=d // 2)))
            self.source_resblocks.append(self._make_resblock(ch, k, dl, g))

        self.resblocks = torch.nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = base_channels // (2 ** (i + 1))
            for k, dl in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(self._make_resblock(ch, k, dl, g))

        # conv_post gets a larger init than the rest of the stack: 12 stacked
        # resblocks (3-stage config) each average 3 residual branches, which
        # shrinks the signal's dynamic range enough that a uniform std=0.02 init
        # leaves conv_post's magnitude-logit channels almost flat (measured
        # std~0.009 around a mean of ~1.0 after exp()) -- a real, trained
        # conv_post would produce properly-varying log-magnitude, but a random
        # near-constant one makes PCC a poor metric here for a reason that has
        # nothing to do with correctness: bf16 rounding error becomes comparable
        # to the signal's own tiny variance, and PCC (a correlation measure)
        # craters on a near-constant signal the same way it does on an exactly
        # constant one (see comp_pcc's own "one tensor is all zero" special
        # case) even though the absolute reconstruction is fine. Matches the
        # istft.py test's own reasoning for using wide-dynamic-range synthetic
        # data instead of plain torch.randn: the data has to actually exercise
        # the numerics being tested.
        final_ch = base_channels // (2**self.num_upsamples)
        self.conv_post = init_(weight_norm(Conv1d(final_ch, n_fft + 2, 7, 1, padding=3)), std=0.5)
        self.stft_window = torch.from_numpy(periodic_hann(n_fft))

    @staticmethod
    def _make_resblock(channels, kernel_size, dilations, g):
        m = torch.nn.Module()
        m.convs1 = torch.nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
                for d in dilations
            ]
        )
        m.convs2 = torch.nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
                for _ in dilations
            ]
        )
        for conv_list in (m.convs1, m.convs2):
            for c in conv_list:
                for p in c.parameters():
                    with torch.no_grad():
                        p.copy_(torch.empty_like(p).normal_(0, 0.02, generator=g))
        alpha1 = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(channels)) for _ in dilations])
        alpha2 = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(channels)) for _ in dilations])
        m.activations1 = torch.nn.ModuleList()
        for a in alpha1:
            act = torch.nn.Module()
            act.alpha = a
            m.activations1.append(act)
        m.activations2 = torch.nn.ModuleList()
        for a in alpha2:
            act = torch.nn.Module()
            act.alpha = a
            m.activations2.append(act)
        return m

    def _resblock_forward(self, m, x):
        for i in range(len(m.convs1)):
            xt = TtSnake.torch_reference(x, m.activations1[i].alpha)
            xt = m.convs1[i](xt)
            xt = TtSnake.torch_reference(xt, m.activations2[i].alpha)
            xt = m.convs2[i](xt)
            x = xt + x
        return x

    def _stft(self, x):
        spec = torch.stft(x, self.n_fft, self.hop_len, self.n_fft, window=self.stft_window, return_complex=True)
        spec = torch.view_as_real(spec)  # [B, F, T, 2]
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        return torch.istft(torch.complex(real, img), self.n_fft, self.hop_len, self.n_fft, window=self.stft_window)

    def decode(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """x: mel [B, 80, T_mel]; s: excitation [B, 1, T_audio] -> waveform [B, L]."""
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = F.pad(x, (1, 0), mode="reflect")

            si = self.source_downs[i](s_stft)
            si = self._resblock_forward(self.source_resblocks[i], si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                out = self._resblock_forward(self.resblocks[i * self.num_kernels + j], x)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, : self.bins, :])
        phase = torch.sin(x[:, self.bins :, :])

        x = self._istft(magnitude, phase)
        return torch.clamp(x, -self.audio_limit, self.audio_limit)


class TtHiFTDecoder:
    """The vocoder's decode() half, on device. Tensors are channels-last `[B, L, C]`."""

    def __init__(self, device, ref: TorchHiFTDecodeRef, dtype=ttnn.bfloat16):
        """Built from a `TorchHiFTDecodeRef` -- shares its exact weights (folding
        weight_norm), so a PCC test against `ref.decode(...)` has zero daylight
        between "the weights differ" and "the op differs"."""
        self.device = device
        self.dtype = dtype
        self.base_channels = ref.conv_pre.out_channels
        self.in_channels = ref.conv_pre.in_channels
        self.n_fft, self.hop_len = ref.n_fft, ref.hop_len
        self.bins = ref.bins
        self.num_upsamples = ref.num_upsamples
        self.num_kernels = ref.num_kernels
        self.upsample_rates = tuple(ref.upsample_rates)
        self.upsample_kernel_sizes = tuple(int(u.kernel_size[0]) for u in ref.ups)  # the attribute noted as missing
        self.lrelu_slope = ref.lrelu_slope
        self.audio_limit = ref.audio_limit

        self.conv_pre = TtConv1d.from_module(device, ref.conv_pre, dtype=dtype)
        self.ups = [TtConvTranspose1d.from_module(device, m, dtype=dtype) for m in ref.ups]
        self.source_downs = [TtConv1d.from_module(device, m, dtype=dtype) for m in ref.source_downs]
        self.source_resblocks = [self._resblock_from_module(device, m, dtype) for m in ref.source_resblocks]
        self.resblocks = [self._resblock_from_module(device, m, dtype) for m in ref.resblocks]
        self.conv_post = TtConv1d.from_module(device, ref.conv_post, dtype=dtype)

        window = ref.stft_window
        self.stft = TtStft(device, self.n_fft, self.hop_len, window=window, dtype=dtype)
        self.istft = TtIStft(device, self.n_fft, self.hop_len, window=window, dtype=dtype)

    @staticmethod
    def _resblock_from_module(device, m, dtype):
        channels = m.convs1[0].out_channels
        dilations = [int(c.dilation[0]) for c in m.convs1]
        return TtResBlock(
            device,
            channels=channels,
            kernel_size=int(m.convs1[0].kernel_size[0]),
            dilations=dilations,
            convs1=list(m.convs1),
            convs2=list(m.convs2),
            alphas1=[a.alpha.detach() for a in m.activations1],
            alphas2=[a.alpha.detach() for a in m.activations2],
            dtype=dtype,
        )

    def decode(self, mel, s, mel_frames: int, batch_size: int = 1):
        """mel: ttnn [B, T_mel, 80]; s: ttnn [B, T_audio, 1] -> [B, L, 1] waveform (NHWC).

        Every length this needs comes from `shape_trace`, called here with this
        instance's *own* `upsample_rates`/`upsample_kernel_sizes` -- explicitly,
        every time, with no default to silently fall back onto. See the module
        docstring for why that specific call site mattered.
        """
        trace = shape_trace(
            mel_frames,
            self.base_channels,
            self.upsample_rates,
            self.upsample_kernel_sizes,
            self.n_fft,
            self.hop_len,
            self.in_channels,
        )

        s_stft, _ = self.stft(s, trace["audio_length"], batch_size)  # [B, 2*bins, T]
        s_stft = ttnn.permute(s_stft, (0, 2, 1))  # [B, T, 2*bins]

        x, _ = self.conv_pre(mel, mel_frames, batch_size)

        for st in trace["stages"]:
            act = ttnn.leaky_relu(x, self.lrelu_slope)
            ttnn.deallocate(x)
            x, _ = self.ups[st.index](act, st.in_length, batch_size)
            ttnn.deallocate(act)

            if st.index == self.num_upsamples - 1:
                # ReflectionPad1d((1, 0)): prepend x[:, 1] -- one sample.
                head = ttnn.slice(x, [0, 1, 0], [batch_size, 2, st.out_channels])
                padded = ttnn.concat([head, x], dim=1)
                ttnn.deallocate(head)
                ttnn.deallocate(x)
                x = padded

            si, _ = self.source_downs[st.index](s_stft, trace["stft_frames"], batch_size)
            si_res = self.source_resblocks[st.index](si, st.source_length, batch_size)
            ttnn.deallocate(si)
            nx = ttnn.add(x, si_res)
            ttnn.deallocate(si_res)
            ttnn.deallocate(x)
            x = nx

            # Three (or however many num_kernels) ResBlocks read the SAME x and
            # their outputs are averaged, so x must outlive all of them.
            acc = None
            for j in range(self.num_kernels):
                out = self.resblocks[st.index * self.num_kernels + j](x, st.padded_length, batch_size)
                if acc is None:
                    acc = out
                else:
                    nacc = ttnn.add(acc, out)
                    ttnn.deallocate(acc)
                    ttnn.deallocate(out)
                    acc = nacc
            ttnn.deallocate(x)
            x = ttnn.multiply(acc, 1.0 / self.num_kernels)
            ttnn.deallocate(acc)

        act = ttnn.leaky_relu(x, 0.01)  # F.leaky_relu default slope, matching upstream
        ttnn.deallocate(x)
        x, _ = self.conv_post(act, trace["conv_post_length"], batch_size)
        ttnn.deallocate(act)

        T = trace["conv_post_length"]
        spec = ttnn.permute(x, (0, 2, 1))
        ttnn.deallocate(x)
        mag_lin = ttnn.slice(spec, [0, 0, 0], [batch_size, self.bins, T])
        pha_lin = ttnn.slice(spec, [0, self.bins, 0], [batch_size, 2 * self.bins, T])
        ttnn.deallocate(spec)

        mag = ttnn.exp(mag_lin)
        ttnn.deallocate(mag_lin)
        mag_c = ttnn.clamp(mag, 0.0, 1e2)
        ttnn.deallocate(mag)

        pha = ttnn.sin(pha_lin)
        ttnn.deallocate(pha_lin)
        cos_p, sin_p = ttnn.cos(pha), ttnn.sin(pha)
        ttnn.deallocate(pha)
        real = ttnn.multiply(mag_c, cos_p)
        imag = ttnn.multiply(mag_c, sin_p)
        ttnn.deallocate(cos_p)
        ttnn.deallocate(sin_p)
        ttnn.deallocate(mag_c)

        wav = self.istft(real, imag)
        ttnn.deallocate(real)
        ttnn.deallocate(imag)
        out = ttnn.clamp(wav, -self.audio_limit, self.audio_limit)
        ttnn.deallocate(wav)
        return out
