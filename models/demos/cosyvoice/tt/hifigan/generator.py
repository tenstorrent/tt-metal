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
from loguru import logger

import ttnn

from .istft import TtIStft
from .stft import TtStft


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

    def __init__(self, device, bag, dtype=ttnn.bfloat16, upsample_rates=(8, 8), resblock_dilations=(1, 3, 5)):
        """Built from a `WeightBag` -- the flat .npz `scripts/export_weights.py`
        emits -- so nothing here imports the CosyVoice package. See tt/weights.py
        for why that boundary is load-bearing."""
        from ..weights import build_conv1d, build_conv_transpose1d, build_resblock
        from .conv import prepare_weights_default

        self.device = device
        self.dtype = dtype
        meta = bag.meta
        self.n_fft = meta["istft_params"]["n_fft"]
        self.hop_len = meta["istft_params"]["hop_len"]
        self.lrelu_slope = meta["lrelu_slope"]
        self.audio_limit = meta["audio_limit"]
        self.num_kernels = meta["num_kernels"]
        self.num_upsamples = meta["num_upsamples"]
        self.bins = self.n_fft // 2 + 1
        self.upsample_rates = tuple(upsample_rates)

        self.conv_pre = build_conv1d(device, bag.sub("conv_pre"), padding=3, dtype=dtype)

        # ups[i]: ConvTranspose1d(k=16, stride=u, padding=(k-u)//2)
        self.ups = []
        for i, u in enumerate(self.upsample_rates):
            sub = bag.sub(f"ups.{i}")
            k = sub.tensor("weight").shape[-1]
            self.ups.append(build_conv_transpose1d(device, sub, stride=u, padding=(k - u) // 2, dtype=dtype))

        # source_downs[i]: k=1/s=1 when the cumulative rate is 1, else k=2u/s=u/pad=u//2
        self.source_downs = []
        for i in range(bag.count("source_downs")):
            sub = bag.sub(f"source_downs.{i}")
            k = sub.tensor("weight").shape[-1]
            stride = k // 2 if k > 1 else 1
            self.source_downs.append(
                build_conv1d(device, sub, stride=stride, padding=(stride // 2 if k > 1 else 0), dtype=dtype)
            )

        self.source_resblocks = [
            build_resblock(device, bag.sub(f"source_resblocks.{i}"), dilations=resblock_dilations, dtype=dtype)
            for i in range(bag.count("source_resblocks"))
        ]
        self.resblocks = [
            build_resblock(device, bag.sub(f"resblocks.{i}"), dilations=resblock_dilations, dtype=dtype)
            for i in range(bag.count("resblocks"))
        ]
        self.conv_post = build_conv1d(device, bag.sub("conv_post"), padding=3, dtype=dtype)

        # The source branch is optional so `decode()` alone stays usable when the
        # excitation is being injected from a golden.
        self.f0_predictor = self.m_source = None
        if bag.sub("f0_predictor").count("condnet"):
            from ..weights import build_f0_predictor

            # fp32, deliberately, and not because the convolutions are delicate.
            # f0 error *integrates* into phase: a 15 Hz error over 72192 samples at
            # 22.05 kHz is 49 whole cycles. Holding phase error under 0.1 cycle
            # needs dF0 < 0.03 Hz, i.e. ~13 mantissa bits at 200 Hz -- bfloat16 has
            # 8. PCC hides this completely, being scale-relative where the
            # consequence is absolute.
            self.f0_predictor = build_f0_predictor(device, bag.sub("f0_predictor"), dtype=ttnn.float32)
            src = bag.sub("m_source")
            if src.sub("l_linear").has("weight"):
                from .source import TtSourceModuleHnNSF

                lw = src.sub("l_linear").tensor("weight")  # [1, H+1]
                self.m_source = TtSourceModuleHnNSF(
                    device,
                    lw,
                    src.sub("l_linear").tensor("bias"),
                    sampling_rate=meta.get("sampling_rate", 22050),
                    # H+1 comes off the merge layer's input width, so nb_harmonics
                    # never has to be carried in the metadata.
                    harmonic_num=int(lw.shape[1]) - 1,
                    dtype=ttnn.float32,  # the whole excitation path stays wide
                )

        # Wormhole disagrees with itself about `prepare_conv_weights` -- see
        # `conv.prepare_weights_default`. Applied here, after everything is built, rather
        # than threaded through four builder signatures, and scoped to the vocoder because
        # the flow estimator shares `TtConv1d` and *is* traced.
        if not prepare_weights_default(device):
            n = self._verify_weight_preparation()
            logger.warning(
                f"Wormhole: conv weight preparation will be verified once per geometry for {n} "
                "vocoder convolutions (prepare_conv_weights disagrees with the op at some lengths)"
            )

        # The window travels in the export rather than being recomputed, so the
        # device cannot disagree with the reference about it.
        window = bag.window
        self.stft = TtStft(device, self.n_fft, self.hop_len, window=window, dtype=dtype)
        self.istft = TtIStft(device, self.n_fft, self.hop_len, window=window, dtype=dtype)

    def _verify_weight_preparation(self) -> int:
        """Arm the prepared-weight check on every `TtConv1d` this generator owns.

        A walk rather than a constructor argument: the convolutions are two and three
        levels down (ResBlocks hold six each, the f0 predictor holds four) and threading a
        flag through `build_conv1d`, `build_resblock`, `build_conv_transpose1d` and
        `build_f0_predictor` would put the same decision in four places. Walking the object
        graph once, here, keeps it in one.
        """
        from .conv import TtConv1d

        seen, count, stack = set(), 0, [self]
        while stack:
            obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if isinstance(obj, TtConv1d):
                obj._verify = True
                count += 1
            elif isinstance(obj, (list, tuple)):
                stack.extend(obj)
            elif hasattr(obj, "__dict__"):
                stack.extend(obj.__dict__.values())
        return count

    @classmethod
    def from_export(cls, device, path: str | None = None, **kw):
        from ..weights import WeightBag, default_weights_path

        return cls(device, WeightBag.load(path or default_weights_path()), **kw)

    # ----------------------------------------------------------------------
    def upsample_f0(self, f0, mel_frames: int, batch_size: int = 1):
        """`nn.Upsample(scale_factor=hop, mode='nearest')` on `[B, T_mel, 1]`.

        Nearest-neighbour upsampling by an integer factor is a **broadcast multiply
        followed by a reshape**: broadcasting `[B, T, 1]` against a row of `n` ones
        gives `[B, T, n]`, and reinterpreting that as `[B, T*n, 1]` lays each
        value's `n` copies down contiguously -- which is the definition.

        `ttnn.upsample` is the obvious candidate and does not apply: it requires a
        tile-aligned tiled input, and this tensor's channel dimension is 1. The
        broadcast form has no alignment constraint because the axis that has to be
        a multiple of 32 is the *repeat count*, which is 256.
        """
        total = self.upsample_rates[0] * self.upsample_rates[1] * self.hop_len
        ones = ttnn.ones((1, 1, total), dtype=f0.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        spread = ttnn.multiply(f0, ones)  # [B, T_mel, total]
        ttnn.deallocate(ones)
        out = ttnn.reshape(spread, (batch_size, mel_frames * total, 1))
        return out, mel_frames * total

    def inference(
        self,
        mel,
        mel_frames: int,
        phase_vec=None,
        sine_noise=None,
        sine_noise_unit=None,
        cache_source=None,
        batch_size: int = 1,
    ):
        """`HiFTGenerator.inference`: mel in, waveform out, source branch included.

        mel is `[B, T_mel, 80]` channels-last; returns `(waveform, n_samples, source)`.

        `phase_vec` and `sine_noise` are `SineGen`'s two draws. The reference makes
        both **unconditionally** -- there is no `if self.training` guard on either --
        so a vocoder that ignores them is not merely unseeded, it is a different
        function. Pass the captured arrays to reproduce a reference run; pass None
        to get the deterministic zero-phase, zero-noise excitation.

        `cache_source` is the previous chunk's excitation tail, spliced over the
        front of this one so streaming does not restart the oscillator's phase at
        every seam. The excitation is returned as well as the waveform because the
        caller needs its tail for the next chunk.
        """
        if self.f0_predictor is None or self.m_source is None:
            raise RuntimeError("this generator was built without the source branch; use decode()")
        f0 = self.f0_predictor(mel, mel_frames, batch_size)
        up, audio_len = self.upsample_f0(f0, mel_frames, batch_size)
        ttnn.deallocate(f0)
        s, _, _ = self.m_source(up, phase_vec=phase_vec, sine_noise=sine_noise, sine_noise_unit=sine_noise_unit)
        ttnn.deallocate(up)

        if cache_source is not None and cache_source.shape[1] > 0:
            # `s[:, :, :cache_len] = cache_source` upstream. The excitation is a
            # phase-continuous oscillator, and a streaming chunk restarts its phase
            # accumulator from zero -- so without splicing the previous chunk's tail
            # back in, every boundary is a phase discontinuity, which is audible as a
            # click. This is the single line that makes streaming sound like speech.
            n = cache_source.shape[1]
            tail = ttnn.slice(s, [0, n, 0], [batch_size, s.shape[1], s.shape[2]])
            # The cache was saved from the *cast* excitation the vocoder consumed,
            # while `s` here is still the wide fp32 one -- concat rejects mixed
            # dtypes, and the error names neither tensor.
            head = cache_source
            if head.dtype != s.dtype:
                head = ttnn.typecast(cache_source, s.dtype)
            spliced = ttnn.concat([head, tail], dim=1)
            if head is not cache_source:
                ttnn.deallocate(head)
            ttnn.deallocate(tail)
            ttnn.deallocate(s)
            s = spliced
        # SineGen accumulates its phase in fp32 on purpose -- the cumsum runs over
        # 72k samples and bfloat16 there is catastrophic (see tt/hifigan/source.py).
        # The rest of the vocoder is bfloat16, so the excitation is cast back at
        # this boundary; leaving it wide makes the first concat inside decode()
        # fail on mixed dtypes, several ops away from the cause.
        if s.dtype != self.dtype:
            cast = ttnn.typecast(s, self.dtype)
            ttnn.deallocate(s)
            s = cast
        wav = self.decode(mel, s, mel_frames, batch_size)
        # Normalise to `[B, T_audio, 1]`. `decode` inherits the iSTFT's rank-4 NHWC
        # from `conv_transpose2d`, which every caller then reshapes away -- and the
        # streaming path slices this on the time axis, so an unnormalised rank here
        # fails inside a helper several frames from the cause.
        if len(wav.shape) != 3:
            reshaped = ttnn.reshape(wav, (batch_size, audio_len, 1))
            ttnn.deallocate(wav)
            wav = reshaped
        # `s` is returned, not freed: streaming needs its tail as the next chunk's
        # cache_source. The caller owns it.
        return wav, audio_len, s

    def decode(self, mel, s, mel_frames: int, batch_size: int = 1):
        """mel: ttnn [B, T_mel, 80]; s: ttnn [B, T_audio, 1] -> [B, L, 1] waveform."""
        trace = shape_trace(mel_frames, n_fft=self.n_fft, hop_len=self.hop_len)

        s_stft, _ = self.stft(s, trace["audio_length"], batch_size)  # [B, 18, T]
        s_stft = ttnn.permute(s_stft, (0, 2, 1))  # [B, T, 18]

        x, _ = self.conv_pre(mel, mel_frames, batch_size)

        for st in trace["stages"]:
            act = ttnn.leaky_relu(x, self.lrelu_slope)
            ttnn.deallocate(x)
            x, _ = self.ups[st.index](act, st.in_length, batch_size)
            ttnn.deallocate(act)

            if st.index == self.num_upsamples - 1:
                # ReflectionPad1d((1, 0)): prepend x[:, 1] -- one sample, so the
                # "reflection" is a single slice, no exchange matmul needed.
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

            # Three ResBlocks read the SAME x and their outputs are averaged, so x
            # must outlive all three -- see the ownership note in resblock.py.
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

        act = ttnn.leaky_relu(x, 0.01)  # F.leaky_relu default slope, as the reference
        ttnn.deallocate(x)
        x, _ = self.conv_post(act, trace["conv_post_length"], batch_size)
        ttnn.deallocate(act)

        # [B, T, 18] -> magnitude/phase, then back to [B, 9, T] for the iSTFT.
        T = trace["conv_post_length"]
        spec = ttnn.permute(x, (0, 2, 1))
        ttnn.deallocate(x)
        mag_lin = ttnn.slice(spec, [0, 0, 0], [batch_size, self.bins, T])
        pha_lin = ttnn.slice(spec, [0, self.bins, 0], [batch_size, 2 * self.bins, T])
        ttnn.deallocate(spec)

        mag = ttnn.exp(mag_lin)
        ttnn.deallocate(mag_lin)
        # The reference clips magnitude at 1e2. exp() is non-negative, so a lower
        # bound of 0 makes this the same operation as torch.clip(max=1e2) without
        # depending on clamp accepting a None bound.
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
