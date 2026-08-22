# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Neural source filter excitation: SineGen + SourceModuleHnNSF.

CosyVoice-300M runs at 22050 Hz, which selects `SineGen` (type 1) rather than
`SineGen2` -- the implementation that integrates phase with a cumsum over the
**audio-rate** signal, 72 192 samples for 3.3 s of speech.

Precision, measured on silicon
------------------------------
This cumsum was flagged up front as a precision risk. It is a larger one than
expected, and in two independent ways -- both invisible to this
module's own short PCC tests and both found only end to end.

**1. `ttnn.cumsum` is far less accurate than torch's.** Against an fp64 reference
over the real 72192-sample f0, on Blackhole:

    device cumsum, fp32    max|d| 5.62e-01    (t=1k 2.3e-07, t=36k 0.114)
    torch  cumsum, fp32    max|d| 2.44e-04

Two thousand times worse. Since phase is `2*pi * (cumsum mod 1)`, an absolute
error of 0.56 is more than half a cycle: the harmonic bank is randomised by the
end of the utterance. At T=1024 and T=8192 the error is ~1e-5 and invisible.
`phase_mod1()` fixes it by reducing each block total mod 1 before accumulating,
which keeps every partial sum O(1) instead of O(T) -- measured 0.843 -> 0.99999745
on the captured f0, and it is also the *faster* path; see PERF.md. (A previous
version of this note claimed the blocked scan was unnecessary and would be
*worse*. That was measured against torch on the host, where it is true; on device
it is the difference between working and not.)

**2. f0 error integrates, so the excitation cannot be reproduced from scratch.**
Phase drift is `sum(delta_f0)/sr` over samples, so holding it under 0.1 cycle
across 72192 samples needs a mean f0 error below **0.03 Hz** -- 1.5e-4 relative at
200 Hz, about 13 mantissa bits. The device's f0 predictor lands at ~16 Hz max
error even with fp32 weights and activations, because Tensix HiFi4 is four
bfloat16 passes rather than true fp32.

That is not a defect to fix; it is a property of the model. **Sample-level
waveform comparison is only meaningful with the reference source injected**, which
is what the PCC gates do. For a self-computed source the honest metrics are the
energy envelope and the spectrum, and those hold up -- the audio is correct, its
phase is simply a different valid realisation, exactly like the RNG draws below.

One structural fact worth recording even though it is not used here: f0 is
upsampled by `nn.Upsample(scale_factor=256)`, whose default mode is *nearest*, so
the audio-rate f0 is piecewise-constant in blocks of 256 -- 282 distinct values
driving 72 192 samples. The scan is therefore 256x redundant. That is a
performance lever for Stage 2/3, not a correctness fix.

Randomness
----------
`SineGen` draws a per-harmonic phase offset from U(-pi, pi) and Gaussian noise
inside its forward pass, and `SourceModuleHnNSF` draws noise again. Seeding cannot
align a TTNN port with torch's RNG stream, so both are accepted as OPTIONAL
ARGUMENTS here: PCC tests pass in the values `gen_golden.py` captured, and
production runs let the module draw its own.
"""
from __future__ import annotations

import math

import torch

import ttnn


class TtSineGen:
    """f0 -> harmonic sine bank. Tensors are [B, T, H+1] (channels-last)."""

    def __init__(
        self,
        device,
        sampling_rate: int = 22050,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 10.0,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.sampling_rate = sampling_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.dtype = dtype

        # harmonic multipliers (i+1)/sr, as a [1, 1, H+1] row so f0 broadcasts.
        # The reference builds this with a Python loop over harmonics writing into
        # a preallocated F_mat; it is an outer product, so it is one multiply here.
        harm = torch.arange(1, harmonic_num + 2, dtype=torch.float32) / sampling_rate
        self.harmonics = ttnn.from_torch(
            harm.reshape(1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )

    BLOCK = 256  # T is always mel_frames * 256 here, so this divides exactly

    def phase_mod1(self, F):
        """`cumsum(F, dim=1) mod 1`, kept accurate over a full utterance.

        A plain `ttnn.cumsum` is not good enough here, and the reason is worth
        stating precisely because the naive version passes every short test.
        Measured on Blackhole against an fp64 reference over 72192 samples:

            device cumsum, fp32    max|d| 5.62e-01   (t=1k 2.3e-07, t=36k 0.114)
            torch  cumsum, fp32    max|d| 2.44e-04

        Two thousand times worse than torch's, and since the phase is
        `2*pi * (cumsum mod 1)`, an absolute error of 0.56 is **more than half a
        cycle** -- the harmonic bank is randomised by the end of the utterance. At
        T=1024 and T=8192 the error is ~1e-5 and invisible, which is exactly why
        this survived the module's own PCC tests and only surfaced end-to-end.

        The fix is arithmetic, not numerical: **only the value mod 1 is ever used**,
        so the accumulator never needs to reach 650. Summing within short blocks,
        reducing each block total mod 1, and accumulating those keeps every partial
        sum O(1) -- and `(a + b) mod 1 == ((a mod 1) + (b mod 1)) mod 1` makes it
        exact. Precision then stops depending on the utterance length at all.

        Do not "simplify" this back to one `ttnn.cumsum`: it is slower as well as
        wrong. The op parallelises only over the axes it is not scanning, so a single
        long scan gets one core; blocking is what gives it rows to spread. See PERF.md.
        """
        b, t, c = F.shape
        blk = self.BLOCK
        if t % blk or t <= blk:
            phase = ttnn.cumsum(F, dim=1, dtype=ttnn.float32)
            out = ttnn.subtract(phase, ttnn.floor(phase))
            ttnn.deallocate(phase)
            return out
        nb = t // blk

        # `x` is a VIEW of F, not a copy: [b, t, c] and [b, nb, blk, c] tile to the same
        # tiles in the same order, so ttnn.reshape returns the same buffer. Deallocating
        # it would free F out from under the caller. Same for `out` at the end of this
        # function -- do not "tidy up" by adding deallocates for either.
        x = ttnn.reshape(F, (b, nb, blk, c))
        within = ttnn.cumsum(x, dim=2, dtype=ttnn.float32)  # [b, nb, blk, c]

        totals = ttnn.slice(within, [0, 0, blk - 1, 0], [b, nb, blk, c])  # [b, nb, 1, c]
        # mod 1 BEFORE accumulating across blocks -- this is the whole trick
        tmod = ttnn.subtract(totals, ttnn.floor(totals))
        ttnn.deallocate(totals)
        incl = ttnn.cumsum(tmod, dim=1, dtype=ttnn.float32)
        offset = ttnn.subtract(incl, tmod)  # exclusive scan: block i excludes itself
        ttnn.deallocate(incl)
        ttnn.deallocate(tmod)
        omod = ttnn.subtract(offset, ttnn.floor(offset))
        ttnn.deallocate(offset)

        phase = ttnn.add(within, omod)  # [b, nb, 1, c] broadcasts over the block
        ttnn.deallocate(within)
        ttnn.deallocate(omod)
        frac = ttnn.subtract(phase, ttnn.floor(phase))
        ttnn.deallocate(phase)
        out = ttnn.reshape(frac, (b, t, c))  # a view of `frac`; freeing `frac` frees `out`
        return out

    def __call__(self, f0, phase_vec=None, noise=None, noise_unit=None):
        """f0: ttnn [B, T, 1] -> (sine_waves [B, T, H+1], uv [B, T, 1], noise).

        phase_vec / noise are the captured RNG draws; pass them in PCC tests.

        **f0 arrives in fp32 or the output is noise.** Not because the arithmetic
        here is delicate, but because f0 error integrates: bfloat16 at 200 Hz
        quantises to 0.78 Hz, and 0.78 Hz over 72192 samples is 2.5 whole cycles of
        phase. The input is widened rather than trusted.
        """
        if f0.dtype != ttnn.float32:
            f0 = ttnn.typecast(f0, ttnn.float32)
        # F[b, t, i] = f0[b, t, 0] * (i+1) / sr
        F = ttnn.multiply(f0, self.harmonics)

        frac = self.phase_mod1(F)
        ttnn.deallocate(F)
        theta = ttnn.multiply(frac, 2.0 * math.pi)
        ttnn.deallocate(frac)

        if phase_vec is not None:
            theta = ttnn.add(theta, phase_vec)
        sine_waves = ttnn.multiply(ttnn.sin(theta), self.sine_amp)
        ttnn.deallocate(theta)

        uv = ttnn.gt(f0, self.voiced_threshold)
        uv = ttnn.typecast(uv, self.dtype)

        if noise is not None:
            # a captured draw, already scaled by the reference's noise_amp
            n = noise
        elif noise_unit is not None:
            # a standard normal from the host, scaled here. This is the production
            # path: the amplitude depends on `uv`, which only exists on device, so
            # the caller supplies unit noise and the scaling happens where the
            # voicing decision lives.
            noise_amp = ttnn.add(ttnn.multiply(uv, self.noise_std - self.sine_amp / 3.0), self.sine_amp / 3.0)
            n = ttnn.multiply(noise_unit, noise_amp)
            ttnn.deallocate(noise_amp)
        else:
            # Deterministic zero. NOT a neutral default: for unvoiced frames `uv`
            # zeroes the sine bank, so the noise is the *entire* excitation there --
            # every fricative and plosive goes silent. Fine for a PCC test against a
            # captured golden, wrong for synthesis.
            noise_amp = ttnn.add(ttnn.multiply(uv, self.noise_std - self.sine_amp / 3.0), self.sine_amp / 3.0)
            n = ttnn.multiply(noise_amp, 0.0)
            ttnn.deallocate(noise_amp)

        out = ttnn.add(ttnn.multiply(sine_waves, uv), n)
        ttnn.deallocate(sine_waves)
        return out, uv, n

    @staticmethod
    def torch_reference(
        f0: torch.Tensor,
        sampling_rate=22050,
        harmonic_num=8,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=10.0,
        phase_vec=None,
        noise=None,
    ):
        """cosyvoice.hifigan.generator.SineGen.forward, with the RNG lifted out.

        Input f0 is [B, T, 1] and the reference immediately transposes to [B, 1, T];
        this stays in [B, T, H+1] to match the device layout, which changes the
        cumsum axis but nothing else.
        """
        harm = torch.arange(1, harmonic_num + 2, dtype=torch.float32) / sampling_rate
        F = f0 * harm.reshape(1, 1, -1)  # [B, T, H+1]
        theta = 2 * math.pi * (torch.cumsum(F.float(), dim=1) % 1)
        if phase_vec is not None:
            theta = theta + phase_vec
        sine_waves = sine_amp * torch.sin(theta)
        uv = (f0 > voiced_threshold).float()
        if noise is None:
            noise = torch.zeros_like(sine_waves)
        return sine_waves * uv + noise, uv, noise


class TtSourceModuleHnNSF:
    """Merges the harmonic bank into a single excitation: tanh(Linear(sines))."""

    def __init__(
        self,
        device,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor,
        sampling_rate: int = 22050,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 10.0,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.sine_amp = sine_amp
        self.sine_gen = TtSineGen(
            device, sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold, dtype=dtype
        )
        # l_linear: (H+1) -> 1. Stored transposed for ttnn.linear.
        self.weight = ttnn.from_torch(
            linear_weight.detach().float().t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.bias = ttnn.from_torch(
            linear_bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    @classmethod
    def from_module(cls, device, module: torch.nn.Module, **kw):
        sg = module.l_sin_gen
        return cls(
            device,
            module.l_linear.weight,
            module.l_linear.bias,
            sampling_rate=sg.sampling_rate,
            harmonic_num=sg.harmonic_num,
            sine_amp=sg.sine_amp,
            add_noise_std=sg.noise_std,
            voiced_threshold=sg.voiced_threshold,
            **kw,
        )

    def __call__(self, f0, phase_vec=None, sine_noise=None, branch_noise=None, sine_noise_unit=None):
        """f0: ttnn [B, T, 1] -> (sine_merge [B, T, 1], noise, uv)."""
        sine_waves, uv, _ = self.sine_gen(f0, phase_vec=phase_vec, noise=sine_noise, noise_unit=sine_noise_unit)
        merged = ttnn.linear(sine_waves, self.weight, bias=self.bias)
        ttnn.deallocate(sine_waves)
        sine_merge = ttnn.tanh(merged)
        ttnn.deallocate(merged)
        noise = branch_noise if branch_noise is not None else ttnn.multiply(uv, 0.0)
        return sine_merge, noise, uv

    @staticmethod
    def torch_reference(f0, linear_weight, linear_bias, phase_vec=None, sine_noise=None, **kw):
        sine_waves, uv, _ = TtSineGen.torch_reference(f0, phase_vec=phase_vec, noise=sine_noise, **kw)
        merged = torch.nn.functional.linear(sine_waves, linear_weight, linear_bias)
        return torch.tanh(merged), uv
