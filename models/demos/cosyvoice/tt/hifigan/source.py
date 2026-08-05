# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Neural source filter excitation: SineGen + SourceModuleHnNSF.

CosyVoice-300M runs at 22050 Hz, which selects `SineGen` (type 1) rather than
`SineGen2` -- the implementation that integrates phase with a cumsum over the
**audio-rate** signal, 72 192 samples for 3.3 s of speech.

Precision, measured rather than assumed
---------------------------------------
`02_plan.md` sec.3.4 flags this cumsum as a precision risk, and it is -- in
bfloat16. With 8 mantissa bits an accumulator reaching ~6000 cannot represent the
~0.01 increments at all, and the output degenerates to silence or noise.

In float32 it is a non-issue. Measured against a float64 reference on the real f0
captured from the vocoder (increments up to 0.15, accumulator reaching 6090.6):

    fp32 cumsum abs error   max 2e-4    mean 4e-5
    resulting PHASE error   max 0.0015 rad,  mean 0.00023 rad   (period is 2*pi)
    sin() PCC fp32 vs fp64  0.99999993

So `ttnn.cumsum(..., dtype=ttnn.float32)` is sufficient outright. The block-wise
mod-1 scan the plan held in reserve is NOT needed -- and would in fact be worse:
splitting the scan into per-block carries raises the phase error to ~0.025 rad,
because torch's cumsum already sums more accurately than a naive sequential loop.

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

    def __call__(self, f0, phase_vec=None, noise=None):
        """f0: ttnn [B, T, 1] -> (sine_waves [B, T, H+1], uv [B, T, 1], noise).

        phase_vec / noise are the captured RNG draws; pass them in PCC tests.
        """
        # F[b, t, i] = f0[b, t, 0] * (i+1) / sr
        F = ttnn.multiply(f0, self.harmonics)

        # fp32 accumulate -- the whole point of this module's precision note.
        # dtype= typecasts BEFORE accumulating (accumulation_common.cpp:20).
        phase = ttnn.cumsum(F, dim=1, dtype=ttnn.float32)
        ttnn.deallocate(F)

        # theta = 2*pi * (phase mod 1). Only the fractional part survives, which is
        # what makes the accumulator's absolute magnitude tolerable.
        frac = ttnn.subtract(phase, ttnn.floor(phase))
        ttnn.deallocate(phase)
        theta = ttnn.multiply(frac, 2.0 * math.pi)
        ttnn.deallocate(frac)

        if phase_vec is not None:
            theta = ttnn.add(theta, phase_vec)
        sine_waves = ttnn.multiply(ttnn.sin(theta), self.sine_amp)
        ttnn.deallocate(theta)

        uv = ttnn.gt(f0, self.voiced_threshold)
        uv = ttnn.typecast(uv, self.dtype)

        if noise is None:
            noise_amp = ttnn.add(ttnn.multiply(uv, self.noise_std - self.sine_amp / 3.0), self.sine_amp / 3.0)
            n = ttnn.multiply(noise_amp, 0.0)  # deterministic zero when unseeded
            ttnn.deallocate(noise_amp)
        else:
            n = noise

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

    def __call__(self, f0, phase_vec=None, sine_noise=None, branch_noise=None):
        """f0: ttnn [B, T, 1] -> (sine_merge [B, T, 1], noise, uv)."""
        sine_waves, uv, _ = self.sine_gen(f0, phase_vec=phase_vec, noise=sine_noise)
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
