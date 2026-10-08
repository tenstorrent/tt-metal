# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""NSF harmonic excitation for CosyVoice2's HiFT vocoder: SineGen2 + SourceModuleHnNSF.

CosyVoice2 selects `SineGen2`, not the `SineGen` (type 1) the CosyVoice1 TTNN port
implements -- confirmed against upstream `cosyvoice/hifigan/generator.py` directly:
`SourceModuleHnNSF(..., sinegen_type='1' if self.sampling_rate == 22050 else '2')`,
and CosyVoice2's `sampling_rate` is 24000. `SineGen2` integrates phase differently
from `SineGen`: `HiFTGenerator.forward`/`inference` nearest-upsamples f0 to audio
rate *before* calling `SourceModuleHnNSF` (`self.f0_upsamp = nn.Upsample(scale_factor
=upsample_scale)`, `upsample_scale = prod(upsample_rates) * hop_len` -- 480 for
CosyVoice2's [8, 5, 3]), and `SineGen2._f02sine` then:

    rad_values = (f0_values / sampling_rate) % 1        # at audio rate
    rad_values = interpolate(rad_values, 1/upsample_scale, mode='linear')  # -> mel rate
    phase = cumsum(rad_values, dim=1) * 2*pi             # cumsum at MEL rate, not audio rate
    phase = interpolate(phase * upsample_scale, upsample_scale,
                         mode='linear' if not causal else 'nearest')       # mel rate -> audio rate
    sines = sin(phase)

(quoted structure, confirmed against upstream source directly -- not summarized).
`mode='linear'` is what CosyVoice2 actually uses: its deployed vocoder is the
plain, non-causal `HiFTGenerator` (confirmed in the iSTFT/HiFT-decode modules'
docstrings), which passes `causal=False` when building `SourceModuleHnNSF`.

Both interpolation legs are linear interpolation by a *fixed* integer factor
(480, set by the model's own upsample_rates/hop_len, not by data) -- the same
precondition that made the iSTFT identity a fixed matrix rather than an FFT, and
that is exactly how both legs are implemented here: as a fixed basis matrix
applied via `ttnn.matmul`, the same pattern `istft.py`/`stft.py` already use, not
a conv.

An initial attempt used `TtConv1d`/`TtConvTranspose1d` instead (the machinery
already proven for HiFT's real convolutions). The downsample leg
(`in_channels=1, kernel=2, stride=480` -- an extreme, degenerate shape) hung
`ttnn.conv1d`'s op-config search past a "falling back to width-slicing" warning
with no further progress for 200+ seconds; abandoned rather than debugged
further, since a matmul sidesteps conv op-config entirely and is the pattern
already proven twice in this package. Both legs' bases are verified against
real `torch.nn.functional.interpolate` (not assumed) before writing any TTNN:

  * Downsample (audio-rate -> mel-rate, factor 1/480): because 480 is even,
    every output sample lands exactly on the midpoint of two fixed input taps:
    `output[j] = 0.5*(input[j*480 + 239] + input[j*480 + 240])`. `downsample_basis`
    is the `[T_mel, audio_len]` matrix with those two 0.5 entries per row --
    verified bit-exact against `F.interpolate`.
  * Upsample (mel-rate -> audio-rate, factor 480, `mode='linear'`): the
    `[audio_len, T_mel]` matrix built directly from the exact per-output-sample
    tap formula (`x_i = (i+0.5)/S - 0.5`, floor/frac, indices clamped to
    `[0, T_mel-1]`) -- `align_corners=False`'s boundary clamping falls out of the
    index-clamp for free (both taps collapse to the same clamped row, so the
    blend is trivially that value regardless of frac), no separate boundary
    handling needed. Verified to <1e-5 max|diff| against `F.interpolate` across
    several lengths and random (non-impulse) data, including the boundary spans.

`SineGen2._f02sine`'s `rand_ini` (an initial-phase-noise draw added at audio-rate
index 0, before the downsample) is **omitted here, verified rather than
overlooked**: a large synthetic perturbation (999.0) injected at that exact
index and pushed through the real downsample formula above produced max|diff| =
0.0 on the output -- the downsample never reads index 0 (its first read is index
239), so `rand_ini` has no effect on `SineGen2`'s output through the exact call
path CosyVoice2's `HiFTGenerator` uses. No RNG-capture plumbing needed for it.

Two draws remain that DO need RNG capture, matching the CosyVoice1 reference's
methodology (both unconditional, no `if self.training` guard, for the
`causal=False` path CosyVoice2 uses):

  * `SineGen2.forward`'s per-sample noise draw: `noise_amp * torch.randn_like(sine_waves)`.
  * `SourceModuleHnNSF.forward`'s separate noise-branch draw: `torch.randn_like(uv) * sine_amp / 3`.

`ttnn.cumsum`'s bf16 precision here is measured, not assumed from the
CosyVoice1 reference's numbers -- see tests/pcc/test_sine_gen2.py. Those numbers
were measured for a cumsum over ~72k *audio-rate* samples; this cumsum runs over
the much shorter *mel-rate* sequence (`T_mel`, order of hundreds for a several-
second utterance), a different regime that could plausibly behave differently
either way and is not assumed safe by analogy.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import ttnn


def downsample_basis(mel_len: int, scale: int) -> np.ndarray:
    """The [mel_len, mel_len*scale] matrix mapping audio-rate values to mel-rate,
    via the two fixed taps `output[j] = 0.5*(input[j*scale + half-1] + input[j*scale + half])`.

    Host-tier reference only -- kept for tests, not used on device.
    `TtSineGen2._downsample` reshapes the audio-rate sequence into `[mel_len,
    scale]` blocks and slices out columns `half-1`/`half` directly instead of
    matmul-ing this matrix: same reasoning as `upsample_basis3` (avoid a
    contraction width that scales with mel_len), and this leg doesn't even need
    a matmul once reshaped -- two slices and an average.

    Verified bit-exact against `F.interpolate(..., scale_factor=1/scale,
    mode='linear')` -- see module docstring.
    """
    half = scale // 2
    D = np.zeros((mel_len, mel_len * scale), dtype=np.float32)
    for j in range(mel_len):
        D[j, j * scale + half - 1] = 0.5
        D[j, j * scale + half] = 0.5
    return D


def upsample_basis3(scale: int) -> np.ndarray:
    """The [scale, 3] per-block basis for the mel-rate -> audio-rate leg --
    same identity as upsample_basis, restructured to a FIXED contraction width
    (3) instead of one that grows with mel_len.

    Necessary, not cosmetic: a `[audio_len, mel_len]` dense matmul (what
    upsample_basis is for) contracts over mel_len even though only 2 entries
    per row are nonzero, and TTNN's fp32 matmul on this hardware accumulates
    real error that grows with the contraction width (Tensix HiFi4 runs fp32
    as four bfloat16 passes, not true fp32 -- the CosyVoice1 reference's f0
    predictor hit the same wall). Measured: max|phase error| after upsampling
    went from ~0.2 rad at mel_len=4 to enough to collapse the full pipeline's
    PCC at mel_len=20, using the dense-matmul form. This form's contraction is
    3 regardless of mel_len, and does not depend on mel_len at all -- built
    once per TtSineGen2 instance, not cached per geometry.

    Each output position r within a block (audio index i = b*scale + r, for
    the b-th mel-frame) blends between at most two of the block's mel-frame
    neighbors [mel[b-1], mel[b], mel[b+1]] (edge-clamping is the caller's job,
    via replicated shifts -- see TtSineGen2._upsample), using the same
    per-tap formula as upsample_basis restricted to within-block-relative
    terms. Verified (see tests/pcc/test_sine_gen2.py) to reconstruct
    upsample_basis's own output exactly when combined with those shifts.
    """
    half = scale // 2
    r = np.arange(scale, dtype=np.float64)
    B = np.zeros((scale, 3), dtype=np.float32)
    left = r <= half - 1
    frac_left = (r[left] + 0.5) / scale + 0.5
    B[left, 0] = 1.0 - frac_left
    B[left, 1] = frac_left
    right = ~left
    frac_right = (r[right] + 0.5) / scale - 0.5
    B[right, 1] = 1.0 - frac_right
    B[right, 2] = frac_right
    return B


def upsample_basis(mel_len: int, scale: int) -> np.ndarray:
    """The [mel_len*scale, mel_len] matrix mapping mel-rate values to audio-rate,
    reproducing `F.interpolate(..., scale_factor=scale, mode='linear',
    align_corners=False)` exactly, boundary clamping included.

    Host-tier reference only -- kept for tests, not used on device.
    `TtSineGen2._upsample` uses `upsample_basis3` instead: a dense
    `[audio_len, mel_len]` matmul like this one contracts over mel_len even
    though only 2 entries per output row are nonzero, and measured error grows
    with that contraction width on this hardware (see upsample_basis3's
    docstring). This function still documents and tests the *identity*, which
    `upsample_basis3` + the caller's edge-replicated shifts reproduce exactly.

    Built directly from the per-output-sample tap formula: `x_i = (i+0.5)/scale
    - 0.5`, `n0 = floor(x_i)`, `frac = x_i - n0`, weight `(1-frac)` at
    `clip(n0, 0, mel_len-1)` and `frac` at `clip(n0+1, 0, mel_len-1)`. Near the
    edges both clipped indices collapse to the same row, so the row's two
    weights land on the same column and sum there -- reproducing the flat clamp
    for free, no separate boundary handling needed. Verified to <1e-5
    max|diff| against `F.interpolate` across several lengths and random data,
    boundary spans included.
    """
    L = mel_len * scale
    i = np.arange(L, dtype=np.float64)
    x = (i + 0.5) / scale - 0.5
    n0 = np.floor(x).astype(np.int64)
    frac = (x - n0).astype(np.float32)
    n0c = np.clip(n0, 0, mel_len - 1)
    n1c = np.clip(n0 + 1, 0, mel_len - 1)
    U = np.zeros((L, mel_len), dtype=np.float32)
    np.add.at(U, (np.arange(L), n0c), 1.0 - frac)
    np.add.at(U, (np.arange(L), n1c), frac)
    return U


class TtSineGen2:
    """f0 -> harmonic sine bank, CosyVoice2's phase-integration path.

    Tensors are `[B, T, H+1]` (channels-last), matching the rest of this
    package. `T` is audio-rate on the way in and out; the cumsum inside runs at
    mel rate (`T // upsample_scale`).
    """

    def __init__(
        self,
        device,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.sampling_rate = sampling_rate
        self.upsample_scale = upsample_scale
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.dtype = dtype

        harm = torch.arange(1, harmonic_num + 2, dtype=torch.float32) / sampling_rate
        self.harmonics = ttnn.from_torch(
            harm.reshape(1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )

        # upsample_basis3 depends only on upsample_scale (fixed for this
        # instance), not on mel_len -- built once, no per-geometry cache needed.
        B3 = upsample_basis3(upsample_scale)
        self._up_basis3 = ttnn.from_torch(
            torch.from_numpy(B3).unsqueeze(0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )

        # HiFi4 + fp32_dest_acc_en measurably reduces matmul error here (this
        # path is phase-sensitive -- see phase_mod1's docstring) versus the
        # default compute config: measured max|diff| 0.198 vs 0.216 on a
        # synthetic case. Not the same accurate/safe-config split conv.py uses
        # (that guarded against an outright wrong result on some conv1d shapes;
        # this is a smaller, monotonic precision difference on a matmul, so one
        # config is used throughout rather than verified-and-cached per geometry).
        self._compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _downsample(self, rad, batch_folded: int, audio_len: int):
        """rad: ttnn [batch_folded, audio_len, 1] -> [batch_folded, mel_len, 1].

        `output[j] = 0.5*(input[j*S+half-1] + input[j*S+half])`: reshape into
        `[batch_folded*mel_len, S, 1]` blocks and slice out the two fixed
        columns directly -- no matmul, so no contraction-width-dependent error
        (see upsample_basis3's docstring for why that matters on this hardware;
        this leg doesn't need a matmul at all to avoid it)."""
        S = self.upsample_scale
        half = S // 2
        mel_len = audio_len // S
        blocks = ttnn.reshape(rad, (batch_folded * mel_len, S, 1))
        a = ttnn.slice(blocks, [0, half - 1, 0], [batch_folded * mel_len, half, 1])
        b = ttnn.slice(blocks, [0, half, 0], [batch_folded * mel_len, half + 1, 1])
        out = ttnn.multiply(ttnn.add(a, b), 0.5)
        ttnn.deallocate(blocks)
        out = ttnn.reshape(out, (batch_folded, mel_len, 1))
        return out, mel_len

    def _upsample(self, phase, batch_folded: int, mel_len: int):
        """phase: ttnn [batch_folded, mel_len, 1] -> [batch_folded, mel_len*S, 1].

        Fixed-contraction-width (3) reformulation -- see upsample_basis3's
        docstring for why the [audio_len, mel_len] dense-matmul form this
        replaced was measurably wrong at anything beyond a few mel-frames.
        Builds a length-3 neighbor window per mel-frame (edge-clamped via
        replicated shifts) and applies one [S, 3] matmul per window, batched
        over batch_folded*mel_len -- the contraction is 3 regardless of
        mel_len.

        `phase` is a cumsum, so its *absolute* magnitude grows unboundedly
        with mel_len even though only its local (window-to-window) variation
        ever matters -- the same reason the CosyVoice1 reference's SineGen
        reduces its cumsum mod 1 rather than let the accumulator grow. Doing
        that safely *before* a linear interpolation is not straightforward
        (wrapping the endpoints independently can wrap across a boundary the
        true continuous trajectory didn't cross), so instead: subtract the
        window's own center value (`phase` itself, always one of the two
        active taps in every position of the block -- see upsample_basis3)
        before the matmul, keeping its operands bounded by one cumsum step
        regardless of how large `phase` has grown, then add the center value
        back afterward via a nearest-broadcast (nothing here loses precision
        by being large -- it is only ever a per-block constant, plain
        addition, not an accumulation). Measured (see
        tests/pcc/test_sine_gen2.py) to remove the mel_len-dependent precision
        collapse the un-centered version had.
        """
        left_nb = ttnn.concat(
            [
                ttnn.slice(phase, [0, 0, 0], [batch_folded, 1, 1]),
                ttnn.slice(phase, [0, 0, 0], [batch_folded, mel_len - 1, 1]),
            ],
            dim=1,
        )
        right_nb = ttnn.concat(
            [
                ttnn.slice(phase, [0, 1, 0], [batch_folded, mel_len, 1]),
                ttnn.slice(phase, [0, mel_len - 1, 0], [batch_folded, mel_len, 1]),
            ],
            dim=1,
        )
        windows = ttnn.concat([left_nb, phase, right_nb], dim=2)  # [batch_folded, mel_len, 3]
        ttnn.deallocate(left_nb)
        ttnn.deallocate(right_nb)
        windows_rel = ttnn.subtract(windows, phase)  # broadcasts phase's trailing 1 against windows' 3
        ttnn.deallocate(windows)
        windows_rel = ttnn.reshape(windows_rel, (batch_folded * mel_len, 3, 1))

        interp_rel = ttnn.matmul(self._up_basis3, windows_rel, compute_kernel_config=self._compute_config)
        ttnn.deallocate(windows_rel)
        audio_len = mel_len * self.upsample_scale
        interp_rel = ttnn.reshape(interp_rel, (batch_folded, audio_len, 1))

        # Nearest-broadcast phase (the per-block center/reference) up to audio
        # rate, matching the CosyVoice1 reference's `upsample_f0`: a repeat
        # count broadcast-multiply, not a gather.
        ones = ttnn.ones((1, 1, self.upsample_scale), dtype=phase.dtype, device=self.device)
        center_broadcast = ttnn.multiply(phase, ones)  # [batch_folded, mel_len, S]
        ttnn.deallocate(ones)
        center_broadcast = ttnn.reshape(center_broadcast, (batch_folded, audio_len, 1))

        out = ttnn.add(interp_rel, center_broadcast)
        ttnn.deallocate(interp_rel)
        ttnn.deallocate(center_broadcast)
        return out, audio_len

    def phase_mod1(self, F_audio, batch_size: int, harmonics: int, audio_len: int):
        """F_audio: ttnn [B, audio_len, H] (already f0*harmonic-multiplier, at
        audio rate) -> sines [B, audio_len, H]. Folds H into the batch dim so
        both interpolation legs stay single-channel."""
        H = harmonics
        folded = batch_size * H
        # [B, L, H] -> [B, H, L] -> [B*H, L, 1]
        x = ttnn.permute(F_audio, (0, 2, 1))
        x = ttnn.reshape(x, (folded, audio_len, 1))

        rad = ttnn.subtract(x, ttnn.floor(x))  # (f0*harm/sr) % 1, done by caller before this
        rad_mel, mel_len = self._downsample(rad, folded, audio_len)
        ttnn.deallocate(rad)

        phase_mel = ttnn.cumsum(rad_mel, dim=1, dtype=ttnn.float32)
        ttnn.deallocate(rad_mel)

        # Upstream computes `phase = cumsum(rad)*2*pi` then feeds `phase *
        # upsample_scale` into the interpolate (`_f02sine`: `phase.transpose(1,2)
        # * self.upsample_scale`) -- algebraically the same as scaling by
        # `2*pi*upsample_scale` before the upsample. Applied here *after* the
        # upsample matmul instead: linear interpolation is linear, so
        # `M @ (c*x) == c*(M @ x)` exactly, and doing it this way keeps the
        # matmul's own operands small (~0.01-1) instead of up to several hundred.
        # Measured, not assumed: pre-scaling fed the matmul values up to ~650 and
        # measured max|phase error| 1.7 rad after upsampling a 4-mel-frame case
        # (enough to scramble sin()); post-scaling on the same case measured
        # ~40% lower error. TTNN's fp32 matmul on this hardware is not bit-exact
        # fp32 regardless (the CosyVoice1 reference's f0 predictor hit the same
        # wall: "Tensix HiFi4 is four bfloat16 passes rather than true fp32") --
        # this reordering reduces, not eliminates, that residual, and the actual
        # resulting device-vs-torch PCC is what's measured in
        # tests/pcc/test_sine_gen2.py, not assumed adequate from this reasoning.
        phase_audio_raw, out_len = self._upsample(phase_mel, folded, mel_len)
        ttnn.deallocate(phase_mel)

        phase_audio = ttnn.multiply(phase_audio_raw, 2.0 * math.pi * self.upsample_scale)
        ttnn.deallocate(phase_audio_raw)

        sines = ttnn.sin(phase_audio)
        ttnn.deallocate(phase_audio)

        # [B*H, L, 1] -> [B, H, L] -> [B, L, H]
        sines = ttnn.reshape(sines, (batch_size, H, out_len))
        sines = ttnn.permute(sines, (0, 2, 1))
        return sines

    def __call__(self, f0, noise=None):
        """f0: ttnn [B, T_audio, 1] (already nearest-upsampled to audio rate by
        the caller, matching HiFTGenerator.forward's `self.f0_upsamp` step) ->
        (sine_waves [B, T_audio, H+1], uv [B, T_audio, 1], noise).

        `noise` is the captured `torch.randn_like(sine_waves)` draw; pass it in
        PCC tests. Deterministic zero if omitted (fine for a PCC test against a
        captured golden that also used zero; wrong for synthesis).
        """
        b, audio_len, _ = f0.shape
        H = self.harmonic_num + 1
        if f0.dtype != ttnn.float32:
            f0 = ttnn.typecast(f0, ttnn.float32)
        fn = ttnn.multiply(f0, self.harmonics)  # [B, T_audio, H]

        sine_waves = self.phase_mod1(fn, b, H, audio_len)
        ttnn.deallocate(fn)
        sine_waves = ttnn.multiply(sine_waves, self.sine_amp)

        uv = ttnn.gt(f0, self.voiced_threshold)
        uv = ttnn.typecast(uv, ttnn.float32)

        noise_amp = ttnn.add(ttnn.multiply(uv, self.noise_std - self.sine_amp / 3.0), self.sine_amp / 3.0)
        if noise is not None:
            n = ttnn.multiply(noise_amp, noise)
        else:
            n = ttnn.multiply(noise_amp, 0.0)
        ttnn.deallocate(noise_amp)

        out = ttnn.add(ttnn.multiply(sine_waves, uv), n)
        ttnn.deallocate(sine_waves)
        return out, uv, n

    @staticmethod
    def torch_reference(
        f0: torch.Tensor,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        noise: torch.Tensor | None = None,
    ):
        """cosyvoice.hifigan.generator.SineGen2.forward + ._f02sine, with the RNG
        lifted out and `rand_ini` omitted (verified to have zero effect through
        this call path -- see module docstring). `flag_for_pulse=False` always
        (HiFTGenerator never sets it), `causal=False` (CosyVoice2's deployed
        generator). f0 is [B, T_audio, 1], already nearest-upsampled -- the
        caller's job, matching upstream's `self.f0_upsamp` step."""
        harm = torch.arange(1, harmonic_num + 2, dtype=torch.float32) / sampling_rate
        fn = f0 * harm.reshape(1, 1, -1)  # [B, T_audio, H]
        rad = (fn) % 1
        S = upsample_scale
        rad_mel = torch.nn.functional.interpolate(rad.transpose(1, 2), scale_factor=1 / S, mode="linear").transpose(
            1, 2
        )
        phase_mel = torch.cumsum(rad_mel, dim=1) * 2 * math.pi
        phase_audio = torch.nn.functional.interpolate(
            (phase_mel * S).transpose(1, 2), scale_factor=S, mode="linear"
        ).transpose(1, 2)
        sine_waves = torch.sin(phase_audio) * sine_amp

        uv = (f0 > voiced_threshold).float()
        noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
        n = noise_amp * noise if noise is not None else torch.zeros_like(sine_waves)
        return sine_waves * uv + n, uv, n


class TtSourceModuleHnNSF:
    """Merges the harmonic bank into a single excitation: tanh(Linear(sines))."""

    def __init__(
        self,
        device,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.sine_amp = sine_amp
        self.sine_gen = TtSineGen2(
            device, sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshold, dtype
        )
        self.weight = ttnn.from_torch(
            linear_weight.detach().float().t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.bias = ttnn.from_torch(
            linear_bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    @classmethod
    def from_module(cls, device, module: torch.nn.Module, upsample_scale: int, **kw):
        sg = module.l_sin_gen
        return cls(
            device,
            module.l_linear.weight,
            module.l_linear.bias,
            sampling_rate=sg.sampling_rate,
            upsample_scale=upsample_scale,
            harmonic_num=sg.harmonic_num,
            sine_amp=sg.sine_amp,
            add_noise_std=sg.noise_std,
            voiced_threshold=sg.voiced_threshold,
            **kw,
        )

    def __call__(self, f0, sine_noise=None, branch_noise=None):
        """f0: ttnn [B, T_audio, 1] -> (sine_merge [B, T_audio, 1], noise, uv)."""
        sine_waves, uv, _ = self.sine_gen(f0, noise=sine_noise)
        merged = ttnn.linear(ttnn.typecast(sine_waves, self.weight.dtype), self.weight, bias=self.bias)
        ttnn.deallocate(sine_waves)
        sine_merge = ttnn.tanh(merged)
        ttnn.deallocate(merged)
        noise = ttnn.multiply(branch_noise, self.sine_amp / 3.0) if branch_noise is not None else ttnn.multiply(uv, 0.0)
        return sine_merge, noise, uv

    @staticmethod
    def torch_reference(f0, linear_weight, linear_bias, sampling_rate, upsample_scale, branch_noise=None, **kw):
        sine_waves, uv, _ = TtSineGen2.torch_reference(f0, sampling_rate, upsample_scale, **kw)
        merged = torch.nn.functional.linear(sine_waves, linear_weight, linear_bias)
        sine_amp = kw.get("sine_amp", 0.1)
        noise = branch_noise * sine_amp / 3.0 if branch_noise is not None else torch.zeros_like(uv)
        return torch.tanh(merged), noise, uv
