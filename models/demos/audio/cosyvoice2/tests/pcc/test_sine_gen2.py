# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""SineGen2 / SourceModuleHnNSF: CosyVoice2's NSF harmonic excitation.

Same two-tier split as the rest of this package: host tests check the
interpolation-as-matmul identity against real `torch.nn.functional.interpolate`
(not against this module's own `torch_reference`, which shares the derivation
and could hide a shared bug -- see tt/hifigan/source.py's module docstring for
why the identity needed re-deriving after an initial conv-based attempt hung);
device tests check the TTNN port against that same `torch_reference`, with no
shared derivation on the device side either.

No CosyVoice2 checkpoint is available yet, so `TtSourceModuleHnNSF` is built
from a randomly-initialised linear layer, matching the pattern in
test_hift_decode.py.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.common.utility_functions import comp_pcc
from models.demos.audio.cosyvoice2.tt.hifigan.source import downsample_basis, upsample_basis

GATE_BF16 = 0.99
S = 480  # CosyVoice2's upsample_scale: prod([8, 5, 3]) * hop_len(4)


def _upsampled_f0(mel_frames: int, scale: int = S, batch: int = 1, seed: int = 0):
    """Synthetic f0 already at audio rate, piecewise-constant in blocks of
    `scale` -- what HiFTGenerator.forward's nearest-upsample (`self.f0_upsamp`)
    actually produces, without needing that step built to test this module.
    Mixes voiced (80-400 Hz) and unvoiced (0 Hz) mel-frames."""
    g = torch.Generator().manual_seed(seed)
    voiced = torch.rand(batch, mel_frames, generator=g) > 0.3
    f0_mel = torch.where(
        voiced, 80.0 + torch.rand(batch, mel_frames, generator=g) * 320.0, torch.zeros(batch, mel_frames)
    )
    return f0_mel.repeat_interleave(scale, dim=1).unsqueeze(-1)  # [B, T_audio, 1]


# --------------------------------------------------------------------------
# host tier -- no device
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mel_len", [4, 6, 20, 50])
def test_downsample_basis_matches_torch_interpolate(mel_len):
    torch.manual_seed(mel_len)
    x = torch.randn(1, mel_len * S) * 0.02  # rad-values scale: (f0*harm/sr) % 1
    want = F.interpolate(x.reshape(1, 1, -1), scale_factor=1 / S, mode="linear").flatten()

    D = torch.from_numpy(downsample_basis(mel_len, S))
    got = D @ x.flatten()
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


@pytest.mark.parametrize("mel_len", [4, 6, 20, 50])
def test_upsample_basis_matches_torch_interpolate(mel_len):
    """Includes the boundary spans (first/last S//2 samples), not just the
    interior -- this is exactly the part a conv_transpose-based attempt got
    wrong before this matrix formulation replaced it."""
    torch.manual_seed(mel_len + 1)
    y = torch.randn(mel_len) * 6.0  # phase scale
    want = F.interpolate(y.reshape(1, 1, mel_len), scale_factor=S, mode="linear").flatten()

    U = torch.from_numpy(upsample_basis(mel_len, S))
    got = U @ y
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()

    half = S // 2
    assert torch.allclose(got[:half], y[0].expand(half), atol=1e-4)
    assert torch.allclose(got[-half:], y[-1].expand(half), atol=1e-4)


@pytest.mark.parametrize("mel_len", [4, 20, 250])
def test_upsample_basis3_relative_offset_matches_torch_interpolate_at_large_phase(mel_len):
    """The fix that made TtSineGen2._upsample precision-independent of mel_len:
    subtract the window's own center value before interpolating (bounding the
    matmul's operands to one cumsum step, not the cumsum's own unbounded
    growth), then add it back via a broadcast. Checked here on host at the
    magnitude that actually matters -- a real cumsum trajectory, which reaches
    into the single digits by mel_len=250 (~5s of audio) -- not on small
    synthetic values that would hide the issue this fix addresses. The device
    version of this exact construction is what closed the PCC gap from 0.38 to
    0.999 at mel_len=20 (see tt/hifigan/source.py's TtSineGen2._upsample
    docstring)."""
    from models.demos.audio.cosyvoice2.tt.hifigan.source import upsample_basis3

    torch.manual_seed(mel_len)
    rad = torch.rand(mel_len) * 0.05
    mel = torch.cumsum(rad, dim=0)  # unbounded growth, same as real phase_mel
    want = F.interpolate(mel.reshape(1, 1, mel_len), scale_factor=S, mode="linear").flatten()

    B = torch.from_numpy(upsample_basis3(S))  # [S, 3]
    left_nb = torch.cat([mel[:1], mel[:-1]])
    right_nb = torch.cat([mel[1:], mel[-1:]])
    windows = torch.stack([left_nb, mel, right_nb], dim=1)  # [mel_len, 3]
    windows_rel = windows - mel.unsqueeze(1)  # bounded regardless of mel_len

    interp_rel = windows_rel @ B.T  # [mel_len, S]
    got = (interp_rel + mel.unsqueeze(1)).flatten()

    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()


def test_rand_ini_has_no_effect_through_downsample():
    """Regression test for the verified claim in source.py's module docstring:
    a perturbation at audio-rate index 0 (where SineGen2._f02sine's `rand_ini`
    lands) is never read by the downsample basis, so omitting rand_ini entirely
    is correct, not an oversight."""
    mel_len = 5
    torch.manual_seed(0)
    base = torch.randn(mel_len * S)
    perturbed = base.clone()
    perturbed[0] += 999.0

    D = torch.from_numpy(downsample_basis(mel_len, S))
    assert torch.equal(D @ base, D @ perturbed)


@pytest.mark.parametrize("mel_frames", [4, 20])
def test_sine_gen2_torch_reference_shape_and_range(mel_frames):
    from models.demos.audio.cosyvoice2.tt.hifigan.source import TtSineGen2

    f0 = _upsampled_f0(mel_frames)
    sine_waves, uv, noise = TtSineGen2.torch_reference(f0, sampling_rate=24000, upsample_scale=S)
    assert sine_waves.shape == (1, mel_frames * S, 9)  # harmonic_num=8 default -> H+1=9
    assert uv.shape == (1, mel_frames * S, 1)
    assert sine_waves.abs().max() <= 0.1 + 1e-4  # sine_amp=0.1 default, no noise


# --------------------------------------------------------------------------
# device tier -- needs silicon
# --------------------------------------------------------------------------
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)


@needs_l1_small
def test_cumsum_precision_at_mel_rate(device):
    """Measured, not assumed from the CosyVoice1 reference's numbers (those were
    for a cumsum over ~72k *audio-rate* samples with the blocked-mod-1 trick
    required; this cumsum runs over the much shorter *mel-rate* sequence).

    Compares ttnn.cumsum(dtype=float32) against a float64 torch reference over
    a realistic utterance length (mel_frames=250, ~5s at CosyVoice2's ~50
    mel-fps), on real rad_values-scale data (fractional part of f0*harmonic/sr,
    magnitude ~1e-3 to a few times 1e-2)."""
    import ttnn

    mel_frames = 250
    torch.manual_seed(42)
    # rad_values scale: fractional part of (f0 * harmonic_idx / sampling_rate)
    rad = torch.rand(1, mel_frames, 1, dtype=torch.float64) * 0.05

    want = torch.cumsum(rad, dim=1)
    want_fp32_torch = torch.cumsum(rad.float(), dim=1).double()

    rad_dev = ttnn.from_torch(rad.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    got_dev = ttnn.cumsum(rad_dev, dim=1, dtype=ttnn.float32)
    got = ttnn.to_torch(got_dev).double()

    err_ttnn = (want - got).abs().max().item()
    err_torch_fp32 = (want - want_fp32_torch).abs().max().item()
    print(
        f"\n  cumsum @ mel-rate (T={mel_frames}): max|err| ttnn.cumsum(fp32)={err_ttnn:.3e}  torch fp32 cumsum={err_torch_fp32:.3e}"
    )

    # Gate: error must stay well under one phase-cycle-worth of drift (2*pi is a
    # full cycle in the eventual sin(); rad_values themselves are mod-1, so an
    # error here should stay a small fraction of 1.0, not accumulate toward it).
    assert err_ttnn < 0.01, f"ttnn.cumsum(dtype=float32) drifted {err_ttnn:.3e} over {mel_frames} mel-frames"


@needs_l1_small
@pytest.mark.parametrize("mel_frames", [4, 20, 250])
def test_device_sine_gen2_matches_real_torch(device, mel_frames):
    """TtSineGen2 vs. TtSineGen2.torch_reference (built on real
    torch.nn.functional.interpolate) -- zero inferential steps, captured noise
    draw shared on both sides.

    mel_frames=250 (~5s at CosyVoice2's ~50 mel-fps, audio_len=120000) closes
    the backlog item from the KV-cache audit: this device path was previously
    only exercised to mel_frames=20, well short of a real utterance length.
    It is not a new precision regime -- test_cumsum_precision_at_mel_rate and
    test_upsample_basis3_relative_offset_matches_torch_interpolate_at_large_phase
    above already check the cumsum and the centering-trick identity at this
    same mel_frames=250 -- this is the first test to run the full TtSineGen2
    device module (cumsum + centering-trick upsample + sin + uv mask) at that
    length end to end, rather than its pieces in isolation."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.source import TtSineGen2

    torch.manual_seed(7)
    f0 = _upsampled_f0(mel_frames, seed=7)
    noise = torch.randn(1, mel_frames * S, 9)  # captured torch.randn_like(sine_waves) draw

    want, want_uv, _ = TtSineGen2.torch_reference(
        f0, sampling_rate=24000, upsample_scale=S, sine_amp=0.1, noise_std=0.003, voiced_threshold=10.0, noise=noise
    )

    sg = TtSineGen2(device, sampling_rate=24000, upsample_scale=S, sine_amp=0.1, noise_std=0.003, voiced_threshold=10.0)
    f0_dev = ttnn.from_torch(f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    noise_dev = ttnn.from_torch(noise, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out, uv, _ = sg(f0_dev, noise=noise_dev)
    got = ttnn.to_torch(out).float()
    got_uv = ttnn.to_torch(uv).float()

    assert got.shape == want.shape, (got.shape, want.shape)
    assert torch.equal(got_uv, want_uv), "uv mask (voiced/unvoiced) mismatch"
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device T_mel={mel_frames} SineGen2 PCC {pcc}")
    assert passed, pcc


@needs_l1_small
def test_device_source_module_matches_real_torch(device):
    """TtSourceModuleHnNSF vs. its torch_reference -- same bar, one level up
    the stack (includes the linear merge + tanh)."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.source import TtSourceModuleHnNSF

    torch.manual_seed(13)
    mel_frames = 10
    harmonic_num = 8
    f0 = _upsampled_f0(mel_frames, seed=13)
    sine_noise = torch.randn(1, mel_frames * S, harmonic_num + 1)
    branch_noise = torch.randn(1, mel_frames * S, 1)

    linear_weight = torch.randn(1, harmonic_num + 1) * 0.1
    linear_bias = torch.randn(1) * 0.1

    want, want_noise, want_uv = TtSourceModuleHnNSF.torch_reference(
        f0,
        linear_weight,
        linear_bias,
        sampling_rate=24000,
        upsample_scale=S,
        harmonic_num=harmonic_num,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=10.0,
        branch_noise=branch_noise,
        noise=sine_noise,
    )

    mod = TtSourceModuleHnNSF(
        device,
        linear_weight,
        linear_bias,
        sampling_rate=24000,
        upsample_scale=S,
        harmonic_num=harmonic_num,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshold=10.0,
    )
    f0_dev = ttnn.from_torch(f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    sine_noise_dev = ttnn.from_torch(sine_noise, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    branch_noise_dev = ttnn.from_torch(branch_noise, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out, noise_out, uv = mod(f0_dev, sine_noise=sine_noise_dev, branch_noise=branch_noise_dev)
    got = ttnn.to_torch(out).float()

    assert got.shape == want.shape, (got.shape, want.shape)
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device SourceModuleHnNSF PCC {pcc}")
    assert passed, pcc
