# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_sinegen.TTSineGen`
vs reference :class:`~models.experimental.kokoro.reference.istftnet.SineGen`."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.istftnet import SineGen
from models.experimental.kokoro.tt.tt_sinegen import TTSineGen, preprocess_tt_sinegen


@contextmanager
def _deterministic_torch_random():
    """Patch ``torch.rand`` / ``torch.randn_like`` to return zeros for the duration of the block.

    ``SineGen`` samples randomness inside its forward (``rand_ini`` and ``randn_like``); for
    deterministic PCC against the TT module we just zero both sides. The TT module defaults to
    zeros when ``rand_ini`` / ``noise_raw`` aren't supplied, so this gives a bit-for-bit comparable
    deterministic path.
    """
    real_rand = torch.rand
    real_randn_like = torch.randn_like

    def fake_rand(*size, **kwargs):
        return torch.zeros(*size, **kwargs)

    def fake_randn_like(t, **kwargs):
        return torch.zeros_like(t, **kwargs)

    torch.rand = fake_rand
    torch.randn_like = fake_randn_like
    try:
        yield
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like


def _run_pcc(
    device,
    *,
    sampling_rate: float,
    upsample_scale: int,
    harmonic_num: int,
    time_len: int,
    B: int,
    seed: int,
    f0_scale: float = 200.0,
):
    torch.manual_seed(seed)
    ref = SineGen(
        samp_rate=sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=harmonic_num,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0.0,
    ).eval()

    # F0 in Hz; allow some unvoiced positions (clamped to 0) to exercise the ``uv`` branch.
    f0 = torch.relu(torch.randn(B, time_len, 1) * f0_scale)

    with torch.no_grad(), _deterministic_torch_random():
        sine_ref, uv_ref, noise_ref = ref(f0)

    params = preprocess_tt_sinegen(
        device=device,
        sampling_rate=sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=harmonic_num,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0.0,
        time_len=time_len,
    )
    tt_mod = TTSineGen(device, params)

    f0_tt = ttnn.from_torch(f0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sine_tt, uv_tt, noise_tt = tt_mod(f0_tt)

    sine_h = ttnn.to_torch(sine_tt).float()
    uv_h = ttnn.to_torch(uv_tt).float()
    noise_h = ttnn.to_torch(noise_tt).float()
    for arr, ref_t in ((sine_h, sine_ref), (uv_h, uv_ref), (noise_h, noise_ref)):
        while arr.dim() > ref_t.dim():
            arr.squeeze_(0)
    ttnn.deallocate(sine_tt)
    ttnn.deallocate(uv_tt)
    ttnn.deallocate(noise_tt)
    ttnn.deallocate(f0_tt)

    assert sine_h.shape == sine_ref.shape, (sine_h.shape, sine_ref.shape)
    assert uv_h.shape == uv_ref.shape, (uv_h.shape, uv_ref.shape)
    assert noise_h.shape == noise_ref.shape, (noise_h.shape, noise_ref.shape)

    _, pcc_uv = comp_pcc(uv_ref, uv_h, pcc=0.0)
    _, pcc_sine = comp_pcc(sine_ref, sine_h, pcc=0.0)
    # ``noise_ref`` is identically zero in deterministic mode; check max-abs instead of PCC.
    noise_abs = noise_h.abs().max().item()
    print(
        f"TTSineGen (sr={sampling_rate}, up={upsample_scale}, harm={harmonic_num}, T={time_len}, B={B}) "
        f"sine PCC: {pcc_sine:.6f}, uv PCC: {pcc_uv:.6f}, noise |max|: {noise_abs:.2e}"
    )
    assert pcc_uv > 0.99, f"uv PCC too low: {pcc_uv}"
    assert pcc_sine > 0.99, f"sine PCC too low: {pcc_sine}"
    assert noise_abs < 1e-3, f"noise should be ~0 in deterministic mode (got {noise_abs})"


def test_tt_sinegen_default_harmonic0(device):
    """Kokoro's default ``harmonic_num=0`` (``dim=1``)."""
    _run_pcc(device, sampling_rate=24000.0, upsample_scale=4, harmonic_num=0, time_len=64, B=1, seed=0)


def test_tt_sinegen_with_harmonics(device):
    """``harmonic_num=2`` → ``dim=3``; exercises the per-harmonic broadcast and the fundamental mask."""
    _run_pcc(device, sampling_rate=24000.0, upsample_scale=4, harmonic_num=2, time_len=64, B=1, seed=1)


def test_tt_sinegen_larger_upsample(device):
    """``upsample_scale=10`` with a longer signal (``T=160``)."""
    _run_pcc(device, sampling_rate=24000.0, upsample_scale=10, harmonic_num=0, time_len=160, B=2, seed=2)


def test_tt_sinegen_noise_path(device):
    """Supply a deterministic ``noise_raw`` tensor and verify the noise-mix formula matches."""
    torch.manual_seed(3)
    sampling_rate = 24000.0
    upsample_scale = 4
    harmonic_num = 0
    time_len = 64
    B = 1
    sine_amp = 0.1
    noise_std = 0.003

    ref = SineGen(
        samp_rate=sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=harmonic_num,
        sine_amp=sine_amp,
        noise_std=noise_std,
        voiced_threshold=0.0,
    ).eval()
    f0 = torch.relu(torch.randn(B, time_len, 1) * 200.0)

    # Use the same noise tensor in both reference and TT.
    fixed_noise = torch.randn(B, time_len, harmonic_num + 1)

    real_randn_like = torch.randn_like
    torch.randn_like = lambda t, **kwargs: fixed_noise.to(t.dtype).expand_as(t).clone()
    real_rand = torch.rand
    torch.rand = lambda *size, **kwargs: torch.zeros(*size, **kwargs)
    try:
        with torch.no_grad():
            sine_ref, uv_ref, noise_ref = ref(f0)
    finally:
        torch.randn_like = real_randn_like
        torch.rand = real_rand

    params = preprocess_tt_sinegen(
        device=device,
        sampling_rate=sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=harmonic_num,
        sine_amp=sine_amp,
        noise_std=noise_std,
        voiced_threshold=0.0,
        time_len=time_len,
    )
    tt_mod = TTSineGen(device, params)
    f0_tt = ttnn.from_torch(f0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    noise_tt_in = ttnn.from_torch(fixed_noise, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sine_tt, uv_tt, noise_tt_out = tt_mod(f0_tt, noise_raw=noise_tt_in)

    sine_h = ttnn.to_torch(sine_tt).float()
    noise_h = ttnn.to_torch(noise_tt_out).float()
    while sine_h.dim() > sine_ref.dim():
        sine_h.squeeze_(0)
    while noise_h.dim() > noise_ref.dim():
        noise_h.squeeze_(0)
    ttnn.deallocate(sine_tt)
    ttnn.deallocate(uv_tt)
    ttnn.deallocate(noise_tt_out)
    ttnn.deallocate(noise_tt_in)
    ttnn.deallocate(f0_tt)

    _, pcc_noise = comp_pcc(noise_ref, noise_h, pcc=0.0)
    _, pcc_sine = comp_pcc(sine_ref, sine_h, pcc=0.0)
    print(f"TTSineGen (noise path) sine PCC: {pcc_sine:.6f}, noise PCC: {pcc_noise:.6f}")
    assert pcc_noise > 0.99, f"noise PCC too low: {pcc_noise}"
    assert pcc_sine > 0.99, f"sine PCC too low: {pcc_sine}"
