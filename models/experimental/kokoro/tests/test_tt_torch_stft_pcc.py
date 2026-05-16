# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_torch_stft.TTTorchSTFT`
vs reference :class:`~models.experimental.kokoro.reference.istftnet.TorchSTFT`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.istftnet import TorchSTFT
from models.experimental.kokoro.tt.tt_torch_stft import TTTorchSTFT, preprocess_tt_torch_stft


# Kokoro's PLBERT-side iSTFT config (``gen_istft_n_fft=20``, ``gen_istft_hop_size=5``).
_N_FFT = 20
_HOP = 5
_WIN = 20


def _make_ref() -> TorchSTFT:
    return TorchSTFT(filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN).eval()


def _to_torch(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(t).float()


def test_tt_torch_stft_transform_matches(device):
    """``transform`` magnitude and phase match the reference."""
    torch.manual_seed(0)
    B, L = 2, 100
    ref = _make_ref()
    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN, input_length=L, device=device
    )
    tt_mod = TTTorchSTFT(device, params)

    x = torch.randn(B, L)
    with torch.no_grad():
        mag_ref, phase_ref = ref.transform(x)

    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mag_tt, phase_tt = tt_mod.transform(x_tt)
    mag_h = _to_torch(mag_tt)
    phase_h = _to_torch(phase_tt)
    while mag_h.dim() > mag_ref.dim():
        mag_h = mag_h.squeeze(0)
    while phase_h.dim() > phase_ref.dim():
        phase_h = phase_h.squeeze(0)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(phase_tt)
    ttnn.deallocate(x_tt)

    assert mag_h.shape == mag_ref.shape, (mag_h.shape, mag_ref.shape)
    assert phase_h.shape == phase_ref.shape, (phase_h.shape, phase_ref.shape)

    _, pcc_mag = comp_pcc(mag_ref, mag_h, pcc=0.0)
    # ``phase`` wraps at ``±π``; compare ``cos(phase)`` to avoid PCC artifacts at the wrap.
    _, pcc_phase = comp_pcc(torch.cos(phase_ref), torch.cos(phase_h), pcc=0.0)
    print(f"TTTorchSTFT.transform magnitude PCC: {pcc_mag:.6f}, cos(phase) PCC: {pcc_phase:.6f}")
    assert pcc_mag > 0.99, f"magnitude PCC too low: {pcc_mag}"
    assert pcc_phase > 0.99, f"cos(phase) PCC too low: {pcc_phase}"


def test_tt_torch_stft_inverse_matches(device):
    """``inverse`` reconstructs the same waveform as the reference for a given (mag, phase)."""
    torch.manual_seed(1)
    B, L = 1, 100
    ref = _make_ref()
    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN, input_length=L, device=device
    )
    tt_mod = TTTorchSTFT(device, params)

    x = torch.randn(B, L)
    with torch.no_grad():
        mag_ref, phase_ref = ref.transform(x)
        y_ref = ref.inverse(mag_ref, phase_ref)  # [B, 1, L]

    mag_tt = ttnn.from_torch(mag_ref, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    phase_tt = ttnn.from_torch(phase_ref, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod.inverse(mag_tt, phase_tt)
    y_h = _to_torch(y_tt)
    while y_h.dim() > y_ref.dim():
        y_h = y_h.squeeze(0)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(phase_tt)

    assert y_h.shape == y_ref.shape, (y_h.shape, y_ref.shape)
    _, pcc = comp_pcc(y_ref, y_h, pcc=0.0)
    print(f"TTTorchSTFT.inverse PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_torch_stft_forward_round_trip(device):
    """Full ``forward`` (transform → inverse) reconstructs the input on TT."""
    torch.manual_seed(2)
    B, L = 2, 100
    ref = _make_ref()
    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN, input_length=L, device=device
    )
    tt_mod = TTTorchSTFT(device, params)

    x = torch.randn(B, L)
    with torch.no_grad():
        y_ref = ref(x)  # [B, 1, L]

    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt)
    y_h = _to_torch(y_tt)
    while y_h.dim() > y_ref.dim():
        y_h = y_h.squeeze(0)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    assert y_h.shape == y_ref.shape, (y_h.shape, y_ref.shape)
    _, pcc = comp_pcc(y_ref, y_h, pcc=0.0)
    print(f"TTTorchSTFT.forward (round trip) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_torch_stft_longer_signal(device):
    """Larger input length (``L=200``) exercises the matmul scaling."""
    torch.manual_seed(3)
    B, L = 1, 200
    ref = _make_ref()
    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN, input_length=L, device=device
    )
    tt_mod = TTTorchSTFT(device, params)

    x = torch.randn(B, L)
    with torch.no_grad():
        y_ref = ref(x)

    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt)
    y_h = _to_torch(y_tt)
    while y_h.dim() > y_ref.dim():
        y_h = y_h.squeeze(0)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    _, pcc = comp_pcc(y_ref, y_h, pcc=0.0)
    print(f"TTTorchSTFT.forward (L=200) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


@pytest.mark.xfail(
    reason=(
        "BH BF16 atan2 noise: near-zero STFT bins have sign-random phase for low-amplitude "
        "harmonic input. Use use_torch_stft_fallback=True to reach PCC > 0.99."
    ),
    strict=False,
)
def test_tt_torch_stft_transform_harmonic_pure_ttnn(device):
    """Informational: documents BH BF16 phase accuracy failure with low-amplitude harmonic input.

    All existing STFT tests use broadband ``torch.randn`` (amplitude O(1)), giving all
    frequency bins meaningful energy — BF16 atan2 noise is negligible relative to the
    signal, so those tests pass.

    Kokoro's ``sine_merge`` is a low-amplitude (~0.1) harmonic signal: energy is
    concentrated in 1–2 bins while the remaining bins have magnitude ~1e-5.  On
    Blackhole hardware, ``atan2`` in the SFPU rounds float32 inputs to BF16 before
    computing.  For near-zero bins whose true X_real/X_imag values are below the BF16
    noise floor (~1e-2 × signal), the rounded inputs become sign-random, giving
    ``atan2`` output that is off by ±π.  This corrupts the STFT phase for these bins
    even though the magnitude is correct.

    This test documents that failure.  Expected: cos(phase) PCC << 0.99.
    See companion ``test_tt_torch_stft_transform_harmonic_torch_fallback`` for the
    passing version with CPU fallback.
    """
    L = 1500
    sr = 24000
    f0 = 200.0
    amplitude = 0.1

    t = torch.arange(L, dtype=torch.float32)
    x = (amplitude * torch.sin(2 * torch.pi * f0 / sr * t)).unsqueeze(0)  # [1, 1500]

    ref = _make_ref()
    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN, input_length=L, device=device
    )
    tt_mod = TTTorchSTFT(device, params, use_torch_stft_fallback=False)

    with torch.no_grad():
        mag_ref, phase_ref = ref.transform(x)

    # float32 input: eliminates input-quantization noise so the test isolates BH conv2d /
    # atan2 BF16 compute error specifically.  BH MACs still round internally to BF16
    # regardless of input dtype, so the test still fails as expected.
    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mag_tt, phase_tt = tt_mod.transform(x_tt)
    mag_h = _to_torch(mag_tt)
    phase_h = _to_torch(phase_tt)
    while mag_h.dim() > mag_ref.dim():
        mag_h = mag_h.squeeze(0)
    while phase_h.dim() > phase_ref.dim():
        phase_h = phase_h.squeeze(0)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(phase_tt)
    ttnn.deallocate(x_tt)

    _, pcc_mag = comp_pcc(mag_ref, mag_h, pcc=0.0)
    _, pcc_phase = comp_pcc(torch.cos(phase_ref), torch.cos(phase_h), pcc=0.0)
    print(
        f"[harmonic f0={f0:.0f}Hz amp={amplitude}, pure-TTNN] "
        f"magnitude PCC: {pcc_mag:.6f}, cos(phase) PCC: {pcc_phase:.6f} "
        f"(expected << 0.99 — BH BF16 atan2 noise on near-zero bins)"
    )
    assert pcc_phase > 0.99, f"cos(phase) PCC too low: {pcc_phase}"


def test_tt_torch_stft_transform_harmonic_torch_fallback(device):
    """cos(phase) PCC > 0.99 with CPU torch.stft fallback for low-amplitude harmonic input.

    Uses the same harmonic signal as ``test_tt_torch_stft_transform_harmonic_pure_ttnn``
    (f0=200 Hz, amplitude=0.1, L=1500) but routes the entire ``transform`` through CPU
    ``torch.stft``, bypassing BH BF16 atan2 noise on near-zero bins.

    Why the fallback is needed: when ``use_torch_stft_fallback=False``, ``atan2`` runs
    on the BH SFPU which rounds float32→BF16 before every op.  For bins with magnitude
    ~1e-5 (well below the BF16 noise floor), the rounded X_real/X_imag become
    sign-random, giving ±π phase error.  ``use_torch_stft_fallback=True`` runs
    ``torch.stft`` in float32 on CPU, so all bins get accurate phase regardless of
    magnitude.
    """
    L = 1500
    sr = 24000
    f0 = 200.0
    amplitude = 0.1

    t = torch.arange(L, dtype=torch.float32)
    x = (amplitude * torch.sin(2 * torch.pi * f0 / sr * t)).unsqueeze(0)  # [1, 1500]

    ref = _make_ref()
    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN, input_length=L, device=device
    )
    tt_mod = TTTorchSTFT(device, params, use_torch_stft_fallback=True)

    with torch.no_grad():
        mag_ref, phase_ref = ref.transform(x)

    # float32 input: the fallback converts the device tensor back to float32 via
    # ttnn.to_torch().float(), so a lossless round-trip means both reference and fallback
    # see identical float32 values → cos(phase) PCC should reach ~1.0.
    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mag_tt, phase_tt = tt_mod.transform(x_tt)
    mag_h = _to_torch(mag_tt)
    phase_h = _to_torch(phase_tt)
    while mag_h.dim() > mag_ref.dim():
        mag_h = mag_h.squeeze(0)
    while phase_h.dim() > phase_ref.dim():
        phase_h = phase_h.squeeze(0)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(phase_tt)
    ttnn.deallocate(x_tt)

    _, pcc_mag = comp_pcc(mag_ref, mag_h, pcc=0.0)
    _, pcc_phase = comp_pcc(torch.cos(phase_ref), torch.cos(phase_h), pcc=0.0)
    print(
        f"[harmonic f0={f0:.0f}Hz amp={amplitude}, torch fallback] "
        f"magnitude PCC: {pcc_mag:.6f}, cos(phase) PCC: {pcc_phase:.6f}"
    )
    assert pcc_mag > 0.99, f"magnitude PCC too low with torch fallback: {pcc_mag}"
    assert pcc_phase > 0.99, f"cos(phase) PCC too low with torch fallback: {pcc_phase}"
