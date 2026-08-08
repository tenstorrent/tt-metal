# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_custom_stft.TTCustomSTFT`
vs reference :class:`~models.experimental.kokoro.reference.custom_stft.CustomSTFT`.

``TTCustomSTFT`` is the on-device port of the istftnet ``disable_complex=True`` STFT — pure
``conv1d`` / ``conv_transpose1d``, no CPU fallback.  These tests compare it to the reference
``CustomSTFT`` on random input (where every DFT bin is well above the BF16 noise floor) and
document the BH BF16 phase ceiling on Kokoro-scale harmonic input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.custom_stft import CustomSTFT
from models.experimental.kokoro.tt.tt_custom_stft import TTCustomSTFT, preprocess_tt_custom_stft

# Kokoro istftnet generator config (``gen_istft_n_fft=20``, ``gen_istft_hop_size=5``).
_N_FFT = 20
_HOP = 5
_WIN = 20


def _make_ref() -> CustomSTFT:
    return CustomSTFT(filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN).eval()


def _make_tt(device) -> TTCustomSTFT:
    params = preprocess_tt_custom_stft(filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN)
    return TTCustomSTFT(device, params)


def _to_torch(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(t).float()


def _squeeze_to(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    while t.dim() > ref.dim():
        t = t.squeeze(0)
    return t


def test_tt_custom_stft_transform_matches(device):
    """``transform`` magnitude and phase match the reference on random input."""
    torch.manual_seed(0)
    B, L = 2, 100
    ref = _make_ref()
    tt_mod = _make_tt(device)

    x = torch.randn(B, L)
    with torch.no_grad():
        mag_ref, phase_ref = ref.transform(x)

    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mag_tt, phase_tt = tt_mod.transform(x_tt)
    mag_h = _squeeze_to(_to_torch(mag_tt), mag_ref)
    phase_h = _squeeze_to(_to_torch(phase_tt), phase_ref)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(phase_tt)
    ttnn.deallocate(x_tt)

    assert mag_h.shape == mag_ref.shape, (mag_h.shape, mag_ref.shape)
    assert phase_h.shape == phase_ref.shape, (phase_h.shape, phase_ref.shape)

    _, pcc_mag = comp_pcc(mag_ref, mag_h, pcc=0.0)
    _, pcc_phase = comp_pcc(torch.cos(phase_ref), torch.cos(phase_h), pcc=0.0)
    print(f"TTCustomSTFT.transform magnitude PCC: {pcc_mag:.6f}, cos(phase) PCC: {pcc_phase:.6f}")
    assert pcc_mag > 0.99, f"magnitude PCC too low: {pcc_mag}"
    assert pcc_phase > 0.99, f"cos(phase) PCC too low: {pcc_phase}"


def test_tt_custom_stft_inverse_matches(device):
    """``inverse`` reconstructs the same waveform as the reference for a given (mag, phase)."""
    torch.manual_seed(1)
    B, L = 1, 100
    ref = _make_ref()
    tt_mod = _make_tt(device)

    x = torch.randn(B, L)
    with torch.no_grad():
        mag_ref, phase_ref = ref.transform(x)
        y_ref = ref.inverse(mag_ref, phase_ref)  # [B, 1, output_length]

    mag_tt = ttnn.from_torch(mag_ref, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    phase_tt = ttnn.from_torch(phase_ref, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod.inverse(mag_tt, phase_tt)
    y_h = _squeeze_to(_to_torch(y_tt), y_ref)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(phase_tt)

    assert y_h.shape == y_ref.shape, (y_h.shape, y_ref.shape)
    _, pcc = comp_pcc(y_ref, y_h, pcc=0.0)
    print(f"TTCustomSTFT.inverse PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_custom_stft_forward_round_trip(device):
    """Full ``forward`` (transform → inverse) matches the reference round trip on TT."""
    torch.manual_seed(2)
    B, L = 2, 100
    ref = _make_ref()
    tt_mod = _make_tt(device)

    x = torch.randn(B, L)
    with torch.no_grad():
        y_ref = ref(x)  # [B, 1, L]

    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt)
    y_h = _squeeze_to(_to_torch(y_tt), y_ref)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    # Reference ``forward`` clamps to ``x.shape[-1]``; the on-device path produces the untrimmed
    # ``(F-1)*hop`` length (equal here since L is a multiple of hop) — compare the common prefix.
    n = min(y_h.shape[-1], y_ref.shape[-1])
    _, pcc = comp_pcc(y_ref[..., :n], y_h[..., :n], pcc=0.0)
    print(f"TTCustomSTFT.forward (round trip) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_custom_stft_longer_signal(device):
    """Larger input length (``L=200``) exercises the conv scaling."""
    torch.manual_seed(3)
    B, L = 1, 200
    ref = _make_ref()
    tt_mod = _make_tt(device)

    x = torch.randn(B, L)
    with torch.no_grad():
        y_ref = ref(x)

    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt)
    y_h = _squeeze_to(_to_torch(y_tt), y_ref)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    n = min(y_h.shape[-1], y_ref.shape[-1])
    _, pcc = comp_pcc(y_ref[..., :n], y_h[..., :n], pcc=0.0)
    print(f"TTCustomSTFT.forward (L=200) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def _harmonic_signal(L: int = 1500, f0: float = 200.0, amplitude: float = 0.1, sr: int = 24_000) -> torch.Tensor:
    """Single-frequency sinusoid matching Kokoro's typical sine-source amplitude."""
    t = torch.arange(L, dtype=torch.float32)
    return (amplitude * torch.sin(2 * torch.pi * f0 / sr * t)).unsqueeze(0)  # [1, L]


def test_tt_custom_stft_transform_harmonic_phase_ceiling(device):
    """``transform`` on Kokoro-scale harmonic input documents the BH BF16 phase ceiling.

    BH rounds float32 → BF16 before every MAC (and the atan2 SFPU).  For a low-amplitude
    sinusoid the off-frequency DFT bins are ~1e-5 — below the BF16 noise floor — so their phase
    is sign-random.  Magnitude (sign-insensitive) stays > 0.99; cos(phase) PCC is printed but
    NOT asserted (documented hardware limitation, no software fallback by design).
    """
    x = _harmonic_signal()
    ref = _make_ref()
    tt_mod = _make_tt(device)

    with torch.no_grad():
        mag_ref, phase_ref = ref.transform(x)

    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mag_tt, phase_tt = tt_mod.transform(x_tt)
    mag_h = _squeeze_to(_to_torch(mag_tt), mag_ref)
    phase_h = _squeeze_to(_to_torch(phase_tt), phase_ref)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(phase_tt)
    ttnn.deallocate(x_tt)

    _, pcc_mag = comp_pcc(mag_ref, mag_h, pcc=0.0)
    _, pcc_phase = comp_pcc(torch.cos(phase_ref), torch.cos(phase_h), pcc=0.0)
    print(
        f"[harmonic 200Hz amp=0.1] mag PCC: {pcc_mag:.6f}, "
        f"cos(phase) PCC: {pcc_phase:.6f}  (BH BF16 near-zero-bin ceiling)"
    )
    assert pcc_mag > 0.99, f"magnitude PCC too low: {pcc_mag}"
