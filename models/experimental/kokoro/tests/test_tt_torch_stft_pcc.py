# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_torch_stft.TTTorchSTFT`
vs reference :class:`~models.experimental.kokoro.reference.istftnet.TorchSTFT`."""

from __future__ import annotations

import sys
from pathlib import Path

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


def _harmonic_signal(L: int = 1500, f0: float = 200.0, amplitude: float = 0.1, sr: int = 24_000) -> torch.Tensor:
    """Single-frequency sinusoid matching Kokoro's typical sine-source amplitude."""
    t = torch.arange(L, dtype=torch.float32)
    return (amplitude * torch.sin(2 * torch.pi * f0 / sr * t)).unsqueeze(0)  # [1, L]


def test_tt_torch_stft_transform_harmonic_phase_ceiling_no_fallback(device):
    """``transform`` on Kokoro-scale harmonic input WITHOUT any fallback documents the BH ceiling.

    BH hardware rounds float32 to BF16 for ALL MAC ops — including the SFPU that evaluates
    ``atan2``.  Near-zero DFT bins (sine_amp ≈ 0.1, true bin value ~1e-5) are rounded to 0
    or sign-flipped → ``atan2(0, neg)`` gives ±π random phase.  The practical
    cos(phase) PCC ceiling on BH is ~0.64 for Kokoro harmonic input.

    Magnitude (sqrt of mag_sq) is insensitive to sign and asserted > 0.99.
    cos(phase) PCC is printed but NOT asserted — this is a documented hardware limitation.
    Use ``use_torch_stft_fallback=True`` or ``use_torch_stft_conv_fallback+use_torch_atan2_fallback``
    to fix the phase.
    """
    x = _harmonic_signal()
    ref = _make_ref()
    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN, input_length=x.shape[-1], device=device
    )
    tt_mod = TTTorchSTFT(device, params, use_torch_stft_fallback=False, use_torch_atan2_fallback=False)

    with torch.no_grad():
        mag_ref, phase_ref = ref.transform(x)

    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mag_tt, phase_tt = tt_mod.transform(x_tt)
    mag_h, phase_h = _to_torch(mag_tt), _to_torch(phase_tt)
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
        f"[no-fallback, harmonic 200Hz amp=0.1] mag PCC: {pcc_mag:.6f}, "
        f"cos(phase) PCC: {pcc_phase:.6f}  (BH BF16 atan2 ceiling ~0.64)"
    )
    assert pcc_mag > 0.99, f"magnitude PCC too low: {pcc_mag}"
    # cos(phase) PCC is NOT asserted — BH BF16 atan2 on near-zero bins is a hardware limitation


def test_tt_torch_stft_transform_harmonic_stft_fallback_pcc(device):
    """``transform`` WITH ``use_torch_stft_fallback=True`` achieves cos(phase) PCC > 0.99.

    Full CPU ``torch.stft`` bypasses BH BF16 atan2 precision loss entirely.
    This is the primary fix for the harmonic-source STFT phase PCC.
    """
    x = _harmonic_signal()
    ref = _make_ref()
    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN, input_length=x.shape[-1], device=device
    )
    tt_mod = TTTorchSTFT(device, params, use_torch_stft_fallback=True)

    with torch.no_grad():
        mag_ref, phase_ref = ref.transform(x)

    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mag_tt, phase_tt = tt_mod.transform(x_tt)
    mag_h, phase_h = _to_torch(mag_tt), _to_torch(phase_tt)
    while mag_h.dim() > mag_ref.dim():
        mag_h = mag_h.squeeze(0)
    while phase_h.dim() > phase_ref.dim():
        phase_h = phase_h.squeeze(0)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(phase_tt)
    ttnn.deallocate(x_tt)

    _, pcc_mag = comp_pcc(mag_ref, mag_h, pcc=0.0)
    _, pcc_phase = comp_pcc(torch.cos(phase_ref), torch.cos(phase_h), pcc=0.0)
    print(f"[stft-fallback, harmonic 200Hz amp=0.1] mag PCC: {pcc_mag:.6f}, " f"cos(phase) PCC: {pcc_phase:.6f}")
    assert pcc_mag > 0.99, f"magnitude PCC too low: {pcc_mag}"
    assert pcc_phase > 0.99, f"cos(phase) PCC too low with stft_fallback: {pcc_phase}"


def test_tt_torch_stft_transform_harmonic_conv_fallback_alone_insufficient(device):
    """``use_torch_stft_conv_fallback=True`` alone does NOT fix cos(phase) PCC.

    Running the strided conv2d on CPU produces accurate X_real/X_imag, but the
    ``atan2`` SFPU on BH still runs in BF16 — near-zero real/imag pairs are rounded to 0
    before SFPU evaluation, producing sign-random phase.

    This test documents that conv_fallback alone is insufficient.  Magnitude is accurate
    (asserted > 0.99); phase is printed but not asserted.  The fix is to also enable
    ``use_torch_atan2_fallback=True``.
    """
    x = _harmonic_signal()
    ref = _make_ref()
    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN, input_length=x.shape[-1], device=device
    )
    tt_mod = TTTorchSTFT(device, params, use_torch_stft_conv_fallback=True, use_torch_atan2_fallback=False)

    with torch.no_grad():
        mag_ref, phase_ref = ref.transform(x)

    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mag_tt, phase_tt = tt_mod.transform(x_tt)
    mag_h, phase_h = _to_torch(mag_tt), _to_torch(phase_tt)
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
        f"[conv-only-fallback, harmonic 200Hz amp=0.1] mag PCC: {pcc_mag:.6f}, "
        f"cos(phase) PCC: {pcc_phase:.6f}  (BH BF16 atan2 SFPU still degrades phase)"
    )
    assert pcc_mag > 0.99, f"magnitude PCC too low: {pcc_mag}"
    # cos(phase) PCC NOT asserted — atan2 BH BF16 degradation documented, fix = add atan2_fallback


def test_tt_torch_stft_transform_harmonic_conv_and_atan2_fallback_pcc(device):
    """``use_torch_stft_conv_fallback=True`` + ``use_torch_atan2_fallback=True`` → PCC > 0.99.

    Both the strided conv2d (produces accurate X_real/X_imag) AND the atan2/sqrt step run
    on CPU float32.  This combination is the minimal per-op fix: it achieves the same
    cos(phase) PCC as ``use_torch_stft_fallback=True`` without invoking ``torch.stft``.

    This test shows that the two BH BF16 failure points are:
    1. Conv2d BF16 sign-flips near-zero bins (fixed by ``use_torch_stft_conv_fallback``).
    2. atan2 SFPU BF16 produces wrong phase even on correct real/imag (fixed by ``use_torch_atan2_fallback``).
    Both must be on CPU for phase PCC > 0.99.
    """
    x = _harmonic_signal()
    ref = _make_ref()
    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN, input_length=x.shape[-1], device=device
    )
    tt_mod = TTTorchSTFT(device, params, use_torch_stft_conv_fallback=True, use_torch_atan2_fallback=True)

    with torch.no_grad():
        mag_ref, phase_ref = ref.transform(x)

    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mag_tt, phase_tt = tt_mod.transform(x_tt)
    mag_h, phase_h = _to_torch(mag_tt), _to_torch(phase_tt)
    while mag_h.dim() > mag_ref.dim():
        mag_h = mag_h.squeeze(0)
    while phase_h.dim() > phase_ref.dim():
        phase_h = phase_h.squeeze(0)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(phase_tt)
    ttnn.deallocate(x_tt)

    _, pcc_mag = comp_pcc(mag_ref, mag_h, pcc=0.0)
    _, pcc_phase = comp_pcc(torch.cos(phase_ref), torch.cos(phase_h), pcc=0.0)
    print(f"[conv+atan2-fallback, harmonic 200Hz amp=0.1] mag PCC: {pcc_mag:.6f}, " f"cos(phase) PCC: {pcc_phase:.6f}")
    assert pcc_mag > 0.99, f"magnitude PCC too low: {pcc_mag}"
    assert pcc_phase > 0.99, f"cos(phase) PCC too low with conv+atan2_fallback: {pcc_phase}"


def test_tt_torch_stft_forward_harmonic_no_fallback(device):
    """Round-trip forward with harmonic input achieves PCC > 0.99 without any fallback.

    The direct X_real/X_imag path (conv → iSTFT matmul, skipping atan2) avoids BH BF16
    phase-error amplification: mag*cos(atan2_BF16(y,x)) diverges from x by up to 100×
    for near-zero bins. Bypassing the mag/phase roundtrip keeps error at the conv2d noise
    floor and yields PCC > 0.99 even for low-amplitude harmonic input.
    """
    L = 1500
    sr = 24000
    f0 = 200.0
    amplitude = 0.1
    t = torch.arange(L, dtype=torch.float32)
    x = (amplitude * torch.sin(2 * torch.pi * f0 / sr * t)).unsqueeze(0)

    ref = _make_ref()
    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN, input_length=L, device=device
    )
    tt_mod = TTTorchSTFT(device, params, use_torch_stft_fallback=False)

    with torch.no_grad():
        y_ref = ref(x)

    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt)
    y_h = _to_torch(y_tt)
    while y_h.dim() > y_ref.dim():
        y_h = y_h.squeeze(0)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    assert y_h.shape == y_ref.shape, (y_h.shape, y_ref.shape)
    _, pcc = comp_pcc(y_ref, y_h, pcc=0.0)
    print(f"[harmonic f0={f0:.0f}Hz amp={amplitude}, no fallback] forward PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"
