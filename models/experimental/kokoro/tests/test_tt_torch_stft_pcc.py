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
