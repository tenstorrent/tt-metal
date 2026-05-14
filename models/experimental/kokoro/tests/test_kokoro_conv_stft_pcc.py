# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN `KokoroConvStft` vs PyTorch `CustomSTFT` (same conv STFT definition)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

ttnn = pytest.importorskip("ttnn")

from models.experimental.kokoro.reference.kokoro_istftnet import CustomSTFT
from models.experimental.kokoro.reference.kokoro_stft_preprocess import preprocess_kokoro_conv_stft_parameters
from models.experimental.kokoro.tt import KokoroConvStft


def _comp_pcc(golden: torch.Tensor, calculated: torch.Tensor, pcc: float = 0.99) -> tuple[bool, float]:
    g = torch.squeeze(golden).detach().float().numpy().flatten()
    c = torch.squeeze(calculated).detach().float().numpy().flatten()
    cal = np.min(np.ma.corrcoef(np.ma.masked_invalid(g), np.ma.masked_invalid(c)))
    if isinstance(cal, np.ma.core.MaskedConstant):
        return True, 1.0
    v = float(np.asarray(cal))
    return v >= pcc, v


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


def test_kokoro_conv_stft_roundtrip_pcc(ttnn_device):
    torch.manual_seed(0)
    B, T = 1, 4800
    ref = CustomSTFT(filter_length=800, hop_length=200, win_length=800, center=True, pad_mode="replicate")
    w_cpu = torch.randn(B, T, dtype=torch.float32)
    mag_ref, phase_ref = ref.transform(w_cpu)
    len_ref = int(w_cpu.shape[-1])
    wav_ref = ref.inverse(mag_ref, phase_ref, length=len_ref)

    params = preprocess_kokoro_conv_stft_parameters(ttnn_device)
    stft = KokoroConvStft(ttnn_device, params)
    w_tt = ttnn.from_torch(
        w_cpu.reshape(B, 1, T),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mag_tt, phase_tt = stft.transform(w_tt)
    wav_tt = stft.inverse(mag_tt, phase_tt, length=len_ref)

    mag_pass, mag_pcc = _comp_pcc(mag_ref, ttnn.to_torch(mag_tt))
    ph_pass, ph_pcc = _comp_pcc(phase_ref, ttnn.to_torch(phase_tt), pcc=0.96)
    wav_pass, wav_pcc = _comp_pcc(wav_ref, ttnn.to_torch(wav_tt))

    assert mag_pass, f"magnitude PCC {mag_pcc} < 0.99"
    assert ph_pass, f"phase PCC {ph_pcc} < 0.96 (SFPU atan2 vs float32 atan2)"
    assert wav_pass, f"waveform PCC {wav_pcc} < 0.99"


# Regression: long-utterance prompts (e.g. "The lazy fox is jumping over the speedy cat.")
# previously failed in the istftnet path with a static-CB / L1 buffer clash at conv2d program load
# (``ttnn_kokoro_stft.py::_StridedStftConv``). The audio fed into the generator STFT is
# ``f0_coarse_time * f0_up_scale``: e.g. ``tf=120 * 300 = 36000`` samples here (the 30000-sample row
# represents the previously-working default text "Hello from Tenstorrent Kokoro full TTNN.").
@pytest.mark.parametrize("audio_len", [30000, 36000, 60000])
def test_kokoro_conv_stft_generator_config_long_sequences(ttnn_device, audio_len: int):
    """STFT with the generator's n_fft=20 / hop=5 config across long inputs (no L1 CB clash)."""
    torch.manual_seed(0)
    B = 1
    n_fft = 20
    hop = 5
    ref = CustomSTFT(filter_length=n_fft, hop_length=hop, win_length=n_fft, center=True, pad_mode="replicate")
    w_cpu = torch.randn(B, audio_len, dtype=torch.float32)
    mag_ref, phase_ref = ref.transform(w_cpu)
    wav_ref = ref.inverse(mag_ref, phase_ref, length=int(w_cpu.shape[-1]))

    params = preprocess_kokoro_conv_stft_parameters(ttnn_device, filter_length=n_fft, hop_length=hop, win_length=n_fft)
    stft = KokoroConvStft(ttnn_device, params)
    w_tt = ttnn.from_torch(
        w_cpu.reshape(B, 1, audio_len),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mag_tt, phase_tt = stft.transform(w_tt)
    wav_tt = stft.inverse(mag_tt, phase_tt, length=audio_len)

    mag_pass, mag_pcc = _comp_pcc(mag_ref, ttnn.to_torch(mag_tt))
    wav_pass, wav_pcc = _comp_pcc(wav_ref, ttnn.to_torch(wav_tt))
    print(f"audio_len={audio_len} mag_pcc={mag_pcc:.6f} wav_pcc={wav_pcc:.6f}")
    assert mag_pass, f"audio_len={audio_len} magnitude PCC {mag_pcc} < 0.99"
    assert wav_pass, f"audio_len={audio_len} waveform PCC {wav_pcc} < 0.99"
