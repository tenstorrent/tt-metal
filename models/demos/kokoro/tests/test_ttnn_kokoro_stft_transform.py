# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN CustomSTFT.transform vs torch reference."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.kokoro.reference.kokoro_istftnet import CustomSTFT
from models.demos.kokoro.tt.ttnn_kokoro_stft import custom_stft_transform, preprocess_custom_stft_transform


@pytest.mark.parametrize("T", [1000, 1200, 1600, 2400])
def test_ttnn_custom_stft_transform_matches_torch(device, T):
    torch.manual_seed(0)
    B = 1
    waveform = torch.randn(B, T, dtype=torch.float32)

    stft = CustomSTFT(filter_length=800, hop_length=200, win_length=800, center=True, pad_mode="replicate").eval()
    ref_mag, ref_phase = stft.transform(waveform)

    params = preprocess_custom_stft_transform(stft, device)
    wav_tt = ttnn.from_torch(waveform, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    mag_tt, phase_tt = custom_stft_transform(waveform_bt=wav_tt, params=params, device=device)
    mag = ttnn.to_torch(mag_tt).to(torch.float32)
    phase = ttnn.to_torch(phase_tt).to(torch.float32)

    # Crop frames to min length (edge padding/stride differences)
    min_frames = min(ref_mag.shape[-1], mag.shape[-1])
    ref_mag = ref_mag[..., :min_frames]
    mag = mag[..., :min_frames]
    ref_phase = ref_phase[..., :min_frames]
    phase = phase[..., :min_frames]

    ok, pcc = comp_pcc(ref_mag, mag, pcc=0.80)
    assert ok, f"stft magnitude pcc low: {pcc}"
    ok, pcc = comp_pcc(ref_phase, phase, pcc=0.70)
    assert ok, f"stft phase pcc low: {pcc}"
