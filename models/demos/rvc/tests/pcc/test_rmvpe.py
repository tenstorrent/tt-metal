# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from models.demos.rvc.torch_impl.rmvpe import RMVPEPitchAlgorithm as RMVPETorch
from models.demos.rvc.tt_impl.rmvpe import RMVPEPitchAlgorithm as RMVPETT
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rmvpe(device):
    torch.manual_seed(0)

    sample_rate = 16000
    hop_size = 160
    audio_num_samples = 16000
    pitch_pcc = 0.96
    periodicity_pcc = 0.96
    f_min = 50.0
    f_max = 1100.0

    torch_predictor = RMVPETorch(
        sample_rate=sample_rate,
        hop_size=hop_size,
        fmin=f_min,
        fmax=f_max,
    )
    tt_predictor = RMVPETT(
        device=device,
        sample_rate=sample_rate,
        hop_size=hop_size,
        fmin=f_min,
        fmax=f_max,
    )

    audio = torch.randn(1, audio_num_samples, dtype=torch.float32)
    audio = audio / audio.abs().max().clamp_min(1e-8)

    # audio_tt = ttnn.from_torch(
    #     audio,
    #     dtype=ttnn.bfloat16,
    #     layout=ttnn.ROW_MAJOR_LAYOUT,
    #     device=device,
    # )

    torch_pitch = torch_predictor.extract_pitch(audio)
    tt_pitch = tt_predictor.extract_pitch(audio)

    assert tuple(torch_pitch.shape) == tuple(tt_pitch.shape)

    assert_with_pcc(torch_pitch.to(torch.float32), tt_pitch.to(torch.float32), pcc=pitch_pcc)
    # assert_with_pcc(
    #     torch_periodicity.to(torch.float32),
    #     tt_periodicity.to(torch.float32),
    #     pcc=periodicity_pcc,
    # )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rmvpe_three_consecutive_runs(device):
    torch.manual_seed(0)

    sample_rate = 16000
    hop_size = 160
    audio_num_samples = 16000
    pitch_pcc = 0.96
    f_min = 50.0
    f_max = 1100.0

    torch_predictor = RMVPETorch(
        sample_rate=sample_rate,
        hop_size=hop_size,
        fmin=f_min,
        fmax=f_max,
    )
    tt_predictor = RMVPETT(
        device=device,
        sample_rate=sample_rate,
        hop_size=hop_size,
        fmin=f_min,
        fmax=f_max,
    )

    audio = torch.randn(1, audio_num_samples, dtype=torch.float32)
    audio = audio / audio.abs().max().clamp_min(1e-8)

    for i in range(3):
        print(f"Run {i}:")
        torch_pitch = torch_predictor.extract_pitch(audio)
        tt_pitch = tt_predictor.extract_pitch(audio)
        assert tuple(torch_pitch.shape) == tuple(tt_pitch.shape)
        assert_with_pcc(torch_pitch.to(torch.float32), tt_pitch.to(torch.float32), pcc=pitch_pcc)
