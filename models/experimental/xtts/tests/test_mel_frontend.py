# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the XTTS-v2 speaker-encoder mel frontend (on-device STFT/mel).

Validates the fully on-device frontend (preemphasis -> gather-frame -> DFT matmul
-> power -> mel matmul) against the pure-PyTorch reference (``torch.stft`` + the
checkpoint's hamming window / mel filterbank). Input is a waveform ``[1, L]``;
output is the linear power mel ``[1, 64, T]``.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/test_mel_frontend.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_mel import build_reference_mel_frontend
from models.experimental.xtts.tt.xtts_mel import TtMelFrontend


@pytest.fixture(scope="module")
def xtts_state_dict():
    return load_xtts_state_dict()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("num_samples", [16000])
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_mel_frontend(device, xtts_state_dict, num_samples, pcc, reset_seeds):
    reference = build_reference_mel_frontend(xtts_state_dict)

    wav = torch.randn(1, num_samples) * 0.1
    with torch.no_grad():
        ref_mel = reference(wav)  # [1, 64, T]

    tt_fe = TtMelFrontend(device, reference)
    wav_dev = ttnn.from_torch(
        wav.reshape(1, num_samples, 1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )
    tt_mel = ttnn.to_torch(tt_fe(wav_dev)).float()

    assert tt_mel.shape == ref_mel.shape, f"shape {tuple(tt_mel.shape)} != {tuple(ref_mel.shape)}"
    does_pass, msg = comp_pcc(ref_mel, tt_mel, pcc)
    logger.info(comp_allclose(ref_mel, tt_mel))
    logger.info(f"mel_frontend num_samples={num_samples}: {msg}")
    assert does_pass, f"mel_frontend PCC below {pcc}: {msg}"
