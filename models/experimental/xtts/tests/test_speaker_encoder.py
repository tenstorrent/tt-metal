# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the XTTS-v2 speaker encoder body (ResNetSpeakerEncoder).

Validates the TTNN SE-ResNet + attentive-statistics pooling against the pure-
PyTorch reference, both from the real coqui/XTTS-v2 checkpoint. Input is a log-mel
``[1, 64, T]`` (the STFT/mel frontend is a later phase); output is the 512-d
L2-normalized speaker embedding ``g``.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/test_speaker_encoder.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_speaker_encoder import build_reference_speaker_encoder
from models.experimental.xtts.tt.xtts_speaker_encoder import TtResNetSpeakerEncoder


@pytest.fixture(scope="module")
def xtts_state_dict():
    return load_xtts_state_dict()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mel_len", [200])
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_speaker_encoder(device, xtts_state_dict, mel_len, pcc, reset_seeds):
    reference = build_reference_speaker_encoder(xtts_state_dict)

    # mel magnitudes [B, 64, T] (positive, as a real mel spectrogram would be).
    mel = torch.randn(1, 64, mel_len).abs() + 0.1
    with torch.no_grad():
        ref_g = reference(mel)  # [1, 512]

    tt_enc = TtResNetSpeakerEncoder(device, reference)
    mel_dev = ttnn.from_torch(mel.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    tt_g = ttnn.to_torch(tt_enc(mel_dev)).float()

    assert tt_g.shape == ref_g.shape, f"shape {tuple(tt_g.shape)} != {tuple(ref_g.shape)}"
    does_pass, msg = comp_pcc(ref_g, tt_g, pcc)
    logger.info(comp_allclose(ref_g, tt_g))
    logger.info(f"speaker_encoder mel_len={mel_len}: {msg}")
    assert does_pass, f"speaker_encoder PCC below {pcc}: {msg}"
