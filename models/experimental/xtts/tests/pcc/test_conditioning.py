# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_conditioning import (
    load_reference_audio,
    reference_conditioning,
    wav_to_mel,
)
from models.experimental.xtts.tt.xtts_conditioning import TtXttsConditioning


@pytest.mark.parametrize("sample", ["en_sample.wav", "es_sample.wav"])
@pytest.mark.parametrize("pcc", [0.99])
def test_xtts_conditioning(device, xtts_state_dict, sample, pcc, reset_seeds):
    """Compare TTNN conditioning output to PyTorch reference via PCC."""
    wav = load_reference_audio(sample=sample)
    mel = wav_to_mel(wav, xtts_state_dict["mel_stats"].cpu())
    logger.info(f"ref audio {sample!r}: wav {tuple(wav.shape)} -> mel {tuple(mel.shape)}")

    reference = reference_conditioning(xtts_state_dict)
    with torch.no_grad():
        reference_output = reference(mel)

    tt_conditioning = TtXttsConditioning(xtts_state_dict, device)
    tt_output = ttnn.to_torch(tt_conditioning(mel)).float()

    does_pass, pcc_message = comp_pcc(reference_output, tt_output, pcc)
    logger.info(comp_allclose(reference_output, tt_output))
    logger.info(f"conditioning latents {tuple(reference_output.shape)}: {pcc_message}")

    assert does_pass, f"XTTS conditioning PCC below {pcc}: {pcc_message}"
