# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the XTTS-v2 audio conditioning path — ref audio -> GPT prompt latents.

Validates the TTNN port of the speaker-conditioning branch (preprocessing ->
conditioning_encoder -> conditioning_perceiver, i.e. everything that produces the
conditioning latents prepended to the GPT input) against the pure-PyTorch
reference, using the *real* weights and a *real* reference wav from
https://huggingface.co/coqui/XTTS-v2 (``model.pth`` + ``samples/en_sample.wav``).

The mel spectrogram is computed once on the host (torch.stft + mel filterbank,
since torchaudio is unavailable) and fed identically to both paths, so the PCC
reflects only the conditioning_encoder + perceiver port.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    # first run downloads ~1.9 GB of XTTS-v2 weights + the sample wav
    pytest models/experimental/xtts/tests/test_conditioning.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_conditioning import (
    load_reference_audio,
    reference_conditioning,
    wav_to_mel,
)
from models.experimental.xtts.tt.xtts_conditioning import TtXttsConditioning


@pytest.fixture(scope="module")
def xtts_state_dict():
    """Load the real XTTS-v2 checkpoint state dict once for the whole module."""
    return load_xtts_state_dict()


@pytest.mark.parametrize("sample", ["en_sample.wav", "es_sample.wav"])
@pytest.mark.parametrize("pcc", [0.99])
def test_xtts_conditioning(device, xtts_state_dict, sample, pcc, reset_seeds):
    # Preprocess: real reference audio -> normalized log-mel [1, 80, s] (the real input).
    wav = load_reference_audio(sample=sample)
    mel = wav_to_mel(wav, xtts_state_dict["mel_stats"].cpu())
    logger.info(f"ref audio {sample!r}: wav {tuple(wav.shape)} -> mel {tuple(mel.shape)}")

    # Reference: conditioning_encoder + perceiver, real weights.
    reference = reference_conditioning(xtts_state_dict)
    with torch.no_grad():
        reference_output = reference(mel)  # [1, 1024, 32]

    # TTNN port of the same conditioning path.
    tt_conditioning = TtXttsConditioning(xtts_state_dict, device)
    tt_output = ttnn.to_torch(tt_conditioning(mel)).float()

    does_pass, pcc_message = comp_pcc(reference_output, tt_output, pcc)
    logger.info(comp_allclose(reference_output, tt_output))
    logger.info(f"conditioning latents {tuple(reference_output.shape)}: {pcc_message}")

    assert does_pass, f"XTTS conditioning PCC below {pcc}: {pcc_message}"
