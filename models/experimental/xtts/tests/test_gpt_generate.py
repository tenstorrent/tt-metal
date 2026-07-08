# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase-1 test for the reference XTTS-v2 GPT greedy generate loop (torch only).

This is the ground truth the TTNN generator (Phase 2) is validated against, so it
must be self-consistent and deterministic. Uses the *real* coqui/XTTS-v2 weights
and *real* conditioning latents (ref audio -> mel -> conditioning encoder), with a
short ``max_new_tokens`` cap for CPU speed.

Checks: codes are valid ids, deterministic across runs, stop token stripped, and
latents align 1:1 with codes (the HiFiGAN decoder input contract).

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/test_gpt_generate.py -s
"""

import pytest
import torch
from loguru import logger

from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_gpt_model import NUM_AUDIO_TOKENS, reference_gpt_model
from models.experimental.xtts.reference.xtts_conditioning import (
    load_reference_audio,
    reference_conditioning,
    wav_to_mel,
)
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.reference.xtts_gpt_generate import STOP_AUDIO_TOKEN, greedy_generate

MAX_NEW_TOKENS = 16


@pytest.fixture(scope="module")
def xtts_state_dict():
    return load_xtts_state_dict()


def test_reference_greedy_generate(xtts_state_dict):
    sd = xtts_state_dict

    # Real conditioning latents [1, 32, 1024] from a real reference wav.
    wav = load_reference_audio(sample="en_sample.wav")
    mel = wav_to_mel(wav, sd["mel_stats"].cpu())
    with torch.no_grad():
        cond_latents = reference_conditioning(sd)(mel).transpose(1, 2)  # [1, 1024, 32] -> [1, 32, 1024]

    text_ids = preprocess_text("hello world", lang="en")
    model = reference_gpt_model(sd)

    codes, latents = greedy_generate(model, text_ids, cond_latents, max_new_tokens=MAX_NEW_TOKENS)
    logger.info(
        f"generated codes {tuple(codes.shape)} -> latents {tuple(latents.shape)}; codes[:8]={codes[0, :8].tolist()}"
    )

    # Latents align 1:1 with codes (decoder input contract).
    assert codes.shape[1] == latents.shape[1], f"code/latent length mismatch {codes.shape} vs {latents.shape}"
    assert codes.numel() > 0, "no codes generated"
    assert latents.shape[-1] == 1024, f"latent hidden dim {latents.shape[-1]} != 1024"

    # Codes are valid audio ids; leading start / trailing stop stripped.
    assert codes.min() >= 0 and codes.max() < NUM_AUDIO_TOKENS, "code id out of range"
    assert (codes != STOP_AUDIO_TOKEN).all(), "stop token not stripped from codes"

    # Greedy is deterministic: same inputs -> identical codes.
    codes2, latents2 = greedy_generate(model, text_ids, cond_latents, max_new_tokens=MAX_NEW_TOKENS)
    assert torch.equal(codes, codes2), "greedy generation is not deterministic"
    assert torch.allclose(latents, latents2), "latents not deterministic"
