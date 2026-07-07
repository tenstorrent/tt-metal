# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC test: reference audio + text -> full XTTS-v2 GPT module.

Exercises the whole GPT front-to-back with the audio conditioning prompt, reusing
the already-ported component modules:

    ref audio --TtXttsConditioning-->  cond latents [b, 1024, 32] -> [b, 32, 1024]
    text      --preprocess_text----->  token ids
    (cond, text_ids, mel_ids) --TtXttsGptModel(cond_latents=...)-->
        emb = [cond | text_emb | mel_emb] -> 30 blocks + ln_f
            -> strip cond prompt -> final_norm -> text_head / mel_head

Validated against the pure-PyTorch reference (``reference_conditioning`` +
``reference_gpt_model``) with the *real* weights and a *real* reference wav from
https://huggingface.co/coqui/XTTS-v2.

Notes:
  * The mel/audio-code stream is random token ids (there is no DVAE here); ref and
    TTNN receive identical ids, so PCC reflects only the GPT-module port.
  * text ids are padded to a tile multiple so the sequence-dim concat/slices are
    tile-aligned; ref and TTNN pad identically.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/test_gpt_with_conditioning.py -s
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_gpt_model import NUM_AUDIO_TOKENS, reference_gpt_model
from models.experimental.xtts.reference.xtts_conditioning import (
    load_reference_audio,
    reference_conditioning,
    wav_to_mel,
)
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_conditioning import TtXttsConditioning
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel

TILE = 32


@pytest.fixture(scope="module")
def xtts_state_dict():
    """Load the real XTTS-v2 checkpoint state dict once for the whole module."""
    return load_xtts_state_dict()


def _pad_to_tile(ids):
    pad = (-ids.shape[1]) % TILE
    return F.pad(ids, (0, pad), value=0) if pad else ids


@pytest.mark.parametrize(
    "input_text, sample, mel_len",
    [
        ("hello world", "en_sample.wav", 64),
        # ("text to speech synthesis on tenstorrent", "es_sample.wav", 96),
    ],
)
@pytest.mark.parametrize("pcc", [0.99])
def test_xtts_gpt_with_conditioning(device, xtts_state_dict, input_text, sample, mel_len, pcc, reset_seeds):
    # --- inputs: reference audio (conditioning) + text ---
    wav = load_reference_audio(sample=sample)
    mel = wav_to_mel(wav, xtts_state_dict["mel_stats"].cpu())
    text_ids = _pad_to_tile(preprocess_text(input_text, lang="en"))
    mel_ids = torch.randint(0, NUM_AUDIO_TOKENS, (1, mel_len), dtype=torch.long)
    logger.info(
        f"text {input_text!r} -> {text_ids.shape[1]} tokens (padded); mel_ids {mel_len}; cond mel {tuple(mel.shape)}"
    )

    # --- reference: conditioning latents -> full GPT ---
    ref_cond = reference_conditioning(xtts_state_dict)
    ref_gpt = reference_gpt_model(xtts_state_dict)
    with torch.no_grad():
        cond_latents = ref_cond(mel).transpose(1, 2)  # [1, 1024, 32] -> [1, 32, 1024]
        ref_text_logits, ref_mel_logits = ref_gpt(text_ids, mel_ids, cond_latents=cond_latents)

    # --- TTNN: reuse TtXttsConditioning + TtXttsGptModel ---
    tt_cond = TtXttsConditioning(xtts_state_dict, device)
    tt_gpt = TtXttsGptModel(xtts_state_dict, device)
    tt_cond_latents = ttnn.permute(tt_cond(mel), (0, 2, 1))  # [1, 1024, 32] -> [1, 32, 1024]
    tt_text_logits, tt_mel_logits = tt_gpt(text_ids, mel_ids, cond_latents=tt_cond_latents)

    tt_text_logits = ttnn.to_torch(tt_text_logits).float()[:, : text_ids.shape[1], : ref_text_logits.shape[-1]]
    tt_mel_logits = ttnn.to_torch(tt_mel_logits).float()[:, :mel_len, : ref_mel_logits.shape[-1]]

    text_pass, text_msg = comp_pcc(ref_text_logits, tt_text_logits, pcc)
    mel_pass, mel_msg = comp_pcc(ref_mel_logits, tt_mel_logits, pcc)
    logger.info(comp_allclose(ref_text_logits, tt_text_logits))
    logger.info(f"text_head: {text_msg}")
    logger.info(comp_allclose(ref_mel_logits, tt_mel_logits))
    logger.info(f"mel_head: {mel_msg}")

    assert text_pass, f"text_head logits PCC below {pcc}: {text_msg}"
    assert mel_pass, f"mel_head logits PCC below {pcc}: {mel_msg}"
