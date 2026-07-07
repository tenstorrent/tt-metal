# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the XTTS-v2 GPT decoder with embeddings and heads.

Validates the TTNN port of the full GPT front-to-back — text/mel token
embeddings + learned position embeddings -> 30 GPT-2 decoder blocks + ln_f ->
final_norm -> text_head / mel_head — against the pure-PyTorch HuggingFace
reference, using the *real* weights from the upstream checkpoint at
https://huggingface.co/coqui/XTTS-v2 (``model.pth``).

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    # first run downloads ~1.9 GB of XTTS-v2 weights to the HF cache
    pytest models/experimental/xtts/tests/test_gpt_model.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_gpt_model import (
    NUM_AUDIO_TOKENS,
    NUM_TEXT_TOKENS,
    reference_gpt_model,
)
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel


@pytest.fixture(scope="module")
def xtts_state_dict():
    """Load the real XTTS-v2 checkpoint state dict once for the whole module."""
    return load_xtts_state_dict()


@pytest.mark.parametrize(
    "text_len, mel_len",
    [
        (64, 96),  # tile-aligned text + mel spans
        (96, 128),
    ],
)
@pytest.mark.parametrize("pcc", [0.99])
def test_xtts_gpt_model(device, xtts_state_dict, text_len, mel_len, pcc, reset_seeds):
    # Reference: embeddings + 30 blocks + ln_f + final_norm + heads, real weights.
    reference = reference_gpt_model(xtts_state_dict)

    # Random token ids within each vocab.
    text_ids = torch.randint(0, NUM_TEXT_TOKENS, (1, text_len), dtype=torch.long)
    mel_ids = torch.randint(0, NUM_AUDIO_TOKENS, (1, mel_len), dtype=torch.long)

    with torch.no_grad():
        ref_text_logits, ref_mel_logits = reference(text_ids, mel_ids)

    # TTNN port of the same model.
    tt_model = TtXttsGptModel(xtts_state_dict, device)
    tt_text_logits, tt_mel_logits = tt_model(text_ids, mel_ids)

    tt_text_logits = ttnn.to_torch(tt_text_logits).float()[:, :text_len, :NUM_TEXT_TOKENS]
    tt_mel_logits = ttnn.to_torch(tt_mel_logits).float()[:, :mel_len, :NUM_AUDIO_TOKENS]

    text_pass, text_msg = comp_pcc(ref_text_logits, tt_text_logits, pcc)
    mel_pass, mel_msg = comp_pcc(ref_mel_logits, tt_mel_logits, pcc)

    logger.info(comp_allclose(ref_text_logits, tt_text_logits))
    logger.info(f"text_head (text_len={text_len}): {text_msg}")
    logger.info(comp_allclose(ref_mel_logits, tt_mel_logits))
    logger.info(f"mel_head (mel_len={mel_len}): {mel_msg}")

    assert text_pass, f"text_head logits PCC below {pcc}: {text_msg}"
    assert mel_pass, f"mel_head logits PCC below {pcc}: {mel_msg}"
