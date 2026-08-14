# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_model import (
    NUM_AUDIO_TOKENS,
    NUM_TEXT_TOKENS,
    reference_gpt_model,
)
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel


@pytest.mark.parametrize(
    "text_len, mel_len",
    [
        (64, 96),
        (96, 128),
    ],
)
# Text head is bf16; mel head weight is bfloat8_b, so they have separate gates.
@pytest.mark.parametrize("pcc", [0.99])
@pytest.mark.parametrize("mel_pcc", [0.99])
def test_xtts_gpt_model(device, xtts_state_dict, text_len, mel_len, pcc, mel_pcc, reset_seeds):
    """Compare TTNN GPT text/mel logits to the PyTorch reference via PCC."""
    reference = reference_gpt_model(xtts_state_dict)

    text_ids = torch.randint(0, NUM_TEXT_TOKENS, (1, text_len), dtype=torch.long)
    mel_ids = torch.randint(0, NUM_AUDIO_TOKENS, (1, mel_len), dtype=torch.long)

    with torch.no_grad():
        ref_text_logits, ref_mel_logits = reference(text_ids, mel_ids)

    tt_model = TtXttsGptModel(xtts_state_dict, device)
    tt_text_logits, tt_mel_logits = tt_model(text_ids, mel_ids)

    tt_text_logits = ttnn.to_torch(tt_text_logits).float()[:, :text_len, :NUM_TEXT_TOKENS]
    tt_mel_logits = ttnn.to_torch(tt_mel_logits).float()[:, :mel_len, :NUM_AUDIO_TOKENS]

    text_pass, text_msg = comp_pcc(ref_text_logits, tt_text_logits, pcc)
    mel_pass, mel_msg = comp_pcc(ref_mel_logits, tt_mel_logits, mel_pcc)

    logger.info(comp_allclose(ref_text_logits, tt_text_logits))
    logger.info(f"text_head (text_len={text_len}): {text_msg}")
    logger.info(comp_allclose(ref_mel_logits, tt_mel_logits))
    logger.info(f"mel_head (mel_len={mel_len}): {mel_msg}")

    assert text_pass, f"text_head logits PCC below {pcc}: {text_msg}"
    assert mel_pass, f"mel_head logits PCC below {mel_pcc}: {mel_msg}"
