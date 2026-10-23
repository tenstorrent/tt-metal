# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_model import reference_gpt_model
from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN, greedy_generate, wrap_text_ids
from models.experimental.xtts.reference.xtts_conditioning import (
    load_reference_audio,
    reference_conditioning,
    wav_to_mel,
)
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel
from models.experimental.xtts.tt.xtts_generator import TtXttsGenerator

TILE = 32
MAX_NEW_TOKENS = 16


@pytest.mark.parametrize("pcc", [0.99])
def test_tt_gpt_generate(device, xtts_state_dict, pcc):
    """Compare TTNN greedy GPT generation latents to the PyTorch reference via PCC."""
    sd = xtts_state_dict

    wav = load_reference_audio(sample="en_sample.wav")
    mel = wav_to_mel(wav, sd["mel_stats"].cpu())
    with torch.no_grad():
        cond = reference_conditioning(sd)(mel).transpose(1, 2)

    wrapped = wrap_text_ids(preprocess_text("hello world", lang="en"))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)

    ref_model = reference_gpt_model(sd)
    ref_codes, ref_latents = greedy_generate(ref_model, wrapped, cond, max_new_tokens=MAX_NEW_TOKENS, wrap_text=False)

    tt_model = TtXttsGptModel(sd, device)
    gen = TtXttsGenerator(tt_model)
    cond_tt = ttnn.from_torch(cond.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    tt_codes, _ = gen.generate(wrapped, cond_tt, max_new_tokens=MAX_NEW_TOKENS)
    ref_list, tt_list = ref_codes[0].tolist(), tt_codes[0].tolist()
    logger.info(f"ref codes ({len(ref_list)}): {ref_list}")
    logger.info(f"tt  codes ({len(tt_list)}): {tt_list}")

    preds, tt_latents_tt = gen.latents_for_codes(wrapped, cond_tt, ref_list)
    tt_latents = ttnn.to_torch(tt_latents_tt).float()

    prefix = 0
    for a, b in zip(tt_list, ref_list):
        if a != b:
            break
        prefix += 1
    forced_agree = sum(int(p == c) for p, c in zip(preds[: len(ref_list)], ref_list))
    logger.info(
        f"free-run exact-match prefix: {prefix}/{len(ref_list)}; "
        f"teacher-forced top-1 agreement: {forced_agree}/{len(ref_list)}"
    )

    assert (
        tt_latents.shape == ref_latents.shape
    ), f"latent shape {tuple(tt_latents.shape)} != {tuple(ref_latents.shape)}"
    does_pass, msg = comp_pcc(ref_latents, tt_latents, pcc)
    logger.info(comp_allclose(ref_latents, tt_latents))
    logger.info(f"latent PCC: {msg}")
    assert does_pass, f"latent PCC below {pcc}: {msg}"

    assert prefix >= 1, f"first code disagrees — loop mis-wired:\n  ref={ref_list}\n  tt ={tt_list}"
