# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_conditioning import (
    load_reference_audio,
    reference_conditioning,
    wav_to_mel,
)
from models.experimental.xtts.reference.xtts_gpt_block import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_HEADS,
    reference_gpt_block,
)
from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN, wrap_text_ids
from models.experimental.xtts.reference.xtts_gpt_model import reference_gpt_model
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel

TILE = 32
NUM_PREFILL_LAYERS = 1


@pytest.mark.parametrize("pcc", [0.99])
def test_tt_gpt_prefill(device, xtts_state_dict, pcc):
    """Compare single-layer TTNN GPT prefill KV cache to the PyTorch reference via PCC."""
    sd = xtts_state_dict

    wav = load_reference_audio(sample="en_sample.wav")
    mel = wav_to_mel(wav, sd["mel_stats"].cpu())
    with torch.no_grad():
        cond = reference_conditioning(sd)(mel).transpose(1, 2)

    wrapped = wrap_text_ids(preprocess_text("hello world", lang="en"))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)
    text_len = wrapped.shape[1]
    prompt_len = cond.shape[1] + text_len
    logger.info(f"single-layer GPT prefill: prompt_len={prompt_len} (cond {cond.shape[1]} + text {text_len})")

    ref_model = reference_gpt_model(sd, num_layers=NUM_PREFILL_LAYERS)
    ref_block = reference_gpt_block(sd, layer_idx=0)
    with torch.no_grad():
        text_pos = torch.arange(text_len)
        text_emb = ref_model.text_embedding(wrapped) + ref_model.text_pos_embedding(text_pos)
        prompt_emb = torch.cat([cond, text_emb], dim=1).float()
        ln1 = ref_block.block.ln_1(prompt_emb)
        qkv = ref_block.block.attn.c_attn(ln1)
        _, k_all, v_all = qkv.split(HIDDEN_SIZE, dim=2)

    def _split_heads(t):
        """Reshape a [B, T, H] tensor into multi-head [B, heads, T, head_dim]."""
        return t.view(1, -1, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()

    ref_k = _split_heads(k_all)
    ref_v = _split_heads(v_all)

    tt_model = TtXttsGptModel(sd, device, num_layers=NUM_PREFILL_LAYERS)
    cond_tt = ttnn.from_torch(cond.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    max_seq = -(-(prompt_len + 4) // TILE) * TILE
    kv = tt_model.prefill(wrapped, cond_tt, max_seq)

    tt_k = ttnn.to_torch(kv[0][0]).float()[:, :, :prompt_len, :]
    tt_v = ttnn.to_torch(kv[0][1]).float()[:, :, :prompt_len, :]

    assert tt_k.shape == ref_k.shape, f"K shape {tuple(tt_k.shape)} != {tuple(ref_k.shape)}"
    assert tt_v.shape == ref_v.shape, f"V shape {tuple(tt_v.shape)} != {tuple(ref_v.shape)}"

    k_pass, k_msg = comp_pcc(ref_k, tt_k, pcc)
    logger.info(comp_allclose(ref_k, tt_k))
    logger.info(f"prefill cache K PCC: {k_msg}")

    v_pass, v_msg = comp_pcc(ref_v, tt_v, pcc)
    logger.info(comp_allclose(ref_v, tt_v))
    logger.info(f"prefill cache V PCC: {v_msg}")

    assert k_pass, f"prefill cache K PCC below {pcc}: {k_msg}"
    assert v_pass, f"prefill cache V PCC below {pcc}: {v_msg}"
