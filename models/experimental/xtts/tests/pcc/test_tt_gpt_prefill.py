# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the TTNN GPT PREFILL step (``TtXttsGptModel.prefill``) vs the torch reference.

Isolates PREFILL — NOT decode — on a SINGLE GPT decoder block (``num_layers=1``, layer 0's weights)
rather than the full 30-layer stack. Prefill's job is to encode the ``[cond | text]`` prompt through
the causal stack and SEED the fixed-size KV cache; its observable product is that per-layer K/V
cache. So the gate is on the cache prefill writes:

  * ``model.prefill(text_ids, cond_latents, max_seq)`` runs the real prefill path — embeddings ->
    ``concat([cond | text])`` -> ``stack.forward_prefill`` (1 block, full causal attention) ->
    ``fill_cache`` — seeding ``model._static_kv`` in place.
  * The reference K/V for the prompt is the layer-0 attention projection of the prompt hiddens
    (``ln_1 -> c_attn -> split heads``); K/V are per-position projections, so the prompt positions'
    K/V depend only on the prompt (independent of any later mel tokens). Head split matches the TT
    path (nlp_create_qkv_heads, transpose_k_heads=False): [q|k|v] blocks, head-major.

Two gates: the cached K and the cached V at the prompt positions (PCC vs the fp32 reference).
Agreement here means the whole prefill pipeline (embeddings + causal block + cache write) is correct.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/pcc/test_tt_gpt_prefill.py -s
"""

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
    load_xtts_state_dict,
    reference_gpt_block,
)
from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN, wrap_text_ids
from models.experimental.xtts.reference.xtts_gpt_model import reference_gpt_model
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel

TILE = 32
NUM_PREFILL_LAYERS = 1  # run a SINGLE GPT decoder block (not the full 30-layer stack)


@pytest.fixture(scope="module")
def xtts_state_dict():
    return load_xtts_state_dict()


@pytest.mark.parametrize("pcc", [0.99])
def test_tt_gpt_prefill(device, xtts_state_dict, pcc):
    sd = xtts_state_dict

    # Real conditioning latents [1, 32, 1024], fed identically to reference + TT.
    wav = load_reference_audio(sample="en_sample.wav")
    mel = wav_to_mel(wav, sd["mel_stats"].cpu())
    with torch.no_grad():
        cond = reference_conditioning(sd)(mel).transpose(1, 2)  # [1, 1024, 32] -> [1, 32, 1024]

    # [START] + [en] + tokens + [STOP], padded to a tile multiple (tile-clean prefill).
    wrapped = wrap_text_ids(preprocess_text("hello world", lang="en"))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)
    text_len = wrapped.shape[1]
    prompt_len = cond.shape[1] + text_len  # cond (32) + text
    logger.info(f"single-layer GPT prefill: prompt_len={prompt_len} (cond {cond.shape[1]} + text {text_len})")

    # Reference (ground truth): build the prompt embeddings [cond | text_emb] exactly as the model
    # does (text positions 0..text_len-1; cond prepended raw), then layer-0's K/V projection.
    ref_model = reference_gpt_model(sd, num_layers=NUM_PREFILL_LAYERS)
    ref_block = reference_gpt_block(sd, layer_idx=0)
    with torch.no_grad():
        text_pos = torch.arange(text_len)
        text_emb = ref_model.text_embedding(wrapped) + ref_model.text_pos_embedding(text_pos)  # [1, text_len, hidden]
        prompt_emb = torch.cat([cond, text_emb], dim=1).float()  # [1, prompt_len, hidden]
        ln1 = ref_block.block.ln_1(prompt_emb)  # GPT2Block applies ln_1 before attention
        qkv = ref_block.block.attn.c_attn(ln1)  # [1, prompt_len, 3*hidden]
        _, k_all, v_all = qkv.split(HIDDEN_SIZE, dim=2)  # each [1, prompt_len, hidden]

    def _split_heads(t):  # [1, S, hidden] -> [1, heads, S, head_dim]
        return t.view(1, -1, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()

    ref_k = _split_heads(k_all)  # [1, heads, prompt_len, head_dim]
    ref_v = _split_heads(v_all)

    # TT: run the real prefill (embeddings -> concat -> causal block -> fill_cache), then read the
    # seeded KV cache at the prompt positions.
    tt_model = TtXttsGptModel(sd, device, num_layers=NUM_PREFILL_LAYERS)
    cond_tt = ttnn.from_torch(cond.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    max_seq = -(-(prompt_len + 4) // TILE) * TILE  # fixed cache length (tile-aligned; room for decode)
    kv = tt_model.prefill(wrapped, cond_tt, max_seq)

    tt_k = ttnn.to_torch(kv[0][0]).float()[:, :, :prompt_len, :]  # [1, heads, prompt_len, head_dim]
    tt_v = ttnn.to_torch(kv[0][1]).float()[:, :, :prompt_len, :]

    assert tt_k.shape == ref_k.shape, f"K shape {tuple(tt_k.shape)} != {tuple(ref_k.shape)}"
    assert tt_v.shape == ref_v.shape, f"V shape {tuple(tt_v.shape)} != {tuple(ref_v.shape)}"

    # Gate 1: cached K at prompt positions.
    k_pass, k_msg = comp_pcc(ref_k, tt_k, pcc)
    logger.info(comp_allclose(ref_k, tt_k))
    logger.info(f"prefill cache K PCC: {k_msg}")

    # Gate 2: cached V at prompt positions.
    v_pass, v_msg = comp_pcc(ref_v, tt_v, pcc)
    logger.info(comp_allclose(ref_v, tt_v))
    logger.info(f"prefill cache V PCC: {v_msg}")

    assert k_pass, f"prefill cache K PCC below {pcc}: {k_msg}"
    assert v_pass, f"prefill cache V PCC below {pcc}: {v_msg}"
