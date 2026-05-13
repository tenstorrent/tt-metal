# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import audio_codebook_embedding_reference
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.audio_tokenizer.model import (
    VoxtralTTAudioTokenizer,
    extract_audio_tokenizer_state_dict,
)
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict


@torch.no_grad()
@pytest.mark.parametrize("time_len", [4, 16])
def test_mm_audio_codebook_embedding_pcc(device, reset_seeds, time_len):
    """``mm_audio_embeddings.audio_codebook_embeddings`` vs ``F.embedding`` reference."""
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")

    emb_key = "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"
    if emb_key not in full:
        pytest.skip(f"Missing {emb_key} in checkpoint.")

    cfg = load_voxtral_config(model_name).audio_tokenizer_args
    sd = extract_audio_tokenizer_state_dict(full)
    try:
        tok = VoxtralTTAudioTokenizer(device, state_dict=sd, tokenizer_cfg=cfg, full_checkpoint=full)
    except Exception as exc:
        pytest.skip(f"Unable to build VoxtralTTAudioTokenizer: {exc}")
    if tok.mm_audio_codebook_embedding is None:
        pytest.skip("MM audio codebook embedding not attached.")

    w = full[emb_key].to(torch.bfloat16)
    n_vocab = int(w.shape[0])
    b = 1
    torch.manual_seed(0)
    idx = torch.randint(0, n_vocab, (b, time_len), dtype=torch.int64)
    ref = audio_codebook_embedding_reference(idx, w)

    idx_tt = ttnn.from_torch(
        idx,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = tok.mm_audio_codebook_embed_forward(idx_tt)
    ttnn.deallocate(idx_tt)

    tt_out = ttnn.to_torch(out_tt).float()
    passing, msg = comp_pcc(ref.float(), tt_out, pcc=0.99)
    assert passing, f"MM audio codebook embedding PCC failed: {msg}"
