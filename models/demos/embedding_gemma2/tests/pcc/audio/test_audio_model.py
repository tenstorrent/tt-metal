# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio head and end-to-end soft tokens.

test_output_proj        output_proj (1024 -> 1536, bias) on the HF last-layer output (teacher forced)
test_embed_audio        embed_audio (RMSNorm no scale + 1536 -> 512) on the HF tower output
test_soft_tokens        free-running TT forward (eager, AudioEncoder.encode) vs the HF fp32
                        reference: soft-token PCC per clip >= 0.99, and the soft-token count. Also
                        logs the tower-output PCC, the per-token cosine minimum / mean and, for
                        context, the same metrics for HF itself run in bf16.
test_reference_matches_full_model
                        the standalone reference (audio_tower + embed_audio) equals the valid rows of
                        `EmbeddingGemma2Model.get_audio_features(...).pooler_output` (CPU only).
"""

import copy

import pytest
import torch

from models.demos.embedding_gemma2.tests.common import PCC_LAYER, log_result, pcc
from models.demos.embedding_gemma2.tests.pcc.audio.audio_common import (
    ALL_CASES,
    LAYER_CASES,
    download_rows,
    upload_rows,
    valid,
)

PCC_SOFT_TOKENS = 0.99


@pytest.fixture(scope="session")
def hf_audio_bf16(hf_audio):
    tower, embed = hf_audio
    return copy.deepcopy(tower).to(torch.bfloat16), copy.deepcopy(embed).to(torch.bfloat16)


@pytest.mark.parametrize("case", LAYER_CASES)
def test_output_proj(case, eg2_device, tt_audio_encoder, audio_reference_for):
    feats, mask, ref = audio_reference_for(case)
    S = tt_audio_encoder.prepare_inputs(feats, mask).seq_len
    N = ref.token_mask.shape[1]
    last = len(tt_audio_encoder.model.layers) - 1
    y = download_rows(tt_audio_encoder.model.output_proj(upload_rows(eg2_device, ref.layer_out[last], S)), N)
    assert log_result("audio_output_proj_pcc", pcc(*valid(ref.hidden, y, ref.token_mask)), PCC_LAYER, case=case)


@pytest.mark.parametrize("case", LAYER_CASES)
def test_embed_audio(case, eg2_device, tt_audio_encoder, audio_reference_for):
    import ttnn

    feats, mask, ref = audio_reference_for(case)
    S = tt_audio_encoder.prepare_inputs(feats, mask).seq_len
    N = ref.token_mask.shape[1]
    h = upload_rows(eg2_device, ref.hidden, S, dtype=ttnn.float32)
    y = download_rows(tt_audio_encoder.model.embedder(h), N)
    assert log_result("audio_embed_pcc", pcc(*valid(ref.soft_all, y, ref.token_mask)), PCC_LAYER, case=case)


@pytest.mark.parametrize("case", ALL_CASES)
def test_soft_tokens(case, tt_audio_encoder, audio_reference_for, hf_audio_bf16):
    from models.demos.embedding_gemma2.reference.hf_audio_reference import run_audio_reference

    feats, mask, ref = audio_reference_for(case)
    soft, hidden = tt_audio_encoder.encode(feats, mask, return_hidden=True)
    ref16 = run_audio_reference(*hf_audio_bf16, feats.to(torch.bfloat16), mask, capture_layers=())
    assert len(soft) == len(ref.soft_tokens)
    ok = True
    for b, (tt, hf) in enumerate(zip(soft, ref.soft_tokens)):
        assert tuple(tt.shape) == tuple(hf.shape), f"clip {b}: {tuple(tt.shape)} vs HF {tuple(hf.shape)}"
        n = hf.shape[0]
        fields = dict(case=case, clip=b, soft_tokens=n)
        cos = torch.nn.functional.cosine_similarity(hf, tt, dim=-1)
        hf16 = ref16.soft_tokens[b].float()
        cos16 = torch.nn.functional.cosine_similarity(hf, hf16, dim=-1)
        log_result("audio_hidden_pcc", pcc(ref.hidden[b, :n], hidden[b]), 0.0, **fields)
        log_result("audio_soft_token_cos_min", float(cos.min()), 0.0, argmin=int(cos.argmin()), **fields)
        log_result("audio_soft_token_cos_mean", float(cos.mean()), 0.0, **fields)
        log_result("hf_bf16_audio_hidden_pcc", pcc(ref.hidden[b, :n], ref16.hidden[b, :n].float()), 0.0, **fields)
        log_result("hf_bf16_audio_soft_token_pcc", pcc(hf, hf16), 0.0, **fields)
        log_result("hf_bf16_audio_soft_token_cos_min", float(cos16.min()), 0.0, argmin=int(cos16.argmin()), **fields)
        ok &= log_result("audio_soft_token_pcc", pcc(hf, tt), PCC_SOFT_TOKENS, **fields)
    assert ok


@pytest.fixture(scope="module")
def tt_audio_encoder_fp32(eg2_device, audio_state_dict, audio_config, eg2_cache_dir, eg2_snapshot):
    import ttnn
    from models.demos.embedding_gemma2.tt.audio.audio_encoder import AudioEncoder
    from models.demos.embedding_gemma2.tt.audio.audio_model import AudioModel

    model = AudioModel.from_state_dict(
        audio_state_dict, audio_config, eg2_device, cache_dir=eg2_cache_dir, act_dtype=ttnn.float32
    )
    return AudioEncoder(model, eg2_device, eg2_snapshot)


@pytest.mark.parametrize("case", ALL_CASES)
def test_soft_tokens_fp32_activations(case, tt_audio_encoder_fp32, audio_reference_for):
    """The fp32-activation variant (AudioModel act_dtype=float32): same metrics as test_soft_tokens.
    Evidence for the precision lever documented in the README; not the default path."""
    feats, mask, ref = audio_reference_for(case)
    soft = tt_audio_encoder_fp32.encode(feats, mask)
    ok = True
    for b, (tt, hf) in enumerate(zip(soft, ref.soft_tokens)):
        assert tuple(tt.shape) == tuple(hf.shape)
        fields = dict(case=case, clip=b, soft_tokens=int(hf.shape[0]), act="fp32")
        cos = torch.nn.functional.cosine_similarity(hf, tt, dim=-1)
        log_result("audio_soft_token_cos_min", float(cos.min()), 0.0, argmin=int(cos.argmin()), **fields)
        log_result("audio_soft_token_cos_mean", float(cos.mean()), 0.0, **fields)
        ok &= log_result("audio_soft_token_pcc", pcc(hf, tt), PCC_SOFT_TOKENS, **fields)
    assert ok


@pytest.mark.parametrize("case", ["s2", "b2_s10_s2"])
def test_reference_matches_full_model(case, eg2_snapshot, audio_reference_for):
    from transformers import AutoModel

    from models.demos.embedding_gemma2.reference.hf_reference import _check_transformers

    _check_transformers()
    feats, mask, ref = audio_reference_for(case)
    model = AutoModel.from_pretrained(str(eg2_snapshot), dtype=torch.float32).eval()
    with torch.no_grad():
        out = model.get_audio_features(feats, mask, return_dict=True)
    full = out.pooler_output[out.attention_mask.bool()]
    mine = torch.cat(ref.soft_tokens, dim=0)
    diff = (full - mine).abs().max().item()
    log_result("audio_reference_vs_full_model_max_abs", diff, 0.0, case=case)
    assert full.shape == mine.shape and diff == 0.0, f"standalone reference differs from the full model by {diff}"
