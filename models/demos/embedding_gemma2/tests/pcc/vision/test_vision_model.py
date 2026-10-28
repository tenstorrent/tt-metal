# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end vision path: processor output -> soft tokens [n_b, 512] per image / video frame.

test_soft_tokens        free-running TT forward (eager, VisionEncoder.encode) vs the HF fp32
                        reference: soft-token PCC per image >= 0.99. Also logs the per-token cosine
                        minimum and, for context, the same metrics for HF itself run in bf16.
test_reference_matches_full_model
                        the standalone reference (vision_tower + embed_vision) equals
                        `EmbeddingGemma2Model.get_image_features` / `get_video_features` (CPU only).
"""

import copy

import pytest
import torch

from models.demos.embedding_gemma2.tests.common import log_result, pcc
from models.demos.embedding_gemma2.tests.pcc.vision.vision_common import ALL_CASES, CASES

PCC_SOFT_TOKENS = 0.99


@pytest.fixture(scope="session")
def hf_vision_bf16(hf_vision):
    tower, embed = hf_vision
    return copy.deepcopy(tower).to(torch.bfloat16), copy.deepcopy(embed).to(torch.bfloat16)


@pytest.mark.parametrize("case", ALL_CASES)
def test_soft_tokens(case, tt_vision_encoder, vision_reference_for, hf_vision_bf16):
    from models.demos.embedding_gemma2.reference.hf_vision_reference import run_vision_reference

    pv, pid, ref = vision_reference_for(case)
    soft = tt_vision_encoder.encode(pv, pid)
    ref16 = run_vision_reference(*hf_vision_bf16, pv.to(torch.bfloat16), pid, capture_layers=())
    assert len(soft) == len(ref.soft_tokens)
    ok = True
    for b, (tt, hf) in enumerate(zip(soft, ref.soft_tokens)):
        assert tuple(tt.shape) == tuple(hf.shape), f"image {b}: {tuple(tt.shape)} vs HF {tuple(hf.shape)}"
        value = pcc(hf, tt)
        cos = torch.nn.functional.cosine_similarity(hf, tt, dim=-1)
        hf16 = ref16.soft_tokens[b].float()
        cos16 = torch.nn.functional.cosine_similarity(hf, hf16, dim=-1)
        fields = dict(case=case, image=b, soft_tokens=int(hf.shape[0]))
        log_result("vision_soft_token_cos_min", float(cos.min()), 0.0, **fields)
        log_result("vision_soft_token_cos_mean", float(cos.mean()), 0.0, **fields)
        log_result("hf_bf16_soft_token_pcc", pcc(hf, hf16), 0.0, **fields)
        log_result("hf_bf16_soft_token_cos_min", float(cos16.min()), 0.0, **fields)
        ok &= log_result("vision_soft_token_pcc", value, PCC_SOFT_TOKENS, **fields)
    assert ok


@pytest.mark.parametrize("case", ["landscape", "b2_square_small", "video8"])
def test_reference_matches_full_model(case, eg2_snapshot, hf_vision, vision_reference_for):
    from transformers import AutoModel

    from models.demos.embedding_gemma2.reference.hf_reference import _check_transformers

    _check_transformers()
    pv, pid, ref = vision_reference_for(case)
    model = AutoModel.from_pretrained(str(eg2_snapshot), torch_dtype=torch.float32).eval()
    with torch.no_grad():
        if CASES[case] == "video":
            out = model.get_video_features(
                pv, pid, num_frames_per_video=torch.tensor([pv.shape[0]]), return_dict=True
            ).pooler_output
            full = torch.cat(list(out), dim=0)
        else:
            full = torch.cat(list(model.get_image_features(pv, pid, return_dict=True).pooler_output), dim=0)
    mine = torch.cat(ref.soft_tokens, dim=0)
    diff = (full - mine).abs().max().item()
    log_result("vision_reference_vs_full_model_max_abs", diff, 0.0, case=case)
    assert full.shape == mine.shape and diff == 0.0, f"standalone reference differs from the full model by {diff}"
