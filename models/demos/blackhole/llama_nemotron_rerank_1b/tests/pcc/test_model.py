# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end reranker scores on device vs the HF fp32 reference."""

import pytest
import torch

from models.demos.blackhole.llama_nemotron_rerank_1b.reference.hf_reference import reference_scores, tokenize_pairs
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model import NemotronReranker
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import RerankConfig

QUERY = "how much protein should a female eat?"
DOCS = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. "
    "But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain.",
    "Calorie intake should not fall below 1,200 a day in women or 1,500 a day in men, except under the supervision "
    "of a health professional.",
    " ".join(
        [
            "Protein needs depend on body weight, activity level and age; athletes may need 1.2 to 2.0 grams per kilogram."
        ]
        * 20
    ),
]
# Largest |device - fp32| logit difference allowed, and the minimum score correlation.
MAX_ABS_ERR = 1.5
MIN_PEARSON = 0.995


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_scores(device, hf_model, tokenizer, state_dict, batch):
    ids = tokenize_pairs(tokenizer, [(QUERY, d) for d in DOCS])
    ref = reference_scores(hf_model, ids)
    model = NemotronReranker(device, RerankConfig(), state_dict, cache_dir=None)
    got = []
    for b in range(0, len(ids), batch):
        got += model.score(ids[b : b + batch])
    err = [abs(a - b) for a, b in zip(got, ref)]
    pear = torch.corrcoef(torch.tensor([got, ref]))[0, 1].item()
    print(
        f"ref {[round(r, 3) for r in ref]}\ntt  {[round(g, 3) for g in got]}\nmax err {max(err):.3f} pearson {pear:.5f}"
    )
    assert sorted(range(len(ref)), key=lambda i: -ref[i]) == sorted(range(len(got)), key=lambda i: -got[i])
    assert max(err) < MAX_ABS_ERR and pear > MIN_PEARSON


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unpadded_batch_runs_unmasked(device, hf_model, tokenizer, state_dict):
    """Pairs that exactly fill their 128-token bucket take the unmasked SDPA path; their scores must match
    the masked path (an all-zero mask) and the HF reference."""
    S = 128
    ids = [x[:S] for x in tokenize_pairs(tokenizer, [(QUERY, d) for d in DOCS]) if len(x) >= S]
    assert ids, "no pair reaches 128 tokens"
    ids = (ids * 4)[:4]
    ref = reference_scores(hf_model, ids)
    model = NemotronReranker(device, RerankConfig(), state_dict, cache_dir=None)
    host, lens = model.prepare_host_inputs(ids, seq_len=S)
    assert not model.needs_mask(lens, S)
    dev = model.to_device(host)
    unmasked = model.scores_from_output(model.forward(*dev, masked=False), lens)
    masked = model.scores_from_output(model.forward(*dev, masked=True), lens)
    print(f"ref {ref}\nunmasked {unmasked}\nmasked {masked}")
    assert max(abs(a - b) for a, b in zip(unmasked, masked)) < 0.1
    assert max(abs(a - b) for a, b in zip(unmasked, ref)) < MAX_ABS_ERR
    assert model.score(ids) == unmasked


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_packed_rows(device, hf_model, tokenizer, state_dict):
    """Several pairs packed back to back into one row score as they do alone: the segment mask keeps each
    pair to its own tokens, and RoPE only sees distances inside a pair."""
    S = 512
    ids = tokenize_pairs(tokenizer, [(QUERY, d) for d in DOCS])
    ref = reference_scores(hf_model, ids)
    model = NemotronReranker(device, RerankConfig(), state_dict, cache_dir=None)
    alone = [model.score([x])[0] for x in ids]
    # first-fit rows of <= 512 tokens, the pairs in input order
    rows = [[]]
    for x in ids:
        if sum(len(y) for y in rows[-1]) + len(x) > S:
            rows.append([])
        rows[-1].append(x)
    assert any(len(r) > 1 for r in rows), "no row holds two pairs"
    host, lens = model.prepare_packed_inputs(rows, S)
    assert model.needs_mask(lens, S)
    dev = model.to_device(host)
    packed = model.scores_from_output(model.forward(*dev), lens)
    print(f"rows {[len(r) for r in rows]}\nref {ref}\nalone {alone}\npacked {packed}")
    # Packed and alone are both bf16 approximations of HF (shapes and offsets differ), so each is checked
    # against HF; packed must be no further from it than alone, within the model's own tolerance.
    err_packed = max(abs(a - b) for a, b in zip(packed, ref))
    err_alone = max(abs(a - b) for a, b in zip(alone, ref))
    assert err_packed < MAX_ABS_ERR and err_packed < err_alone + 0.25
    assert sorted(range(len(ref)), key=lambda i: -ref[i]) == sorted(range(len(packed)), key=lambda i: -packed[i])
