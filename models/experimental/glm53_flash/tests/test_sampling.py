# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host tests for ``tt/sampling.py``: the sampler reference against transformers' logits warpers, the seeded per-slot
``HostSampler``, and the split top-k candidate model of the device path (no device needed)."""

from __future__ import annotations

import json
import os

import pytest
import torch

from models.experimental.glm53_flash.tt.sampling import (
    CHECKPOINT_DEFAULT,
    GREEDY,
    MAX_TOP_K,
    SAMPLER_PLAN,
    TILE,
    VOCAB_SIZE,
    HostSampler,
    SamplingParams,
    candidate_sample_reference,
    greedy,
    mask_padded_vocab,
    nucleus_size,
    probabilities,
    tp_padded_vocab,
    vocab_splits_single_device,
    warp,
)

generation = pytest.importorskip("transformers.generation.logits_process")

SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
)


def hf_warp(logits: torch.Tensor, p: SamplingParams) -> torch.Tensor:
    """transformers' sampling warpers, in the order ``GenerationMixin._get_logits_processor`` builds them."""
    procs = generation.LogitsProcessorList()
    if p.temperature not in (0.0, 1.0):
        procs.append(generation.TemperatureLogitsWarper(p.temperature))
    if p.top_k:
        procs.append(generation.TopKLogitsWarper(top_k=p.top_k, min_tokens_to_keep=1))
    if p.top_p < 1.0:
        procs.append(generation.TopPLogitsWarper(top_p=p.top_p, min_tokens_to_keep=1))
    return procs(torch.zeros(logits.shape[0], 1, dtype=torch.long), logits.float().clone())


def peaked_logits(rows: int, vocab: int, seed: int) -> torch.Tensor:
    """LM-like rows: a few strong candidates over a long tail, plus exact ties (bf16 logits tie often)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, vocab, generator=g) * 2.0
    x[:, :8] += 6.0 * torch.rand(rows, 8, generator=g)
    x[:, 5] = x[:, 3]
    return x.bfloat16().float()


# ---------------------------------------------------------------------------------------------
# SamplingParams
# ---------------------------------------------------------------------------------------------


def test_params_validation(expect_error):
    for bad in (dict(temperature=-0.1), dict(top_k=-1), dict(top_p=0.0), dict(top_p=1.5)):
        with expect_error(ValueError, ""):
            SamplingParams(**bad)


def test_params_greedy_and_device_values():
    assert GREEDY.is_greedy and GREEDY.device_mode == "greedy" and GREEDY.device_values() == (1, 0.0, 1.0)
    assert SamplingParams(temperature=0.7, top_k=1).is_greedy
    p = SamplingParams(temperature=0.7, top_k=5, top_p=0.9)
    assert not p.is_greedy and p.device_mode == "sample"
    assert p.device_values() == (5, 0.9, 0.7)
    assert SamplingParams(temperature=1.0, top_k=500).device_values()[0] == MAX_TOP_K
    assert CHECKPOINT_DEFAULT.device_values() == (MAX_TOP_K, 0.95, 1.0)


@pytest.mark.skipif(not os.path.isfile(os.path.join(SNAPSHOT, "generation_config.json")), reason="no local snapshot")
def test_checkpoint_default_matches_generation_config():
    with open(os.path.join(SNAPSHOT, "generation_config.json")) as f:
        gen = json.load(f)
    p = SamplingParams.from_generation_config(gen, seed=3)
    assert (p.temperature, p.top_k, p.top_p, p.seed) == (
        CHECKPOINT_DEFAULT.temperature,
        CHECKPOINT_DEFAULT.top_k,
        CHECKPOINT_DEFAULT.top_p,
        3,
    )


# ---------------------------------------------------------------------------------------------
# warp / greedy / probabilities vs transformers
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [1.0, 0.6, 1.3])
@pytest.mark.parametrize("top_k", [None, 1, 4, 50])
@pytest.mark.parametrize("top_p", [1.0, 0.95, 0.5, 0.01])
def test_warp_matches_transformers(temperature, top_k, top_p):
    logits = peaked_logits(6, 2048, seed=11)
    p = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p)
    ours, ref = warp(logits, p), hf_warp(logits, p)
    assert torch.equal(torch.isinf(ours), torch.isinf(ref))
    finite = torch.isfinite(ref)
    torch.testing.assert_close(ours[finite], ref[finite], rtol=0, atol=0)
    assert (torch.isfinite(ours).sum(-1) >= 1).all()


def test_warp_single_row_matches_batched():
    logits = peaked_logits(3, 512, seed=12)
    p = SamplingParams(temperature=0.8, top_k=20, top_p=0.9)
    batched = warp(logits, p)
    for r in range(3):
        assert torch.equal(warp(logits[r], p), batched[r])


def test_greedy_ties_resolve_to_lowest_id():
    x = torch.zeros(2, 64)
    x[0, [7, 3, 40]] = 5.0
    x[1, [63, 62]] = 1.0
    assert greedy(x).tolist() == [3, 62]


def test_probabilities():
    logits = peaked_logits(4, 1000, seed=13)
    one_hot = probabilities(logits, GREEDY)
    assert torch.equal(one_hot.argmax(-1), greedy(logits)) and torch.equal(one_hot.sum(-1), torch.ones(4))
    p = SamplingParams(temperature=0.9, top_k=10, top_p=0.9)
    probs = probabilities(logits, p)
    torch.testing.assert_close(probs.sum(-1), torch.ones(4))
    torch.testing.assert_close(probs, hf_warp(logits, p).softmax(-1), rtol=0, atol=0)
    assert ((probs > 0).sum(-1) <= 10).all()


def test_mask_padded_vocab(expect_error):
    x = torch.randn(2, VOCAB_SIZE + 768)
    m = mask_padded_vocab(x)
    assert torch.isneginf(m[:, VOCAB_SIZE:]).all() and torch.equal(m[:, :VOCAB_SIZE], x[:, :VOCAB_SIZE])
    assert mask_padded_vocab(x[:, :VOCAB_SIZE]) is not None
    with expect_error(ValueError, ""):
        mask_padded_vocab(x[:, : VOCAB_SIZE - 1])


def test_nucleus_size():
    x = torch.full((1, 100), -1e4)
    x[0, :3] = torch.tensor([10.0, 9.0, 8.0])
    assert nucleus_size(x, 0.5).item() == 1
    assert nucleus_size(x, 0.99).item() == 3
    assert nucleus_size(x, 1.0).item() == 100


# ---------------------------------------------------------------------------------------------
# HostSampler
# ---------------------------------------------------------------------------------------------


def test_host_sampler_greedy_rows_are_argmax():
    logits = peaked_logits(3, 1000, seed=14)
    s = HostSampler(max_batch=4)
    assert torch.equal(s.sample(logits, [GREEDY] * 3, [0, 1, 2]), greedy(logits))


def test_host_sampler_seeded_is_reproducible_and_slot_local():
    logits = peaked_logits(2, 1000, seed=15)
    p = SamplingParams(temperature=1.0, top_p=0.95, seed=1234)

    def run(order):
        s = HostSampler(max_batch=4)
        for slot in (0, 3):
            s.reset_slot(slot, p.seed)
        return torch.stack([s.sample(logits, [p, p], order) for _ in range(20)])

    a, b = run([0, 3]), run([0, 3])
    assert torch.equal(a, b)
    # the same seed in two slots gives the same stream, whatever row each slot occupies
    assert torch.equal(a[:, 0], run([3, 0])[:, 0])


def test_host_sampler_state_restore():
    logits = peaked_logits(1, 1000, seed=16)
    p = SamplingParams(temperature=1.0)
    s = HostSampler(max_batch=1, base_seed=5)
    state = s.state()
    first = torch.stack([s.sample(logits, [p], [0]) for _ in range(10)])
    s.restore(state)
    assert torch.equal(first, torch.stack([s.sample(logits, [p], [0]) for _ in range(10)]))


def test_host_sampler_distribution():
    """Empirical frequencies follow the warped distribution, and nothing outside its support is ever drawn."""
    logits = torch.tensor([[2.0, 1.5, 1.0, 0.5, -1.0, -3.0, -8.0, -8.0]])
    p = SamplingParams(temperature=0.8, top_k=5, top_p=0.95, seed=0)
    s = HostSampler(max_batch=1)
    s.reset_slot(0, p.seed)
    n = 20000
    toks = torch.stack([s.sample(logits, [p], [0]) for _ in range(n)]).flatten()
    freq = torch.bincount(toks, minlength=8).float() / n
    want = probabilities(logits, p)[0]
    assert (freq[want == 0] == 0).all()
    assert (freq - want).abs().max() < 0.015


def test_host_sampler_rejects_mismatched_batch(expect_error):
    s = HostSampler(max_batch=2)
    with expect_error(ValueError, ""):
        s.sample(torch.zeros(2, 10), [GREEDY], [0, 1])
    with expect_error(ValueError, ""):
        s.sample(torch.zeros(10), [GREEDY], [0])


# ---------------------------------------------------------------------------------------------
# Device algorithm model
# ---------------------------------------------------------------------------------------------


def test_vocab_splits_and_padding():
    assert vocab_splits_single_device(VOCAB_SIZE) == 4
    assert (VOCAB_SIZE // 4) % TILE == 0 and VOCAB_SIZE // 4 <= 64 * 1024
    assert vocab_splits_single_device(1000) == 2 or vocab_splits_single_device(1000) is None
    assert tp_padded_vocab(VOCAB_SIZE, 8) == VOCAB_SIZE
    assert tp_padded_vocab(VOCAB_SIZE + 1, 8) == VOCAB_SIZE + 256
    assert SAMPLER_PLAN["single_chip_path"]["candidates"] == 4 * MAX_TOP_K


@pytest.mark.parametrize("n_shards", [4, 8])
def test_candidate_greedy_is_exact(n_shards):
    logits = peaked_logits(8, VOCAB_SIZE, seed=17)
    logits[1, 100] = logits[1, VOCAB_SIZE - 5] = logits[1].max() + 1  # cross-shard tie -> the lower id
    for r in range(logits.shape[0]):
        tok, _, probs = candidate_sample_reference(logits[r], GREEDY, n_shards=n_shards)
        assert tok.item() == greedy(logits[r]).item()
        assert probs.sum().item() == 1.0
    assert candidate_sample_reference(logits[1], GREEDY, n_shards=n_shards)[0].item() == 100


@pytest.mark.parametrize("top_k,top_p", [(16, 0.9), (32, 1.0), (4, 0.5)])
def test_candidate_distribution_matches_full_vocab_for_small_k(top_k, top_p):
    """top_k <= MAX_TOP_K and no tie at the k-th logit: the candidate set holds every token the full-vocab warp
    keeps, with the same probabilities."""
    g = torch.Generator().manual_seed(18)
    logits = torch.randn(VOCAB_SIZE, generator=g) * 2.0
    logits[:8] += 6.0 * torch.rand(8, generator=g)
    p = SamplingParams(temperature=0.7, top_k=top_k, top_p=top_p)
    _, ids, probs = candidate_sample_reference(logits, p, n_shards=4, generator=torch.Generator().manual_seed(0))
    full = probabilities(logits, p)
    torch.testing.assert_close(probs, full[ids], rtol=1e-5, atol=1e-7)
    assert set(full.nonzero().flatten().tolist()) <= set(ids.tolist())


def test_candidate_truncates_ties_at_the_kth_logit():
    """transformers' TopKLogitsWarper keeps every token tied with the k-th logit; the candidate path keeps exactly k
    (``ttnn.sampling`` semantics). bf16 logits over 154,880 tokens tie often, so this is where "exact for
    top_k <= 32" stops holding."""
    logits = torch.full((VOCAB_SIZE,), -10.0)
    logits[:6] = torch.tensor([5.0, 4.0, 3.0, 2.0, 2.0, 2.0])
    p = SamplingParams(temperature=1.0, top_k=4)
    _, ids, probs = candidate_sample_reference(logits, p, n_shards=4)
    assert (probs > 0).sum().item() == 4
    assert (hf_warp(logits[None], p).isfinite()).sum().item() == 6
    assert (probabilities(logits, p) > 0).sum().item() == 6


def test_candidate_padded_vocab_never_samples_padding(expect_error):
    logits = torch.cat([peaked_logits(1, VOCAB_SIZE, seed=19)[0], torch.full((768,), 100.0)])
    tok, ids, _ = candidate_sample_reference(logits, GREEDY, n_shards=8, vocab_size=VOCAB_SIZE)
    assert tok.item() < VOCAB_SIZE
    with expect_error(ValueError, ""):
        candidate_sample_reference(logits[:-1], GREEDY, n_shards=8)
