# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free checks of tt/host_sampling.py against vLLM's penalty semantics and seeding contract."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.demos.laguna.tt import host_sampling as H


def _vllm_reference(logits, prompt, output, presence, frequency, repetition):
    """vLLM's torch reference (vllm/_custom_ops.apply_repetition_penalties_torch + layers.utils)."""
    from vllm.model_executor.layers.utils import get_token_bin_counts_and_mask

    logits = logits.clone()
    rows, vocab = logits.shape
    pad = lambda t: torch.where(t < 0, torch.full_like(t, vocab), t)  # plugin -1 -> vLLM's vocab padding
    _, prompt_mask = get_token_bin_counts_and_mask(pad(prompt), vocab, rows)
    counts, output_mask = get_token_bin_counts_and_mask(pad(output), vocab, rows)
    rep = repetition.unsqueeze(1).repeat(1, vocab)
    penalties = torch.where(prompt_mask | output_mask, rep, 1.0)
    logits *= torch.where(logits > 0, 1.0 / penalties, penalties)
    logits -= frequency.unsqueeze(1) * counts
    logits -= presence.unsqueeze(1) * output_mask
    return logits


def test_penalties_match_vllm_reference():
    g = torch.Generator().manual_seed(0)
    rows, vocab = 3, 50
    logits = torch.randn(rows, vocab, generator=g) * 3
    prompt = torch.tensor([[1, 2, 2, -1], [5, 6, 7, 8], [-1, -1, -1, -1]])
    output = torch.tensor([[2, 3, 3, -1, -1], [9, 9, 9, 1, -1], [4, -1, -1, -1, -1]])
    presence = torch.tensor([0.5, 0.0, 1.0])
    frequency = torch.tensor([0.25, 0.75, 0.0])
    repetition = torch.tensor([1.3, 1.0, 2.0])
    got = H.apply_penalties(logits, prompt, output, presence, frequency, repetition)
    want = _vllm_reference(logits, prompt, output, presence, frequency, repetition)
    assert torch.allclose(got, want, atol=1e-6)


def test_penalties_active_detection():
    neutral = SimpleNamespace(repetition_penalty=torch.ones(4), presence_penalty=torch.zeros(4), frequency_penalty=torch.zeros(4))
    assert not H.penalties_active(neutral)
    assert not H.penalties_active(None)
    for name, value in (("repetition_penalty", 1.2), ("presence_penalty", 0.5), ("frequency_penalty", -0.5)):
        sp = SimpleNamespace(**{k: v.clone() for k, v in vars(neutral).items()})
        getattr(sp, name)[2] = value
        assert H.penalties_active(sp)


def test_greedy_rows_take_argmax_and_seeded_rows_reproduce():
    logits = torch.zeros(3, 10)
    logits[:, 7] = 5.0
    logits[:, 3] = 4.9
    sp = SimpleNamespace(
        temperature=torch.tensor([0.0, 1.0, 1.0]),
        top_k=torch.tensor([0, 0, 0]),
        top_p=torch.tensor([1.0, 1.0, 1.0]),
        seed=torch.tensor([-1, 1234, 1234]),
        presence_penalty=torch.zeros(3),
        frequency_penalty=torch.zeros(3),
        repetition_penalty=torch.ones(3),
    )
    a = H.sample_penalized(logits, sp, None, None, torch.tensor([10, 10, 10]))
    b = H.sample_penalized(logits, sp, None, None, torch.tensor([10, 10, 10]))
    assert a[0] == 7  # greedy
    assert a[1] == a[2] == b[1] == b[2]  # same seed + position -> same token


def test_top_k_one_is_argmax_and_top_p_keeps_the_top_token():
    logits = torch.tensor([0.0, 3.0, 2.9, -1.0])
    for step in range(20):
        assert H.sample_row(logits, temperature=1.0, top_k=1, top_p=1.0, seed=None, step=step) == 1
        assert H.sample_row(logits, temperature=1.0, top_k=0, top_p=1e-6, seed=None, step=step) == 1


def test_unseeded_rows_vary():
    logits = torch.zeros(64)
    draws = {H.sample_row(logits, temperature=1.0, top_k=0, top_p=1.0, seed=None, step=5) for _ in range(20)}
    assert len(draws) > 1


def test_repetition_penalty_changes_greedy_choice():
    logits = torch.tensor([[1.0, 0.95, 0.1]])
    sp = SimpleNamespace(
        temperature=torch.tensor([0.0]), top_k=torch.tensor([0]), top_p=torch.tensor([1.0]), seed=torch.tensor([-1]),
        presence_penalty=torch.zeros(1), frequency_penalty=torch.zeros(1), repetition_penalty=torch.tensor([1.5]),
    )
    out = H.sample_penalized(logits, sp, torch.tensor([[-1]]), torch.tensor([[0, 0]]), torch.tensor([3]))
    assert out[0] == 1  # token 0 was generated before; 1.0/1.5 < 0.95


def test_device_sampler_treats_plugin_seed_sentinel_as_unseeded():
    # Regression: -1 (the plugin's "no seed") was used as a literal seed, so unseeded requests were
    # deterministic and the plugin's no-seed variety test failed.
    from models.demos.laguna.tt.generator_vllm import LagunaForCausalLM

    sp = SimpleNamespace(temperature=torch.tensor([1.0]), top_k=torch.tensor([0]), top_p=torch.tensor([1.0]),
                         seed=torch.tensor([-1]))
    seeds = {LagunaForCausalLM._sampling_row_params(sp, 0)[3] for _ in range(10)}
    assert len(seeds) > 1 and all(s >= 0 for s in seeds)
    sp.seed = torch.tensor([77])
    assert LagunaForCausalLM._sampling_row_params(sp, 0)[3] == 77


def test_device_sampler_receives_inverse_temperature():
    # Regression: the device sampler scales logits by its temp input (1/T, as format_sampling_params passes);
    # Laguna passed T, so T=2.0 sampled like T=0.5.
    from models.demos.laguna.tt.generator_vllm import LagunaForCausalLM

    for temperature, expected in ((2.0, 0.5), (0.5, 2.0), (1.0, 1.0)):
        sp = SimpleNamespace(temperature=torch.tensor([temperature]), top_k=torch.tensor([10]),
                             top_p=torch.tensor([1.0]), seed=torch.tensor([3]))
        k, p, inv_t, s = LagunaForCausalLM._sampling_row_params(sp, 0)
        assert (k, p, s) == (10, 1.0, 3)
        assert inv_t == pytest.approx(expected)
    greedy = SimpleNamespace(temperature=torch.tensor([0.0]), top_k=torch.tensor([0]), top_p=torch.tensor([1.0]),
                             seed=torch.tensor([-1]))
    assert LagunaForCausalLM._sampling_row_params(greedy, 0) == (1, 1.0, 1.0, 0)
