# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for the autoregressive text-generation loop (tt/generate.py). Pure torch
# (mock forward_logits_fn) — no device — so the sampling / stage-transition / stopping
# logic is validated independently of the resident backbone. The device adapter
# (make_backbone_logits_fn) is exercised separately on-box.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_generate.py -v -s

import torch

from models.experimental.hunyuan_image_3_0.tt.generate import (
    SamplingConfig,
    generate_text,
    sample_next_token,
)

V = 64


def const_logits_fn(token_logits):
    """forward_logits_fn that always returns the same fixed logits [V]."""
    base = torch.full((V,), -10.0)
    for tok, val in token_logits.items():
        base[tok] = val

    def fn(ids):
        return base.unsqueeze(0).repeat(ids.shape[0], 1)

    return fn


def ramp_logits_fn():
    """Logits that favor `last_token + 1` — produces a counting sequence under greedy."""

    def fn(ids):
        B, _ = ids.shape
        out = torch.full((B, V), -10.0)
        for i in range(B):
            nxt = (int(ids[i, -1].item()) + 1) % V
            out[i, nxt] = 10.0
        return out

    return fn


def test_greedy_counts_up():
    cfg = SamplingConfig(do_sample=False, max_new_tokens=5)
    out = generate_text(ramp_logits_fn(), torch.tensor([[3]]), config=cfg)
    assert out["new_tokens"][0] == [4, 5, 6, 7, 8]
    assert out["sequences"].tolist() == [[3, 4, 5, 6, 7, 8]]


def test_stop_token_halts():
    cfg = SamplingConfig(do_sample=False, max_new_tokens=50)
    out = generate_text(ramp_logits_fn(), torch.tensor([[0]]), config=cfg, final_stop_tokens=[4])
    # counts 1,2,3,4 then stops (4 is a stop token, included)
    assert out["new_tokens"][0] == [1, 2, 3, 4]


def test_sampling_is_seed_reproducible():
    cfg = SamplingConfig(do_sample=True, temperature=1.0, max_new_tokens=8)
    fn = const_logits_fn({10: 2.0, 20: 1.5, 30: 1.0})
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = generate_text(fn, torch.tensor([[0]]), config=cfg, generator=g1)
    b = generate_text(fn, torch.tensor([[0]]), config=cfg, generator=g2)
    assert a["new_tokens"] == b["new_tokens"]


def test_top_k_restricts_support():
    cfg = SamplingConfig(do_sample=True, top_k=1, max_new_tokens=6)
    fn = const_logits_fn({42: 5.0, 7: 4.0, 9: 3.0})
    g = torch.Generator().manual_seed(0)
    out = generate_text(fn, torch.tensor([[0]]), config=cfg, generator=g)
    assert set(out["new_tokens"][0]) == {42}  # top_k=1 => argmax token only


def test_stage_transition_forces_injection():
    # When token 5 is emitted, force [50, 51] before resuming.
    cfg = SamplingConfig(do_sample=False, max_new_tokens=8)

    # forward favors token 5 from prompt token 4; afterwards favors last+1
    def fn(ids):
        B = ids.shape[0]
        out = torch.full((B, V), -10.0)
        for i in range(B):
            last = int(ids[i, -1].item())
            nxt = 5 if last == 4 else (last + 1) % V
            out[i, nxt] = 10.0
        return out

    out = generate_text(
        fn,
        torch.tensor([[4]]),
        config=cfg,
        stage_transitions=[(5, [50, 51])],
        final_stop_tokens=[55],
    )
    gen = out["new_tokens"][0]
    # emits 5, then forced 50, 51, then resumes counting 52, 53, 54, 55(stop)
    assert gen[:3] == [5, 50, 51], gen
    assert gen[3:] == [52, 53, 54, 55], gen


def test_repetition_penalty_demotes_seen():
    fn = const_logits_fn({1: 1.0, 2: 1.0, 3: 1.0})
    ids = torch.tensor([[1, 1, 1]])  # token 1 heavily seen
    logits = fn(ids)
    cfg = SamplingConfig(do_sample=False, repetition_penalty=2.0)
    nxt = sample_next_token(logits, ids, cfg)
    assert int(nxt[0].item()) != 1  # penalized token 1 should not win
