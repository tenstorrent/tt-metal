# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): device-sampled verify ids of Qwen36MTPContractForCausalLM (tt/spec_sample.py).

* Draw then match is exact rejection sampling: per-column draws from vLLM's sampling distribution, walked by the
  plugin's own id walk (vllm_tt_plugin.spec_decode.accept_greedy_drafts), commit tokens distributed as the target at
  every position (chi-square), for drafts the target likes, dislikes and rules out, and for rows without drafts.
* sampling_plan: which rows draw, their device encoding, the per-position seeds of a seeded row, the SKIP seed.
* The engine, the model and the class: the plan reaches the verify sampler, draws replace a sampled row's ids only,
  a narrow device-sampled step returns tokens (and the sampled logprobs when asked), and the declarations.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm_tt_plugin.spec_decode import accept_greedy_drafts

import ttnn
from models.common.sampling.generator import MAX_UINT32, _hash_request_seed_to_device_seed
from models.demos.blackhole.qwen36.tests.sampling_reference import chi_square_p, target_probs
from models.demos.blackhole.qwen36.tt import spec_engine
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_sample import (
    SAMPLER_TOP_K,
    enables_logprobs,
    sampled_token_logprobs,
    sampling_plan,
    vocab_chunks,
    with_draws,
)

P_MIN = 1e-4  # every chi-square below is seeded, so a pass is deterministic; the bound only has to clear chance


def _params(**rows):
    """A plugin-style per-row ``sampling_params`` (lists over the rows)."""
    n = max(len(v) for v in rows.values())
    fill = dict(temperature=0.0, top_k=1, top_p=1.0, seed=None, enable_log_probs=False)
    return SimpleNamespace(**{k: list(rows.get(k, [fill[k]] * n)) for k in fill})


# ------------------------------------------------------------------------------ draw then match is rejection sampling


def _walk_draws(probs, drafts, num_valid, n, generator):
    """``n`` verifies of one row: one draw per column from ``probs[j]``, walked as the plugin walks argmax ids."""
    T = len(probs)
    draws = torch.stack([torch.multinomial(p, n, replacement=True, generator=generator) for p in probs], dim=1)
    d = torch.tensor(drafts + [-1] * (T - 1 - len(drafts)), dtype=torch.int64).expand(n, T - 1)
    committed, counts = accept_greedy_drafts(draws, d, torch.full((n,), num_valid, dtype=torch.int32))
    return committed.to(torch.int64), counts.to(torch.int64)


@pytest.mark.parametrize(
    "temperature, top_k, top_p",
    [(1.0, 20, 1.0), (0.7, 8, 0.9), (1.3, 32, 0.5), (1.0, 3, 1.0)],
)
def test_draw_then_match_commits_the_target_distribution(temperature, top_k, top_p):
    """Per position, the committed token is a draw from the target, whatever the drafts: the first committed token
    follows p_0, and given that draft j was accepted, the token after it follows p_(j+1). Drafts: the target's mode, a
    token of small but nonzero mass, a token the filters rule out; with every draft accepted, the bonus follows p_3."""
    g = torch.Generator().manual_seed(int(temperature * 100) + top_k)
    V, n = 64, 120_000
    probs = [target_probs(torch.randn(V, generator=g) * 2.5, temperature, top_k, top_p) for _ in range(4)]
    ranked = [torch.argsort(p, descending=True) for p in probs]
    drafts = [int(ranked[0][0]), int(ranked[1][min(2, top_k - 1)]), int(ranked[2][-1])]
    committed, counts = _walk_draws(probs, drafts, 3, n, g)

    assert chi_square_p(torch.bincount(committed[:, 0], minlength=V), probs[0]) > P_MIN
    for j in range(2):
        given = counts >= j + 2  # draft j accepted, so the committed token at j + 1 exists
        assert int(given.sum()) > 1000
        assert (
            abs(float(given.float().mean()) / float((counts >= j + 1).float().mean()) - float(probs[j][drafts[j]]))
            < 0.01
        )
        assert chi_square_p(torch.bincount(committed[given, j + 1], minlength=V), probs[j + 1]) > P_MIN
    assert float(probs[2][drafts[2]]) == 0 and not bool((counts == 4).any()), "a ruled-out draft is never accepted"


def test_a_row_without_drafts_commits_one_draw_and_every_draft_can_be_the_bonus():
    g = torch.Generator().manual_seed(7)
    V, n = 32, 80_000
    probs = [target_probs(torch.randn(V, generator=g) * 2.0, 0.9, 16, 0.95) for _ in range(4)]
    committed, counts = _walk_draws(probs, [], 0, n, g)
    assert bool((counts == 1).all())
    assert chi_square_p(torch.bincount(committed[:, 0], minlength=V), probs[0]) > P_MIN
    likely = [int(torch.argmax(p)) for p in probs[:3]]
    committed, counts = _walk_draws(probs, likely, 3, n, g)
    full = counts == 4
    assert int(full.sum()) > 1000
    assert chi_square_p(torch.bincount(committed[full, 3], minlength=V), probs[3]) > P_MIN


# ------------------------------------------------------------------------------ sampling_plan


def test_the_plan_draws_every_column_of_a_sampled_row_only():
    sp = _params(
        temperature=[0.7, 0.0, 1.0, 1.0, 0.8],
        top_k=[20, 1, 1, 32, 5],
        top_p=[0.95, 1.0, 1.0, 0.9, 1.0],
        seed=[None, None, None, 1234, None],
    )
    plan = sampling_plan(sp, [100, 50, 10, 7, -1], 4, fresh_seed=iter(range(1, 100)).__next__)
    rows = lambda u: plan.sampled[u * 4 : (u + 1) * 4]
    assert rows(0) == [True] * 4 and rows(3) == [True] * 4
    assert rows(1) == rows(2) == rows(4) == [False] * 4, "greedy, top-k 1 and padding rows keep their argmax ids"
    assert len(plan.sampled) == 20 and plan.any_sampled
    assert plan.k[:4] == [20] * 4 and plan.k[12:16] == [32] * 4 and plan.k[4:8] == [1] * 4
    assert plan.temp[0] == pytest.approx(1 / 0.7) and plan.p[0] == pytest.approx(0.95) and plan.p[4] == 0.0
    # a seeded row draws column j with the device seed of a plain decode at that input position; idle rows skip
    assert plan.seeds[12:16] == [_hash_request_seed_to_device_seed(1234, 7 + j + 1) for j in range(4)]
    assert plan.seeds[:4] == [1, 2, 3, 4], "an unseeded row's columns take fresh, distinct seeds"
    assert set(plan.seeds[4:12] + plan.seeds[16:]) == {MAX_UINT32}
    assert not sampling_plan(_params(temperature=[0.0, 0.0], top_k=[1, 1]), [3, 4], 4).any_sampled


def test_the_plan_refuses_what_the_sampler_cannot_draw(expect_error):
    with expect_error(ValueError, "top_k 64 outside"):
        sampling_plan(_params(temperature=[1.0], top_k=[64]), [5], 4)
    with expect_error(ValueError, "exceed the sampler"):
        sampling_plan(_params(temperature=[1.0] * 9, top_k=[5] * 9), [5] * 9, 4)
    assert SAMPLER_TOP_K == 32 and vocab_chunks(248320) == 4
    with expect_error(ValueError, "tile-aligned"):
        vocab_chunks(248321)


def test_draws_replace_the_sampled_rows_ids_only():
    plan = sampling_plan(_params(temperature=[1.0, 0.0], top_k=[5, 1]), [10, 20], 2)
    ids = torch.tensor([[1, 2], [3, 4]])
    assert with_draws(ids, torch.arange(100, 132), plan).tolist() == [[100, 101], [3, 4]]


def test_sampled_logprobs_are_raw_log_softmax_and_rank():
    logits = torch.tensor([[1.0, 3.0, 3.0, -1.0], [0.0, 0.0, 0.0, 5.0]])
    out = sampled_token_logprobs(logits, torch.tensor([2, 0]))
    assert torch.allclose(out.sampled_logprobs, torch.log_softmax(logits, -1)[[0, 1], [2, 0]])
    assert out.selected_token_ranks.tolist() == [2, 4], "the rank counts every score at or above the token's"
    assert enables_logprobs(_params(temperature=[1.0], enable_log_probs=[False])) is False
    assert enables_logprobs(SimpleNamespace(enable_log_probs=torch.tensor([False, True])))


# ------------------------------------------------------------------------------ engine and model


@pytest.fixture
def engine():
    B, K = 4, 3
    log = []
    model = SimpleNamespace(
        mtp=SimpleNamespace(attention=SimpleNamespace()),
        num_devices=8,
        # MTPBlockEngine builds a compute kernel config from the mesh's arch, which needs the real enum.
        mesh_device=MagicMock(**{"arch.return_value": ttnn.device.Arch.BLACKHOLE}),
        args=SimpleNamespace(vocab_size=32, max_batch_size=B, dim=64),
        layers=[SimpleNamespace(is_full_attention=False, attention=SimpleNamespace())],
        gdn_blocks=SimpleNamespace(verify_tokens=K + 1, set_verify_blocks=lambda s: None),
        _fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt).to(torch.int32),
        spec_feed_rows=lambda h: h,
    )

    def verify_traced(tokens, positions, mi, read_logits=False, page_tables=None, inactive=None, **kw):
        log.append(kw)
        ids = list(range(B * (K + 1)))
        if "sample_plan" in kw:
            ids = with_draws(torch.tensor(ids).reshape(B, K + 1), torch.arange(1000, 1032), kw["sample_plan"])
            ids = ids.reshape(-1).tolist()
        return ids, "feed", None

    model.verify_traced = verify_traced
    eng = spec_engine.MTPBlockEngine(model, B, K)
    eng.captured, eng.nb, eng.block_size, eng._scratch_block = True, 128, 64, 10**6
    eng._active |= {0, 1, 3}
    return eng, log


def _decode(eng, spec_mode, sampling):
    T = eng.T
    tokens = torch.tensor([[1, 7, 8, 0], [2, 9, 9, 9], [0, 0, 0, 0], [5, 6, 1, 2]], dtype=torch.int32)
    start_pos = torch.tensor(
        [[10 + j for j in range(T)], [20 + j for j in range(T)], [-1] * T, [30 + j for j in range(T)]]
    )
    return eng.decode_forward(
        tokens,
        start_pos,
        num_valid_drafts=torch.tensor([2, 3, 0, 3], dtype=torch.int32),
        accepted_counts=torch.ones(eng.B, dtype=torch.int32),
        page_table=torch.ones(eng.B, eng.nb, dtype=torch.int32),
        spec_mode=spec_mode,
        mamba_state="S",
        sampling=sampling,
    )


def test_a_device_sampled_verify_returns_the_draws_of_its_sampled_rows(engine, expect_error):
    eng, log = engine
    sampled = _params(temperature=[1.0, 0.0, 1.0, 0.0], top_k=[20, 1, 20, 1], seed=[None, None, None, None])
    with expect_error(ValueError, "can_sample_on_device"):
        _decode(eng, "argmax_ids", sampled)
    eng.sample_on_device = True
    res = _decode(eng, "argmax_ids", sampled)
    plan = log[-1]["sample_plan"]
    assert plan.sampled[:4] == [True] * 4 and not any(plan.sampled[4:]), "the padding row 2 draws nothing"
    assert res.argmax_ids.tolist() == [[1000, 1001, 1002, -1], [4, 5, 6, 7], [8, -1, -1, -1], [12, 13, 14, 15]]
    # an all-greedy device-sampled step runs no sampler
    _decode(eng, "argmax_ids", _params(temperature=[0.0] * 4, top_k=[1] * 4))
    assert "sample_plan" not in log[-1]


def test_the_model_reads_draws_only_from_a_sampling_trace(expect_error):
    model = SimpleNamespace(_vfy_trace_id=1, _vfy_T=4, _vfy_B=1, _vfy_sampler=None)
    with expect_error(ValueError, "sample=True"):
        Qwen36Model.verify_traced(model, [[1, 2, 3, 4]], [0], [0], sample_plan=object())
