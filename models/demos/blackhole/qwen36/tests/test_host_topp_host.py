# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of tt/host_topp.py against vLLM's apply_top_k_top_p and, when importable with _top_k_candidates, the
vllm-tt-plugin host sampler.

When topp_candidates answers, its kept tokens and their probabilities equal vLLM's full-vocabulary path; otherwise it
returns None and the full path runs.
"""

import sys
import types

import pytest
import torch

from models.demos.blackhole.qwen36.tt import host_topp

vllm_ops = pytest.importorskip("vllm.v1.sample.ops.topk_topp_sampler")

V = 248320


def _rows(vocab, seed=1):
    """Flat to peaked rows, and a Zipf-shaped row, all without ties."""
    g = torch.Generator().manual_seed(seed)
    rows = [torch.randn(vocab, generator=g) * scale for scale in (1.0, 3.0, 8.0, 15.0)]
    zipf = -torch.log(torch.arange(1, vocab + 1, dtype=torch.float64)) * 1.2
    rows.append((zipf[torch.randperm(vocab, generator=g)] * 4.0).float())
    return [r.reshape(1, -1).float() for r in rows]


def _full_path(logits, p):
    """vLLM's filtered logits for top_k disabled and per-row top_p ``p``."""
    k = torch.full((logits.shape[0],), logits.shape[1])
    return vllm_ops.apply_top_k_top_p_pytorch(
        logits.clone(), k, torch.as_tensor(p, dtype=torch.float32), allow_cpu_sync=True
    )


def _check_equal(logits, p, cand):
    """The candidates keep vLLM's tokens with vLLM's probabilities."""
    values, ids = cand
    full = _full_path(logits, p)
    for row in range(logits.shape[0]):
        kept = torch.isfinite(values[row])
        assert set(ids[row][kept].tolist()) == set(torch.nonzero(torch.isfinite(full[row])).reshape(-1).tolist())
        got = values[row : row + 1].softmax(-1, dtype=torch.float32)[0, kept]
        want = full[row : row + 1].softmax(-1, dtype=torch.float32)[0, ids[row][kept]]
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("vocab", [12000, V])
@pytest.mark.parametrize("p", [0.5, 0.8, 0.9, 0.95, 0.99])
def test_kept_set_and_probabilities_match_vllm(vocab, p):
    used = 0
    for logits in _rows(vocab):
        cand = host_topp.topp_candidates(logits, None, torch.tensor([p]))
        if cand is not None:
            used += 1
            _check_equal(logits, [p], cand)
    assert used >= 2, f"the candidates served only {used} rows"


def test_batch_with_per_row_top_p_matches_vllm():
    logits = torch.cat(_rows(V)[2:])
    p = [0.9, 0.95, 0.8]
    cand = host_topp.topp_candidates(logits, torch.full((3,), V), torch.tensor(p))
    assert cand is not None
    _check_equal(logits, p, cand)


def test_input_not_modified():
    logits = _rows(V)[3]
    before = logits.clone()
    assert host_topp.topp_candidates(logits, None, torch.tensor([0.95])) is not None
    assert torch.equal(logits, before)


def test_sampled_frequencies_match_full_path():
    logits = _rows(12000, seed=3)[3]
    p = 0.9
    values, ids = host_topp.topp_candidates(logits, None, torch.tensor([p]))
    draws = 20000
    torch.manual_seed(0)
    probs = values.softmax(-1, dtype=torch.float32).expand(draws, -1).contiguous()
    tokens = ids[0][vllm_ops.random_sample(probs, {})]
    counts = torch.bincount(tokens, minlength=logits.shape[1]).double() / draws
    want = _full_path(logits, [p]).softmax(-1, dtype=torch.float64)[0]
    assert set(torch.nonzero(counts).reshape(-1).tolist()) <= set(torch.nonzero(want).reshape(-1).tolist())
    bound = 5.0 * (want * (1.0 - want) / draws).sqrt() + 1e-4
    assert bool(((counts - want).abs() <= bound).all())


def _probs_to_logits(probs, vocab):
    """Logits whose softmax has ``probs`` on the first tokens and spreads the rest over distinct small logits."""
    head = torch.tensor(probs, dtype=torch.float64)
    rest = vocab - head.numel()
    tail = torch.linspace(0.0, 1e-3, rest, dtype=torch.float64).exp()
    tail = tail / tail.sum() * (1.0 - head.sum())
    return torch.cat([head, tail]).log().float().reshape(1, -1)


def test_cut_within_margin_takes_full_path():
    logits = _probs_to_logits([0.6, 0.3], 4096)
    margin = host_topp.TOPP_MARGIN
    assert host_topp.topp_candidates(logits, None, torch.tensor([0.6 + margin / 2])) is None
    assert host_topp.topp_candidates(logits, None, torch.tensor([0.6 - margin / 2])) is None
    cand = host_topp.topp_candidates(logits, None, torch.tensor([0.6 + 10 * margin]))
    assert cand is not None
    _check_equal(logits, [0.6 + 10 * margin], cand)


def test_tie_across_cut_takes_full_path():
    logits = _probs_to_logits([0.5, 0.2, 0.2], 4096)
    assert logits[0, 1] == logits[0, 2]
    assert host_topp.topp_candidates(logits, None, torch.tensor([0.6])) is None
    # the tie inside the kept set does not matter
    cand = host_topp.topp_candidates(logits, None, torch.tensor([0.85]))
    assert cand is not None
    _check_equal(logits, [0.85], cand)


def test_declines():
    logits = torch.cat(_rows(V)[2:4])
    f = host_topp.topp_candidates
    assert f(logits, torch.tensor([20, V]), torch.tensor([0.9, 0.9])) is None  # a row with top_k
    assert f(logits, None, None) is None
    assert f(logits, torch.full((2,), V), None) is None
    assert f(logits, None, torch.tensor([1.0, 0.9])) is None  # a row without top_p sorts in vLLM
    assert f(logits, None, torch.tensor([0.0, 0.9])) is None
    assert f(logits, None, torch.tensor([0.9])) is None  # wrong length
    assert f(logits[:0], None, torch.ones(0)) is None
    assert f(logits[0], None, torch.tensor([0.9])) is None
    assert f(torch.zeros(1, V), None, torch.tensor([0.5])) is None  # every logit tied
    assert f(_rows(V)[0] * 0.01, None, torch.tensor([0.99])) is None  # the cut is past the candidates


class _FakeSampler:
    logprobs_mode = "raw_logprobs"
    calls = 0

    def _top_k_candidates(self, logits, k, p):
        type(self).calls += 1
        return None


def _fake_class(name, **attrs):
    return type(name, (_FakeSampler,), dict(attrs, calls=0))


def test_install_wraps_top_k_candidates():
    cls = _fake_class("Sampler")
    assert host_topp.install_plugin_topp_candidates(cls)
    assert host_topp.install_plugin_topp_candidates(cls)  # idempotent: one wrapper
    assert cls._top_k_candidates.__wrapped__ is _FakeSampler._top_k_candidates
    logits = _rows(V)[3]
    sampler = cls()
    cand = sampler._top_k_candidates(logits, None, torch.tensor([0.95]))
    assert cand is not None and cls.calls == 1
    _check_equal(logits, [0.95], cand)
    sampler.logprobs_mode = "processed_logprobs"
    assert sampler._top_k_candidates(logits, None, torch.tensor([0.95])) is None


def test_install_keeps_plugin_top_k_result():
    sentinel = (torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.long))
    cls = _fake_class("Sampler", _top_k_candidates=lambda self, logits, k, p: sentinel)
    assert host_topp.install_plugin_topp_candidates(cls)
    assert cls()._top_k_candidates(_rows(V)[3], None, torch.tensor([0.95])) is sentinel


def test_install_skips_sampler_without_top_k_candidates():
    cls = type("Sampler", (), {"logprobs_mode": "raw_logprobs"})
    assert not host_topp.install_plugin_topp_candidates(cls)
    assert not hasattr(cls, "_top_k_candidates")


def test_install_skips_plugin_with_own_top_p_candidates(monkeypatch):
    module = types.ModuleType("fake_host_sampler")
    module._sample_top_p_candidates = lambda logits, k, p: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cls = _fake_class("Sampler")
    cls.__module__ = module.__name__
    orig = cls._top_k_candidates
    assert not host_topp.install_plugin_topp_candidates(cls)
    assert cls._top_k_candidates is orig


def test_plugin_sampler_uses_candidates(monkeypatch):
    hs = pytest.importorskip("vllm_tt_plugin.host_sampler")
    cls = hs.TTTopKTopPSampler
    if not hasattr(cls, "_top_k_candidates"):
        pytest.skip("this vllm-tt-plugin has no TTTopKTopPSampler._top_k_candidates")
    # registers the restore of the class attribute
    monkeypatch.setattr(cls, "_top_k_candidates", cls._top_k_candidates)
    assert host_topp.install_plugin_topp_candidates()

    def full_path(*args, **kwargs):
        raise AssertionError("the candidates must serve this batch")

    monkeypatch.setattr(hs.TopKTopPSampler, "forward_native", full_path)
    sampler = hs.TTTopKTopPSampler()
    logits = _rows(V)[3]
    p = torch.tensor([0.9])
    kept = torch.isfinite(_full_path(logits, [0.9])[0])
    torch.manual_seed(0)
    for _ in range(50):
        tokens, scores = sampler.forward_native(logits.clone(), {}, None, p)
        assert scores is None and bool(kept[tokens].all())
