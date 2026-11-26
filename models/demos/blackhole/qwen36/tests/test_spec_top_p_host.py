# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): top-p with top-k off on the verify sampler (tt/spec_top_p.py,
vllm-tt-plugin supports_spec_device_top_p).

* The certificate rule: where q - top_p >= CERT_EPS, the nucleus cut inside the top-32 at 1 - top_p / q (the float32
  arithmetic of the certificate trace, mirrored on the host) keeps exactly the tokens of vLLM's full-vocabulary nucleus
  in (value descending, id ascending) order, ties at the cut included, with their probabilities; a row with a nucleus
  past the top-32 fails.
* The host redo draws vLLM's distribution (chi-square over seeds) and repeats for a seed.
* The engine: a step with a certificate row draws it with top-k 32 and top-p 1 under the certificate trace, runs no
  device acceptance and drafter, and its read redoes a failed row from the step's logits (with exact logprobs).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.sampling_reference import chi_square_p, target_probs
from models.demos.blackhole.qwen36.tt import spec_engine
from models.demos.blackhole.qwen36.tt.spec_logprobs import exact_logprobs
from models.demos.blackhole.qwen36.tt.spec_sample import SAMPLER_TOP_K, sampling_plan, with_draws
from models.demos.blackhole.qwen36.tt.spec_top_p import (
    CERT_EPS,
    CertificateRows,
    certificate_cut,
    certificate_rows,
    failed_rows,
    redo_draws,
    row_distribution,
    sampler_params,
)

P_MIN = 1e-4


def _logits(g, V, kind):
    scale = {"random": 2.5, "peaked": 1.5, "flat": 0.4, "tied": 2.0}[kind]
    x = torch.randn(V, generator=g) * scale
    if kind == "tied":
        x = torch.round(x / 0.5) * 0.5
    if kind == "peaked":
        idx = torch.randint(0, V, (5,), generator=g)
        x[idx] += torch.tensor([12.0, 10.0, 9.0, 8.0, 7.0])
    return x.to(torch.bfloat16).float()


def _certificate_distribution(x, T, p, M=SAMPLER_TOP_K):
    """The certificate trace's arithmetic on one row, float32: (ids [M] in (value desc, id asc) order, prob [M] float64
    over the kept prefix, q, fail, the margin to the nearest cut decision)."""
    v = x.to(torch.float32)
    order = torch.sort(v, descending=True, stable=True).indices[:M]
    top = v[order]
    t = 1.0 / T
    e = torch.exp((top - top[0]) * t)
    full = torch.exp((v - top[0]) * t).sum()
    q, omp, fail = certificate_cut(e.sum(), full, torch.tensor(p))
    cum = (e / e.sum()).flip(0).cumsum(0).flip(0)
    drop = cum <= omp
    drop[0] = False
    prob = torch.where(~drop, torch.exp((top.double() - top[0].double()) * t), torch.zeros(M, dtype=torch.float64))
    return order, prob / prob.sum(), float(q), bool(fail), float((cum - omp).abs().min())


def _nucleus(x, T, p):
    """vLLM's full-vocabulary nucleus in (value desc, id asc) order, dense [V] float64 (``row_distribution``)."""
    ids, prob = row_distribution(x, T, 0, p)
    return torch.zeros(x.numel(), dtype=torch.float64).index_put((ids,), prob)


@pytest.mark.parametrize("kind", ["random", "peaked", "flat", "tied"])
def test_a_passing_certificate_keeps_vllms_full_vocabulary_nucleus(kind):
    g = torch.Generator().manual_seed({"random": 1, "peaked": 2, "flat": 3, "tied": 4}[kind])
    V = 3000
    passed = failed = 0
    for _ in range(40):
        x = _logits(g, V, kind)
        for p in (0.5, 0.8, 0.9, 0.95, 0.99):
            for T in (0.3, 0.6, 1.0, 1.5):
                order, prob, q, fail, margin = _certificate_distribution(x, T, p)
                want = _nucleus(x, T, p)
                fits = float(want[order].sum()) > 1 - 1e-9
                if fail:
                    failed += 1
                    continue
                if margin < 1e-5:
                    continue  # a cut decision within float32 error
                passed += 1
                assert fits, f"{kind} T={T} p={p}: the certificate passed (q={q}) with a nucleus past the top-32"
                got = torch.zeros(V, dtype=torch.float64)
                got[order] = prob
                assert torch.allclose(got, want, atol=1e-6), f"{kind} T={T} p={p}: the device nucleus differs"
    if kind == "flat":
        assert failed > 0, "a flat row's nucleus reaches past the top-32"
    else:
        assert passed > 0


def test_certificate_rows_and_the_sampler_parameters_of_their_step():
    V = 1000
    params = SimpleNamespace(
        temperature=[1.0, 0.7, 0.0, 0.8, 0.9],
        top_k=[V, 20, 1, 0, V],
        top_p=[0.9, 0.9, 1.0, 0.95, 1.0],
        seed=[None, 3, None, 4, None],
    )
    cert = certificate_rows(params, 5, V)
    assert cert.rows == [True, False, False, True, False] and cert.top_p[0] == 0.9 and cert.temperature[3] == 0.8
    sp = sampler_params(params, cert)
    assert sp.top_k == [32, 20, 1, 32, V] and sp.top_p == [1.0, 0.9, 1.0, 1.0, 1.0]
    assert sp.temperature == [1.0, 0.7, 0.0, 1.0, 0.9] and sp.seed is params.seed
    plan = sampling_plan(sp, [10, 20, 30, 40, -1], 2)
    assert plan.sampled == [True] * 4 + [False] * 2 + [True] * 2 + [False] * 2
    assert certificate_rows(SimpleNamespace(temperature=[1.0], top_k=[20], top_p=[0.9], seed=[None]), 1, V) is None
    assert sampler_params(params, None) is params


def test_failed_rows_are_certificate_rows_below_the_margin():
    cert = CertificateRows(rows=[True, False], top_p=[0.9, 1.0], temperature=[1.0, 1.0])
    q = torch.tensor([0.95, 0.9 + CERT_EPS / 2, 0.5, 0.5])
    assert failed_rows(q, cert, 2).tolist() == [False, True, False, False]


def test_a_redo_draws_vllms_distribution_and_repeats_for_a_seed():
    g = torch.Generator().manual_seed(9)
    V, T, N = 400, 2, 3000
    x = _logits(g, V, "flat").reshape(1, V).expand(2, V).contiguous()
    cert = CertificateRows(rows=[True], top_p=[0.9], temperature=[0.8])
    fail = torch.tensor([False, True])
    draws = [int(redo_draws(torch.zeros(1, T), x, fail, cert, [0, s], T)[0, 1]) for s in range(N)]
    assert chi_square_p(torch.bincount(torch.tensor(draws), minlength=V), target_probs(x[1], 0.8, 0, 0.9)) > P_MIN
    again = redo_draws(torch.tensor([[7, 8]]), x, fail, cert, [0, 123], T)
    assert int(again[0, 0]) == 7 and torch.equal(again, redo_draws(torch.tensor([[7, 8]]), x, fail, cert, [0, 123], T))
    ids, prob = row_distribution(x[1], 0.8, 0, 0.9)
    assert torch.allclose(torch.zeros(V, dtype=torch.float64).index_put((ids,), prob), target_probs(x[1], 0.8, 0, 0.9))


# ------------------------------------------------------------------------------ the engine


@pytest.fixture
def engine(monkeypatch):
    """MTPBlockEngine over a model whose deferred verify records its calls; the device reads are host tensors."""
    B, K, V = 2, 3, 64
    T = K + 1
    log = {}
    g = torch.Generator().manual_seed(5)
    full = torch.randn(B * T, V, generator=g) * 0.3  # flat rows: their nuclei reach past 32 tokens
    full[:T] = torch.randn(T, V, generator=g) * 4.0  # user 0: peaked rows
    model = SimpleNamespace(
        mtp=SimpleNamespace(attention=SimpleNamespace()),
        num_devices=8,
        mesh_device=MagicMock(**{"arch.return_value": ttnn.device.Arch.BLACKHOLE}),
        args=SimpleNamespace(vocab_size=V, max_batch_size=B, dim=64),
        layers=[SimpleNamespace(is_full_attention=False, attention=SimpleNamespace())],
        gdn_blocks=SimpleNamespace(verify_tokens=T, set_verify_blocks=lambda s: None),
        _fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt).to(torch.int32),
        spec_feed_rows=lambda h: h,
        _vfy_rows_out="feed",
        _vfy_ids_out=torch.arange(32, dtype=torch.int64).reshape(1, 1, 1, 32),
        _vfy_sampler=SimpleNamespace(draws=torch.full((1, 1, 1, 32), 3, dtype=torch.int64)),
        _vfy_top_p=SimpleNamespace(q=torch.tensor([0.99] * T + [0.5] * T + [1.0] * (32 - 2 * T))),
        verify_logits_source=lambda bucket=None: (full, False),
    )

    def verify_enqueue(*args, **kw):
        log["enqueue"] = kw

    model.verify_enqueue = verify_enqueue
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda t: [t])
    monkeypatch.setattr(ttnn, "from_device", lambda t, blocking=True: t)
    monkeypatch.setattr(ttnn, "record_event", lambda mesh, cq_id: None)
    monkeypatch.setattr(ttnn, "event_synchronize", lambda event: None)
    monkeypatch.setattr(ttnn, "to_torch", lambda t, **kw: t)
    eng = spec_engine.MTPBlockEngine(model, B, K)
    eng.captured, eng.nb, eng.block_size, eng._scratch_block = True, 128, 64, 10**6
    eng.sample_on_device = True
    eng._active |= {0, 1}
    return eng, log, full


def test_a_failed_certificate_row_is_drawn_again_from_the_steps_logits(engine):
    eng, log, full = engine
    T, V = eng.T, full.shape[1]
    params = SimpleNamespace(temperature=[0.9, 1.0], top_k=[V, V], top_p=[0.9, 0.8], seed=[11, None])
    res = eng.decode_forward(
        torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32),
        torch.tensor([[10 + j for j in range(T)], [20 + j for j in range(T)]]),
        num_valid_drafts=torch.tensor([3, 3], dtype=torch.int32),
        accepted_counts=torch.ones(2, dtype=torch.int32),
        page_table=torch.ones(2, 128, dtype=torch.int32),
        spec_mode="argmax_ids",
        mamba_state="S",
        sampling=params,
        logprobs=False,
    )
    pending = res.argmax_ids
    kw = log["enqueue"]
    assert "accept_rows" not in kw, "a certificate step runs no device acceptance and drafter"
    assert kw["top_p"].rows == [True, True] and kw["sample_plan"].k[: 2 * T] == [SAMPLER_TOP_K] * (2 * T)
    assert kw["sample_plan"].p[: 2 * T] == [1.0] * (2 * T)
    assert kw["sample_plan"].temp[: 2 * T] == [1.0] * (2 * T), "certificate rows draw at temperature 1"
    ids = pending.result()
    # user 0 passes (q = 0.99 > 0.9): the device draws; user 1 fails (q = 0.5 < 0.8): each column drawn again.
    assert ids[0].tolist() == [3, 3, 3, 3]
    assert pending.redone.tolist() == [False] * T + [True] * T
    seeds = kw["sample_plan"].seeds
    want = redo_draws(torch.full((2, T), 3), full, pending.redone, kw["top_p"], seeds, T)
    assert torch.equal(ids.to(torch.int64), want)
    for j in range(T):
        assert float(target_probs(full[T + j], 1.0, 0, 0.8)[int(ids[1, j])]) > 0


def test_a_redone_rows_logprobs_come_from_the_full_logits(engine):
    eng, log, full = engine
    T, V = eng.T, full.shape[1]
    params = SimpleNamespace(temperature=[0.9, 1.0], top_k=[V, V], top_p=[0.9, 0.8], seed=[11, 12])
    model = eng.model
    cand = torch.sort(full, dim=1, descending=True, stable=True)
    rows = 2 * T
    model._vfy_logprobs = SimpleNamespace(
        outputs=[
            cand.values[:, :SAMPLER_TOP_K].reshape(1, 1, rows, -1),
            cand.indices[:, :SAMPLER_TOP_K].reshape(1, 1, rows, -1),
            torch.exp(full - full.max(dim=1, keepdim=True).values).sum(1).reshape(1, 1, rows, 1),
        ]
    )
    res = eng.decode_forward(
        torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32),
        torch.tensor([[10 + j for j in range(T)], [20 + j for j in range(T)]]),
        num_valid_drafts=torch.tensor([3, 3], dtype=torch.int32),
        accepted_counts=torch.ones(2, dtype=torch.int32),
        page_table=torch.ones(2, 128, dtype=torch.int32),
        spec_mode="argmax_ids",
        mamba_state="S",
        sampling=params,
        logprobs=True,
    )
    pending = res.argmax_ids
    # the device's draws for user 0 (token 3 at every column) must be among its candidates
    model._vfy_sampler.draws[0, 0, 0, :T] = cand.indices[:T, 0]
    ids = pending.result()
    lp, rank, top_lp, top_ids = pending.logprobs_result()
    for j in range(T):
        want = exact_logprobs(full[T + j], ids[1, j])
        assert torch.allclose(lp[1, j], want[0]) and int(rank[1, j]) == int(want[1])
        assert torch.equal(top_ids[1, j], want[3])
    assert torch.allclose(
        lp[0].double(), torch.log_softmax(full[:T].double(), -1).gather(1, ids[0, :, None].long())[:, 0]
    )


def test_no_prefill_runs_behind_an_unread_certificate_verify(engine):
    eng, log, full = engine
    T, V = eng.T, full.shape[1]
    params = SimpleNamespace(temperature=[0.9, 1.0], top_k=[V, V], top_p=[0.9, 0.8], seed=[11, None])
    eng.decode_forward(
        torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32),
        torch.tensor([[10 + j for j in range(T)], [20 + j for j in range(T)]]),
        num_valid_drafts=torch.tensor([3, 3], dtype=torch.int32),
        accepted_counts=torch.ones(2, dtype=torch.int32),
        page_table=torch.ones(2, 128, dtype=torch.int32),
        spec_mode="argmax_ids",
        mamba_state="S",
        sampling=params,
        defer=True,
    )
    with pytest.raises(ValueError, match="certificate rows"):
        eng.prefill_over_pending([5])


def test_the_plan_of_a_step_without_certificate_rows_is_unchanged():
    params = SimpleNamespace(temperature=[0.9, 0.0], top_k=[20, 1], top_p=[0.9, 1.0], seed=[1, None])
    plan = sampling_plan(sampler_params(params, certificate_rows(params, 2, 1000)), [5, 6], 2)
    assert plan.k[:2] == [20, 20] and plan.p[0] == pytest.approx(0.9, abs=1e-3)
    assert with_draws(torch.zeros(2, 2), torch.arange(4), plan).tolist() == [[0, 1], [0, 0]]
