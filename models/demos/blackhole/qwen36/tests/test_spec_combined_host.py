# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): one deferred verify whose rows combine the device sampling features of
Qwen36MTPContractForCausalLM (vllm-tt-plugin ``supports_spec_device_mask`` with ``supports_spec_device_logprobs``,
``supports_spec_device_penalties`` and ``supports_spec_device_top_p``).

A token-masked verify (``decode_forward(defer_sampling=True)``) enqueues its forward at submission and the rest at
``sample_deferred``: the mask, the penalties, the sampler (or the certificate sampler) and the logprobs trace, in that
order (``Qwen36Model.verify_enqueue_post``).

* Logprobs: the payload describes the logits the draws came from: an unmasked, unpenalized row's payload is its raw
  logits', and a masked or penalized row's draw is among its edited candidates (the payload of such a row is not
  published: the plugin verifies a masked or penalized row that asks for logprobs on the ``logits`` path).
* Penalties after the mask: vLLM masks to -inf and then penalizes, which leaves a masked token at -inf; on the device
  a masked logit stays far below every allowed one, so the order changes no draw. Penalizing after the mask makes the
  float32 argmax that a greedy penalty row takes the masked one, so such a row does not draw (a top-k 1 draw over the
  bfloat16 penalized logits could break a float32 near-tie the other way).
* Certificate rows (top-k off, top_p < 1) on a deferred verify: the certificate sampler replaces the sampler at
  sample_deferred, the verify runs no device acceptance and drafter, and the logits a failed row is drawn again from
  are read after the mask and the penalties edited them, so the redo draws from the same masked, penalized target.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_step_host import _bare_verify_model
from models.demos.blackhole.qwen36.tt import spec_engine
from models.demos.blackhole.qwen36.tt.spec_mask import MASK_NEG, WORD_BITS, allowed_tokens, mask_words
from models.demos.blackhole.qwen36.tt.spec_sample import SAMPLER_TOP_K

B, K, V, SHARDS = 2, 3, 64, 2
T = K + 1


def _pack(allowed):
    """Plugin packing int32 [R, ceil(V/32)] of a bool [R, V]."""
    R, V_ = allowed.shape
    words = mask_words(V_)
    bits = torch.zeros(R, words * WORD_BITS, dtype=torch.int64)
    bits[:, :V_] = allowed.to(torch.int64)
    packed = (bits.reshape(R, words, WORD_BITS) << torch.arange(WORD_BITS)).sum(-1)
    return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)


def _mask(constrained):
    """[B, T, W] plugin mask: -1 everywhere but {(u, j): allowed ids}."""
    m = torch.full((B, T, mask_words(V)), -1, dtype=torch.int32)
    for (u, j), allowed in constrained.items():
        bits = torch.zeros(1, V, dtype=torch.bool)
        bits[0, list(allowed)] = True
        m[u, j] = _pack(bits)[0]
    return m


def _params(**rows):
    fill = dict(temperature=0.0, top_k=1, top_p=1.0, seed=None, repetition_penalty=1.0)
    return SimpleNamespace(**{k: list(rows.get(k, [fill[k]] * B)) for k in fill})


def _candidates(x):
    """The logprobs trace's outputs for logits x [R, V]: per shard the stable top-32 with global ids, and the float32
    sums of exp(x - M) over the vocabulary, M the row maximum (spec_logprobs.VerifyLogprobs.body)."""
    per = V // SHARDS
    vals, ids, parts = [], [], []
    m = x.max(dim=1, keepdim=True).values
    for s in range(SHARDS):
        sh = x[:, s * per : (s + 1) * per]
        o = torch.sort(sh, dim=1, descending=True, stable=True).indices[:, :SAMPLER_TOP_K]
        vals.append(sh.gather(1, o))
        ids.append(o + s * per)
        parts.append((sh - m).exp().sum(1))
    R = x.shape[0]
    sums = torch.stack(parts).to(torch.float32).sum(0)
    return [
        torch.cat(vals, 1).to(torch.bfloat16).reshape(1, 1, R, -1),
        torch.cat(ids, 1).reshape(1, 1, R, -1),
        sums.reshape(1, 1, R, 1),
    ]


@pytest.fixture
def deferred(monkeypatch):
    """MTPBlockEngine of B = 2 users, K = 3, vocabulary 64 over a stand-in model that records its enqueue calls. Its
    reads give what the device holds after the post replays over ``device["logits"]`` [B*T, V]: the verify argmax of
    the raw logits, the draws of the drawn rows (top-k 1: the masked argmax) and the logprobs trace's outputs over the
    masked logits."""
    calls, reads, device = [], [], {}
    model = SimpleNamespace(
        mtp=SimpleNamespace(attention=SimpleNamespace()),
        num_devices=8,
        mesh_device=MagicMock(),
        args=SimpleNamespace(vocab_size=V, max_batch_size=B, dim=64),
        layers=[SimpleNamespace(is_full_attention=False, attention=SimpleNamespace())],
        gdn_blocks=SimpleNamespace(verify_tokens=T, set_verify_blocks=lambda s: None),
        _fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt).to(torch.int32),
        rope=SimpleNamespace(rope_delta=0),
        _vfy_mask=object(),
        _vfy_sampler=SimpleNamespace(draws="draws"),
        _vfy_logprobs=SimpleNamespace(outputs=["lp_vals", "lp_ids", "lp_sums"]),
        _vfy_penalties=SimpleNamespace(
            rebuild=lambda rows: device.setdefault("rebuild", []).append(rows),
            add=lambda tokens: device.setdefault("add", []).append(tokens),
            stage=lambda penalties, greedy, **kw: device.setdefault("staged", []).append((penalties, greedy)),
        ),
        _vfy_top_p=SimpleNamespace(q="q"),
        _vfy_ids_out="ids",
        _vfy_rows_out="rows",
        verify_logits_source=lambda bucket: ("shard", True),
        verify_enqueue=lambda *a, **kw: calls.append(("forward", kw)),
        verify_enqueue_post=lambda **kw: calls.append(("post", kw)),
    )
    eng = spec_engine.MTPBlockEngine(model, B, K)
    eng.captured, eng.nb, eng.block_size, eng._scratch_block = True, 16, 64, 10**6
    eng.sample_on_device = True
    eng._active = {0, 1}

    def read(tensors):
        reads.append(list(tensors))
        raw = device["logits"]
        post = calls[-1][1]
        masked = raw.clone()
        if post.get("mask") is not None:
            masked = torch.where(allowed_tokens(post["mask"], V), raw, torch.tensor(MASK_NEG))
        lanes = torch.zeros(1, 32, dtype=torch.int64)
        lanes[0, : B * T] = raw.argmax(dim=1)
        out = {"ids": lanes, "draws": masked.argmax(dim=1), "q": device.get("q", torch.ones(1, 1, B * T, 1))}
        out.update(zip(["lp_vals", "lp_ids", "lp_sums"], _candidates(masked)))
        host = [out[t] for t in tensors]
        return SimpleNamespace(result=lambda: host, event="event")

    def shards(source):
        raw, mask = device["logits"], calls[-1][1].get("mask")
        edited = raw if mask is None else torch.where(allowed_tokens(mask, V), raw, torch.tensor(MASK_NEG))
        return list(edited.reshape(1, 1, B * T, V).chunk(SHARDS, dim=-1))

    monkeypatch.setattr(eng, "_read_device0_async", read)
    monkeypatch.setattr(ttnn, "get_device_tensors", shards)
    monkeypatch.setattr(ttnn, "from_device", lambda t, blocking=True: t)
    monkeypatch.setattr(ttnn, "to_torch", lambda t: t)
    return eng, calls, reads, device


def _forward(eng, logprobs=False, nv=(3, 3), history=True):
    start = torch.tensor([[100 + j for j in range(T)], [200 + j for j in range(T)]])
    return eng.decode_forward(
        torch.ones(B, T, dtype=torch.int32),
        start,
        num_valid_drafts=torch.tensor(nv, dtype=torch.int32),
        accepted_counts=torch.ones(B, dtype=torch.int32),
        page_table=torch.arange(1, B * eng.nb + 1, dtype=torch.int32).reshape(B, eng.nb),
        spec_mode="argmax_ids",
        mamba_state=SimpleNamespace(),
        defer_sampling=True,
        logprobs=logprobs,
        prompt_tokens=torch.tensor([[1, 2, -1], [3, 4, 5]]) if history else None,
        output_tokens=torch.tensor([[6, -1], [7, 8]]) if history else None,
    )


def test_a_deferred_verify_with_logprobs_replays_the_logprobs_trace_after_the_masked_draws(deferred):
    """User 0 is masked (greedy: it draws the masked argmax), user 1 asks for logprobs and is not masked. The forward
    is enqueued without the logprobs trace; sample_deferred replays it after the mask and the sampler and reads its
    outputs after the ids and draws. User 1's payload is the raw logits', and user 0's masked draw is a candidate."""
    eng, calls, reads, device = deferred
    g = torch.Generator().manual_seed(5)
    device["logits"] = (torch.randn(B * T, V, generator=g) * 3).to(torch.bfloat16).float()
    payload = _forward(eng, logprobs=True)
    assert [c[0] for c in calls] == ["forward"] and calls[0][1]["post"] is False
    assert "logprobs" not in calls[0][1], "the logprobs trace reads the logits after the mask edits them"
    assert payload.step.logprobs

    allowed = {(0, j): (3 + j, 40 + j) for j in range(T)}
    step = eng.sample_deferred(payload, sampling=_params(), mask=_mask(allowed))

    post = calls[-1][1]
    assert calls[-1][0] == "post" and post["logprobs"] is True and post["mask"] is not None
    assert post["sample_plan"].sampled == [True] * T + [False] * T, "only the masked user draws"
    assert reads[-1] == ["ids", "draws", "lp_vals", "lp_ids", "lp_sums"]
    raw = device["logits"]
    ids = step.result()
    for j in range(T):
        a, b = allowed[(0, j)]
        assert int(ids[0, j]) == (a if raw[j, a] >= raw[j, b] else b), "user 0 takes its masked argmax"
    assert ids[1].tolist() == raw[T:].argmax(dim=1).tolist()
    lp, rank, top_lp, top_ids = step.logprobs_result()
    want = torch.log_softmax(raw[T:].double(), dim=-1)
    assert torch.allclose(lp[1].double(), want.max(dim=1).values, atol=2e-5)
    assert torch.allclose(top_lp[1, :, 0].double(), want.max(dim=1).values, atol=2e-5)
    assert rank[1].tolist() == [1] * T
    masked_lp = torch.log_softmax(
        torch.where(allowed_tokens(post["mask"], V), raw, torch.tensor(MASK_NEG)).double(), dim=-1
    )
    assert torch.allclose(lp[0].double(), masked_lp[:T].gather(1, ids[0].reshape(-1, 1)).reshape(-1), atol=2e-5)


def test_a_deferred_verify_without_logprobs_reads_no_payload(deferred):
    eng, calls, reads, device = deferred
    device["logits"] = torch.zeros(B * T, V)
    step = eng.sample_deferred(_forward(eng), sampling=_params(), mask=None)
    assert calls[-1][1]["logprobs"] is False and reads[-1] == ["ids"]
    assert step.logprobs_result() is None


def test_a_greedy_penalty_row_takes_the_masked_penalized_argmax_and_a_sampled_one_draws(deferred):
    """User 0: greedy, masked, repetition penalty: it does not draw, so the penalty trace's float32 argmax (over the
    logits the mask edited first) gives its ids. User 1: sampled with a penalty: it draws from the penalized logits.
    Both rows' histories are staged at sample_deferred from what decode_forward was given at submission."""
    eng, calls, reads, device = deferred
    device["logits"] = torch.zeros(B * T, V)
    payload = _forward(eng)
    sampling = _params(temperature=[0.0, 0.8], top_k=[1, 20], repetition_penalty=[1.3, 1.2])
    step = eng.sample_deferred(payload, sampling=sampling, mask=_mask({(0, 1): (4, 5)}))
    post = calls[-1][1]
    assert post["penalties"] is True and post["mask"] is not None
    assert post["sample_plan"].sampled == [False] * T + [True] * T, "the greedy penalty user takes the argmax"
    penalties, greedy = device["staged"][-1]
    assert penalties.users == [0, 1] and greedy == [1] * T + [0] * T
    assert [sorted(r) for r in device["rebuild"]] == [[0, 1]], "both rows' counts are written from their history"
    assert step.sample_plan is post["sample_plan"]


def test_a_deferred_verify_with_penalties_and_no_history_raises_before_it_changes_anything(deferred, expect_error):
    eng, calls, reads, device = deferred
    payload = _forward(eng, history=False)
    with expect_error(ValueError, "prompt_tokens and output_tokens at submission"):
        eng.sample_deferred(payload, sampling=_params(repetition_penalty=[1.3, 1.0]))
    assert eng._deferred is payload and [c[0] for c in calls] == ["forward"]


@pytest.mark.parametrize("bucket", [None, 2])
def test_the_post_replays_mask_then_penalties_then_sampler_then_logprobs(monkeypatch, expect_error, bucket):
    m, _ = _bare_verify_model(4, 4, monkeypatch)
    m._vfy_mask, m._vfy_sampler = MagicMock(name="mask"), MagicMock(name="sampler")
    m._vfy_mask_trace_id, m._vfy_pen_trace_id = "mask", "pen"
    m._vfy_sample_trace_id, m._vfy_logprobs_trace_id = "sample", "logprobs"
    m._vfy_bucket_mask_traces, m._vfy_bucket_pen_traces = {2: "mask2"}, {2: "pen2"}
    m._vfy_bucket_sample_traces, m._vfy_bucket_logprobs_traces = {2: "sample2"}, {2: "logprobs2"}
    traces = []
    monkeypatch.setattr(ttnn, "execute_trace", lambda dev, trace, **kw: traces.append(trace))
    rows = torch.full((16, 2), -1, dtype=torch.int32)
    m.verify_enqueue_post(sample_plan="plan", bucket=bucket, mask=rows, penalties=True, logprobs=True)
    tag = "" if bucket is None else "2"
    assert traces == [f"mask{tag}", f"pen{tag}", f"sample{tag}", f"logprobs{tag}"]
    traces.clear()
    m.verify_enqueue_post(bucket=bucket, mask=rows, penalties=True)
    assert traces == [f"mask{tag}", f"pen{tag}"], "greedy penalty rows alone take the masked argmax"
    with expect_error(ValueError, "sample_plan that draws its rows or the penalties"):
        m.verify_enqueue_post(bucket=bucket, mask=rows)


def test_a_certificate_row_on_a_masked_verify_is_drawn_again_from_the_masked_logits(deferred):
    """User 0 has top-k off and top_p 0.9 (a certificate row) and a mask at column 0; user 1 is masked and greedy.
    The certificate sampler replaces the sampler at sample_deferred, the step runs no device acceptance (its
    submission offered one), and the reads take the certificate's q after the draws. User 0's column 0 fails the
    certificate: its redo draws from the logits read back after the mask, so it takes an allowed token."""
    eng, calls, reads, device = deferred
    eng._one_enqueue_ready = lambda: True
    g = torch.Generator().manual_seed(9)
    device["logits"] = (torch.randn(B * T, V, generator=g) * 2).to(torch.bfloat16).float()
    q = torch.full((1, 1, B * T, 1), 0.99)
    q[0, 0, 0, 0] = 0.5
    device["q"] = q
    payload = _forward(eng, logprobs=True)
    assert payload.one_enqueue
    sampling = _params(temperature=[0.7, 0.0], top_k=[-1, 1], top_p=[0.9, 1.0], seed=[11, None])
    step = eng.sample_deferred(payload, sampling=sampling, mask=_mask({(0, 0): (3, 9), (1, 1): (4, 5)}))

    post = calls[-1][1]
    assert post["top_p"] is step.cert and step.cert.rows == [True, False] and step.cert.top_p[0] == 0.9
    assert post["accept_rows"] is None and step.plan is None, "a certificate verify runs no device acceptance"
    plan = post["sample_plan"]
    assert plan.sampled == [True] * (2 * T) and plan.k[:T] == [32] * T, "the certificate row draws its top-32"
    assert reads[-1] == ["ids", "draws", "q", "lp_vals", "lp_ids", "lp_sums"]
    ids = step.result()
    assert bool(step.redone[0]) and int(step.redone.sum()) == 1
    assert int(ids[0, 0]) in (3, 9), "the redo draws from the masked logits"
    lp = step.logprobs_result()[0]
    masked0 = torch.where(allowed_tokens(post["mask"][:1], V), device["logits"][:1], torch.tensor(MASK_NEG))
    want = torch.log_softmax(masked0.double(), dim=-1)[0, int(ids[0, 0])]
    assert abs(float(lp[0, 0]) - float(want)) < 1e-4, "a redone row's payload comes from the logits it was drawn from"
