# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): one deferred verify whose rows combine the device sampling features of
Qwen36MTPContractForCausalLM (vllm-tt-plugin ``supports_spec_device_mask`` with ``supports_spec_device_logprobs``).

A token-masked verify (``decode_forward(defer_sampling=True)``) enqueues its forward at submission and the rest at
``sample_deferred``. A verify sent ``spec_num_logprobs`` too replays the logprobs trace there, after the mask and the
sampler, so the payload describes the logits the draws came from: an unmasked row's payload is its raw logits', and a
masked row's draw is among its masked candidates (the payload of a masked row is not published: the plugin verifies a
masked row that asks for logprobs on the ``logits`` path).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

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
    fill = dict(temperature=0.0, top_k=1, top_p=1.0, seed=None)
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
        _vfy_ids_out="ids",
        _vfy_rows_out="rows",
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
        out = {"ids": lanes, "draws": masked.argmax(dim=1)}
        out.update(zip(["lp_vals", "lp_ids", "lp_sums"], _candidates(masked)))
        host = [out[t] for t in tensors]
        return SimpleNamespace(result=lambda: host, event="event")

    monkeypatch.setattr(eng, "_read_device0_async", read)
    return eng, calls, reads, device


def _forward(eng, logprobs=False, nv=(3, 3)):
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
