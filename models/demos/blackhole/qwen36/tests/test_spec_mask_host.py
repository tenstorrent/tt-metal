# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the token mask of a deferred verify (tt/spec_mask.py, vllm-tt-plugin
``supports_spec_device_mask``).

* The bit-plane layout: the device op chain of ``VerifyMask.apply_`` (shift by 0..31, bit 0, flatten, where, add),
  emulated in torch on the staged words of each shard, gives the host reference ``masked_logits`` bit for bit.
* ``sampling_plan`` draws every column of a masked user, a greedy one with top-k 1 (the masked argmax).
* MTPBlockEngine: ``decode_forward(defer_sampling=True)`` enqueues the forward only; ``sample_deferred`` validates
  the payload, the mask and the valid drafts before any state changes, then enqueues the mask, the sampler and the
  reads, and its ids carry the draws of the masked users.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from models.demos.blackhole.qwen36.tt import spec_engine
from models.demos.blackhole.qwen36.tt.spec_mask import (
    MASK_NEG,
    WORD_BITS,
    allowed_tokens,
    mask_words,
    masked_logits,
    normalise_rows,
    shard_bitplanes,
    shard_words,
)
from models.demos.blackhole.qwen36.tt.spec_sample import SKIP_SEED, sampling_plan


def _pack(allowed):
    """Plugin packing int32 [R, ceil(V/32)] of a bool [R, V]."""
    R, V = allowed.shape
    words = mask_words(V)
    bits = torch.zeros(R, words * WORD_BITS, dtype=torch.int64)
    bits[:, :V] = allowed.to(torch.int64)
    packed = (bits.reshape(R, words, WORD_BITS) << torch.arange(WORD_BITS)).sum(-1)
    return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)


def _device_chain(x, planes):
    """Torch emulation of VerifyMask.apply_ on one device: ``x`` bf16 [R, 32 * W], ``planes`` int32 [R, W]."""
    R, W = planes.shape
    flags = (
        (planes.reshape(R, 1, W) >> torch.arange(WORD_BITS, dtype=torch.int32).reshape(1, WORD_BITS, 1)) & 1
    ).bool()
    offset = torch.where(flags.reshape(R, WORD_BITS * W), 0.0, MASK_NEG).to(torch.bfloat16)
    return x + offset


@pytest.mark.parametrize("shards", [1, 2, 4, 8])
def test_the_device_chain_over_bit_planes_is_the_host_mask(shards):
    g = torch.Generator().manual_seed(shards)
    V, R = 64 * 8 * 4, 5  # whole 32-bit words in every shard up to 8 shards
    allowed = torch.rand(R, V, generator=g) < 0.3
    allowed[1] = True  # an unconstrained row
    allowed[2] = False
    allowed[2, V - 1] = True  # the last token alone
    logits = torch.randn(R, V, generator=g) * 4
    words = shard_words(V, shards)
    planes = shard_bitplanes(_pack(allowed), shards, words)  # [shards, R, words]
    want = masked_logits(logits, _pack(allowed), V)
    width = WORD_BITS * words
    for s in range(shards):
        got = _device_chain(logits[:, s * width : (s + 1) * width].to(torch.bfloat16), planes[s])
        assert torch.equal(got, want[:, s * width : (s + 1) * width]), f"shard {s}"
    assert torch.equal(want[1], logits[1].to(torch.bfloat16)), "an all-ones row keeps every logit"
    assert int((want[2] > MASK_NEG).sum()) == 1


def test_mask_rows_are_padded_and_cut_to_the_vocabulary():
    V = 100  # 4 words, the last holding 4 tokens
    m = normalise_rows(torch.full((2, 3), -1, dtype=torch.int32), V)
    assert m.shape == (2, 4) and m[0, :3].tolist() == [-1] * 3
    assert m[0, 3].item() == 0, "words past the plugin's mask are disallowed"
    m = normalise_rows(torch.full((1, 4), -1, dtype=torch.int32), V)
    assert allowed_tokens(m, 128).sum().item() == V, "bits past the vocabulary are cleared"
    with pytest.raises(ValueError, match="wider"):
        normalise_rows(torch.zeros(1, 5, dtype=torch.int32), V)
    with pytest.raises(ValueError, match="whole 32-bit words"):
        shard_words(248320, 32)
    assert shard_words(248320, 8) == 970 and shard_words(248320, 4) == 1940


def _params(**rows):
    n = max(len(v) for v in rows.values())
    fill = dict(temperature=0.0, top_k=1, top_p=1.0, seed=None)
    return SimpleNamespace(**{k: list(rows.get(k, [fill[k]] * n)) for k in fill})


def test_a_masked_user_draws_every_column_a_greedy_one_the_masked_argmax():
    sp = _params(temperature=[0.0, 0.7, 0.0, 0.0], top_k=[1, 20, 1, 5], top_p=[1.0, 0.9, 1.0, 1.0])
    plan = sampling_plan(sp, [10, 20, 30, -1], 4, fresh_seed=lambda: 7, masked=[True, True, False, True])
    rows = lambda u: plan.sampled[u * 4 : (u + 1) * 4]
    assert rows(0) == [True] * 4 and rows(1) == [True] * 4
    assert rows(2) == rows(3) == [False] * 4, "an unmasked greedy user and a padding user keep their argmax ids"
    assert plan.k[:4] == [1] * 4 and plan.p[:4] == [0.0] * 4 and plan.temp[:4] == [1.0] * 4, "top-k 1: the argmax"
    assert plan.seeds[:4] == [SKIP_SEED] * 4 and plan.user_ids[:4] == [0, 1, 2, 3]
    assert plan.k[4:8] == [20] * 4 and plan.seeds[4:8] == [7] * 4, "a sampled masked user draws as before"
    assert sampling_plan(sp, [10, 20, 30, -1], 4, masked=[False] * 4).sampled[:4] == [False] * 4


# ------------------------------------------------------------------------------ the engine's deferred verify


@pytest.fixture
def deferred_engine(monkeypatch):
    """MTPBlockEngine of B = 2 users, K = 3, vocabulary 64 (2 words), over a stand-in model that records its enqueue
    calls; reads return ids 5 everywhere and draws 40 + row."""
    B, K, V = 2, 3, 64
    calls = []
    sampler = SimpleNamespace(draws="draws")
    model = SimpleNamespace(
        mtp=SimpleNamespace(attention=SimpleNamespace()),
        num_devices=8,
        mesh_device=MagicMock(),
        args=SimpleNamespace(vocab_size=V, max_batch_size=B, dim=64),
        layers=[SimpleNamespace(is_full_attention=False, attention=SimpleNamespace())],
        gdn_blocks=SimpleNamespace(verify_tokens=K + 1, set_verify_blocks=lambda s: None),
        _fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt).to(torch.int32),
        rope=SimpleNamespace(rope_delta=0),
        _vfy_mask=object(),
        _vfy_sampler=sampler,
        _vfy_ids_out="ids",
        _vfy_rows_out="rows",
        verify_enqueue=lambda *a, **kw: calls.append(("forward", kw)),
        verify_enqueue_post=lambda **kw: calls.append(("post", kw)),
    )
    eng = spec_engine.MTPBlockEngine(model, B, K)
    eng.captured, eng.nb, eng.block_size, eng._scratch_block = True, 16, 64, 10**6
    eng.sample_on_device = True
    eng._active = {0, 1}

    def reads(tensors):
        host = [torch.full((1, 32), 5, dtype=torch.int64)]
        if "draws" in tensors:
            host.append(torch.arange(40, 72))
        return SimpleNamespace(result=lambda: host, event="event")

    monkeypatch.setattr(eng, "_read_device0_async", reads)
    return eng, calls


def _forward(eng, nv=(3, 1)):
    B, T = eng.B, eng.T
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
    )


def _mask(B, T, V, constrained=None):
    m = torch.full((B, T, mask_words(V)), -1, dtype=torch.int32)
    for (u, j), allowed in (constrained or {}).items():
        bits = torch.zeros(1, V, dtype=torch.bool)
        bits[0, list(allowed)] = True
        m[u, j] = _pack(bits)[0]
    return m


def test_a_deferred_verify_enqueues_its_forward_only_and_one_payload_at_a_time(deferred_engine):
    eng, calls = deferred_engine
    payload = _forward(eng)
    assert isinstance(payload, spec_engine.DeferredVerify)
    assert [c[0] for c in calls] == ["forward"] and calls[0][1]["post"] is False
    with pytest.raises(RuntimeError, match="deferred sampling is still to come"):
        _forward(eng)
    with pytest.raises(RuntimeError, match="deferred sampling is still to come"):
        eng.prefill_over_pending([0])


def test_sample_deferred_validates_before_it_changes_anything(deferred_engine):
    eng, calls = deferred_engine
    payload = _forward(eng, nv=(3, 1))
    sp = _params(temperature=[0.0, 0.0])
    with pytest.raises(ValueError, match="not the engine's pending"):
        eng.sample_deferred(SimpleNamespace(), sampling=sp)
    with pytest.raises(ValueError, match=r"allows no token at \(row, column\) \[\[1, 1\]\]"):
        eng.sample_deferred(payload, sampling=sp, mask=_mask(2, 4, 64, {(1, 1): ()}))
    with pytest.raises(ValueError, match="exceed the submitted"):
        eng.sample_deferred(payload, sampling=sp, num_valid_drafts=torch.tensor([3, 2]))
    with pytest.raises(ValueError, match="must be"):
        eng.sample_deferred(payload, sampling=sp, mask=torch.full((2, 3, 2), -1, dtype=torch.int32))
    assert eng._deferred is payload and [c[0] for c in calls] == ["forward"], "the payload is still pending"


def test_sample_deferred_draws_the_masked_users_and_reads_their_draws(deferred_engine):
    eng, calls = deferred_engine
    payload = _forward(eng, nv=(3, 1))
    # user 1's column 3 lies past its valid drafts: an empty mask there is never drawn from
    mask = _mask(2, 4, 64, {(0, 0): (7, 9), (1, 3): ()})
    step = eng.sample_deferred(
        payload, sampling=_params(temperature=[0.0, 0.0]), mask=mask, num_valid_drafts=torch.tensor([2, 1])
    )
    assert eng._deferred is None and eng.masked_steps == 1
    post = calls[-1][1]
    assert calls[-1][0] == "post" and post["bucket"] is None and post["accept_rows"] is None
    assert post["mask"].shape == (8, 2) and torch.equal(post["mask"][0], mask[0, 0])
    assert (post["mask"][1:] == -1).all(), "unconstrained columns, and the column past user 1's drafts"
    plan = post["sample_plan"]
    assert plan.sampled == [True] * 4 + [False] * 4 and plan.k[:4] == [1] * 4
    assert step.result().tolist() == [[40, 41, 42, -1], [5, 5, -1, -1]], "draws of user 0, its valid drafts cut to 2"
    with pytest.raises(ValueError, match="not the engine's pending"):
        eng.sample_deferred(payload, sampling=_params(temperature=[0.0, 0.0]))


def test_an_unconstrained_greedy_deferred_verify_draws_nothing(deferred_engine):
    eng, calls = deferred_engine
    step = eng.sample_deferred(_forward(eng), sampling=_params(temperature=[0.0, 0.0]), mask=None)
    assert calls[-1][1]["sample_plan"] is None and calls[-1][1]["mask"] is None and eng.masked_steps == 0
    assert step.result().tolist() == [[5, 5, 5, 5], [5, 5, -1, -1]]
