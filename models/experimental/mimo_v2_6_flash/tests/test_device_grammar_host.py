# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM adapter's device grammar (generator_vllm.MiMoV2ForCausalLM, vllm-tt-plugin docs/DEVICE_GRAMMAR_CONTRACT.md)
on fake lanes (no device): a deferred step holds its arguments until ``sample_deferred_decode`` runs it once with the
rows' sampling parameters and vLLM's mask, padded to every decode row; one step may be pending; a call that fails its
checks leaves the step pending and nothing changed; a failure after the step started is final. Per lane, the sampler
gets the mask rows of its active constrained rows, and a lane whose last mask flagged rows has the flags lifted on its
next device-sampled step without one."""

from types import SimpleNamespace

import pytest
import torch

from models.experimental.mimo_v2_6_flash.tt.device_sampler import MAX_USERS
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import DeferredDecode, MiMoV2ForCausalLM

DP, PER_LANE, VOCAB = 4, 4, 152576
ROWS = DP * PER_LANE
WORDS = -(-VOCAB // 32)


class FakeSampler:
    def __init__(self):
        self.grammar = []  # each set_grammar call: (mask, rows), or None (lifted)
        self.params = 0

    def host_params(self, temperature, top_k, top_p, states=None):
        return None, bool(((temperature != 0) & (top_k != 1)).any())

    def set_params(self, hosts):
        self.params += 1

    def set_grammar(self, mask=None, rows=None):
        self.grammar.append(None if mask is None else (mask.clone(), rows.clone()))


class FakeLane:
    def __init__(self):
        self.max_batch = 32
        self.sampler = FakeSampler()
        self.decode_trace = object()
        self.enqueued = []

    def set_vllm_page_tables(self, rows):
        pass

    def decode_enqueue(self, tok, pos, sample):
        self.enqueued.append(("reload", sample))
        return "handle"

    def decode_enqueue_resident(self, pos, sample):
        self.enqueued.append(("steady", sample))
        return "handle"


def params(rows=ROWS, temperature=1.0, top_k=-1, top_p=0.95, logprobs=False):
    return SimpleNamespace(
        temperature=[temperature] * rows,
        top_k=[top_k] * rows,
        top_p=[top_p] * rows,
        seed=None,
        enable_log_probs=[logprobs] * rows,
    )


def bitmask(constrained):
    """vLLM's mask for the decode rows: rows ``constrained`` allow tokens 0-31 only, the others everything."""
    mask = torch.full((ROWS, WORDS), -1, dtype=torch.int32)
    for r in constrained:
        mask[r] = 0
        mask[r, 0] = -1
    return mask


@pytest.fixture
def adapter():
    gen = SimpleNamespace(
        models=[FakeLane() for _ in range(DP)], dp=DP, users_per_replica=PER_LANE, cfg=SimpleNamespace(vocab_size=VOCAB)
    )
    model = MiMoV2ForCausalLM(gen, 4096)
    model.layer_groups = [0]
    model.enable_device_grammar()
    return model


@pytest.fixture
def recorded(adapter, monkeypatch):
    """The adapter with ``_decode`` recorded instead of run."""
    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return {"step": len(calls)}

    monkeypatch.setattr(adapter, "_decode", record)
    return adapter, calls


def submit(model, positions=None):
    tokens = torch.arange(ROWS, dtype=torch.int32).reshape(ROWS, 1)
    positions = torch.arange(ROWS, dtype=torch.int32) + 100 if positions is None else positions
    table = torch.zeros(ROWS, 4, dtype=torch.int32)
    return model.decode_forward(tokens, positions, table, None, read_from_device=False, defer_device_sampling=True)


def test_capability_and_enable(adapter):
    assert MiMoV2ForCausalLM.model_capabilities["supports_device_grammar"] is True
    assert adapter.device_grammar_enabled and callable(adapter.sample_deferred_decode)
    wide = MiMoV2ForCausalLM(
        SimpleNamespace(models=[], dp=1, users_per_replica=MAX_USERS + 1, cfg=SimpleNamespace(vocab_size=VOCAB)), 4096
    )
    wide.enable_device_grammar()
    assert not wide.device_grammar_enabled, "past MAX_USERS a lane has no device sampler"


def test_deferred_step_runs_once_with_the_padded_mask(recorded, expect_error):
    model, calls = recorded
    payload = submit(model)
    assert isinstance(payload, DeferredDecode) and calls == []
    with expect_error(RuntimeError, "pending"):
        submit(model)
    mask = bitmask([1, 6])[:10]  # vLLM's rows of the step; the rest are padded as unconstrained
    out = model.sample_deferred_decode(payload, params(), grammar_bitmask=mask)
    assert out == {"step": 1} and len(calls) == 1
    call = calls[0]
    assert call["read_from_device"] is False and call["sampling_params"] is not None
    grammar = call["grammar"]
    assert grammar.shape == (ROWS, WORDS) and grammar.dtype == torch.int32
    assert torch.equal(grammar[:10], mask) and bool((grammar[10:] == -1).all())
    assert call["tokens"].reshape(-1).tolist() == list(range(ROWS))
    with expect_error(ValueError, "pending deferred"):
        model.sample_deferred_decode(payload, params(), grammar_bitmask=mask)
    assert isinstance(submit(model), DeferredDecode), "a new step after the last one ran"


def test_deferred_inputs_are_copies(recorded):
    model, calls = recorded
    tokens = torch.arange(ROWS, dtype=torch.int32).reshape(ROWS, 1)
    positions = torch.arange(ROWS, dtype=torch.int32)
    payload = model.decode_forward(
        tokens,
        positions,
        torch.zeros(ROWS, 4, dtype=torch.int32),
        None,
        read_from_device=False,
        defer_device_sampling=True,
    )
    tokens.zero_()
    positions.fill_(-1)
    model.sample_deferred_decode(payload, params(), grammar_bitmask=bitmask([0]))
    assert calls[0]["tokens"].reshape(-1).tolist() == list(range(ROWS))
    assert calls[0]["start_pos"].tolist() == list(range(ROWS))


@pytest.mark.parametrize(
    "bad",
    [
        lambda m, p: m.sample_deferred_decode(p, None, grammar_bitmask=bitmask([0])),
        lambda m, p: m.sample_deferred_decode(p, params(), grammar_bitmask=None),
        lambda m, p: m.sample_deferred_decode(p, params(logprobs=True), grammar_bitmask=bitmask([0])),
        lambda m, p: m.sample_deferred_decode(p, params(top_k=-1, top_p=1.0), grammar_bitmask=bitmask([0])),
        lambda m, p: m.sample_deferred_decode(p, params(top_k=100000), grammar_bitmask=bitmask([0])),
        lambda m, p: m.sample_deferred_decode(p, params(), grammar_bitmask=bitmask([0])[:, :-1]),
        lambda m, p: m.sample_deferred_decode(DeferredDecode(p.args), params(), grammar_bitmask=bitmask([0])),
    ],
    ids=["no-params", "no-mask", "logprobs", "unbounded", "top-k", "mask-width", "stale"],
)
def test_checks_leave_the_step_pending(recorded, bad, expect_error):
    model, calls = recorded
    payload = submit(model)
    with expect_error((ValueError, TypeError), ""):
        bad(model, payload)
    assert calls == [] and model._deferred is payload and model._grammar_failure is None
    model.sample_deferred_decode(payload, params(), grammar_bitmask=bitmask([0]))
    assert len(calls) == 1


def test_failure_after_submission_is_final(adapter, monkeypatch, expect_error):
    def fail(**kwargs):
        raise RuntimeError("device error")

    monkeypatch.setattr(adapter, "_decode", fail)
    payload = submit(adapter)
    with expect_error(RuntimeError, "device error"):
        adapter.sample_deferred_decode(payload, params(), grammar_bitmask=bitmask([0]))
    with expect_error(RuntimeError, "failed after a decode submission"):
        submit(adapter)
    with expect_error(RuntimeError, "failed after a decode submission"):
        adapter.sample_deferred_decode(payload, params(), grammar_bitmask=bitmask([0]))


def test_defer_needs_grammar_and_a_plain_step(adapter, expect_error):
    table = torch.zeros(ROWS, 4, dtype=torch.int32)
    tokens, positions = torch.zeros(ROWS, 1, dtype=torch.int32), torch.zeros(ROWS, dtype=torch.int32)
    for kwargs in ({"sampling_params": params()}, {"read_from_device": True}, {"spec_mode": "argmax_ids"}):
        with expect_error(ValueError, "deferred decode"):
            adapter.decode_forward(
                tokens, positions, table, None, **{"read_from_device": False, **kwargs}, defer_device_sampling=True
            )
    adapter.device_grammar_enabled = False
    with expect_error(ValueError, "enable_device_grammar"):
        submit(adapter)
    assert adapter._deferred is None


def test_steady_deferred_step_needs_positions(recorded, expect_error):
    model, calls = recorded
    steady = dict(read_from_device=False, defer_device_sampling=True, reload_inputs=False)
    payload = model.decode_forward(None, None, None, None, **steady)
    with expect_error(RuntimeError, "steady decode before any reloading"):
        model.sample_deferred_decode(payload, params(), grammar_bitmask=bitmask([0]))
    assert model._deferred is payload and calls == []
    model.decode_positions = [torch.full((32,), -1, dtype=torch.int32) for _ in range(DP)]
    model.sample_deferred_decode(payload, params(), grammar_bitmask=bitmask([0]))
    assert calls[0]["reload_inputs"] is False


def test_lanes_get_their_constrained_rows(adapter):
    """Through the real ``_decode``: lane 1's constrained active rows are flagged with their mask rows; a reloading
    device-sampled step re-uploads the parameters without flags, so nothing is left to lift; an unconstrained lane
    never touches its mask."""
    positions = torch.arange(ROWS, dtype=torch.int32) + 100
    positions[7] = -1  # an idle slot of lane 1: never flagged
    payload = submit(adapter, positions)
    adapter.sample_deferred_decode(payload, params(), grammar_bitmask=bitmask([4, 6, 7]))
    lane1, lane0 = adapter.models[1].sampler, adapter.models[0].sampler
    ((mask, rows),) = lane1.grammar
    assert rows.tolist() == [True, False, True, False]
    assert mask.shape == (PER_LANE, WORDS) and int(mask[0, 0]) == -1 and int(mask[0, 1]) == 0
    assert lane0.grammar == [] and adapter.lane_grammar == [False, True, False, False]
    assert adapter.models[1].enqueued[-1] == ("reload", True)
    # Next step: an ordinary device-sampled decode that reloads the parameters.
    out = adapter.decode_forward(
        torch.zeros(ROWS, 1, dtype=torch.int32),
        positions + 1,
        torch.zeros(ROWS, 4, dtype=torch.int32),
        None,
        read_from_device=False,
        sampling_params=params(temperature=0.0),
        reload_sampling_params=True,
    )
    assert out["tokens"] and len(lane1.grammar) == 1 and adapter.lane_grammar == [False] * DP
    assert lane1.params == 2, "the reload uploaded the parameters again, flags off: nothing left to lift"
    count = len(lane1.grammar)
    adapter.decode_forward(
        None,
        None,
        torch.zeros(ROWS, 4, dtype=torch.int32),
        None,
        read_from_device=False,
        sampling_params=params(temperature=0.0),
        reload_inputs=False,
    )
    assert len(lane1.grammar) == count, "no flags left to lift"


def test_steady_mask_without_reload_lifts_the_flags(adapter):
    """A steady deferred step after a constrained one, with the request's mask now allowing everything: the lane's
    flags are lifted (the parameters were not re-uploaded)."""
    payload = submit(adapter)
    adapter.sample_deferred_decode(payload, params(), grammar_bitmask=bitmask([5]))
    lane1 = adapter.models[1].sampler
    assert lane1.grammar[-1] is not None
    payload = adapter.decode_forward(
        None, None, None, None, read_from_device=False, defer_device_sampling=True, reload_inputs=False
    )
    adapter.sample_deferred_decode(payload, params(), grammar_bitmask=bitmask([]))
    assert lane1.grammar[-1] is None and lane1.params == 1
    assert adapter.models[1].enqueued[-1] == ("steady", True)
