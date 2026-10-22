# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM adapter (generator_vllm.MiMoV2ForCausalLM) under the plugin's asynchronous scheduling, on fake lanes whose
device is the host mirrors (no device; ttnn may be a stub; needs vllm-tt-plugin, not vLLM). The lanes are
test_spec_chain_adapter_host.py's (an oracle target, the accept and chain steps' mirrors, a drafter wrong at every
third bonus position) with an ordinary decode added: a resident step reads the tokens and positions the last one fed
back, as the decode trace does.

The calls follow the plugin's async contract (vllm-tt-plugin docs/SPEC_DECODE_CONTRACT.md, sections 4c and 4j):
ordinary steps are submitted without reading (``read_from_device=False``) and read a step later, and the propose for
a step comes when it commits, after the next step's submission. Checked: the declared capabilities, a verify asked to
defer its read answering host tensors, the reads passing a verify's host output through, and a propose for an older
ordinary step enqueueing nothing (placeholder drafts that make the plugin drain and propose again)."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.test_spec_chain_adapter_host import (
    DP,
    MAX_SEQ_LEN,
    PER_LANE,
    SEQ,
    SPEC_ROW,
    TABLE,
    VOCAB,
    FakeLane,
    draft_of,
    propose,
    start,
    verify,
)
from models.experimental.mimo_v2_6_flash.reference.spec_decode import MAX_DRAFTS
from models.experimental.mimo_v2_6_flash.tt import generator_vllm
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import MiMoV2ForCausalLM

pytest.importorskip("vllm_tt_plugin")

ROWS = DP * PER_LANE


class FakeTokens:
    """A device token row: ``cpu`` copies it, as the read enqueued behind the step would."""

    def __init__(self, values):
        self.values = torch.tensor(values, dtype=torch.int64)

    def cpu(self, blocking=True):
        return FakeTokens(self.values.tolist())


class AsyncLane(FakeLane):
    """FakeLane with the ordinary decode step: the oracle's token after each active slot's position, fed back into
    the resident tokens and positions (TtMiMoModel._feed_back)."""

    def __init__(self):
        super().__init__()
        self.decode_trace = {"id": 0}
        self.resident = None
        self.decodes = 0
        self.sampler = SimpleNamespace(host_params=lambda *a, **k: (None, False), set_params=lambda hosts: None)

    def _decode(self, pos):
        self.decodes += 1
        tokens = [SEQ[int(p) + 1] if p >= 0 else 0 for p in pos.tolist()]
        self.resident = (tokens, torch.where(pos >= 0, pos + 1, pos))
        return {"next_tokens": FakeTokens(tokens)}

    def decode_enqueue(self, tokens, positions, sample=False):
        for s, p in enumerate(positions.tolist()):
            assert p < 0 or int(tokens[s]) == SEQ[p], f"slot {s}: token {int(tokens[s])} is not the one at {p}"
        return self._decode(positions.clone())

    def decode_enqueue_resident(self, positions, sample=False):
        assert torch.equal(positions, self.resident[1]), "the adapter's positions are the device's"
        return self._decode(self.resident[1].clone())


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda t: [t], raising=False)
    monkeypatch.setattr(ttnn, "to_torch", lambda t: t.values, raising=False)
    gen = SimpleNamespace(
        models=[AsyncLane() for _ in range(DP)],
        dp=DP,
        users_per_replica=PER_LANE,
        cfg=SimpleNamespace(vocab_size=VOCAB),
    )
    model = MiMoV2ForCausalLM(gen, MAX_SEQ_LEN)
    model.layer_groups = [0]
    return model


GREEDY = SimpleNamespace(temperature=[0.0] * ROWS, top_k=[1] * ROWS, top_p=[1.0] * ROWS, seed=None)


def submit(model, live, reload=True):
    """An ordinary step submitted the way the plugin overlaps it: ``live`` {row: position} (the token there is the
    oracle's), unread."""
    tokens = torch.zeros(ROWS, 1, dtype=torch.int32)
    positions = torch.full((ROWS,), -1, dtype=torch.int32)
    for row, p in live.items():
        tokens[row, 0], positions[row] = SEQ[p], p
    return model.decode_forward(
        tokens,
        positions,
        TABLE,
        None,
        sampling_params=GREEDY,
        read_from_device=False,
        reload_inputs=reload,
    )


def read(model, out):
    host, events = model.read_decode_output(out, async_read=True)
    return model.process_decode_output_host(host, is_tokens=True)


def propose_plain(model, sampled, live):
    """The plugin's propose after an ordinary step it committed: each live row's token at its position + 1."""
    committed = torch.full((ROWS, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
    positions = torch.full((ROWS, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
    for row, p in live.items():
        committed[row, 0], positions[row, 0] = int(sampled[row]), p + 1
    return model.propose_draft_tokens(MAX_DRAFTS, committed, positions, torch.ones(ROWS, dtype=torch.int32))


def test_capabilities(monkeypatch):
    caps = MiMoV2ForCausalLM.model_capabilities
    assert caps["supports_async_decode"] and caps["supports_async_spec_decode"] == generator_vllm.ASYNC_SPEC
    assert generator_vllm.ASYNC_SPEC, "MIMO_ASYNC_SPEC defaults to on"


def test_a_propose_for_an_older_ordinary_step_enqueues_nothing(adapter):
    """Lane 1 holds one request, lane 0 two. Step n + 1 is submitted before step n is read (an overlapped ordinary
    step); the propose for n then finds lane 1's source to be n + 1's: placeholder drafts, nothing enqueued. After the
    plugin drains n + 1, its propose drafts on device from n + 1's hidden."""
    lane0, lane1 = adapter.models
    p = 300
    live = {0: 40, 1: 90, SPEC_ROW: p}
    out = submit(adapter, live)
    for k in range(4):
        nxt = submit(adapter, {row: q + k + 1 for row, q in live.items()}, reload=False)
        sampled = read(adapter, out)
        assert [int(sampled[row]) for row in live] == [SEQ[q + k + 1] for q in live.values()]
        here = {row: q + k for row, q in live.items()}
        drafts = propose_plain(adapter, sampled, here)
        assert int(drafts.num_valid[SPEC_ROW]) == MAX_DRAFTS and int(drafts.num_valid.sum()) == MAX_DRAFTS
        assert drafts.draft_token_ids[SPEC_ROW].tolist() == [SEQ[p + k + 1]] * MAX_DRAFTS, "placeholders"
        assert lane1.mtp.planned == 0 and lane0.mtp.planned == 0, "a stale propose enqueues no draft"
        out = nxt
    # The plugin drains the step in flight and proposes for it: the MTP drafts from that step's resident hidden.
    sampled = read(adapter, out)
    here = {row: q + 4 for row, q in live.items()}
    drafts = propose_plain(adapter, sampled, here)
    q = p + 4 + 1
    assert drafts.draft_token_ids[SPEC_ROW].tolist() == [draft_of(q, k) for k in range(MAX_DRAFTS)]
    assert int(drafts.num_valid.sum()) == MAX_DRAFTS and lane1.mtp.planned == 1 and lane0.mtp.planned == 0
    assert adapter.chain_stats["stale"] == 4 and adapter.chain_stats["misplaced"] == 0


def test_lanes_of_several_requests_propose_without_the_device(adapter):
    """The concurrency-16 shape: every lane holds several requests, so no lane drafts and a propose touches no
    device."""
    live = {row: 100 + 7 * row for row in range(ROWS) if row % PER_LANE < 3}
    out = submit(adapter, live)
    for k in range(3):
        nxt = submit(adapter, {row: q + k + 1 for row, q in live.items()}, reload=False)
        sampled = read(adapter, out)
        drafts = propose_plain(adapter, sampled, {row: q + k for row, q in live.items()})
        assert int(drafts.num_valid.sum()) == 0
        out = nxt
    assert all(lane.mtp.planned == 0 and lane.reads == 0 for lane in adapter.models)
    assert adapter.chain_stats["stale"] == adapter.chain_stats["misplaced"] == 0


def test_a_deferred_verify_answers_host_tensors(adapter):
    """``defer_read`` on a fused_sample verify: host tensors (the plugin reads it synchronously, chaining nothing),
    which ``read_decode_output`` passes through, as it passes an argmax_ids or logits verify's tensor."""
    drafts = start(adapter, 600)
    rows, width = ROWS, 1 + MAX_DRAFTS
    tokens = torch.zeros(rows, width, dtype=torch.int32)
    positions = torch.full((rows, width), -1, dtype=torch.int32)
    tokens[SPEC_ROW] = torch.tensor([SEQ[600]] + drafts, dtype=torch.int32)
    positions[SPEC_ROW] = torch.arange(600, 600 + width, dtype=torch.int32)
    valid = torch.zeros(rows, dtype=torch.int32)
    valid[SPEC_ROW] = MAX_DRAFTS
    out = adapter.decode_forward(
        tokens, positions, TABLE, None, spec_mode="fused_sample", num_valid_drafts=valid, defer_read=True
    )
    assert out.handle is None and out.accepted_token_ids is not None
    assert adapter.read_decode_output(out, async_read=True) == (out, [])
    assert adapter.read_decode_output(out) is out
    ids = torch.zeros(rows, width, dtype=torch.int32)
    assert adapter.read_decode_output(ids, async_read=True)[0] is ids
    with pytest.raises(ValueError, match="defer_read"):
        adapter.decode_forward(tokens, positions, TABLE, None, spec_mode="argmax_ids", num_valid_drafts=valid,
                               defer_read=True)  # fmt: skip
    with pytest.raises(ValueError, match="chain"):
        adapter.decode_forward(
            tokens,
            positions,
            TABLE,
            None,
            spec_mode="fused_sample",
            num_valid_drafts=valid,
            defer_read=True,
            chain_rows=torch.zeros(rows, dtype=torch.bool),
            chain_tip=1,
        )


def test_synchronous_proposes_are_unchanged(adapter):
    """Under synchronous scheduling each propose follows its step: a plain step's propose drafts on device."""
    lane = adapter.models[1]
    for k in range(3):
        sampled = read(adapter, submit(adapter, {SPEC_ROW: 700 + k}, reload=k == 0))
        drafts = propose_plain(adapter, sampled, {SPEC_ROW: 700 + k})
        assert drafts.draft_token_ids[SPEC_ROW].tolist() == [draft_of(701 + k, j) for j in range(MAX_DRAFTS)]
        assert lane.mtp.planned == k + 1
    assert adapter.chain_stats["stale"] == 0
    # And a verify after them proposes from its chain as before.
    committed, positions, counts, _ = verify(adapter, SEQ[703], 703, drafts.draft_token_ids[SPEC_ROW].tolist())
    assert propose(adapter, committed, positions, counts) == [
        draft_of(703 + int(counts[SPEC_ROW]), j) for j in range(MAX_DRAFTS)
    ]
