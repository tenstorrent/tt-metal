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

import threading
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


class FakeMesh:
    def num_program_cache_entries(self):
        return 7  # a served prefill compiles nothing: no decode trace recapture


class AsyncLane(FakeLane):
    """FakeLane with the ordinary decode step: the oracle's token after each active slot's position, fed back into
    the resident tokens and positions (TtMiMoModel._feed_back)."""

    def __init__(self):
        super().__init__()
        self.decode_trace = {"id": 0}
        self.mesh_device = FakeMesh()
        self.resident = None
        self.decodes = 0
        self.table_slots = []  # the slots of each set_vllm_page_tables call
        self.prefills = []
        self.mtp_resets = 0
        self.mtp.reset = self._reset_mtp
        self.sampler = SimpleNamespace(host_params=lambda *a, **k: (None, False), set_params=lambda hosts: None)

    def _reset_mtp(self):
        self.mtp_resets += 1

    def _decode(self, pos):
        self.decodes += 1
        tokens = [SEQ[int(p) + 1] if p >= 0 else 0 for p in pos.tolist()]
        self.resident = (tokens, torch.where(pos >= 0, pos + 1, pos))
        return {"next_tokens": FakeTokens(tokens)}

    def set_vllm_page_tables(self, rows):
        self.table_slots.append(sorted(rows))

    def prefill_enqueue(self, chunk, slot, start_pos=0, history_continues=False):
        self.prefills.append((slot, start_pos, chunk.numel()))
        return {"logits": torch.nn.functional.one_hot(torch.tensor(SEQ[start_pos + chunk.numel()]), VOCAB).float()}

    def prefill_read(self, handle):
        return [handle["logits"].reshape(1, -1)]

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


def test_a_deferred_verify_answers_host_tensors(adapter, monkeypatch):
    """``defer_read`` on a fused_sample verify with MIMO_SPEC_DEFER=0: host tensors (the plugin reads it
    synchronously, chaining nothing), which ``read_decode_output`` passes through, as it passes an argmax_ids or
    logits verify's tensor."""
    monkeypatch.setattr(generator_vllm, "SPEC_DEFER", False)
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


# ---------------------------------------------------------------------- deferred verifies (MIMO_SPEC_DEFER)
def block(last, p, drafts):
    """The plugin's candidate block for the speculating row alone: its last committed token at p and its drafts."""
    width = 1 + MAX_DRAFTS
    tokens = torch.zeros(ROWS, width, dtype=torch.int32)
    positions = torch.full((ROWS, width), -1, dtype=torch.int32)
    valid = torch.zeros(ROWS, dtype=torch.int32)
    tokens[SPEC_ROW, : 1 + len(drafts)] = torch.tensor([last] + list(drafts), dtype=torch.int32)
    positions[SPEC_ROW, : 1 + len(drafts)] = torch.arange(p, p + 1 + len(drafts), dtype=torch.int32)
    valid[SPEC_ROW] = len(drafts)
    return tokens, positions, valid


# Every row's block table: real blocks (vLLM's null block 0 nowhere), so a chain row's window is covered.
FULL_TABLE = 1 + torch.arange(ROWS * (MAX_SEQ_LEN // 64), dtype=torch.int32).reshape(ROWS, -1)


def deferred(model, tokens, positions, valid, counts=None, table=FULL_TABLE, **chain):
    return model.decode_forward(
        tokens,
        positions,
        table,
        None,
        spec_mode="fused_sample",
        num_valid_drafts=valid,
        accepted_counts=torch.ones(ROWS, dtype=torch.int32) if counts is None else counts,
        defer_read=True,
        **chain,
    )


def chained(model, tip, bound, hint=MAX_DRAFTS, row=SPEC_ROW, table=FULL_TABLE, **overrides):
    """A verify chained onto deferred verify ``tip``: row ``row``'s block is the device's (placeholder tokens,
    accepted count -1, positions laid out from the provisional ``bound``); every other row a gap."""
    width = 1 + MAX_DRAFTS
    tokens = torch.full((ROWS, width), -1, dtype=torch.int32)
    positions = torch.full((ROWS, width), -1, dtype=torch.int32)
    positions[row, : 1 + hint] = torch.arange(bound, bound + 1 + hint, dtype=torch.int32)
    valid = torch.zeros(ROWS, dtype=torch.int32)
    valid[row] = hint
    counts = torch.ones(ROWS, dtype=torch.int32)
    counts[row] = -1
    chain_rows = torch.zeros(ROWS, dtype=torch.bool)
    chain_rows[row] = True
    args = dict(tokens=tokens, positions=positions, valid=valid, counts=counts, chain_rows=chain_rows, chain_tip=tip)
    args.update(overrides)
    return deferred(model, args.pop("tokens"), args.pop("positions"), args.pop("valid"), table=table, **args)


def read_on_thread(model, handle):
    """``read_decode_output(handle)`` where the plugin resolves deferred output: another thread."""
    box = []
    reader = threading.Thread(target=lambda: box.append(model.read_decode_output(handle)))
    reader.start()
    reader.join(timeout=30)
    assert box, "the read did not finish"
    return box[0]


def commit(readout):
    """The plugin's propose arguments after committing a FusedReadout: positions from p0 (p0 + 1 + j, -1 on gaps)."""
    width = readout.accepted_token_ids.shape[1]
    first = readout.p0.to(torch.int64).unsqueeze(1)
    positions = (first + torch.arange(1, width + 1)).masked_fill(first < 0, -1).to(torch.int32)
    return readout.accepted_token_ids, positions, readout.accepted_counts


@pytest.fixture
def defer(monkeypatch):
    monkeypatch.setattr(generator_vllm, "SPEC_DEFER", True)


def test_deferred_verifies_follow_the_oracle(adapter, defer):
    """Each fused verify returns a handle (increasing tip ids, no host tensors); its read, on another thread, is the
    synchronous answer with p0 the submitted column 0; its propose returns the drafts its page carries, enqueueing
    nothing. Greedy decoding through them reproduces the oracle."""
    from vllm_tt_plugin.spec_decode import FusedReadout

    lane = adapter.models[1]
    p = 200
    drafts = start(adapter, p)
    seq, tips = SEQ[: p + 1], []
    while len(seq) < 400:
        tokens, positions, valid = block(seq[-1], len(seq) - 1, drafts)
        reads = lane.reads
        out = deferred(adapter, tokens, positions, valid)
        assert out.accepted_token_ids is None and out.accepted_counts is None
        handle = out.handle
        tips.append(handle.tip_id)
        assert lane.reads == reads + 1, "one page copy enqueued behind the pass"
        readout = read_on_thread(adapter, handle)
        assert isinstance(readout, FusedReadout) and adapter.read_decode_output(handle) is readout
        assert readout.p0.dtype == torch.int32 and torch.equal(readout.p0, positions[:, 0])
        m = int(readout.accepted_counts[SPEC_ROW])
        assert readout.accepted_token_ids[SPEC_ROW, :m].tolist() == SEQ[len(seq) : len(seq) + m]
        assert readout.accepted_counts.tolist() == [1] * SPEC_ROW + [m] + [1] * (ROWS - SPEC_ROW - 1)
        seq += readout.accepted_token_ids[SPEC_ROW, :m].tolist()
        planned = lane.mtp.planned
        out = adapter.propose_draft_tokens(MAX_DRAFTS, *commit(readout), hidden=handle)
        drafts = out.draft_token_ids[SPEC_ROW].tolist()
        assert int(out.num_valid.sum()) == MAX_DRAFTS and lane.mtp.planned == planned, "nothing enqueued"
        assert drafts == [draft_of(len(seq) - 1, k) for k in range(MAX_DRAFTS)]
        with pytest.raises(RuntimeError, match="proposed a second time"):
            adapter.propose_draft_tokens(MAX_DRAFTS, *commit(readout), hidden=handle)
    assert seq == SEQ[: len(seq)]
    assert tips == sorted(set(tips)) and all(isinstance(t, int) for t in tips)
    assert adapter.chain_stats["chained"] == len(tips) and adapter.chain_stats["host"] == 1


def test_a_handle_outlives_later_steps(adapter, defer):
    """A handle's read and propose come after later steps ran on its lane (a plain step here): its page was copied
    behind its own pass. A propose without a handle drafts nothing from a deferred verify, and a row whose request is
    gone (position -1) is offered none."""
    lane = adapter.models[1]
    drafts = start(adapter, 900)
    out = deferred(adapter, *block(SEQ[900], 900, drafts))
    planned = lane.mtp.planned
    tokens, positions, _ = block(SEQ[900], 900, [])
    plain = adapter.propose_draft_tokens(MAX_DRAFTS, tokens, positions, torch.ones(ROWS, dtype=torch.int32))
    assert int(plain.num_valid.sum()) == 0 and lane.mtp.planned == planned
    assert int(read(adapter, submit(adapter, {SPEC_ROW: 950}))[SPEC_ROW]) == SEQ[951]
    readout = read_on_thread(adapter, out.handle)
    m = int(readout.accepted_counts[SPEC_ROW])
    assert readout.accepted_token_ids[SPEC_ROW, :m].tolist() == SEQ[901 : 901 + m]
    got = adapter.propose_draft_tokens(MAX_DRAFTS, *commit(readout), hidden=out.handle)
    assert got.draft_token_ids[SPEC_ROW].tolist() == [draft_of(900 + m, k) for k in range(MAX_DRAFTS)]
    out = deferred(adapter, *block(SEQ[900 + m], 900 + m, got.draft_token_ids[SPEC_ROW].tolist()))
    committed, positions, counts = commit(read_on_thread(adapter, out.handle))
    positions[SPEC_ROW] = -1
    gone = adapter.propose_draft_tokens(MAX_DRAFTS, committed, positions, counts, hidden=out.handle)
    assert int(gone.num_valid.sum()) == 0


@pytest.mark.parametrize("why", ["no_chain", "no_rope_tables"])
def test_without_the_chain_a_deferred_verify_answers_host_tensors(adapter, defer, monkeypatch, why):
    if why == "no_chain":
        monkeypatch.setattr(generator_vllm, "SPEC_CHAIN", False)
    else:
        adapter.models[0].rope_tables = None
    drafts = start(adapter, 1200)
    out = deferred(adapter, *block(SEQ[1200], 1200, drafts))
    assert out.handle is None and out.accepted_token_ids is not None


# ---------------------------------------------------------------------- chained verifies (the relay step)
def run_chained(model, p, steps_to=400):
    """The plugin's chained lane loop for the request alone in lane 1: a host-built deferred verify, then each verify
    chained onto the one in flight (provisional column 0: the applied length plus 1 + K for the verify in flight,
    minus one), the older one read (on another thread), committed and proposed after the newer one's submission.
    Returns the committed sequence and each step's count."""
    drafts = start(model, p)
    seq, counts = SEQ[: p + 1], []
    pending = deferred(model, *block(seq[-1], len(seq) - 1, drafts)).handle
    while len(seq) < steps_to:
        bound = len(seq) - 1 + (1 + MAX_DRAFTS)
        newer = chained(model, pending.tip_id, bound).handle
        assert newer.tip_id > pending.tip_id
        readout = read_on_thread(model, pending)
        m = int(readout.accepted_counts[SPEC_ROW])
        assert int(readout.p0[SPEC_ROW]) == len(seq) - 1, "the device's column 0: one past the applied history"
        assert readout.accepted_token_ids[SPEC_ROW, :m].tolist() == SEQ[len(seq) : len(seq) + m]
        seq += readout.accepted_token_ids[SPEC_ROW, :m].tolist()
        counts.append(m)
        # The chained verify consumed these drafts on device; the plugin proposes anyway and publishes nothing.
        out = model.propose_draft_tokens(MAX_DRAFTS, *commit(readout), hidden=pending)
        assert out.draft_token_ids[SPEC_ROW].tolist() == [draft_of(len(seq) - 1, k) for k in range(MAX_DRAFTS)]
        pending = newer
    readout = read_on_thread(model, pending)
    m = int(readout.accepted_counts[SPEC_ROW])
    seq += readout.accepted_token_ids[SPEC_ROW, :m].tolist()
    counts.append(m)
    model.propose_draft_tokens(MAX_DRAFTS, *commit(readout), hidden=pending)
    return seq, counts


def run_synchronous(model, p, steps):
    """The same request on the synchronous path: each fused verify read before the next is built."""
    drafts = start(model, p)
    seq, counts = SEQ[: p + 1], []
    for _ in range(steps):
        committed, positions, c, _ = verify(model, seq[-1], len(seq) - 1, drafts, "fused_sample")
        m = int(c[SPEC_ROW])
        seq += committed[SPEC_ROW, :m].tolist()
        counts.append(m)
        drafts = propose(model, committed, positions, c)
    return seq, counts


def test_chained_greedy_equals_the_synchronous_path(adapter, defer):
    lane = adapter.models[1]
    seq, counts = run_chained(adapter, 200)
    assert seq == SEQ[: len(seq)]
    assert lane.mtp.planned == 1, "only the first draft is host-planned: every other one chained on device"
    assert adapter.chain_stats["uncovered"] == 0
    sync = MiMoV2ForCausalLM(adapter.gen, MAX_SEQ_LEN)
    sync.layer_groups = [0]
    sync_seq, sync_counts = run_synchronous(sync, 200, len(counts))
    assert sync_seq == seq and sync_counts == counts, "token for token, step for step"
    assert sum(counts) > len(counts), "some drafts accepted"


def test_the_chain_survives_the_ring_and_reads_its_pages_late(adapter, defer):
    """Across the MTP ring's wrap (256), with several chained verifies enqueued before the oldest is read: each
    handle's page was copied behind its own pass."""
    drafts = start(adapter, 240)
    seq = SEQ[:241]
    handles = [deferred(adapter, *block(seq[-1], 240, drafts)).handle]
    for k in range(5):
        # Unread steps in flight: the provisional position counts each at its full width.
        handles.append(chained(adapter, handles[-1].tip_id, 240 + (k + 1) * (1 + MAX_DRAFTS)).handle)
    for h in handles:
        readout = read_on_thread(adapter, h)
        m = int(readout.accepted_counts[SPEC_ROW])
        assert int(readout.p0[SPEC_ROW]) == len(seq) - 1
        assert readout.accepted_token_ids[SPEC_ROW, :m].tolist() == SEQ[len(seq) : len(seq) + m]
        seq += readout.accepted_token_ids[SPEC_ROW, :m].tolist()
        adapter.propose_draft_tokens(MAX_DRAFTS, *commit(readout), hidden=h)
    assert seq == SEQ[: len(seq)] and len(seq) > 256


def test_fewer_drafts_than_the_device_made(adapter, defer):
    """A chain row verifying fewer drafts than the device drafted (a plugin K below MAX_DRAFTS, or a draft cap)."""
    drafts = start(adapter, 300)
    pending = deferred(adapter, *block(SEQ[300], 300, drafts)).handle
    newer = chained(adapter, pending.tip_id, 300 + 1 + MAX_DRAFTS, hint=1).handle
    first = read_on_thread(adapter, pending)
    m = int(first.accepted_counts[SPEC_ROW])
    second = read_on_thread(adapter, newer)
    m2 = int(second.accepted_counts[SPEC_ROW])
    assert int(second.p0[SPEC_ROW]) == 300 + m and 1 <= m2 <= 2
    assert second.accepted_token_ids[SPEC_ROW, :m2].tolist() == SEQ[301 + m : 301 + m + m2]


def contract_case(adapter, case):
    drafts = start(adapter, 400)
    tip = deferred(adapter, *block(SEQ[400], 400, drafts)).handle.tip_id
    bound = 400 + 1 + MAX_DRAFTS
    if case == "tip":
        chained(adapter, tip + 1, bound)
    elif case == "after_plain":
        read(adapter, submit(adapter, {0: 30}))
        chained(adapter, tip, bound)
    elif case == "no_source":
        chained(adapter, tip, bound, row=0)
    elif case == "host_tokens":
        chained(adapter, tip, bound, tokens=block(SEQ[404], bound, [1, 2, 3])[0])
    elif case == "count":
        counts = torch.ones(ROWS, dtype=torch.int32)
        chained(adapter, tip, bound, counts=counts)
    elif case == "hint":
        chained(adapter, tip, bound, hint=0)
    elif case == "bound":
        chained(adapter, tip, bound - 1)
    elif case == "neighbour":
        args = chained_inputs = None
        del args, chained_inputs
        positions = torch.full((ROWS, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
        positions[SPEC_ROW] = torch.arange(bound, bound + 1 + MAX_DRAFTS)
        positions[SPEC_ROW + 1, 0] = 50
        chained(adapter, tip, bound, positions=positions)
    elif case == "not_deferred":
        rows = torch.zeros(ROWS, dtype=torch.bool)
        adapter.decode_forward(
            *block(SEQ[404], bound, [])[:2],
            FULL_TABLE,
            None,
            spec_mode="fused_sample",
            num_valid_drafts=torch.zeros(ROWS, dtype=torch.int32),
            chain_rows=rows,
            chain_tip=tip,
        )


@pytest.mark.parametrize(
    "case, match",
    [
        ("tip", "chain_tip"),
        ("after_plain", "chain_tip"),
        ("no_source", "drafted nothing"),
        ("host_tokens", "host tokens"),
        ("count", "accepted_counts -1"),
        ("hint", "asks to verify 0"),
        ("bound", "below the device"),
        ("neighbour", "beside chain row"),
        ("not_deferred", "deferred"),
    ],
)
def test_a_chained_verify_outside_the_contract_raises(adapter, defer, case, match):
    with pytest.raises(ValueError, match=match):
        contract_case(adapter, case)


def test_a_chain_onto_a_declined_draft_raises_at_the_read(adapter, defer):
    """The step chained onto declined its draft at the context limit (the plugin stops chaining 2 (1 + K) short of
    max_model_len, so only a contract slip gets here): the relay idles the pass and the read raises."""
    p = MAX_SEQ_LEN - 1 - MAX_DRAFTS - 2
    drafts = [SEQ[p + 1 + k] for k in range(MAX_DRAFTS)]  # every draft accepted: the bonus reaches the limit
    start(adapter, p)
    pending = deferred(adapter, *block(SEQ[p], p, drafts)).handle
    newer = chained(adapter, pending.tip_id, p + 1 + MAX_DRAFTS).handle
    readout = read_on_thread(adapter, pending)
    assert int(readout.accepted_counts[SPEC_ROW]) == 1 + MAX_DRAFTS
    with pytest.raises(RuntimeError, match="drafted nothing on device"):
        adapter.read_decode_output(newer)


def test_a_hole_in_a_chain_row_s_page_tables_is_counted(adapter, defer):
    """vLLM frees sliding-window blocks by the scheduler's optimistic length: a block the device's real window still
    reads, freed (null block 0), is counted and logged."""
    drafts = start(adapter, 1000)
    tip = deferred(adapter, *block(SEQ[1000], 1000, drafts)).handle.tip_id
    table = FULL_TABLE.clone()
    table[SPEC_ROW, (1001 - 127) // 64] = 0
    chained(adapter, tip, 1000 + 1 + MAX_DRAFTS, table=table)
    assert adapter.chain_stats["uncovered"] == 1


# ---------------------------------------------------------------------- finishing slots and a prefill over a verify
def test_capabilities_of_the_chained_lane_path():
    caps = MiMoV2ForCausalLM.model_capabilities
    assert caps["supports_finishing_slots"] and caps["supports_prefill_over_pending_spec"]


def test_finishing_slots_are_taken_on_any_decode_step(adapter, defer):
    drafts = start(adapter, 300)
    out = deferred(adapter, *block(SEQ[300], 300, drafts), finishing_slots=[1, 2])
    assert out.handle is not None and adapter.chain_stats["finishing"] == 2
    read(adapter, submit(adapter, {SPEC_ROW: 310}))
    tokens = torch.zeros(ROWS, 1, dtype=torch.int32)
    positions = torch.full((ROWS,), -1, dtype=torch.int32)
    adapter.decode_forward(tokens, positions, TABLE, None, sampling_params=GREEDY, finishing_slots=[0])
    with pytest.raises(ValueError, match="finishing_slots"):
        adapter.decode_forward(tokens, positions, TABLE, None, sampling_params=GREEDY, finishing_slots=[3, 1])


def test_a_prefill_over_a_deferred_verify(adapter, defer, monkeypatch):
    """M15: a prefill into another slot runs while a deferred verify is in flight. It writes only its slot's page
    tables, the verify is read and proposed after it as if nothing had happened, and nothing chains onto it."""
    monkeypatch.setattr(generator_vllm, "PREFILL_BATCH", False)
    lane = adapter.models[1]
    drafts = start(adapter, 500)
    pending = deferred(adapter, *block(SEQ[500], 500, drafts)).handle
    lane.table_slots.clear()
    prompt = torch.tensor([SEQ[:40]], dtype=torch.int32)
    logits = adapter.prefill_forward(
        prompt, torch.zeros(1, dtype=torch.int32), torch.tensor([40]), FULL_TABLE, None, empty_slots=[SPEC_ROW + 2]
    )
    assert int(logits[0, -1].argmax()) == SEQ[40] and lane.prefills == [(2, 0, 40)]
    assert lane.table_slots == [[2]], "the prefill writes only its slot's page-table rows"
    assert lane.mtp_resets == 1 and adapter.models[0].mtp_resets == 0
    readout = read_on_thread(adapter, pending)
    m = int(readout.accepted_counts[SPEC_ROW])
    assert readout.accepted_token_ids[SPEC_ROW, :m].tolist() == SEQ[501 : 501 + m]
    out = adapter.propose_draft_tokens(MAX_DRAFTS, *commit(readout), hidden=pending)
    assert out.draft_token_ids[SPEC_ROW].tolist() == [draft_of(500 + m, k) for k in range(MAX_DRAFTS)]
    with pytest.raises(ValueError, match="chain_tip"):
        chained(adapter, pending.tip_id, 500 + 1 + MAX_DRAFTS)
