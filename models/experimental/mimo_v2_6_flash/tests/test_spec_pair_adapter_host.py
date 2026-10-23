# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM adapter's pair draft (generator_vllm.MiMoV2ForCausalLM: a lane of two requests drafts one token for each)
on fake lanes whose device is the host mirrors (test_spec_chain_adapter_host.py's and test_spec_async_adapter_host.py's:
an oracle target, the accept, chain, drafts and relay steps' mirrors, a drafter that drafts the oracle's tokens with
errors that depend on the bonus and on the MTP ring its cache positions address); no device, needs vllm-tt-plugin.

The plugin's side is played by ``Plugin``: one request per lane slot, the draft cap per lane's live requests
(``spec_plan``'s k_by_rows: 3 for one, 1 for two, 0 here for more, which MiMo drafts nothing for), the carried drafts
trimmed to fit the lane's 4 slots, prompts prefilled into their slots. Lane 1 goes through 1 -> 2 -> 1 -> 3 -> 2
requests, synchronously and under async scheduling with deferred verifies chained onto each other where the plugin
chains them (an unchanged layout whose every row the device drafted for). Checked: every request commits the oracle's
tokens step for step (speculative greedy equals plain greedy), every propose offers the drafts the lane's population
calls for (MAX_DRAFTS alone, one each in a pair, none for three) and they are those drafted in the request's own ring,
the pair draft runs where expected, and the relayed backfill of a pair source going on alone."""

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference.spec_decode import MAX_DRAFTS, SPEC_SLOTS
from models.experimental.mimo_v2_6_flash.tests.test_spec_async_adapter_host import (  # noqa: F401
    FULL_TABLE,
    ROWS,
    adapter,
    commit,
    defer,
    read_on_thread,
    with_history,
)
from models.experimental.mimo_v2_6_flash.tests.test_spec_chain_adapter_host import PER_LANE, SEQ, draft_of
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import MiMoV2ForCausalLM, MtpRing

pytest.importorskip("vllm_tt_plugin")
from vllm_tt_plugin.spec_decode import accept_greedy_drafts  # noqa: E402

LANE = 1
FIRST = LANE * PER_LANE
WIDTH = 1 + MAX_DRAFTS


@pytest.fixture(autouse=True)
def eager_prefill(monkeypatch):
    """Prompts prefill one by one (the fake lanes have no batched prefill)."""
    from models.experimental.mimo_v2_6_flash.tt import generator_vllm

    monkeypatch.setattr(generator_vllm, "PREFILL_BATCH", False)


def test_spec_plan_caps_drafts_per_lane_rows(monkeypatch):
    from models.experimental.mimo_v2_6_flash.tt import generator_vllm

    monkeypatch.setattr(generator_vllm, "SPEC_DECODE", "mtp")
    monkeypatch.setattr(generator_vllm, "_untraced_decode", lambda config: None)
    plan = MiMoV2ForCausalLM.spec_plan(None, 4, 3)
    assert plan.k_by_rows == ((1, 3), (2, 1), (4, 1)) and plan.lanes_per_request == 4
    assert [plan.draft_cap(n) for n in (1, 2, 3, 4)] == [3, 1, 1, 1]
    monkeypatch.setattr(generator_vllm, "SPEC_PAIR", False)
    assert MiMoV2ForCausalLM.spec_plan(None, 4, 3).k_by_rows == ()


class Plugin:
    """The plugin's view of lane 1: {slot: [the last committed position, the drafts it carries]}."""

    def __init__(self, model):
        self.model, self.reqs = model, {}
        self.steps = []  # (population, counts per slot) per verify

    def join(self, *slots, length=None):
        """Prompts of ``length`` tokens prefilled into ``slots`` (the oracle's sequence): their first token at
        position length."""
        for slot in slots:
            n = length or 300 + 37 * slot
            prompt = torch.tensor([SEQ[:n]], dtype=torch.int32)
            logits = self.model.prefill_forward(
                prompt,
                torch.zeros(1, dtype=torch.int32),
                torch.tensor([n]),
                FULL_TABLE,
                None,
                empty_slots=[FIRST + slot],
            )
            assert int(logits[0, -1].argmax()) == SEQ[n]
            self.reqs[slot] = [n, []]

    def leave(self, *slots):
        for slot in slots:
            del self.reqs[slot]

    def cap(self):
        return {1: MAX_DRAFTS, 2: 1}.get(len(self.reqs), 0)

    def block(self):
        """The candidate block of the lane: each request's drafts cut to the cap, then trimmed (the last slot first)
        until the lane fits SPEC_SLOTS slots."""
        drafts = {slot: d[: self.cap()] for slot, (_, d) in self.reqs.items()}
        while sum(1 + len(d) for d in drafts.values()) > SPEC_SLOTS:
            slot = max(s for s, d in drafts.items() if d)
            drafts[slot] = drafts[slot][:-1]
        tokens = torch.zeros(ROWS, WIDTH, dtype=torch.int32)
        positions = torch.full((ROWS, WIDTH), -1, dtype=torch.int32)
        valid = torch.zeros(ROWS, dtype=torch.int32)
        for slot, (p, _) in self.reqs.items():
            r, d = FIRST + slot, drafts[slot]
            tokens[r, : 1 + len(d)] = torch.tensor([SEQ[p]] + d, dtype=torch.int32)
            positions[r, : 1 + len(d)] = torch.arange(p, p + 1 + len(d), dtype=torch.int32)
            valid[r] = len(d)
        return tokens, positions, valid

    def expect(self, valid):
        """How many drafts each request is offered after a step that verified ``valid``: MAX_DRAFTS alone, one each
        for a pair that verified at most one each, else none."""
        n = len(self.reqs)
        if n == 1:
            return MAX_DRAFTS
        if n == 2 and all(int(valid[FIRST + s]) <= 1 for s in self.reqs):
            return 1
        return 0

    def apply(self, accepted, counts, drafts, valid, check_drafts=True):
        """Commit the step (the oracle's tokens) and take the proposed drafts."""
        want = self.expect(valid)
        self.steps.append((len(self.reqs), {s: int(counts[FIRST + s]) for s in self.reqs}))
        for slot, req in self.reqs.items():
            r, p = FIRST + slot, req[0]
            m = int(counts[r])
            assert accepted[r, :m].tolist() == SEQ[p + 1 : p + 1 + m], (slot, p, m)
            req[0] = p + m
            got = drafts.draft_token_ids[r, : int(drafts.num_valid[r])].tolist()
            if check_drafts:
                assert len(got) == want, (slot, len(self.reqs), got)
                assert got == [draft_of(req[0], k, slot) for k in range(want)], (slot, req[0], got)
            req[1] = got
        for r in range(ROWS):
            if r // PER_LANE != LANE or r - FIRST not in self.reqs:
                assert int(drafts.num_valid[r]) == 0

    def step(self, mode):
        tokens, positions, valid = self.block()
        out = self.model.decode_forward(tokens, positions, FULL_TABLE, None, spec_mode=mode, num_valid_drafts=valid)
        if mode == "fused_sample":
            accepted, counts = out.accepted_token_ids, out.accepted_counts
        elif mode == "logits":
            accepted, counts = accept_greedy_drafts(out.logits.argmax(-1).to(torch.int32), tokens[:, 1:], valid)
        else:
            accepted, counts = accept_greedy_drafts(out.argmax_ids, tokens[:, 1:], valid)
        first = positions[:, 0].unsqueeze(1).to(torch.int64)
        cpos = (first + torch.arange(1, WIDTH + 1)).masked_fill(first < 0, -1).to(torch.int32)
        drafts = self.model.propose_draft_tokens(MAX_DRAFTS, accepted, cpos, counts)
        self.apply(accepted, counts, drafts, valid)


SCHEDULE = [("join", (0,)), 6, ("join", (1,)), 6, ("leave", (1,)), 6, ("join", (1, 2)), 4, ("leave", (0,)), 6]


@pytest.mark.parametrize("check", [False, True], ids=["", "chain_check"])
@pytest.mark.parametrize("mode", ["argmax_ids", "fused_sample", "logits"])
def test_lane_transitions_follow_the_oracle(adapter, mode, check):
    """1 -> 2 -> 1 -> 3 -> 2 requests, synchronous: every request commits the oracle's tokens; one request drafts
    three (from its ring), a pair one each (from theirs, the pair draft chained behind each verify, or after a logits
    verify drafted from the host plan), three none. MIMO_SPEC_CHAIN_CHECK drafts every chained draft again from the
    host plan, pairs with the pair draft: no difference."""
    adapter.chain_check = check
    plugin, lane = Plugin(adapter), adapter.models[LANE]
    for item in SCHEDULE:
        if isinstance(item, tuple):
            getattr(plugin, item[0])(*item[1])
            continue
        for _ in range(item):
            plugin.step(mode)
    populations = [n for n, _ in plugin.steps]
    assert populations == [1] * 6 + [2] * 6 + [1] * 6 + [3] * 4 + [2] * 6
    counts = {c for n, cs in plugin.steps if n == 2 for c in cs.values()}
    assert counts == {1, 2}, "pair drafts accepted and rejected"
    assert lane.mtp.pair_runs >= 8
    assert lane.mtp.chained >= 20 if mode != "logits" else lane.mtp.chained == 0 and lane.mtp.planned >= 20
    assert adapter.models[0].mtp.chained == adapter.models[0].mtp.planned == 0
    stats = adapter.chain_stats
    assert stats["mismatched"] == 0 and (stats["checked"] > 0) == (check and mode != "logits")


def test_lane_transitions_with_the_pair_draft_off(adapter, monkeypatch):
    """MIMO_SPEC_PAIR=0: a lane of two requests drafts nothing (the A/B baseline)."""
    from models.experimental.mimo_v2_6_flash.tt import generator_vllm

    monkeypatch.setattr(generator_vllm, "SPEC_PAIR", False)
    plugin = Plugin(adapter)
    plugin.expect = lambda valid: MAX_DRAFTS if len(plugin.reqs) == 1 else 0
    for item in SCHEDULE[:6]:
        if isinstance(item, tuple):
            getattr(plugin, item[0])(*item[1])
        else:
            for _ in range(item):
                plugin.step("fused_sample")
    assert adapter.models[LANE].mtp.pair_runs == 0


class AsyncPlugin(Plugin):
    """The plugin under async scheduling: deferred fused verifies, the next one chained onto the one in flight when
    the lane's layout is unchanged and the device drafted for every row (its provisional column 0: the applied
    position plus the in-flight step's full width), else read first and built on the host."""

    def __init__(self, model):
        super().__init__(model)
        self.pending = None  # (handle, the block's valid drafts, the population it ran)
        self.chained = 0

    def resolve(self):
        """Read the verify in flight (on a reader thread), commit it and propose from its handle."""
        handle, valid, reqs = self.pending
        readout = read_on_thread(self.model, handle)
        accepted, cpos, counts = commit(readout)
        drafts = self.model.propose_draft_tokens(MAX_DRAFTS, accepted, cpos, counts, hidden=handle)
        saved, self.reqs = self.reqs, reqs
        self.apply(accepted, counts, drafts, valid)
        self.reqs = {s: reqs.get(s, saved[s]) for s in saved}
        self.pending = None

    def flush(self):
        if self.pending is not None:
            self.resolve()

    def step(self):
        pending = self.pending
        reqs = {s: list(v) for s, v in self.reqs.items()}
        layout = pending is not None and set(pending[2]) == set(self.reqs) and self.expect(pending[1]) > 0
        if layout:
            handle, valid, _ = pending
            hint = min(self.cap(), self.expect(valid))
            tokens = torch.full((ROWS, WIDTH), -1, dtype=torch.int32)
            positions = torch.full((ROWS, WIDTH), -1, dtype=torch.int32)
            nvalid = torch.zeros(ROWS, dtype=torch.int32)
            counts = torch.ones(ROWS, dtype=torch.int32)
            chain_rows = torch.zeros(ROWS, dtype=torch.bool)
            for slot, (p, _) in self.reqs.items():
                r = FIRST + slot
                bound = p + 1 + int(valid[r])
                positions[r, : 1 + hint] = torch.arange(bound, bound + 1 + hint, dtype=torch.int32)
                nvalid[r], counts[r], chain_rows[r] = hint, -1, True
            out = self.model.decode_forward(
                tokens,
                positions,
                FULL_TABLE,
                None,
                spec_mode="fused_sample",
                num_valid_drafts=nvalid,
                accepted_counts=counts,
                defer_read=True,
                chain_rows=chain_rows,
                chain_tip=handle.tip_id,
            )
            self.chained += 1
            self.resolve()  # the older one, after the newer one's submission
            self.pending = (out.handle, nvalid, {s: list(v) for s, v in self.reqs.items()})
            return
        self.flush()
        tokens, positions, valid = self.block()
        out = self.model.decode_forward(
            tokens,
            positions,
            FULL_TABLE,
            None,
            spec_mode="fused_sample",
            num_valid_drafts=valid,
            accepted_counts=torch.ones(ROWS, dtype=torch.int32),
            defer_read=True,
        )
        del reqs
        self.pending = (out.handle, valid, {s: list(v) for s, v in self.reqs.items()})


@pytest.mark.parametrize("check", [False, True], ids=["", "chain_check"])
def test_lane_transitions_chained_follow_the_synchronous_path(adapter, defer, check):
    """The same schedule under async scheduling, chained where the plugin chains: the oracle's tokens, the drafts of
    each population, and the same counts step for step as the synchronous path. MIMO_SPEC_CHAIN_CHECK redrafts every
    deferred verify's drafts (a pair's with the pair draft) right after its enqueue: no difference."""
    adapter.chain_check = check
    plugin, lane = AsyncPlugin(adapter), adapter.models[LANE]
    for item in SCHEDULE:
        if isinstance(item, tuple):
            plugin.flush()
            getattr(plugin, item[0])(*item[1])
            continue
        for _ in range(item):
            plugin.step()
    plugin.flush()
    assert [n for n, _ in plugin.steps] == [1] * 6 + [2] * 6 + [1] * 6 + [3] * 4 + [2] * 6
    assert plugin.chained >= 15, "single and pair verifies chained"
    assert (lane.mtp.planned == 0) == (not check), "every draft drafted on device behind its verify"
    assert adapter.chain_stats["mismatched"] == 0 and (adapter.chain_stats["checked"] > 0) == check
    adapter.chain_check = False
    sync = MiMoV2ForCausalLM(adapter.gen, adapter.max_seq_len)
    sync.layer_groups = [0]
    for lane_model in adapter.models:
        lane_model.mtp.reset()
    replay = Plugin(sync)
    for item in SCHEDULE:
        if isinstance(item, tuple):
            getattr(replay, item[0])(*item[1])
            continue
        for _ in range(item):
            replay.step("fused_sample")
    assert replay.steps == plugin.steps, "token for token, step for step"


def test_a_pair_source_going_on_alone_backfills_on_device(adapter, defer):
    """A verify chained onto a pair's with one of its rows (the other request finished): the request now drafts
    alone, and with every MTP layer its layers 1 and 2 never saw the pair's rows: the backfill runs, reading its first
    row from the relayed upload (verify slot 0). With one layer the ring continues."""
    lane = adapter.models[LANE]
    with_history(lane)
    plugin = AsyncPlugin(adapter)
    plugin.join(0, 1)
    plugin.step()
    plugin.flush()  # host-built: the pair's first drafts (and its backfills after the prompts)
    plugin.step()  # host-built deferred, the pair drafted behind it
    handle, valid, _ = plugin.pending
    backfills = list(lane.mtp.backfills)
    assert [b[:2] for b in backfills] == [(0, 300), (1, 337)]
    # Slot 1's request finished: its row is gone from the chained verify.
    p = plugin.reqs[0][0]
    tokens = torch.full((ROWS, WIDTH), -1, dtype=torch.int32)
    positions = torch.full((ROWS, WIDTH), -1, dtype=torch.int32)
    positions[FIRST, :2] = torch.tensor([p + 2, p + 3], dtype=torch.int32)
    nvalid, counts, rows = (
        torch.zeros(ROWS, dtype=torch.int32),
        torch.ones(ROWS, dtype=torch.int32),
        torch.zeros(ROWS, dtype=torch.bool),
    )
    nvalid[FIRST], counts[FIRST], rows[FIRST] = 1, -1, True
    out = adapter.decode_forward(
        tokens,
        positions,
        FULL_TABLE,
        None,
        spec_mode="fused_sample",
        num_valid_drafts=nvalid,
        accepted_counts=counts,
        defer_read=True,
        chain_rows=rows,
        chain_tip=handle.tip_id,
    )
    new = lane.mtp.backfills[len(backfills) :]
    if lane.mtp.multi:
        assert [b[:2] for b in new] == [(0, ("relay", 0))]
        assert adapter.mtp_ring[LANE][0] == MtpRing(slot=0, handle=out.handle)
    else:
        assert new == []
    readout = read_on_thread(adapter, out.handle)
    m = int(readout.accepted_counts[FIRST])
    p0 = int(readout.p0[FIRST])
    assert readout.accepted_token_ids[FIRST, :m].tolist() == SEQ[p0 + 1 : p0 + 1 + m]
    got = adapter.propose_draft_tokens(MAX_DRAFTS, *commit(readout), hidden=out.handle)
    assert got.draft_token_ids[FIRST].tolist() == [draft_of(p0 + m, k, 0) for k in range(MAX_DRAFTS)]


def test_a_pair_chain_row_verifies_at_most_one_draft(adapter, defer, expect_error):
    plugin = AsyncPlugin(adapter)
    plugin.join(0, 1)
    plugin.step()
    plugin.flush()
    plugin.step()
    handle = plugin.pending[0]
    positions = torch.full((ROWS, WIDTH), -1, dtype=torch.int32)
    rows, nvalid, counts = (
        torch.zeros(ROWS, dtype=torch.bool),
        torch.zeros(ROWS, dtype=torch.int32),
        torch.ones(ROWS, dtype=torch.int32),
    )
    for slot, (p, _) in plugin.reqs.items():
        positions[FIRST + slot, :3] = torch.arange(p + 2, p + 5, dtype=torch.int32)
        rows[FIRST + slot], nvalid[FIRST + slot], counts[FIRST + slot] = True, 2, -1
    with expect_error(ValueError, "the device drafted 1"):
        adapter.decode_forward(
            torch.full((ROWS, WIDTH), -1, dtype=torch.int32),
            positions,
            FULL_TABLE,
            None,
            spec_mode="fused_sample",
            num_valid_drafts=nvalid,
            accepted_counts=counts,
            defer_read=True,
            chain_rows=rows,
            chain_tip=handle.tip_id,
        )
