# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM adapter (generator_vllm.MiMoV2ForCausalLM) with adaptive draft length (MIMO_SPEC_SLOTS=8
MIMO_SPEC_GROUPS=2,4): a lane step drafts 3 tokens per request in 2 groups of 4 slots while the lane holds up to two
requests, and 1 per request in 4 groups of 2 slots with three or four. Fake lanes whose device is the host mirrors (no
device; needs vllm-tt-plugin with SpecPlan.k_by_rows); skipped unless the process runs with both variables set.

Each fake lane keeps a result page per layout and runs a verify pass's argmax as an oracle target (the token after
position q is ``SEQ[q + 1]``), the accept trace as reference.sample_verify.verify_pass followed by the layout's chain
step, and the draft trace as a drafter that drafts the oracle's tokens (wrong at every third bonus) for each drafting
group of the layout, ending with the layout's drafts step. Checked: the plan's per-lane cap; a lane of two requests
drafts 3 tokens each, chained behind every verify; a third request switches the lane to the K = 1 layout (its groups
afresh, every request backfilled in it) and each request then drafts 1 token, chained; when the lane drops back to two
it switches back, backfills again and drafts 3; every step follows the oracle; past four requests the lane drafts for
none."""

from types import SimpleNamespace

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference.sample_verify import control_words, verify_pass
from models.experimental.mimo_v2_6_flash.reference.spec_chain import (
    PAGES,
    chain_step,
    drafts_step,
    relay_step,
    unpack_result,
)
from models.experimental.mimo_v2_6_flash.reference.spec_decode import ADAPTIVE, LAYOUT, LAYOUTS, MAX_DRAFTS, SPEC_SLOTS
from models.experimental.mimo_v2_6_flash.reference.spec_inputs import DESC_A, DESC_COUNT, pack_mtp_commits, pack_verify
from models.experimental.mimo_v2_6_flash.tt import generator_vllm
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import MiMoV2ForCausalLM

pytestmark = pytest.mark.skipif(
    not ADAPTIVE or SPEC_SLOTS != 8 or [layout.groups for layout in LAYOUTS] != [2, 4],
    reason="adaptive draft length: run with MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=2,4",
)
pytest.importorskip("vllm_tt_plugin")
from vllm_tt_plugin.spec_decode import accept_greedy_drafts  # noqa: E402

B, DP, PER_LANE, VOCAB = 32, 2, 8, 1000
MAX_SEQ_LEN = 8192
SEQ = [(7 * q * q + 3 * q + 11) % VOCAB for q in range(MAX_SEQ_LEN + 8)]  # the oracle target's sequence
LANE = 1
K3, K1 = LAYOUTS if len(LAYOUTS) == 2 else (None, None)  # the module skips a launch of one layout


def draft_of(bonus, k):
    """The fake drafter's draft k after the bonus position: the oracle's token, wrong at every third bonus."""
    t = SEQ[bonus + 1 + k]
    return (t + 1) % VOCAB if bonus % 3 == 0 and k == 1 else t


class Chain:
    """One layout's result page."""

    def __init__(self, layout):
        self.layout, self.words = layout, [0] * PAGES[layout].result_words

    def read(self):
        return unpack_result(self.words, self.layout)

    def read_async(self):
        return list(self.words)

    def unpack(self, host):
        return unpack_result(host, self.layout)


class AdaptiveMTP:
    """Every MTP layer in K = 3's groups, layer 0 in K = 1's: per drafting group of the description, its drafts at its
    anchor (the group's last slot) in that layout's page; backfills recorded per (slot, first row, group, layout)."""

    def __init__(self, lane):
        self.lane, self.last_inputs_s = lane, 0.0
        self.chained = self.planned = 0
        self.layers, self.multi, self.history = [0, 1, 2], True, object()
        self.drafting = {K3: [0, 1, 2], K1: [0]}
        self.backfills = []

    def _run(self, layout):
        mtp, gs = self.lane.mtp_words, layout.group_slots
        heads = [[0] * B for _ in range(layout.max_drafts)]
        for g in range(layout.groups):
            desc = mtp[g * B : (g + 1) * B]
            if desc[0] == 0:
                continue
            bonus, anchor = desc[DESC_A] + desc[DESC_COUNT], (g + 1) * gs - 1
            for k in range(layout.max_drafts):
                heads[k][anchor] = draft_of(bonus, k)
        chain = self.lane.spec_chains[layout]
        chain.words[PAGES[layout].drafts_at :] = drafts_step(mtp, heads, layout.max_drafts, layout)

    def enqueue_chained(self, layout=LAYOUT):
        self.chained += 1
        self._run(layout)

    def enqueue_layers(self, commits, layout=LAYOUT):
        self.planned += 1
        self.lane.mtp_words = pack_mtp_commits(B, commits, layout).tolist()
        self._run(layout)

    def enqueue_backfill(self, slot, a, group, layout=LAYOUT):
        self.backfills.append((slot, a, group, layout))

    def read_drafts(self, anchor, group, layout=LAYOUT):
        chain = self.lane.spec_chains[layout].read().groups[group]
        assert chain.draft_anchor == anchor == (group + 1) * layout.group_slots - 1
        return chain.drafts


class Lane:
    """The TtMiMoModel surface the adapter's speculative step uses, with the oracle target and a page per layout."""

    def __init__(self):
        self.max_batch = B
        self.rope_tables = object()
        self.verify_trace = {"accept": 1}
        self.decode_trace = {"id": 0}
        self.mesh_device = None
        self.spec_chains = {layout: Chain(layout) for layout in LAYOUTS}
        self.spec_chain = self.spec_chains[LAYOUT]
        self.verify_sampler = SimpleNamespace(tally=lambda out: None, groups=0, truncated=0)
        self.mtp, self.dflash = AdaptiveMTP(self), None
        self.mtp_words = pack_mtp_commits(B, []).tolist()
        self.verify_words = self.argmax = None
        self.ctrl = control_words([])
        self.verify_layouts = []

    def set_vllm_page_tables(self, rows):
        pass

    def verify_enqueue(self, tokens, positions, chain=None, accept=None, layout=LAYOUT):
        self.verify_layouts.append(layout)
        if accept is not None:
            self.ctrl = control_words(accept[1])
        page = self.spec_chains[layout].words
        upload = pack_verify(tokens, positions, chain).tolist()
        self.verify_words, self.ctrl = relay_step(upload, page, self.ctrl, layout)
        self.argmax = [SEQ[int(p) + 1] if p >= 0 else 0 for p in self.verify_words[B : 2 * B]]
        return {"next_tokens": self.argmax}

    def verify_accept_enqueue(self, slots=None, groups=None, layout=LAYOUT):
        if groups is not None:
            self.ctrl = control_words(groups)
        accept = verify_pass(None, None, self.argmax, self.ctrl, VOCAB)
        self.mtp_words, header = chain_step(self.verify_words, accept, self.argmax, layout.max_drafts, layout)
        self.spec_chains[layout].words[: PAGES[layout].drafts_at] = header

    def decode_enqueue(self, tokens, positions, sample=False):
        return {"next_tokens": [SEQ[int(p) + 1] if p >= 0 else 0 for p in positions.tolist()]}

    def read_next_tokens(self, handle):
        return torch.tensor(handle["next_tokens"])

    def decode_read_rows(self, handle):
        return torch.nn.functional.one_hot(torch.tensor(handle["next_tokens"][: self.max_batch]), VOCAB).float()


def adapter():
    gen = SimpleNamespace(
        models=[Lane() for _ in range(DP)], dp=DP, users_per_replica=PER_LANE, cfg=SimpleNamespace(vocab_size=VOCAB)
    )
    model = MiMoV2ForCausalLM(gen, MAX_SEQ_LEN)
    model.layer_groups = [0]
    return model


TABLE = torch.zeros(DP * PER_LANE, 4, dtype=torch.int32)
ROW = LANE * PER_LANE  # the lane's slot 0


def plain_step(model, at):
    """A plain decode step of the lane's requests ``at`` {row: position} (input SEQ[position]), then the propose of
    MAX_DRAFTS (the plugin's num_speculative_tokens). Returns {row: drafts offered}."""
    rows = DP * PER_LANE
    tokens = torch.zeros(rows, 1, dtype=torch.int32)
    positions = torch.full((rows,), -1, dtype=torch.int32)
    for r, p in at.items():
        tokens[r, 0], positions[r] = SEQ[p], p
    model.decode_forward(tokens, positions, TABLE, None, sampling_params=None, read_from_device=False)
    committed = torch.full((rows, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
    committed_positions = torch.full((rows, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
    for r, p in at.items():
        committed[r, 0], committed_positions[r, 0] = SEQ[p + 1], p + 1
    out = model.propose_draft_tokens(MAX_DRAFTS, committed, committed_positions, torch.ones(rows, dtype=torch.int32))
    return {r: out.draft_token_ids[r].tolist()[: int(out.num_valid[r])] for r in at}


def verify_step(model, seqs, drafts, mode):
    """A verify of the lane's requests (``seqs`` {row: its committed sequence}, ``drafts`` {row: drafts}); returns
    {row: committed tokens} and the next drafts."""
    rows, width = DP * PER_LANE, 1 + MAX_DRAFTS
    tokens = torch.zeros(rows, width, dtype=torch.int32)
    positions = torch.full((rows, width), -1, dtype=torch.int32)
    valid = torch.zeros(rows, dtype=torch.int32)
    for r, seq in seqs.items():
        d = drafts[r]
        tokens[r] = torch.tensor([seq[-1]] + d + [0] * (MAX_DRAFTS - len(d)))
        positions[r, : 1 + len(d)] = torch.arange(len(seq) - 1, len(seq) + len(d))
        valid[r] = len(d)
    out = model.decode_forward(tokens, positions, TABLE, None, spec_mode=mode, num_valid_drafts=valid)
    if mode == "fused_sample":
        committed, counts = out.accepted_token_ids, out.accepted_counts
    else:
        committed, counts = accept_greedy_drafts(out.argmax_ids, tokens[:, 1:], valid)
    first = positions[:, 0].unsqueeze(1)
    committed_positions = (first + torch.arange(1, width + 1)).masked_fill(first < 0, -1).to(torch.int32)
    nxt = model.propose_draft_tokens(MAX_DRAFTS, committed, committed_positions, counts)
    got = {r: committed[r, : int(counts[r])].tolist() for r in seqs}
    return got, {r: nxt.draft_token_ids[r].tolist()[: int(nxt.num_valid[r])] for r in seqs}


def run_verifies(model, seqs, drafts, mode, steps, k):
    """``steps`` verifies: each request follows the oracle and its next k drafts come chained behind the verify."""
    mtp = model.models[LANE].mtp
    for _ in range(steps):
        before = (mtp.chained, mtp.planned)
        got, drafts = verify_step(model, seqs, drafts, mode)
        assert (mtp.chained, mtp.planned) == (before[0] + 1, before[1]), "the drafts chained behind the verify"
        for r in seqs:
            seqs[r] += got[r]
            assert seqs[r] == SEQ[: len(seqs[r])]
            assert drafts[r] == [draft_of(len(seqs[r]) - 1, j) for j in range(k)]
    return drafts


def test_the_plan_caps_each_lane_by_its_load(monkeypatch):
    monkeypatch.setattr(generator_vllm, "SPEC_DECODE", "mtp")
    plan = MiMoV2ForCausalLM.spec_plan(SimpleNamespace(additional_config={}), 8, 3)
    assert plan.effective_k == 3 and plan.k_by_rows == ((2, 3), (8, 1))
    assert [plan.draft_cap(n) for n in range(1, 9)] == [3, 3, 1, 1, 1, 1, 1, 1]


@pytest.mark.parametrize("mode", ["argmax_ids", "fused_sample"])
def test_the_lane_switches_layout_with_its_load(mode):
    model = adapter()
    lane = model.models[LANE]
    mtp = lane.mtp
    two = {ROW: 200, ROW + 1: 1000}

    # Two requests: K = 3, both drafts in one host-planned program, then chained behind every verify.
    drafts = plain_step(model, two)
    assert model.lane_layout[LANE] == K3 and mtp.planned == 1
    assert sorted(b[2:] for b in mtp.backfills) == [(0, K3), (1, K3)]
    for r, p in two.items():
        assert drafts[r] == [draft_of(p + 1, j) for j in range(3)]
    seqs = {r: SEQ[: p + 2] for r, p in two.items()}
    drafts = run_verifies(model, seqs, drafts, mode, 6, 3)
    assert set(lane.verify_layouts) == {K3}

    # A third request joins: the step is the plain decode (it has no drafts), in K = 1's four groups of 2 slots.
    seqs[ROW + 2] = SEQ[:3001]
    drafts = plain_step(model, {r: len(s) - 1 for r, s in seqs.items()})
    for r in seqs:
        seqs[r] = SEQ[: len(seqs[r]) + 1]
        assert drafts[r] == [draft_of(len(seqs[r]) - 1, 0)], "one draft each"
    assert model.lane_layout[LANE] == K1 and model.chain_stats["layout_switches"] == 1
    assert sorted(b[2:] for b in mtp.backfills[2:]) == [(0, K1), (1, K1), (2, K1)], "every request backfilled"
    assert sorted(model.group_owners[LANE][:3]) == [0, 1, 2] and len(model.group_owners[LANE]) == 4
    lane.verify_layouts.clear()
    drafts = run_verifies(model, seqs, drafts, mode, 6, 1)
    assert set(lane.verify_layouts) == {K1}

    # The third finishes: the two left verify their one draft each in K = 3's layout, backfill again, and draft 3.
    del seqs[ROW + 2], drafts[ROW + 2]
    lane.verify_layouts.clear()
    got, drafts = verify_step(model, seqs, drafts, mode)
    for r in seqs:
        seqs[r] += got[r]
        assert seqs[r] == SEQ[: len(seqs[r])]
        assert drafts[r] == [draft_of(len(seqs[r]) - 1, j) for j in range(3)]
    assert model.lane_layout[LANE] == K3 and model.chain_stats["layout_switches"] == 2
    assert sorted(b[2:] for b in mtp.backfills[5:]) == [(0, K3), (1, K3)]
    drafts = run_verifies(model, seqs, drafts, mode, 6, 3)
    assert set(lane.verify_layouts) == {K3}


def test_past_four_requests_the_lane_drafts_for_none():
    model = adapter()
    at = {ROW + s: 100 * (s + 1) for s in range(5)}
    drafts = plain_step(model, at)
    assert model.lane_layout[LANE] == K1 and all(d == [] for d in drafts.values())
    assert model.models[LANE].mtp.planned == 0
