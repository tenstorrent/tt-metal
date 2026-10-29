# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM adapter (generator_vllm.MiMoV2ForCausalLM) with draft groups (MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=2): two
requests of a lane speculate in one verify pass, each in its draft group. Fake lanes whose device is the host mirrors
(no device; needs vllm-tt-plugin); skipped unless the process runs with both variables set.

Each fake lane runs a verify pass's argmax as an oracle target (the token after position q is ``SEQ[q + 1]``), the
accept trace as reference.sample_verify.verify_pass (greedy) followed by the chain step of its drafter (every MTP
layer: reference.spec_chain.chain_step; DFlash: dflash_chain_step), and the draft trace as a drafter that reads each
group's anchor and bonus from the packed upload (whoever wrote it) and drafts the oracle's tokens, wrong at every
third bonus, ending with the drafts step. Checked: both requests follow the oracle on drafts chained behind every
verify (one host plan each, at the start), in argmax_ids and fused_sample verifies; each request keeps its draft
group (and its MTP backfill / DFlash ring) across steps; DFlash context rows are the taps the last step left and each
group's ring runs without a gap below its anchor; a third request takes the group of the one that finished."""

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference.sample_verify import control_words, verify_pass
from models.experimental.mimo_v2_6_flash.reference.spec_chain import (
    DRAFTS_AT,
    RESULT_WORDS,
    DFlashLayout,
    chain_step,
    dflash_chain_step,
    dflash_drafts_step,
    dflash_inputs,
    drafts_step,
    relay_step,
    unpack_result,
)
from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    LAYOUT,
    DRAFT_GROUPS,
    GROUP_SLOTS,
    MAX_DRAFTS,
    SPEC_SLOTS,
)
from models.experimental.mimo_v2_6_flash.reference.spec_inputs import DESC_A, DESC_COUNT, pack_mtp_commits, pack_verify
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import MiMoV2ForCausalLM
from types import SimpleNamespace

pytestmark = pytest.mark.skipif(
    DRAFT_GROUPS != 2 or SPEC_SLOTS != 8, reason="draft groups: run with MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=2"
)
pytest.importorskip("vllm_tt_plugin")
from vllm_tt_plugin.spec_decode import accept_greedy_drafts  # noqa: E402

B, DP, PER_LANE, VOCAB = 32, 2, 8, 1000
MAX_SEQ_LEN = 4096
SEQ = [(7 * q * q + 3 * q + 11) % VOCAB for q in range(MAX_SEQ_LEN + 8)]  # the oracle target's sequence
LANE = 1
DFLASH = DFlashLayout(ring=64, block=8, heads=2, last=MAX_SEQ_LEN - 1, groups=DRAFT_GROUPS)


def draft_of(bonus, k):
    """The fake drafter's draft k after the bonus position: the oracle's token, wrong at every third bonus."""
    t = SEQ[bonus + 1 + k]
    return (t + 1) % VOCAB if bonus % 3 == 0 and k == 1 else t


class Chain:
    def __init__(self, lane):
        self.lane, self.words = lane, [0] * RESULT_WORDS

    def read(self):
        return unpack_result(self.words)

    def read_async(self):
        return list(self.words)

    @staticmethod
    def unpack(host):
        return unpack_result(host)


class GroupsMTP:
    """Every MTP layer: per drafting group of the description, its drafts at its anchor (the group's last slot); a
    backfill is recorded per (slot, first row, group)."""

    def __init__(self, lane):
        self.lane, self.last_inputs_s = lane, 0.0
        self.chained = self.planned = 0
        self.layers, self.multi, self.history = [0, 1, 2], True, object()
        self.drafting = {LAYOUT: self.layers}
        self.backfills = []

    def _run(self):
        lane, mtp = self.lane, self.lane.mtp_words
        heads = [[0] * B for _ in range(MAX_DRAFTS)]
        for g in range(DRAFT_GROUPS):
            desc = mtp[g * B : (g + 1) * B]
            if desc[0] == 0:
                continue
            bonus, anchor = desc[DESC_A] + desc[DESC_COUNT], (g + 1) * GROUP_SLOTS - 1
            for k in range(MAX_DRAFTS):
                heads[k][anchor] = draft_of(bonus, k)
        lane.chain.words[DRAFTS_AT:] = drafts_step(mtp, heads, layers=MAX_DRAFTS)

    def enqueue_chained(self, layout=None):
        self.chained += 1
        self._run()

    def enqueue_layers(self, commits, layout=None):
        self.planned += 1
        self.lane.mtp_words = pack_mtp_commits(B, commits).tolist()
        self._run()

    def enqueue_backfill(self, slot, a, group, layout=None):
        self.backfills.append((slot, a, group))

    def read_drafts(self, anchor, group, layout=None):
        chain = self.lane.chain.read().groups[group]
        assert chain.draft_anchor == anchor == (group + 1) * GROUP_SLOTS - 1
        return chain.drafts


class GroupsDFlash:
    """DFlash: each context row must be a tap row the lane's last step left at that position, in the group's ring;
    each drafting group's ring runs without a gap up to its anchor's position."""

    def __init__(self, lane):
        self.lane, self.last_inputs_s = lane, 0.0
        self.chained = self.planned = 0
        self.resets = [0] * DRAFT_GROUPS
        self.rings = [[] for _ in range(DRAFT_GROUPS)]
        self.head = [0] * B

    def reset(self, group):
        self.resets[group] += 1
        self.rings[group] = []

    def _run(self):
        lane, w = self.lane, self.lane.dflash_words
        for row in range(32):
            ring_row = w[DFLASH.RING_ROWS + row]
            if ring_row != DFLASH.no_write:
                g, pos = ring_row // DFLASH.ring, w[DFLASH.CONTEXT + row]
                assert lane.tap_positions[row] == pos and ring_row == g * DFLASH.ring + pos % DFLASH.ring
                self.rings[g] = sorted(set(self.rings[g]) | {pos})
        self.head = [0] * B
        for g in range(DRAFT_GROUPS):
            if w[g * DFLASH.block + 1] == 0:
                continue
            start = w[DFLASH.BLOCK + g * DFLASH.block]
            assert w[g * DFLASH.block] == SEQ[start], "the anchor is the bonus token"
            below = [q for q in self.rings[g] if q < start]
            assert below and below == list(range(below[0], start)), f"group {g}'s ring {self.rings[g]} below {start}"
            self.head[g * MAX_DRAFTS : (g + 1) * MAX_DRAFTS] = [draft_of(start, k) for k in range(MAX_DRAFTS)]
        lane.chain.words[DRAFTS_AT:] = dflash_drafts_step(w, self.head, DFLASH)

    def enqueue_chained(self, layout=None):
        self.chained += 1
        self._run()

    def enqueue(self, drafts):
        self.planned += 1
        self.lane.dflash_words = dflash_inputs(DFLASH, drafts)
        self._run()

    def read_drafts(self, group):
        return self.head[group * MAX_DRAFTS : (group + 1) * MAX_DRAFTS]


class Lane:
    """The TtMiMoModel surface the adapter's speculative step uses, with the oracle target."""

    def __init__(self, drafter):
        self.max_batch = B
        self.rope_tables = object()
        self.verify_trace = {"accept": 1}
        self.decode_trace = {"id": 0}
        self.mesh_device = None
        self.chain = self.spec_chain = Chain(self)
        self.spec_chains = {LAYOUT: self.spec_chain}
        self.verify_sampler = SimpleNamespace(tally=lambda out: None, groups=0, truncated=0)
        self.mtp = GroupsMTP(self) if drafter == "mtp" else None
        self.dflash = GroupsDFlash(self) if drafter == "dflash" else None
        self.mtp_words = pack_mtp_commits(B, []).tolist()
        self.dflash_words = dflash_inputs(DFLASH)
        self.verify_words = self.argmax = None
        self.tap_positions = [-1] * B
        self.ctrl = control_words([])
        self.decode_tokens = None

    def set_vllm_page_tables(self, rows):
        pass

    def verify_enqueue(self, tokens, positions, chain=None, accept=None, layout=None):
        if accept is not None:
            self.ctrl = control_words(accept[1])
        self.verify_words, self.ctrl = relay_step(
            pack_verify(tokens, positions, chain).tolist(), self.chain.words, self.ctrl
        )
        self.tap_positions = list(self.verify_words[B : 2 * B])
        self.argmax = [SEQ[int(p) + 1] if p >= 0 else 0 for p in self.verify_words[B : 2 * B]]
        return {"next_tokens": self.argmax}

    def verify_accept_enqueue(self, slots=None, groups=None, layout=None):
        if groups is not None:
            self.ctrl = control_words(groups)
        accept = verify_pass(None, None, self.argmax, self.ctrl, VOCAB)
        if self.mtp is not None:
            self.mtp_words, header = chain_step(self.verify_words, accept, self.argmax, MAX_DRAFTS)
        else:
            self.dflash_words, header = dflash_chain_step(self.verify_words, accept, self.argmax, DFLASH)
        self.chain.words[:DRAFTS_AT] = header

    def decode_enqueue(self, tokens, positions, sample=False):
        self.tap_positions = positions.tolist()
        return {"next_tokens": [SEQ[int(p) + 1] if p >= 0 else 0 for p in positions.tolist()]}

    def read_next_tokens(self, handle):
        return torch.tensor(handle["next_tokens"])

    def decode_read_rows(self, handle):
        return torch.nn.functional.one_hot(torch.tensor(handle["next_tokens"][: self.max_batch]), VOCAB).float()


def adapter_of(drafter):
    gen = SimpleNamespace(
        models=[Lane(drafter) for _ in range(DP)],
        dp=DP,
        users_per_replica=PER_LANE,
        cfg=SimpleNamespace(vocab_size=VOCAB),
    )
    model = MiMoV2ForCausalLM(gen, MAX_SEQ_LEN)
    model.layer_groups = [0]
    return model


TABLE = torch.zeros(DP * PER_LANE, 4, dtype=torch.int32)
ROWS = [LANE * PER_LANE, LANE * PER_LANE + 1]  # lane 1, slots 0 and 1


def plain_step(model, at):
    """A plain decode step of the lane's requests ``at`` {row: position} (input SEQ[position]), then the propose."""
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


@pytest.mark.parametrize("drafter", ["mtp", "dflash"])
@pytest.mark.parametrize("mode", ["argmax_ids", "fused_sample"])
def test_two_requests_follow_the_oracle_on_chained_drafts(drafter, mode):
    model = adapter_of(drafter)
    lane = model.models[LANE]
    d = lane.mtp or lane.dflash
    starts = {ROWS[0]: 200, ROWS[1]: 1000}
    if drafter == "dflash":
        # Each request's prompt context, as its prefill wrote it into its group's ring.
        for r, p in starts.items():
            g = model._draft_groups(LANE, [r - LANE * PER_LANE], prompt=True)[r - LANE * PER_LANE]
            d.rings[g] = list(range(p - 30, p))
            model.dflash_ring[LANE][g] = (r - LANE * PER_LANE, p - 1)
    drafts = plain_step(model, starts)
    assert d.planned == 1, "one host-planned draft program for both requests"
    groups = {r: model.group_owners[LANE].index(r - LANE * PER_LANE) for r in starts}
    seqs = {r: SEQ[: p + 2] for r, p in starts.items()}
    for r, p in starts.items():
        assert drafts[r] == [draft_of(p + 1, k) for k in range(MAX_DRAFTS)]
    steps = 0
    while min(len(s) for s in seqs.values()) < 1100 and steps < 60:
        before = (d.chained, d.planned)
        got, drafts = verify_step(model, seqs, drafts, mode)
        assert (d.chained, d.planned) == (before[0] + 1, before[1]), "both drafts chained behind the verify"
        for r in seqs:
            seqs[r] += got[r]
            assert seqs[r] == SEQ[: len(seqs[r])]
            assert drafts[r] == [draft_of(len(seqs[r]) - 1, k) for k in range(MAX_DRAFTS)]
        steps += 1
    owners = model.group_owners[LANE]
    assert sorted(owners) == [0, 1]
    assert all(owners[g] == r - LANE * PER_LANE for r, g in groups.items()), "each request kept its group"
    if drafter == "mtp":
        assert sorted(g for *_, g in d.backfills) == [0, 1], "one backfill per request, into its own group"
    else:
        assert d.resets == [0, 0]
    assert model.chain_stats["chained"] == 2 * steps and model.chain_stats["host"] == 2


def test_a_new_request_takes_the_finished_ones_group():
    model = adapter_of("mtp")
    a, b = 0, 1
    g = model._draft_groups(LANE, [a], prompt=True)[a]
    h = model._draft_groups(LANE, [b], prompt=True)[b]
    assert {g, h} == {0, 1}
    model._seen(LANE, [a, b])
    model._seen(LANE, [a])  # b finished
    c = 2
    assert model._draft_groups(LANE, [c], prompt=True)[c] == h
    assert model._draft_groups(LANE, [a, c]) == {a: g, c: h}
    # Three live requests: the third has no group (the step drafts for none), the two others keep theirs.
    assert model._draft_groups(LANE, [a, c, 5])[5] is None
    assert model.group_owners[LANE][g] == a and model.group_owners[LANE][h] == c


def fused(model, tokens, positions, valid, counts, **chain):
    return model.decode_forward(
        tokens,
        positions,
        TABLE,
        None,
        spec_mode="fused_sample",
        num_valid_drafts=valid,
        accepted_counts=counts,
        defer_read=True,
        **chain,
    )


def commit(readout):
    """The plugin's propose arguments after committing a FusedReadout: positions from p0."""
    width = readout.accepted_token_ids.shape[1]
    first = readout.p0.to(torch.int64).unsqueeze(1)
    positions = (first + torch.arange(1, width + 1)).masked_fill(first < 0, -1).to(torch.int32)
    return readout.accepted_token_ids, positions, readout.accepted_counts


@pytest.mark.parametrize("drafter", ["mtp", "dflash"])
def test_two_requests_chain_deferred_verifies(drafter):
    """The plugin's chained lane loop with both requests: a host-built deferred verify, then each verify chained onto
    the one in flight (both rows relayed on device from their groups of its page), the older one read, committed and
    proposed after the newer one's submission. Both follow the oracle on the device's drafts."""
    model = adapter_of(drafter)
    lane = model.models[LANE]
    d = lane.mtp or lane.dflash
    starts = {ROWS[0]: 300, ROWS[1]: 2000}
    if drafter == "dflash":
        for r, p in starts.items():
            s = r - LANE * PER_LANE
            g = model._draft_groups(LANE, [s], prompt=True)[s]
            d.rings[g] = list(range(p - 30, p))
            model.dflash_ring[LANE][g] = (s, p - 1)
    drafts = plain_step(model, starts)
    seqs = {r: SEQ[: p + 2] for r, p in starts.items()}
    rows, width = DP * PER_LANE, 1 + MAX_DRAFTS
    tokens = torch.zeros(rows, width, dtype=torch.int32)
    positions = torch.full((rows, width), -1, dtype=torch.int32)
    valid = torch.zeros(rows, dtype=torch.int32)
    for r, seq in seqs.items():
        tokens[r] = torch.tensor([seq[-1]] + drafts[r])
        positions[r] = torch.arange(len(seq) - 1, len(seq) + MAX_DRAFTS)
        valid[r] = MAX_DRAFTS
    pending = fused(model, tokens, positions, valid, torch.ones(rows, dtype=torch.int32)).handle
    for _ in range(25):
        chain_tokens = torch.full((rows, width), -1, dtype=torch.int32)
        chain_positions = torch.full((rows, width), -1, dtype=torch.int32)
        counts = torch.ones(rows, dtype=torch.int32)
        chain_rows = torch.zeros(rows, dtype=torch.bool)
        for r, seq in seqs.items():
            bound = len(seq) - 1 + width
            chain_positions[r] = torch.arange(bound, bound + width)
            counts[r], chain_rows[r] = -1, True
        newer = fused(
            model, chain_tokens, chain_positions, valid, counts, chain_rows=chain_rows, chain_tip=pending.tip_id
        ).handle
        readout = pending.readout()
        for r in seqs:
            m = int(readout.accepted_counts[r])
            assert int(readout.p0[r]) == len(seqs[r]) - 1, "the device's column 0: one past the applied history"
            seqs[r] += readout.accepted_token_ids[r, :m].tolist()
            assert seqs[r] == SEQ[: len(seqs[r])]
        out = model.propose_draft_tokens(MAX_DRAFTS, *commit(readout), hidden=pending)
        for r in seqs:
            assert out.draft_token_ids[r].tolist() == [draft_of(len(seqs[r]) - 1, k) for k in range(MAX_DRAFTS)]
        pending = newer
    assert d.planned == 1 and d.chained == 26, "every draft after the first chained on device"
