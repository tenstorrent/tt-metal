# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM adapter's chained speculative step (generator_vllm.MiMoV2ForCausalLM._verify / propose_draft_tokens) on
fake lanes whose device is the host mirrors (no device; ttnn may be a stub; needs vllm-tt-plugin).

Each fake lane runs a verify pass's argmax as an oracle target (the token after position q is ``SEQ[q + 1]``), the
accept trace as reference.sample_verify.verify_pass (greedy requests) followed by reference.spec_chain.chain_step, and
the MTP draft trace as a drafter that reads its anchor and bonus from the packed upload (whoever wrote it) and drafts
the oracle's tokens, wrong at every third bonus position, ending with drafts_step. The plugin's side is its greedy walk
(vllm_tt_plugin.spec_decode.accept_greedy_drafts) and its committed positions. Checked: the argmax_ids and fused
answers, which drafts come from the chain and which from the host plan (and that both drafters agree), the fallbacks
(a commit other than the device's, a logits verify, no decode RoPE tables, the context limit, MIMO_SPEC_CHAIN=0), and
that speculative greedy decoding reproduces the oracle sequence."""

from types import SimpleNamespace

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference.sample_verify import control_words, verify_pass
from models.experimental.mimo_v2_6_flash.reference.spec_chain import (
    RESULT_WORDS,
    chain_step,
    drafts_step,
    unpack_result,
)
from models.experimental.mimo_v2_6_flash.reference.spec_decode import MAX_DRAFTS, SPEC_SLOTS, DraftSource
from models.experimental.mimo_v2_6_flash.reference.spec_inputs import MTP_ROWS, pack_mtp, pack_verify
from models.experimental.mimo_v2_6_flash.tt import generator_vllm
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import MiMoV2ForCausalLM

pytest.importorskip("vllm_tt_plugin")
from vllm_tt_plugin.spec_decode import accept_greedy_drafts  # noqa: E402

B, DP, PER_LANE, VOCAB = 32, 2, 4, 1000
MAX_SEQ_LEN = 4096
SEQ = [(7 * q * q + 3 * q + 11) % VOCAB for q in range(MAX_SEQ_LEN + 8)]  # the oracle target's sequence
SPEC_ROW = PER_LANE  # lane 1, slot 0


def draft_of(bonus, k):
    """The fake drafter's draft k after the bonus position: the oracle's token, wrong at every third bonus."""
    t = SEQ[bonus + 1 + k]
    return (t + 1) % VOCAB if bonus % 3 == 0 and k == 1 else t


class FakeChain:
    def __init__(self, lane):
        self.lane, self.words = lane, [0] * RESULT_WORDS

    def read(self):
        self.lane.reads += 1
        return unpack_result(self.words)

    def read_async(self):
        """The copy enqueued behind the pass: the page as the pass left it."""
        self.lane.reads += 1
        return list(self.words)

    @staticmethod
    def unpack(host):
        return unpack_result(host)


class FakeMTP:
    def __init__(self, lane):
        self.lane, self.last_inputs_s = lane, 0.0
        self.chained = self.planned = 0

    def _run(self):
        """The three passes and the drafts step on the upload the lane holds."""
        lane, mtp = self.lane, self.lane.mtp_words
        heads = [[0] * B for _ in range(MAX_DRAFTS)]
        for s in range(SPEC_SLOTS):
            if mtp[2 * B + s] >= 0:
                bonus = mtp[5 * B + s]
                assert mtp[s] == SEQ[bonus], "the anchor slot holds the bonus token"
                for k in range(MAX_DRAFTS):
                    heads[k][s] = draft_of(bonus, k)
        lane.chain.words[64:] = drafts_step(mtp, heads)

    def enqueue_chained(self):
        assert self.lane.rope_tables is not None
        self.chained += 1
        self._run()

    def enqueue(self, tokens, plan):
        self.planned += 1
        self.lane.mtp_words = pack_mtp(B, tokens, plan).tolist()
        self._run()

    def read_drafts(self, anchor):
        result = self.lane.chain.read()
        assert result.draft_anchor == anchor
        return result.drafts


class FakeLane:
    """The TtMiMoModel surface the adapter's speculative step uses."""

    def __init__(self):
        self.max_batch = B
        self.rope_tables = object()
        self.verify_trace = {"accept": 1}
        self.mesh_device = None
        self.chain = self.spec_chain = FakeChain(self)
        self.mtp = FakeMTP(self)
        self.verify_sampler = SimpleNamespace(tally=lambda out: None, groups=0, truncated=0)
        self.mtp_words = pack_mtp(B).tolist()
        self.verify_words = self.argmax = None
        self.accepts = self.reads = 0

    def set_vllm_page_tables(self, rows):
        pass

    def verify_enqueue(self, tokens, positions, chain=None):
        self.verify_words = pack_verify(tokens, positions, chain).tolist()
        self.argmax = [SEQ[int(p) + 1] if p >= 0 else 0 for p in positions.tolist()]
        return {"next_tokens": self.argmax}

    def verify_accept_enqueue(self, slots, groups):
        assert slots == {}, "greedy requests only"
        self.accepts += 1
        accept = verify_pass(None, None, self.argmax, control_words(groups), VOCAB)
        self.mtp_words, header = chain_step(self.verify_words, accept, self.argmax)
        self.chain.words[:64] = header

    def read_next_tokens(self, handle):
        return torch.tensor(handle["next_tokens"])

    def decode_read_rows(self, handle):
        return torch.nn.functional.one_hot(torch.tensor(handle["next_tokens"][: self.max_batch]), VOCAB).float()


@pytest.fixture
def adapter():
    gen = SimpleNamespace(
        models=[FakeLane() for _ in range(DP)], dp=DP, users_per_replica=PER_LANE, cfg=SimpleNamespace(vocab_size=VOCAB)
    )
    model = MiMoV2ForCausalLM(gen, MAX_SEQ_LEN)
    model.layer_groups = [0]
    return model


TABLE = torch.zeros(DP * PER_LANE, 4, dtype=torch.int32)


def verify(model, last, p, drafts, mode="argmax_ids", others=()):
    """A verify of the speculating row (last committed token ``last`` at p, its drafts) and ``others`` (row, token,
    position) without drafts; returns the plugin's (committed, counts) after its walk, as the plugin hands them to
    propose_draft_tokens, and the verify's output."""
    rows, width = DP * PER_LANE, 1 + MAX_DRAFTS
    tokens = torch.zeros(rows, width, dtype=torch.int32)
    positions = torch.full((rows, width), -1, dtype=torch.int32)
    valid = torch.zeros(rows, dtype=torch.int32)
    tokens[SPEC_ROW] = torch.tensor([last] + drafts + [0] * (MAX_DRAFTS - len(drafts)))
    positions[SPEC_ROW, : 1 + len(drafts)] = torch.arange(p, p + 1 + len(drafts))
    valid[SPEC_ROW] = len(drafts)
    for row, tok, pos in others:
        tokens[row, 0], positions[row, 0] = tok, pos
    out = model.decode_forward(tokens, positions, TABLE, None, spec_mode=mode, num_valid_drafts=valid)
    out_positions = positions
    if mode == "fused_sample":
        committed, counts = out.accepted_token_ids, out.accepted_counts
    elif mode == "logits":
        committed, counts = accept_greedy_drafts(out.logits.argmax(-1).to(torch.int32), tokens[:, 1:], valid)
    else:
        committed, counts = accept_greedy_drafts(out.argmax_ids, tokens[:, 1:], valid)
    first = out_positions[:, 0].unsqueeze(1)
    committed_positions = (first + torch.arange(1, width + 1)).masked_fill(first < 0, -1).to(torch.int32)
    return committed, committed_positions, counts, out


def propose(model, committed, positions, counts, num_drafts=MAX_DRAFTS):
    out = model.propose_draft_tokens(num_drafts, committed, positions, counts)
    return out.draft_token_ids[SPEC_ROW].tolist()[: int(out.num_valid[SPEC_ROW])]


def start(model, p):
    """As after a plain decode step of the request alone on lane 1 (slot 0), which committed SEQ[p] at p."""
    model.spec_sources[1] = DraftSource(row=SPEC_ROW, slots=[0])
    rows = DP * PER_LANE
    committed = torch.full((rows, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
    positions = torch.full((rows, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
    committed[SPEC_ROW, 0], positions[SPEC_ROW, 0] = SEQ[p], p
    return propose(model, committed, positions, torch.ones(rows, dtype=torch.int32))


@pytest.mark.parametrize("mode", ["argmax_ids", "fused_sample"])
def test_speculative_greedy_follows_the_oracle_on_chained_drafts(adapter, mode):
    lane = adapter.models[1]
    p = 200
    drafts = start(adapter, p)
    assert lane.mtp.planned == 1 and drafts == [draft_of(p, k) for k in range(MAX_DRAFTS)]
    seq, steps = SEQ[: p + 1], 0
    while len(seq) < 400:
        before = (lane.mtp.chained, lane.mtp.planned, lane.reads)
        # Lane 0 holds two requests without drafts: one pass of two greedy groups, no draft source.
        committed, positions, counts, out = verify(
            adapter, seq[-1], len(seq) - 1, drafts, mode, others=[(0, SEQ[50], 50), (1, SEQ[90], 90)]
        )
        m = int(counts[SPEC_ROW])
        assert committed[SPEC_ROW, :m].tolist() == SEQ[len(seq) : len(seq) + m]
        assert committed[0, 0] == SEQ[51] and committed[1, 0] == SEQ[91] and counts[:2].tolist() == [1, 1]
        if mode == "argmax_ids":
            want = [SEQ[len(seq) + j] for j in range(1 + MAX_DRAFTS)]
            assert out.argmax_ids[SPEC_ROW].tolist() == want, "every cell, past the count too"
        # One result read per pass and lane, the draft chained behind the verify.
        assert (lane.mtp.chained, lane.mtp.planned, lane.reads) == (before[0] + 1, before[1], before[2] + 1)
        seq += committed[SPEC_ROW, :m].tolist()
        drafts = propose(adapter, committed, positions, counts)
        assert lane.mtp.planned == before[1], "the propose returned the chained drafts"
        assert drafts == [draft_of(len(seq) - 1, k) for k in range(MAX_DRAFTS)]
        steps += 1
    assert seq == SEQ[: len(seq)]
    assert adapter.chain_stats["chained"] == steps and adapter.chain_stats["host"] == 1
    assert adapter.models[0].mtp.chained == adapter.models[0].mtp.planned == 0


def test_the_chained_drafts_equal_the_host_plans(adapter, monkeypatch):
    """MIMO_SPEC_CHAIN_CHECK: every chained draft is drafted again from the host plan and compared."""
    adapter.chain_check = True
    p = 1000
    drafts = start(adapter, p)
    seq = SEQ[: p + 1]
    for _ in range(12):
        committed, positions, counts, _ = verify(adapter, seq[-1], len(seq) - 1, drafts)
        seq += committed[SPEC_ROW, : int(counts[SPEC_ROW])].tolist()
        drafts = propose(adapter, committed, positions, counts)
    stats = adapter.chain_stats
    assert stats["checked"] == stats["chained"] == 12 and stats["mismatched"] == 0
    assert adapter.models[1].mtp.planned == 13


def test_a_different_commit_drafts_from_the_host_plan(adapter):
    drafts = start(adapter, 300)
    committed, positions, counts, _ = verify(adapter, SEQ[300], 300, drafts)
    m = int(counts[SPEC_ROW])
    assert m >= 2
    # The plugin commits one token fewer than the device did (as if it had walked differently).
    committed[SPEC_ROW, m - 1] = -1
    counts[SPEC_ROW] = m - 1
    planned = adapter.models[1].mtp.planned
    drafts = propose(adapter, committed, positions, counts)
    assert adapter.models[1].mtp.planned == planned + 1
    assert drafts == [draft_of(300 + m - 1, k) for k in range(MAX_DRAFTS)]


def test_logits_verifies_and_lanes_without_rope_tables_chain_nothing(adapter):
    lane = adapter.models[1]
    drafts = start(adapter, 400)
    committed, positions, counts, _ = verify(adapter, SEQ[400], 400, drafts, "logits")
    assert lane.mtp.chained == 0 and lane.accepts == 0
    p = 400 + int(counts[SPEC_ROW])
    drafts = propose(adapter, committed, positions, counts)
    assert lane.mtp.planned == 2 and drafts == [draft_of(p, k) for k in range(MAX_DRAFTS)]
    lane.rope_tables = None
    committed, positions, counts, _ = verify(adapter, SEQ[p], p, drafts)
    assert lane.mtp.chained == 0 and lane.accepts == 1
    propose(adapter, committed, positions, counts)
    assert lane.mtp.planned == 3


def test_without_the_chain_an_argmax_verify_runs_no_accept_step(adapter, monkeypatch):
    monkeypatch.setattr(generator_vllm, "SPEC_CHAIN", False)
    lane = adapter.models[1]
    drafts = start(adapter, 500)
    committed, positions, counts, out = verify(adapter, SEQ[500], 500, drafts)
    assert lane.accepts == 0 and lane.mtp.chained == 0
    assert out.argmax_ids[SPEC_ROW].tolist() == SEQ[501:505]
    propose(adapter, committed, positions, counts)
    assert lane.mtp.planned == 2


@pytest.mark.parametrize("num_drafts", range(1, MAX_DRAFTS + 1))
def test_the_context_limit(adapter, num_drafts):
    """Near max_seq_len the chain declines where a K-wide propose declines; a narrower propose that still fits drafts
    from the host plan, and one that does not fit offers nothing."""
    lane = adapter.models[1]
    for p in range(MAX_SEQ_LEN - 12, MAX_SEQ_LEN - 1 - MAX_DRAFTS):
        start(adapter, p)
        drafts = [SEQ[p + 1 + k] for k in range(MAX_DRAFTS)]  # every draft accepted: the bonus at p + 4
        before = (lane.mtp.chained, lane.mtp.planned)
        committed, positions, counts, _ = verify(adapter, SEQ[p], p, drafts)
        bonus = p + int(counts[SPEC_ROW])
        chained = bonus + MAX_DRAFTS < MAX_SEQ_LEN
        assert lane.mtp.chained == before[0] + 1 and lane.chain.read().drafting == chained
        out = propose(adapter, committed, positions, counts, num_drafts)
        fits = bonus + num_drafts < MAX_SEQ_LEN
        assert len(out) == (num_drafts if fits else 0)
        assert lane.mtp.planned == before[1] + int(fits and not chained), p
        if fits:
            assert out == [draft_of(bonus, k) for k in range(num_drafts)]
