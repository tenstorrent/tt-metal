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
that speculative greedy decoding reproduces the oracle sequence. With DFlash drafting (MIMO_SPEC_DECODE=dflash) the
accept trace ends with reference.spec_chain.dflash_chain_step and the fake drafter checks every context row against
the taps the last step left and that its ring holds the request's positions without a gap below the anchor."""

from types import SimpleNamespace

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference.mtp import MtpProgramMirror
from models.experimental.mimo_v2_6_flash.reference.sample_verify import control_words, verify_pass
from models.experimental.mimo_v2_6_flash.reference.spec_chain import (
    CHAIN_HSLOTS,
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
from models.experimental.mimo_v2_6_flash.reference.spec_decode import LAYOUT, MAX_DRAFTS, SPEC_SLOTS, DraftSource
from models.experimental.mimo_v2_6_flash.reference.spec_inputs import (
    pack_mtp,
    pack_mtp_commits,
    pack_verify,
)
from models.experimental.mimo_v2_6_flash.tests.seeded_rng import SeededRng
from models.experimental.mimo_v2_6_flash.tt import generator_vllm
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import MiMoV2ForCausalLM
from models.experimental.mimo_v2_6_flash.reference import spec_decode as spec_layout

# The MTP drafter's 4-slot verify (the draft groups: test_spec_groups_host.py, test_spec_groups_adapter_host.py).
pytestmark = pytest.mark.skipif(
    (spec_layout.SPEC_SLOTS, spec_layout.DRAFT_GROUPS) != (4, 1),
    reason="MTP's 4-slot layouts: run without MIMO_SPEC_SLOTS / MIMO_SPEC_GROUPS",
)


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
        self.layers, self.multi = [0], False
        self.drafting = {LAYOUT: self.layers}
        self.history, self.backfills = None, []

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

    def enqueue_chained(self, layout=None):
        assert self.lane.rope_tables is not None
        self.chained += 1
        self._run()

    def enqueue(self, tokens, plan):
        self.planned += 1
        self.lane.mtp_words = pack_mtp(B, tokens, plan).tolist()
        self._run()

    def read_drafts(self, anchor, group=0, layout=None):
        result = self.lane.chain.read()
        assert group == 0 and result.draft_anchor == anchor
        return result.drafts


class FakeLane:
    """The TtMiMoModel surface the adapter's speculative step uses."""

    def __init__(self):
        self.max_batch = B
        self.rope_tables = object()
        self.verify_trace = {"accept": 1}
        self.mesh_device = None
        self.chain = self.spec_chain = FakeChain(self)
        self.spec_chains = {LAYOUT: self.spec_chain}
        self.mtp = FakeMTP(self)
        self.dflash = None
        self.verify_sampler = SimpleNamespace(tally=lambda out: None, groups=0, truncated=0)
        self.mtp_words = pack_mtp(B).tolist()
        self.verify_words = self.argmax = None
        self.ctrl = control_words([])
        self.accepts = self.reads = 0

    def set_vllm_page_tables(self, rows):
        pass

    def verify_enqueue(self, tokens, positions, chain=None, accept=None, layout=None):
        """The verify trace: the relay step on the upload (and the control page, staged first), then the oracle."""
        if accept is not None:
            self.verify_accept_stage(*accept)
        self.verify_words, self.ctrl = relay_step(
            pack_verify(tokens, positions, chain).tolist(), self.chain.words, self.ctrl
        )
        self.argmax = [SEQ[int(p) + 1] if p >= 0 else 0 for p in self.verify_words[B : 2 * B]]
        return {"next_tokens": self.argmax}

    def verify_accept_stage(self, slots, groups):
        assert slots == {}, "greedy requests only"
        self.ctrl = control_words(groups)

    def verify_accept_enqueue(self, slots=None, groups=None, layout=None):
        if groups is not None:
            self.verify_accept_stage(slots, groups)
        self.accepts += 1
        accept = verify_pass(None, None, self.argmax, self.ctrl, VOCAB)
        self.mtp_words, header = chain_step(self.verify_words, accept, self.argmax, len(self.mtp.layers))
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
    model.spec_sources[1] = [DraftSource(row=SPEC_ROW, slots=[0])]
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


# ---------------------------------------------------------------------- every MTP layer (MIMO_MTP_LAYERS=3)
HIDDEN = 64


def hidden_of(q, token):
    """The fake target's final normed hidden at position q with input token ``token``."""
    return torch.randn(HIDDEN, generator=torch.Generator().manual_seed(q * 1000003 + int(token)))


def accepts(q):
    """Whether the fake target takes the draft after position q (60% of positions, pseudo-random): counts of every
    size."""
    return SeededRng(q).random() < 0.6


class LayersMTP:
    """The every-layer draft trace on a fake lane: reference.mtp.MtpProgramMirror on the packed upload and the last
    step's final norm rows (whoever wrote the upload: the chain step, or the host's description)."""

    def __init__(self, lane, mtps, embedding, lm_head):
        self.lane, self.last_inputs_s = lane, 0.0
        self.chained = self.planned = 0
        self.layers, self.multi = list(mtps), True
        self.drafting = {LAYOUT: self.layers}
        self.history = None
        self.program = MtpProgramMirror(mtps, embedding, lm_head, table_rows=B)

    def _run(self):
        self.lane.chain.words[64:] = self.program.run(self.lane.mtp_words, self.lane.step_hidden)

    def enqueue_chained(self, layout=None):
        self.chained += 1
        self._run()

    def enqueue_layers(self, commits, layout=None):
        self.planned += 1
        self.lane.mtp_words = pack_mtp_commits(B, commits).tolist()
        self._run()

    def read_drafts(self, anchor, group=0, layout=None):
        result = self.lane.chain.read()
        assert group == 0 and result.draft_anchor == anchor == SPEC_SLOTS - 1
        return result.drafts


class LayersLane(FakeLane):
    """A fake lane whose target takes a draft per ``accepts`` (else a token off the drafts) and keeps each slot's
    final normed hidden (``hidden_of``) as the draft program's step rows."""

    def __init__(self, mtps, embedding, lm_head):
        super().__init__()
        self.mtp = LayersMTP(self, mtps, embedding, lm_head)
        self.step_hidden = torch.zeros(B, HIDDEN)

    def verify_enqueue(self, tokens, positions, chain=None, accept=None, layout=None):
        """The verify trace: the relay step (a chained verify's tokens and positions from the lane's last page), then
        the target over the slots."""
        if accept is not None:
            self.verify_accept_stage(*accept)
        self.verify_words, self.ctrl = relay_step(
            pack_verify(tokens, positions, chain).tolist(), self.chain.words, self.ctrl
        )
        tok, pos = self.verify_words[:B], self.verify_words[B : 2 * B]
        self.argmax = [0] * B
        for j in range(B):
            if pos[j] < 0:
                continue
            self.step_hidden[j] = hidden_of(pos[j], tok[j])
            nxt = tok[j + 1] if j + 1 < SPEC_SLOTS and pos[j + 1] == pos[j] + 1 else None
            self.argmax[j] = nxt if nxt is not None and accepts(pos[j]) else (SEQ[pos[j] + 1] + 3 * j + 1) % VOCAB
            if nxt is not None and not accepts(pos[j]) and self.argmax[j] == nxt:
                self.argmax[j] = (nxt + 1) % VOCAB
        return {"next_tokens": self.argmax}


@pytest.fixture
def layers_adapter(monkeypatch):
    from models.experimental.mimo_v2_6_flash.reference.mtp import ReferenceMTPDrafter
    from models.experimental.mimo_v2_6_flash.tests.test_spec_decode_host import mtp_drafter, tiny_config

    g = torch.Generator().manual_seed(5)
    cfg = tiny_config()
    target = SimpleNamespace(
        embedding=torch.randn(VOCAB, HIDDEN, generator=g), lm_head=torch.randn(VOCAB, HIDDEN, generator=g)
    )
    mtps = mtp_drafter(cfg, target, g, MAX_DRAFTS).mtps
    lanes = [LayersLane(mtps, target.embedding, target.lm_head) for _ in range(DP)]
    gen = SimpleNamespace(models=lanes, dp=DP, users_per_replica=PER_LANE, cfg=SimpleNamespace(vocab_size=VOCAB))
    model = MiMoV2ForCausalLM(gen, MAX_SEQ_LEN)
    model.layer_groups = [0]
    model.reference = ReferenceMTPDrafter(mtps, target.embedding, target.lm_head)
    return model


def reference_drafts(model, source, committed, positions, m, slot_hidden):
    """The host reference drafter's drafts for the plugin's commit of the speculating row."""
    ref = model.reference
    plan = ref.plan(source, positions, m)
    slot_tokens = {source.slots[c]: committed[c] for c in range(m)}
    return ref.draft(plan, slot_tokens, slot_hidden)


def layers_start(model, p):
    """A plain decode step of the request alone on lane 1 (slot 0) at p - 1, committing SEQ[p] at p."""
    lane = model.models[1]
    lane.step_hidden[0] = hidden_of(p - 1, SEQ[p - 1])
    drafts = start(model, p)
    want = reference_drafts(
        model, DraftSource(row=SPEC_ROW, slots=[0]), [SEQ[p]], [p, -1, -1, -1], 1, {0: lane.step_hidden[0].clone()}
    )
    assert drafts == want
    return drafts


@pytest.mark.parametrize("mode", ["argmax_ids", "fused_sample"])
def test_layers_chained_drafts_are_the_reference_drafters(layers_adapter, mode):
    """Every MTP layer: each propose after a verify returns the drafts the device chained (the chain step's
    description, the plan step on the device stash), and they are the host reference drafter's drafts for the
    plugin's commit, over accepted counts of every size."""
    model, lane = layers_adapter, layers_adapter.models[1]
    p = 300
    drafts = layers_start(model, p)
    seq, counts = SEQ[: p + 1], set()
    for _ in range(60):
        before = (lane.mtp.chained, lane.mtp.planned)
        q = len(seq) - 1
        committed, positions, counts_t, _ = verify(
            model, seq[-1], q, drafts, mode, others=[(0, SEQ[50], 50), (1, SEQ[90], 90)]
        )
        m = int(counts_t[SPEC_ROW])
        counts.add(m)
        new = committed[SPEC_ROW, :m].tolist()
        source = DraftSource(row=SPEC_ROW, slots=list(range(1 + MAX_DRAFTS)))
        slot_hidden = {c: hidden_of(q + c, ([seq[-1]] + drafts)[c]) for c in range(m)}
        drafts = propose(model, committed, positions, counts_t)
        assert (lane.mtp.chained, lane.mtp.planned) == (before[0] + 1, before[1]), "returned from the chain"
        assert drafts == reference_drafts(model, source, new, positions[SPEC_ROW].tolist(), m, slot_hidden)
        seq += new
    assert counts == set(range(1, MAX_DRAFTS + 2))
    assert model.chain_stats["chained"] == 60 and model.chain_stats["host"] == 1


def test_layers_chain_check_redrafts_the_same(layers_adapter):
    """MIMO_SPEC_CHAIN_CHECK with every layer: the host description of the same commit, run after the chained draft
    (the stash then holds that draft's rows), drafts the same tokens."""
    model = layers_adapter
    model.chain_check = True
    drafts = layers_start(model, 700)
    seq = SEQ[:701]
    for _ in range(30):
        committed, positions, counts, _ = verify(model, seq[-1], len(seq) - 1, drafts)
        seq += committed[SPEC_ROW, : int(counts[SPEC_ROW])].tolist()
        drafts = propose(model, committed, positions, counts)
    stats = model.chain_stats
    assert stats["checked"] == stats["chained"] == 30 and stats["mismatched"] == 0


def test_layers_a_different_commit_drafts_from_the_host_description(layers_adapter):
    """The plugin commits one token fewer than the device: the host's description of its commit, planned on the
    device stash the chained draft left, drafts the reference drafter's tokens for that commit."""
    model, lane = layers_adapter, layers_adapter.models[1]
    drafts = layers_start(model, 900)
    seq = SEQ[:901]
    for _ in range(40):
        q = len(seq) - 1
        committed, positions, counts, _ = verify(model, seq[-1], q, drafts)
        m = int(counts[SPEC_ROW])
        if m >= 2:
            break
        seq += committed[SPEC_ROW, :m].tolist()
        source = DraftSource(row=SPEC_ROW, slots=list(range(1 + MAX_DRAFTS)))
        slot_hidden = {c: hidden_of(q + c, ([seq[-2]] + drafts)[c]) for c in range(m)}
        drafts = propose(model, committed, positions, counts)
        reference_drafts(model, source, committed[SPEC_ROW, :m].tolist(), positions[SPEC_ROW].tolist(), m, slot_hidden)
    assert m >= 2
    committed[SPEC_ROW, m - 1] = -1
    counts[SPEC_ROW] = m - 1
    planned = lane.mtp.planned
    source = DraftSource(row=SPEC_ROW, slots=list(range(1 + MAX_DRAFTS)))
    slot_hidden = {c: hidden_of(q + c, ([seq[-1]] + drafts)[c]) for c in range(m - 1)}
    got = propose(model, committed, positions, counts)
    assert lane.mtp.planned == planned + 1
    want = reference_drafts(
        model, source, committed[SPEC_ROW, : m - 1].tolist(), positions[SPEC_ROW].tolist(), m - 1, slot_hidden
    )
    assert got == want


# ---------------------------------------------------------------------- the ring backfill's bookkeeping
def test_the_backfill_runs_where_the_rings_do_not_continue(adapter):
    """With the MTP history (tt/mtp_backfill.py): the first draft after the prompt (the rings reset) and a draft
    after steps without drafts backfill first; drafts after a verify (chained, re-drafted by the chain check, or
    host-planned on a differing commit) do not. Every verify pass's chain row names its slots' request slot."""
    lane = adapter.models[1]
    lane.mtp.history = object()
    lane.mtp.enqueue_backfill = lambda slot, a, group=0, layout=None: lane.mtp.backfills.append((slot, a))
    adapter.chain_check = True
    rows_seen = []
    enqueue = lane.verify_enqueue

    def verify_enqueue(tokens, positions, chain=None, accept=None, layout=None):
        rows_seen.append(chain[CHAIN_HSLOTS : CHAIN_HSLOTS + SPEC_SLOTS].tolist())
        return enqueue(tokens, positions, chain, accept)

    lane.verify_enqueue = verify_enqueue
    p = 600
    drafts = start(adapter, p)
    assert lane.mtp.backfills == [(0, p - 1)]
    seq = SEQ[: p + 1]
    for _ in range(5):
        committed, positions, counts, _ = verify(adapter, seq[-1], len(seq) - 1, drafts)
        seq += committed[SPEC_ROW, : int(counts[SPEC_ROW])].tolist()
        drafts = propose(adapter, committed, positions, counts)
    assert lane.mtp.backfills == [(0, p - 1)] and adapter.chain_stats["backfills"] == 1
    assert rows_seen == [[1, 1, 1, 1]] * 5, "slot 0's request in every verify slot (slot + 1)"
    # Steps without drafts (the lane served several requests): the next draft, after a plain step, backfills.
    q = len(seq) + 7
    start(adapter, q)
    assert lane.mtp.backfills[-1] == (0, q - 1) and adapter.chain_stats["backfills"] == 2
    # A plain step right after a draft continues the rings: no backfill.
    start(adapter, q + 1)
    assert adapter.chain_stats["backfills"] == 2


# ---------------------------------------------------------------------- DFlash drafting (MIMO_SPEC_DECODE=dflash)
DFLASH = DFlashLayout(ring=64, block=8, heads=2, last=MAX_SEQ_LEN - 1)


class FakeDFlash:
    """TtDFlash's surface: the draft trace on the packed upload the lane holds (the chain step's or the host's). Each
    context row must be a tap row the lane's last step left at that position; the ring keeps the positions written
    since ``reset``, which must run without a gap up to the anchor's; the drafts are the oracle's tokens after the
    anchor (wrong at every third bonus), written by the drafts step."""

    def __init__(self, lane):
        self.lane, self.last_inputs_s = lane, 0.0
        self.chained = self.planned = self.resets = 0
        self.ring = []  # the positions the ring holds, sorted
        self.head = [0] * B

    def reset(self, group=0):
        assert group == 0
        self.resets += 1
        self.ring = []

    def _run(self):
        lane, w = self.lane, self.lane.dflash_words
        for row in range(32):
            if w[DFLASH.RING_ROWS + row] != DFLASH.ring:
                pos = w[DFLASH.CONTEXT + row]
                assert lane.tap_positions[row] == pos, f"tap row {row} holds {lane.tap_positions[row]}, not {pos}"
                assert w[DFLASH.RING_ROWS + row] == pos % DFLASH.ring
                self.ring = sorted(set(self.ring) | {pos})
        self.head = [0] * B
        if w[1] != 0:
            start = w[DFLASH.BLOCK]
            assert w[0] == SEQ[start], "the anchor is the bonus token"
            below = [q for q in self.ring if q < start]
            assert below and below == list(range(below[0], start)), f"the ring {self.ring} below the anchor {start}"
            self.head[:MAX_DRAFTS] = [draft_of(start, k) for k in range(MAX_DRAFTS)]
        lane.chain.words[DRAFTS_AT:] = dflash_drafts_step(w, self.head, DFLASH)

    def enqueue_chained(self, layout=None):
        assert self.lane.rope_tables is not None
        self.chained += 1
        self._run()

    def enqueue(self, drafts):
        self.planned += 1
        self.lane.dflash_words = dflash_inputs(DFLASH, drafts)
        self._run()

    def read_drafts(self, group=0):
        assert group == 0
        return self.head[:MAX_DRAFTS]


class DFlashLane(FakeLane):
    """FakeLane drafting with DFlash: the taps hold each verify slot's row at its position."""

    def __init__(self):
        super().__init__()
        self.mtp, self.dflash = None, FakeDFlash(self)
        self.dflash_words = dflash_inputs(DFLASH)
        self.tap_positions = [-1] * B

    def verify_enqueue(self, tokens, positions, chain=None, accept=None, layout=None):
        out = super().verify_enqueue(tokens, positions, chain, accept)
        self.tap_positions = list(self.verify_words[B : 2 * B])
        return out

    def verify_accept_enqueue(self, slots=None, groups=None, layout=None):
        if groups is not None:
            self.verify_accept_stage(slots, groups)
        self.accepts += 1
        accept = verify_pass(None, None, self.argmax, self.ctrl, VOCAB)
        self.dflash_words, self.chain.words[:DRAFTS_AT] = dflash_chain_step(
            self.verify_words, accept, self.argmax, DFLASH
        )


@pytest.fixture
def dflash_adapter():
    gen = SimpleNamespace(
        models=[DFlashLane() for _ in range(DP)],
        dp=DP,
        users_per_replica=PER_LANE,
        cfg=SimpleNamespace(vocab_size=VOCAB),
    )
    model = MiMoV2ForCausalLM(gen, MAX_SEQ_LEN)
    model.layer_groups = [0]
    return model


def dflash_start(model, p):
    """``start`` on a DFlash lane: the plain step's tap row 0 holds position p - 1."""
    model.models[1].tap_positions[0] = p - 1
    return start(model, p)


@pytest.mark.parametrize("mode", ["argmax_ids", "fused_sample"])
def test_dflash_greedy_follows_the_oracle_on_chained_drafts(dflash_adapter, mode):
    """The first draft after a plain step is host-planned and resets the ring; every later one chains behind its
    verify (its context the verify's committed slots), continues the ring, and is what the propose returns."""
    lane = dflash_adapter.models[1].dflash
    p = 200
    drafts = dflash_start(dflash_adapter, p)
    assert (lane.planned, lane.resets) == (1, 1) and drafts == [draft_of(p, k) for k in range(MAX_DRAFTS)]
    seq = SEQ[: p + 1]
    while len(seq) < 400:
        before = (lane.chained, lane.planned)
        committed, positions, counts, _ = verify(dflash_adapter, seq[-1], len(seq) - 1, drafts, mode)
        m = int(counts[SPEC_ROW])
        assert committed[SPEC_ROW, :m].tolist() == SEQ[len(seq) : len(seq) + m]
        assert (lane.chained, lane.planned) == (before[0] + 1, before[1])
        seq += committed[SPEC_ROW, :m].tolist()
        drafts = propose(dflash_adapter, committed, positions, counts)
        assert drafts == [draft_of(len(seq) - 1, k) for k in range(MAX_DRAFTS)]
    assert seq == SEQ[: len(seq)]
    assert lane.resets == 1 and lane.ring == list(range(p - 1, len(seq) - 1))
    assert dflash_adapter.chain_stats["host"] == 1 and dflash_adapter.chain_stats["dflash_resets"] == 1


def test_dflash_chain_check_and_a_different_commit(dflash_adapter):
    """MIMO_SPEC_CHAIN_CHECK re-drafts every chained draft from the host plan (the same ring rows, no reset); a commit
    other than the device's drafts from the host plan; a plain step past a gap resets the ring."""
    adapter, lane = dflash_adapter, dflash_adapter.models[1].dflash
    adapter.chain_check = True
    p = 1000
    drafts = dflash_start(adapter, p)
    seq = SEQ[: p + 1]
    for _ in range(8):
        committed, positions, counts, _ = verify(adapter, seq[-1], len(seq) - 1, drafts)
        seq += committed[SPEC_ROW, : int(counts[SPEC_ROW])].tolist()
        drafts = propose(adapter, committed, positions, counts)
    stats = adapter.chain_stats
    assert stats["checked"] == stats["chained"] == 8 and stats["mismatched"] == 0
    assert (lane.planned, lane.resets) == (9, 1)
    adapter.chain_check = False
    committed, positions, counts, _ = verify(adapter, seq[-1], len(seq) - 1, drafts)
    m = int(counts[SPEC_ROW])
    assert m >= 2
    committed[SPEC_ROW, m - 1] = -1
    counts[SPEC_ROW] = m - 1
    drafts = propose(adapter, committed, positions, counts)
    assert lane.planned == 10 and drafts == [draft_of(len(seq) - 1 + m - 1, k) for k in range(MAX_DRAFTS)]
    assert lane.resets == 1
    q = len(seq) + 20
    dflash_start(adapter, q)
    assert lane.resets == 2 and lane.ring == [q - 1]
