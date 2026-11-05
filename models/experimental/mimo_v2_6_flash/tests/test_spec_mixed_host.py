# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The mixed group layout (MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=adaptive, MIMO_SPEC_MIXED on by default): three requests
of a lane speculate in one verify pass of 8 slots as groups 0 (slots 0-3, 3 drafts), 2 (slots 4-5) and 3 (slots 6-7,
1 draft each), reference/spec_decode.py MIXED_LAYOUT. No device; skipped unless the process runs with both variables.

Checked on the host mirrors: the layouts per request count, the verify pass and its draft sources, the accept page's
groups, the chain / plan / tokens / drafts steps of every MTP layer on each group at its own width (the wide group's
re-prefill and layer windows; the idle group index 1 never touches the others' slots), the relay of every group, the
backfill into a narrow group's stash slots; and in the adapter (fake lanes, oracle target): the layout transitions on
host-built and relayed passes, the wide group going to the request with the best first-draft acceptance, and three
requests following the oracle on drafts chained behind every verify, synchronous and deferred."""

from types import SimpleNamespace

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference.mtp_backfill import BACKFILL_DRAFTING, BACKFILL_GROUP, backfill_row
from models.experimental.mimo_v2_6_flash.reference.sample_verify import (
    commit_pass,
    control_words,
    pass_inputs,
    pass_rows,
    unpack_out,
    verify_pass,
)
from models.experimental.mimo_v2_6_flash.reference.spec_chain import (
    DRAFTS_AT,
    NO_SLOT,
    STATUS_AT,
    STATUS_WORDS,
    chain_row,
    chain_step,
    chained_draft,
    drafts_step,
    plan_step,
    relay_step,
    stash_page,
    tokens_step,
    unpack_result,
)
from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    ADAPTIVE,
    DRAFT_GROUPS,
    LAYOUTS,
    MAX_DRAFTS,
    MIXED,
    MIXED_LAYOUT,
    SPEC_SLOTS,
    DraftSource,
    MtpStash,
    accept_greedy,
    draft_sources,
    layout_width,
    mtp_cache_position,
    mtp_layer_tokens,
    mtp_stash,
    pass_layout,
    plan_mtp_layers,
    plan_verify_passes,
    uniform_layout,
    wide_rows,
)
from models.experimental.mimo_v2_6_flash.reference.spec_inputs import DESC_A, DESC_COUNT, pack_mtp_commits
from models.experimental.mimo_v2_6_flash.tests.seeded_rng import SeededRng

pytestmark = pytest.mark.skipif(
    not (ADAPTIVE and MIXED), reason="mixed layout: run with MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=adaptive"
)

B = 32
VOCAB = 152576
MAX_SEQ_LEN = 1 << 20
LIMIT = MAX_SEQ_LEN - MAX_DRAFTS
WIDE, NARROW = uniform_layout(4), uniform_layout(2)


def row(r, p, n, rng):
    """A verify row: its last committed token and n drafts at positions p .. p + n."""
    return (r, [rng.randrange(VOCAB) for _ in range(n + 1)], list(range(p, p + n + 1)), n)


def mixed_pass(rng, p=(100, 500, 9000), n=(3, 1, 1), groups=(0, 2, 3), rows=(5, 6, 7)):
    """One mixed pass of ``rows`` at ``p`` with ``n`` drafts in ``groups``, and its sources by row."""
    passes = plan_verify_passes(
        [row(r, pp, nn, rng) for r, pp, nn in zip(rows, p, n)],
        groups=dict(zip(rows, groups)),
        group_slots=layout_width(MIXED_LAYOUT),
        layout=MIXED_LAYOUT,
    )
    assert len(passes) == 1
    return passes[0], {s.row: s for s in draft_sources(passes, group_slots=2, layout=MIXED_LAYOUT)}


def committed_positions(p, m):
    return [p + 1 + c for c in range(m)] + [-1] * (1 + MAX_DRAFTS - m)


def test_layouts():
    assert MIXED_LAYOUT in LAYOUTS and WIDE in LAYOUTS and NARROW in LAYOUTS
    assert [pass_layout(n) for n in range(1, 9)] == [WIDE, WIDE, MIXED_LAYOUT] + [NARROW] * 5
    assert wide_rows() == 3 and layout_width(MIXED_LAYOUT) == 2


def test_the_mixed_pass_and_its_sources(expect_error):
    rng = SeededRng(0)
    vp, src = mixed_pass(rng)
    assert vp.width == 2
    assert vp.positions == [100, 101, 102, 103, 500, 501, 9000, 9001]
    assert vp.table_rows == [5, 5, 5, 5, 6, 6, 7, 7]
    assert src[5] == DraftSource(row=5, slots=[0, 1, 2, 3], group=0, width=4) and src[5].anchor == 3
    assert src[6] == DraftSource(row=6, slots=[4, 5], group=2, width=2) and src[6].anchor == 5
    assert src[7] == DraftSource(row=7, slots=[6, 7], group=3, width=2) and src[7].anchor == 7
    # The accept page's groups are the draft groups: first slot // 2.
    assert sorted(pass_rows(vp)) == [(0, 5), (2, 6), (3, 7)]
    _, groups = pass_inputs(vp, None)
    assert [g if g is None else g[0] for g in groups] == [0, None, 4, 6]
    # Fewer drafts than the group holds: the group's first slots; a wide row's drafts never fit a narrow group.
    vp, src = mixed_pass(rng, n=(1, 0, 1))
    assert vp.positions == [100, 101, -1, -1, 500, -1, 9000, 9001]
    assert src[5].slots == [0, 1] and src[5].width == 4 and src[6].slots == [4]
    with expect_error(ValueError, "3 drafts need 4 slots"):
        mixed_pass(rng, n=(1, 3, 1))


@pytest.mark.parametrize("seed", range(20))
def test_the_accept_page_commits_each_row(seed):
    rng = SeededRng(seed)
    vp, _ = mixed_pass(rng, n=(rng.randrange(4), rng.randrange(2), rng.randrange(2)))
    argmax = [rng.randrange(VOCAB) for _ in range(B)]
    for s in range(SPEC_SLOTS - 1):
        if rng.random() < 0.6:
            argmax[s] = vp.tokens[s + 1]
    out = verify_pass(None, None, argmax, control_words(pass_inputs(vp, None)[1]), 0)
    committed = torch.full((8, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
    counts = torch.zeros(8, dtype=torch.int32)
    commit_pass(vp, unpack_out(out), committed, counts)
    for r in (5, 6, 7):
        slots = sorted(s for s, rr, _ in vp.answers if rr == r)
        want = accept_greedy([argmax[s] for s in slots], [vp.tokens[s] for s in slots[1:]], len(slots) - 1)
        assert committed[r, : int(counts[r])].tolist() == want


def accept_page(groups):
    """The accept page of {group: (count, committed tokens)}."""
    from models.experimental.mimo_v2_6_flash.reference.sample_verify import NO_TOKEN, OUT_WORDS, GROUPS

    page = [0] * (GROUPS * OUT_WORDS)
    for g, (m, tokens) in groups.items():
        page[g * OUT_WORDS] = m
        for c in range(1 + MAX_DRAFTS):
            page[g * OUT_WORDS + 1 + c] = tokens[c] if c < m else NO_TOKEN
    return page


def upload(vp, chain):
    from models.experimental.mimo_v2_6_flash.reference.spec_inputs import pack_verify

    tokens = torch.zeros(B, dtype=torch.int32)
    positions = torch.full((B,), -1, dtype=torch.int32)
    tokens[:SPEC_SLOTS] = torch.tensor(vp.tokens)
    positions[:SPEC_SLOTS] = torch.tensor(vp.positions)
    return pack_verify(tokens, positions, torch.tensor(chain, dtype=torch.int32)).tolist()


@pytest.mark.parametrize("bonus", [3, 255, 256, 1023, 4096, MAX_SEQ_LEN - 1 - MAX_DRAFTS])
def test_the_chain_writes_each_groups_description_at_its_width(bonus):
    """Every commit of the three groups: the chain step's upload is pack_mtp_commits of all of them, each group's
    status its anchor (its last slot at its width); the drafts step and chained_draft read each group back."""
    rng = SeededRng(bonus)
    bonuses = {5: bonus, 6: bonus + 40 if bonus < 1000 else bonus - 40, 7: max(2, bonus - 2)}
    for m5 in range(1, min(4, bonuses[5]) + 1):
        for m6 in (1, 2):
            for m7 in (1, 2):
                ms = {5: m5, 6: m6, 7: m7}
                p = {r: bonuses[r] - ms[r] for r in ms}
                vp, src = mixed_pass(rng, p=(p[5], p[6], p[7]))
                tokens = {r: [rng.randrange(VOCAB) for _ in range(ms[r])] for r in ms}
                accept = accept_page({src[r].group: (ms[r], tokens[r]) for r in ms})
                verify = upload(vp, chain_row(B, list(src.values()), LIMIT))
                mtp, header = chain_step(verify, accept, [0] * B, layers=MAX_DRAFTS)
                commits = [(src[r], tokens[r], committed_positions(p[r], ms[r]), ms[r]) for r in (5, 6, 7)]
                assert mtp == pack_mtp_commits(B, commits).tolist()
                for r in ms:
                    g = src[r].group
                    status = header[STATUS_AT + g * STATUS_WORDS : STATUS_AT + (g + 1) * STATUS_WORDS]
                    assert status == [1, src[r].anchor, p[r] + ms[r], 0]
                idle = header[STATUS_AT + STATUS_WORDS : STATUS_AT + 2 * STATUS_WORDS]
                assert idle == [0, NO_SLOT, 0, 0], "group index 1 is no group of the mixed layout"
                heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
                result = unpack_result(header + drafts_step(mtp, heads, MAX_DRAFTS))
                for r in ms:
                    d = chained_draft(result, src[r], vp.positions, layers=MAX_DRAFTS)
                    assert d.tokens == tokens[r] and d.drafts == [h[src[r].anchor] for h in heads]


@pytest.mark.parametrize("seed", range(40))
def test_plan_and_tokens_steps_run_each_group_at_its_width(seed):
    """The plan step on the three descriptions writes plan_mtp_layers' passes of each group in its own slots (the wide
    group re-runs its own stash rows; a narrow group drafts one token and re-runs none), the new stash is each group's
    mtp_stash; the tokens step gives the wide group's rows layer i's window, and the narrow groups' rows are never
    written by another group (the idle group index 1, read as 4 slots wide, comes before groups 2 and 3)."""
    rng = SeededRng(seed)
    stash_pos, stash_tok, commits = [-1] * SPEC_SLOTS, [0] * SPEC_SLOTS, []
    for r, g, w in ((5, 0, 4), (6, 2, 2), (7, 3, 2)):
        lo = g * w
        m = rng.randrange(1, w + 1)
        a = rng.choice([2, 255, 256, 3000, MAX_SEQ_LEN - 10])
        src = DraftSource(row=r, slots=list(range(lo, lo + m)), group=g, width=w)
        tokens = [rng.randrange(VOCAB) for _ in range(m)]
        commits.append((src, tokens, committed_positions(a, m), m))
        rows = rng.randrange(0, w + 1)
        sp = [-1] * (w - rows) + [a - rows + j for j in range(rows)]
        stash_pos[lo : lo + w] = sp
        stash_tok[lo : lo + w] = [rng.randrange(VOCAB) if q >= 0 else 0 for q in sp]
    page = stash_page(B, stash_pos, stash_tok)
    mtp = pack_mtp_commits(B, commits).tolist()
    got, new = plan_step(mtp, page, B)
    want_pos, want_hidden, want_tokens = [-1] * B, [B + j for j in range(B)], [0] * B
    want_stash = list(page)
    for src, tokens, positions, m in commits:
        lo, w = src.group * src.width, src.width
        plan = plan_mtp_layers(src, positions, m, MtpStash(stash_pos), layers=min(MAX_DRAFTS, w - 1), group_slots=w)
        assert all(lo <= j < lo + w for j in plan.rows) and plan.anchor == lo + w - 1
        for j in plan.rows:
            kind, k = plan.sources[j]
            assert kind == "step" or lo <= k < lo + w, "a group re-runs its own stash rows"
            want_pos[j] = mtp_cache_position(plan.positions[j])
            want_hidden[j] = k if kind == "step" else B + k
            want_tokens[j] = tokens[src.slots.index(k)] if kind == "step" else stash_tok[k]
        fresh = mtp_stash(plan).positions
        for j in range(lo, lo + w):
            want_stash[j] = fresh[j]
            want_stash[SPEC_SLOTS + j] = want_tokens[j] if fresh[j] >= 0 else 0
    assert got["pos"] == want_pos and got["hidden"] == want_hidden and got["tokens0"] == want_tokens
    assert new == want_stash
    heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
    for layer in range(1, MAX_DRAFTS):
        out = tokens_step(got["tokens0"], heads, layer, mtp)
        used = [j for j in range(4) if got["pos"][j] >= 0]
        window = mtp_layer_tokens([got["tokens0"][j] for j in used], [h[3] for h in heads], layer)
        assert [out[j] for j in used] == window, "the wide group's layer window"
        for lo in (4, 6):  # a narrow group's slots: its own tokens shifted, then its anchor's drafts
            assert out[lo] == (got["tokens0"][lo + 1] if layer == 1 else heads[layer - 2][lo + 1])
            assert out[lo + 1] == heads[layer - 1][lo + 1]


@pytest.mark.parametrize("seed", range(20))
def test_the_relay_writes_each_relayed_group(seed):
    """A chained mixed pass: the relay step writes each group's block from its group of the last page."""
    rng = SeededRng(seed)
    bound = {5: 1000, 6: 3000, 7: 70000}
    vp, src = mixed_pass(rng, p=(bound[5], bound[6], bound[7]))
    tokens = {r: [rng.randrange(VOCAB) for _ in range(rng.randrange(1, src[r].width + 1))] for r in src}
    prev, prev_src = mixed_pass(rng, p=tuple(bound[r] - 10 for r in (5, 6, 7)))
    accept = accept_page({prev_src[r].group: (len(tokens[r]), tokens[r]) for r in src})
    verify = upload(prev, chain_row(B, list(prev_src.values()), LIMIT))
    mtp, header = chain_step(verify, accept, [0] * B, layers=MAX_DRAFTS)
    heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
    page = header + drafts_step(mtp, heads, MAX_DRAFTS)
    chained = upload(vp, chain_row(B, list(src.values()), LIMIT, relay=True))
    out, ctrl = relay_step(chained, page, control_words(pass_inputs(vp, None)[1]))
    for r in src:
        m, anchor = len(tokens[r]), prev_src[r].anchor
        bonus = bound[r] - 10 + m
        for c, s in enumerate(src[r].slots):
            assert out[B + s] == bonus + c
            assert out[s] == (tokens[r][m - 1] if c == 0 else heads[c - 1][anchor])


def test_the_backfill_upload_names_the_groups_width():
    row = backfill_row(B, slot=2, a=900, group=2, width=2)
    assert row[BACKFILL_GROUP] == 2 and row[BACKFILL_DRAFTING] == 2


# ---------------------------------------------------------------------- the adapter
pytest.importorskip("vllm_tt_plugin")
from vllm_tt_plugin.spec_decode import accept_greedy_drafts  # noqa: E402

import ttnn  # noqa: E402

from models.experimental.mimo_v2_6_flash.tests import test_spec_groups_adapter_host as groups_host  # noqa: E402
from models.experimental.mimo_v2_6_flash.tt import generator_vllm  # noqa: E402
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import MiMoV2ForCausalLM  # noqa: E402

SEQ, draft_of = groups_host.SEQ, groups_host.draft_of
DP, PER_LANE, LANE = groups_host.DP, groups_host.PER_LANE, groups_host.LANE
ROWS = [LANE * PER_LANE + s for s in range(3)]


@pytest.fixture(autouse=True)
def host_events(monkeypatch):
    monkeypatch.setattr(ttnn, "record_event", lambda *args, **kwargs: object(), raising=False)
    monkeypatch.setattr(ttnn, "event_synchronize", lambda event: None, raising=False)


def adapter(layout, owners):
    """A one-lane adapter at group ``layout`` whose groups belong to ``owners`` (lane slots), with an MTP drafter that
    records the layouts its page table is pointed at."""
    m = MiMoV2ForCausalLM.__new__(MiMoV2ForCausalLM)
    m.set_layouts = []
    m.models = [SimpleNamespace(mtp=SimpleNamespace(set_layout=m.set_layouts.append))]
    m.lane_layout = [layout]
    m.group_owners = [list(owners)]
    m.slot_seen = [{}]
    m.per_lane = 8
    m.accept_ema = [None] * 8
    m.mtp_ring = [["ring"] * DRAFT_GROUPS]
    m.dflash_ring = [[None] * DRAFT_GROUPS]
    m.chain_stats = {"width_changes": 0, "width_holds": 0}
    return m


def test_a_third_request_takes_the_mixed_layout_on_a_host_built_pass():
    m = adapter(WIDE, [0, 1, None, None])
    assert m._set_layout(0, 3) == MIXED_LAYOUT
    assert m.group_owners[0] == [None] * DRAFT_GROUPS and m.mtp_ring[0] == [None] * DRAFT_GROUPS
    assert m.set_layouts == [MIXED_LAYOUT] and m.chain_stats["width_changes"] == 1


def test_a_relay_pass_keeps_or_narrows_the_mixed_layout():
    # Three requests relayed onto a pass of groups of 4: groups of 2 keep both groups' owners and rings.
    m = adapter(WIDE, [0, 1, None, None])
    assert m._set_layout(0, 3, hold=True) == NARROW
    assert m.group_owners[0] == [0, 1, None, None] and m.mtp_ring[0] == ["ring"] * DRAFT_GROUPS
    # The mixed layout holds two relayed requests until a host-built pass; four narrow it, keeping groups 0, 2, 3.
    m = adapter(MIXED_LAYOUT, [0, None, 1, 2])
    assert m._set_layout(0, 2, hold=True) == MIXED_LAYOUT and m.chain_stats["width_holds"] == 1
    assert m._set_layout(0, 4, hold=True) == NARROW
    assert m.group_owners[0] == [0, None, 1, 2] and m.mtp_ring[0] == ["ring"] * DRAFT_GROUPS
    assert m.set_layouts == [NARROW]
    # Host-built: two requests widen.
    assert m._set_layout(0, 2) == WIDE and m.group_owners[0] == [None] * DRAFT_GROUPS


def test_the_wide_group_goes_to_the_best_acceptance():
    m = adapter(MIXED_LAYOUT, [None] * DRAFT_GROUPS)
    m.accept_ema[:3] = [0.4, 0.9, None]
    assert m._draft_groups(0, [0, 1, 2]) == {1: 0, 0: 2, 2: 3}
    # Sticky: a better acceptance later moves nothing while the layout stays.
    m.accept_ema[2] = 1.0
    assert m._draft_groups(0, [0, 1, 2]) == {0: 2, 1: 0, 2: 3}
    # Without averages: the lowest slot takes the wide group.
    m = adapter(MIXED_LAYOUT, [None] * DRAFT_GROUPS)
    assert m._draft_groups(0, [4, 3, 5]) == {3: 0, 4: 2, 5: 3}


def test_the_acceptance_average():
    rng = SeededRng(3)
    vp, src = mixed_pass(rng, rows=(0, 1, 2), n=(3, 1, 0))
    accept = accept_page({0: (3, [1, 2, 3]), 2: (1, [4]), 3: (1, [5])})
    result = SimpleNamespace(accept=unpack_out(accept))
    ema = [None] * 4
    generator_vllm._update_accept_ema(ema, [(0, vp, result)])
    assert ema == [1.0, 0.0, None, None], "a row without drafts is not counted"
    generator_vllm._update_accept_ema(ema, [(0, vp, result)])
    assert ema[1] == 0.0 and ema[0] == 1.0
    accept = accept_page({0: (1, [1]), 2: (2, [4, 6]), 3: (1, [5])})
    generator_vllm._update_accept_ema(ema, [(0, vp, SimpleNamespace(accept=unpack_out(accept)))])
    rate = generator_vllm.ACCEPT_EMA_RATE
    assert ema[0] == pytest.approx(1 - rate) and ema[1] == pytest.approx(rate)


class MixedMTP(groups_host.GroupsMTP):
    """Every MTP layer: per drafting group of the description, its drafts at its anchor (its last slot at its width,
    the description's drafting word); the layouts its page table is pointed at."""

    def __init__(self, lane):
        super().__init__(lane)
        self.layouts = []

    def set_layout(self, layout):
        self.layouts.append(tuple(layout))

    def _run(self):
        lane, mtp = self.lane, self.lane.mtp_words
        heads = [[0] * B for _ in range(MAX_DRAFTS)]
        for g in range(DRAFT_GROUPS):
            desc = mtp[g * B : (g + 1) * B]
            if desc[0] == 0:
                continue
            bonus, anchor = desc[DESC_A] + desc[DESC_COUNT], (g + 1) * desc[0] - 1
            for k in range(MAX_DRAFTS):
                heads[k][anchor] = draft_of(bonus, k)
        lane.chain.words[DRAFTS_AT:] = drafts_step(mtp, heads, layers=MAX_DRAFTS)

    def read_drafts(self, anchor, group):
        chain = self.lane.chain.read().groups[group]
        assert chain.draft_anchor == anchor
        return chain.drafts


class MixedLane(groups_host.Lane):
    """The fake lane, recording each verify's attention variant (attention.verify_variant; None: the widest)."""

    def __init__(self):
        super().__init__("mtp")
        self.mtp = MixedMTP(self)
        self.variants = []

    def verify_enqueue(self, tokens, positions, chain=None, accept=None, drain=False, groups=None):
        self.variants.append(groups)
        return super().verify_enqueue(tokens, positions, chain, accept=accept, drain=drain)


def adapter_of():
    lanes = [MixedLane() for _ in range(DP)]
    gen = SimpleNamespace(models=lanes, dp=DP, users_per_replica=PER_LANE, cfg=SimpleNamespace(vocab_size=1000))
    model = MiMoV2ForCausalLM(gen, groups_host.MAX_SEQ_LEN)
    model.layer_groups = [0]
    return model


def expected(model, seqs):
    """Each request's drafts at its group's width."""
    out = {}
    for r, seq in seqs.items():
        s = r - LANE * PER_LANE
        g = model.group_owners[LANE].index(s)
        out[r] = [draft_of(len(seq) - 1, k) for k in range(model.lane_layout[LANE][g] - 1)]
    return out


@pytest.mark.parametrize("mode", ["argmax_ids", "fused_sample"])
def test_three_requests_follow_the_oracle_on_chained_drafts(mode):
    model = adapter_of()
    d = model.models[LANE].mtp
    model.accept_ema[ROWS[1]] = 0.9  # the best acceptance: the wide group
    starts = {ROWS[0]: 200, ROWS[1]: 1000, ROWS[2]: 2500}
    drafts = groups_host.plain_step(model, starts)
    assert model.lane_layout[LANE] == MIXED_LAYOUT and d.layouts == [MIXED_LAYOUT] and d.planned == 1
    assert model.group_owners[LANE] == [1, None, 0, 2]
    seqs = {r: SEQ[: p + 2] for r, p in starts.items()}
    assert drafts == expected(model, seqs)
    assert [len(drafts[r]) for r in ROWS] == [1, 3, 1]
    steps = 0
    while min(len(s) for s in seqs.values()) < 300 and steps < 60:
        before = (d.chained, d.planned)
        got, drafts = groups_host.verify_step(model, seqs, drafts, mode)
        assert (d.chained, d.planned) == (before[0] + 1, before[1]), "every draft chained behind the verify"
        for r in seqs:
            seqs[r] += got[r]
            assert seqs[r] == SEQ[: len(seqs[r])]
        assert drafts == expected(model, seqs)
        steps += 1
    assert model.group_owners[LANE] == [1, None, 0, 2], "each request kept its group"
    variants = set(model.models[LANE].variants)
    assert variants <= {None, (2, 4, False), (2, 4, True)} and variants != {None}, "the 4 groups of 2 slots"
    assert sorted(g for *_, g in d.backfills) == [0, 2, 3], "one backfill per request, into its own group"
    assert model.chain_stats["chained"] == 3 * steps and model.chain_stats["host"] == 3
    assert model.accept_ema[ROWS[1]] > 0.5


def test_three_requests_chain_deferred_verifies():
    """The plugin's chained lane loop: a host-built deferred verify, then each verify chained onto the one in flight,
    each row carrying the drafts its group verifies (the wide row 3, the narrow ones 1)."""
    model = adapter_of()
    d = model.models[LANE].mtp
    starts = {ROWS[0]: 300, ROWS[1]: 2000, ROWS[2]: 777}
    drafts = groups_host.plain_step(model, starts)
    seqs = {r: SEQ[: p + 2] for r, p in starts.items()}
    rows, width = DP * PER_LANE, 1 + MAX_DRAFTS
    tokens = torch.zeros(rows, width, dtype=torch.int32)
    positions = torch.full((rows, width), -1, dtype=torch.int32)
    valid = torch.zeros(rows, dtype=torch.int32)
    for r, seq in seqs.items():
        n = len(drafts[r])
        tokens[r, : 1 + n] = torch.tensor([seq[-1]] + drafts[r])
        positions[r, : 1 + n] = torch.arange(len(seq) - 1, len(seq) + n)
        valid[r] = n
    assert valid[ROWS].tolist() == [3, 1, 1], "the lowest slot takes the wide group without averages"
    pending = groups_host.fused(model, tokens, positions, valid, torch.ones(rows, dtype=torch.int32)).handle
    for _ in range(25):
        chain_tokens = torch.full((rows, width), -1, dtype=torch.int32)
        chain_positions = torch.full((rows, width), -1, dtype=torch.int32)
        counts = torch.ones(rows, dtype=torch.int32)
        chain_rows = torch.zeros(rows, dtype=torch.bool)
        for r, seq in seqs.items():
            bound = len(seq) - 1 + 1 + int(valid[r])
            chain_positions[r, : 1 + int(valid[r])] = torch.arange(bound, bound + 1 + int(valid[r]))
            counts[r], chain_rows[r] = -1, True
        newer = groups_host.fused(
            model, chain_tokens, chain_positions, valid, counts, chain_rows=chain_rows, chain_tip=pending.tip_id
        ).handle
        readout = pending.readout()
        for r in seqs:
            m = int(readout.accepted_counts[r])
            assert int(readout.p0[r]) == len(seqs[r]) - 1
            seqs[r] += readout.accepted_token_ids[r, :m].tolist()
            assert seqs[r] == SEQ[: len(seqs[r])]
        out = model.propose_draft_tokens(MAX_DRAFTS, *groups_host.commit(readout), hidden=pending)
        for r in seqs:
            n = int(out.num_valid[r])
            assert n == int(valid[r]) and out.draft_token_ids[r].tolist()[:n] == expected(model, seqs)[r]
        pending = newer
    assert d.planned == 1 and d.chained == 26
    assert model.lane_layout[LANE] == MIXED_LAYOUT
