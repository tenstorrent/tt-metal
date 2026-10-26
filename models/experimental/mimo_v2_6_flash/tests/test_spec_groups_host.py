# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Draft groups (MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=2): a verify pass of 8 slots holds two requests of a lane, each in
a group of 4 slots with 3 drafts, and the drafter drafts for both (reference/spec_decode.py, reference/spec_chain.py).
No device; skipped unless the process runs with both variables set (the layouts are fixed at import).

Checked: rows take their sticky groups and the accept page's groups follow them; the chain step writes each group's
commit (every MTP layer: its description in row g; DFlash: its block and context rows) exactly as the host plan of
that commit, declines each group on its own, and the drafts step, the result page and chained_draft read each group
back; the plan and tokens steps run plan_mtp_layers / mtp_layer_tokens per group on its own stash slots; the relay
writes each relayed group's block from its group of the page."""

import pytest

from models.experimental.mimo_v2_6_flash.reference.sample_verify import (
    NO_GROUP,
    GROUP_WORDS,
    commit_pass,
    control_words,
    pass_inputs,
    pass_rows,
    verify_pass,
)
from models.experimental.mimo_v2_6_flash.reference.spec_chain import (
    CHAIN_GROUP_WORDS,
    CHAIN_SOURCE,
    DRAFT_WORDS,
    DRAFTS_AT,
    NO_SLOT,
    STATUS_AT,
    STATUS_WORDS,
    DFlashLayout,
    chain_row,
    chain_step,
    chained_draft,
    dflash_chain_step,
    dflash_drafts_step,
    dflash_inputs,
    drafts_step,
    plan_step,
    relay_step,
    stash_page,
    tokens_step,
    unpack_result,
)
from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    DRAFT_GROUPS,
    GROUP_SLOTS,
    MAX_DRAFTS,
    SPEC_SLOTS,
    DraftSource,
    MtpStash,
    accept_greedy,
    draft_sources,
    mtp_cache_position,
    mtp_layer_tokens,
    mtp_stash,
    plan_mtp_layers,
    plan_verify_passes,
)
from models.experimental.mimo_v2_6_flash.reference.spec_inputs import pack_mtp_commits
from models.experimental.mimo_v2_6_flash.tests.seeded_rng import SeededRng
from models.experimental.mimo_v2_6_flash.tests.test_spec_chain_host import accept_page, upload

pytestmark = pytest.mark.skipif(
    DRAFT_GROUPS != 2 or SPEC_SLOTS != 8, reason="draft groups: run with MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=2"
)

B = 32
VOCAB = 152576
MAX_SEQ_LEN = 1 << 20
LIMIT = MAX_SEQ_LEN - MAX_DRAFTS
DFLASH = DFlashLayout(ring=1024, block=8, heads=8, last=MAX_SEQ_LEN - 1, groups=DRAFT_GROUPS)


def row(r, p, n, rng):
    """A verify row: its last committed token and n drafts at positions p .. p + n."""
    return (r, [rng.randrange(VOCAB) for _ in range(n + 1)], list(range(p, p + n + 1)), n)


def two_rows(rng, p0, n0, p1, n1, groups=None):
    """One pass of rows 5 (at p0, n0 drafts) and 6 (at p1, n1 drafts), and its sources by row."""
    (vp,) = plan_verify_passes([row(5, p0, n0, rng), row(6, p1, n1, rng)], groups=groups)
    return vp, {s.row: s for s in draft_sources([vp])}


def committed_positions(p, m):
    return [p + 1 + c for c in range(m)] + [-1] * (1 + MAX_DRAFTS - m)


def test_rows_take_their_groups():
    rng = SeededRng(0)
    vp, sources = two_rows(rng, 100, 3, 500, 1, groups={5: 1, 6: 0})
    assert sources[5] == DraftSource(row=5, slots=[4, 5, 6, 7], group=1)
    assert sources[6] == DraftSource(row=6, slots=[0, 1], group=0)
    assert vp.positions == [500, 501, -1, -1, 100, 101, 102, 103]
    assert sorted(pass_rows(vp)) == [(0, 6), (1, 5)]
    _, groups = pass_inputs(vp, None)
    assert [g[0] for g in groups] == [0, 4]
    # A row alone in group 1: the accept page's group 0 is unused.
    (one,) = plan_verify_passes([row(7, 30, 2, rng)], groups={7: 1})
    (src,) = draft_sources([one])
    assert src.group == 1 and src.slots == [4, 5, 6]
    words = control_words(pass_inputs(one, None)[1])
    assert words[0] == NO_GROUP and words[GROUP_WORDS] == 4
    # Rows without a group take the free ones; a third row starts a second pass, and nothing drafts.
    passes = plan_verify_passes([row(1, 10, 3, rng), row(2, 20, 0, rng), row(3, 30, 1, rng)])
    assert len(passes) == 2 and draft_sources(passes) == []
    assert [s.group for s in draft_sources(passes[:1])] == [0, 1]


@pytest.mark.parametrize("seed", range(20))
def test_the_accept_page_commits_each_row(seed):
    """Greedy accept over both groups, through commit_pass: each row commits its own greedy walk."""
    rng = SeededRng(seed)
    import torch

    n0, n1 = rng.randrange(4), rng.randrange(4)
    vp, _ = two_rows(rng, 1000, n0, 7000, n1, groups=rng.choice([None, {5: 1, 6: 0}]))
    argmax = [rng.randrange(VOCAB) for _ in range(B)]
    for s in range(SPEC_SLOTS - 1):
        if rng.random() < 0.6:
            argmax[s] = vp.tokens[s + 1]
    out = verify_pass(None, None, argmax, control_words(pass_inputs(vp, None)[1]), 0)
    from models.experimental.mimo_v2_6_flash.reference.sample_verify import unpack_out

    committed = torch.full((8, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
    counts = torch.zeros(8, dtype=torch.int32)
    commit_pass(vp, unpack_out(out), committed, counts)
    for r in (5, 6):
        slots = sorted(s for s, rr, _ in vp.answers if rr == r)
        want = accept_greedy([argmax[s] for s in slots], [vp.tokens[s] for s in slots[1:]], len(slots) - 1)
        assert committed[r, : int(counts[r])].tolist() == want


@pytest.mark.parametrize("bonus", [3, 255, 256, 257, 1023, 4096, MAX_SEQ_LEN - 1 - MAX_DRAFTS])
def test_layers_chain_writes_each_groups_description(bonus):
    """Every pair of commits: the chain step's upload is pack_mtp_commits of both, each group's status its own; a
    group past the limit declines alone; the drafts step and chained_draft read each group back."""
    rng = SeededRng(bonus)
    b0, b1 = bonus, bonus + 50 if bonus < 1000 else bonus - 50  # the two bonuses, both below LIMIT
    for m0 in range(1, min(GROUP_SLOTS, b0) + 1):
        for m1 in range(1, GROUP_SLOTS + 1):
            vp, src = two_rows(rng, b0 - m0, MAX_DRAFTS, b1 - m1, MAX_DRAFTS, groups={5: 0, 6: 1})
            c0, c1 = [rng.randrange(VOCAB) for _ in range(m0)], [rng.randrange(VOCAB) for _ in range(m1)]
            accept = accept_page([(m0, c0), (m1, c1)])
            for lim in (LIMIT, max(b0, b1), min(b0, b1)):
                drafting = (b0 < lim, b1 < lim)
                verify = upload(vp, chain_row(B, list(src.values()), lim))
                mtp, header = chain_step(verify, accept, [0] * B, layers=MAX_DRAFTS)
                commits = []
                for g, (r, c, m, p) in enumerate(((5, c0, m0, b0 - m0), (6, c1, m1, b1 - m1))):
                    status = header[STATUS_AT + g * STATUS_WORDS : STATUS_AT + (g + 1) * STATUS_WORDS]
                    if drafting[g]:
                        commits.append((src[r], c, committed_positions(p, m), m))
                        assert status == [1, (g + 1) * GROUP_SLOTS - 1, p + m, 0]
                    else:
                        assert status == [0, NO_SLOT, 0, 0]
                assert mtp == pack_mtp_commits(B, commits).tolist(), (m0, m1, lim)
                heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
                result = unpack_result(header + drafts_step(mtp, heads, MAX_DRAFTS))
                for g, r in enumerate((5, 6)):
                    d = chained_draft(result, src[r], vp.positions, layers=MAX_DRAFTS)
                    if not drafting[g]:
                        assert d is None
                        continue
                    anchor = (g + 1) * GROUP_SLOTS - 1
                    assert d.tokens == (c0, c1)[g] and d.drafts == [h[anchor] for h in heads]


def group_stash(rng, a, lo):
    """A stash layout of one group: its last draft's rows (1 .. 4, anchor at a - 1) or unrelated or empty."""
    rows = rng.randrange(0, GROUP_SLOTS + 1)
    positions = [-1] * (GROUP_SLOTS - rows) + [a - rows + j for j in range(rows)]
    if rng.random() < 0.2:
        positions = [a - 40 + j for j in range(GROUP_SLOTS)]
    return positions


@pytest.mark.parametrize("seed", range(40))
def test_plan_and_tokens_steps_run_each_group_on_its_stash(seed):
    """The plan step on two descriptions writes plan_mtp_layers' passes of each group on its own stash slots (the
    other group's rows never re-run), and the new stash is each group's mtp_stash; the tokens step shifts each group's
    window by the layer, its anchor's drafts appended."""
    rng = SeededRng(seed)
    sources, plans, stash_pos, stash_tok, commits = [], [], [-1] * SPEC_SLOTS, [0] * SPEC_SLOTS, []
    for g in range(DRAFT_GROUPS):
        m = rng.randrange(1, GROUP_SLOTS + 1)
        a = rng.choice([2, 255, 256, 3000, MAX_SEQ_LEN - 10])
        slots = list(range(g * GROUP_SLOTS, g * GROUP_SLOTS + m)) if rng.random() < 0.7 else [g * GROUP_SLOTS]
        m = min(m, len(slots))
        src = DraftSource(row=5 + g, slots=slots, group=g)
        positions = committed_positions(a, m)
        tokens = [rng.randrange(VOCAB) for _ in range(m)]
        commits.append((src, tokens, positions, m))
        sp = group_stash(rng, a, g * GROUP_SLOTS)
        stash_pos[g * GROUP_SLOTS : (g + 1) * GROUP_SLOTS] = sp
        stash_tok[g * GROUP_SLOTS : (g + 1) * GROUP_SLOTS] = [rng.randrange(VOCAB) if p >= 0 else 0 for p in sp]
        sources.append(src)
    stash_tok = [t if p >= 0 else 0 for p, t in zip(stash_pos, stash_tok)]
    page = stash_page(B, stash_pos, stash_tok)
    got, new = plan_step(pack_mtp_commits(B, commits).tolist(), page, B)
    want_pos, want_hidden, want_tokens = [-1] * B, [B + j for j in range(B)], [0] * B
    want_stash = list(page)
    for src, tokens, positions, m in commits:
        plan = plan_mtp_layers(src, positions, m, MtpStash(stash_pos))
        lo = src.group * GROUP_SLOTS
        assert all(lo <= j < lo + GROUP_SLOTS for j in plan.rows) and plan.anchor == lo + GROUP_SLOTS - 1
        for j in plan.rows:
            kind, k = plan.sources[j]
            assert kind == "step" or lo <= k < lo + GROUP_SLOTS, "a group re-runs its own stash rows"
            want_pos[j] = mtp_cache_position(plan.positions[j])
            want_hidden[j] = k if kind == "step" else B + k
            want_tokens[j] = tokens[src.slots.index(k)] if kind == "step" else stash_tok[k]
        fresh = mtp_stash(plan).positions
        for j in range(lo, lo + GROUP_SLOTS):
            want_stash[j] = fresh[j]
            want_stash[SPEC_SLOTS + j] = want_tokens[j] if fresh[j] >= 0 else 0
    assert got["pos"] == want_pos and got["hidden"] == want_hidden and got["tokens0"] == want_tokens
    assert new == want_stash
    heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
    for layer in range(1, MAX_DRAFTS):
        out = tokens_step(got["tokens0"], heads, layer)
        for g in range(DRAFT_GROUPS):
            lo, anchor = g * GROUP_SLOTS, (g + 1) * GROUP_SLOTS - 1
            used = [j for j in range(lo, lo + GROUP_SLOTS) if got["pos"][j] >= 0]
            window = mtp_layer_tokens([got["tokens0"][j] for j in used], [h[anchor] for h in heads], layer)
            assert [out[j] for j in used] == window


@pytest.mark.parametrize("seed", range(30))
def test_the_relay_writes_each_relayed_group(seed):
    """A verify chained onto a pass of two requests: each relayed group's columns take its last committed token at
    the bonus and its drafts, its control group the drafts; a group whose chain declined goes idle alone."""
    rng = SeededRng(seed)
    m0, m1 = rng.randrange(1, GROUP_SLOTS + 1), rng.randrange(1, GROUP_SLOTS + 1)
    vp, src = two_rows(rng, 2000 - m0, MAX_DRAFTS, 9000 - m1, MAX_DRAFTS, groups={5: 0, 6: 1})
    c0, c1 = [rng.randrange(VOCAB) for _ in range(m0)], [rng.randrange(VOCAB) for _ in range(m1)]
    limit = rng.choice([LIMIT, 9000])  # the second: group 1 declines (its bonus is 9000)
    mtp, header = chain_step(
        upload(vp, chain_row(B, list(src.values()), limit)), accept_page([(m0, c0), (m1, c1)]), [0] * B, MAX_DRAFTS
    )
    heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
    page = header + drafts_step(mtp, heads, MAX_DRAFTS)
    # The next verify's provisional layout, above the real positions.
    nxt, nsrc = two_rows(rng, 2100, MAX_DRAFTS, 9100, MAX_DRAFTS, groups={5: 0, 6: 1})
    ctrl = control_words(pass_inputs(nxt, None)[1])
    verify, ctrl = relay_step(upload(nxt, chain_row(B, list(nsrc.values()), LIMIT, relay=True)), page, ctrl)
    for g, (c, bonus) in enumerate(((c0, 2000), (c1, 9000))):
        lo, anchor = g * GROUP_SLOTS, (g + 1) * GROUP_SLOTS - 1
        drafted = g == 0 or limit == LIMIT
        drafts = [h[anchor] for h in heads]
        if drafted:
            assert verify[lo : lo + GROUP_SLOTS] == [c[-1]] + drafts
            assert verify[B + lo : B + lo + GROUP_SLOTS] == [bonus + j for j in range(GROUP_SLOTS)]
            assert ctrl[g * GROUP_WORDS + 4 : g * GROUP_WORDS + 4 + MAX_DRAFTS] == drafts
        else:
            assert verify[lo : lo + GROUP_SLOTS] == [0] * GROUP_SLOTS
            assert verify[B + lo : B + lo + GROUP_SLOTS] == [-1] * GROUP_SLOTS
            assert verify[2 * B + g * CHAIN_GROUP_WORDS + CHAIN_SOURCE] == 0


@pytest.mark.parametrize("bonus", [1, 1023, 1024, 1025, 70000])
def test_dflash_chain_writes_each_groups_draft(bonus):
    """The DFlash chain step for two commits: dflash_inputs of both host commits (each group's block at its rows, its
    context rows into its ring); the drafts step reads group g's drafts at rows g * MAX_DRAFTS .. of the head."""
    rng = SeededRng(bonus)
    for m0 in range(1, GROUP_SLOTS + 1):
        m1 = rng.randrange(1, GROUP_SLOTS + 1)
        if bonus - m0 < 0:
            continue
        vp, src = two_rows(rng, bonus - m0, MAX_DRAFTS, bonus + 3 - m1, MAX_DRAFTS, groups={5: 1, 6: 0})
        c5, c6 = [rng.randrange(VOCAB) for _ in range(m0)], [rng.randrange(VOCAB) for _ in range(m1)]
        accept = accept_page([(m1, c6), (m0, c5)])  # group 0 is row 6's
        packed, header = dflash_chain_step(upload(vp, chain_row(B, list(src.values()), LIMIT)), accept, [0] * B, DFLASH)
        drafts = []
        for r, c, m, p in ((5, c5, m0, bonus - m0), (6, c6, m1, bonus + 3 - m1)):
            s = src[r]
            positions = committed_positions(p, m)
            context = [(s.slots[k], positions[k] - 1) for k in range(m)]
            drafts.append((s.group, context, c[m - 1], positions[m - 1], s.slots[m - 1]))
        assert packed == dflash_inputs(DFLASH, drafts)
        head = [rng.randrange(VOCAB) for _ in range(B)]
        section = dflash_drafts_step(packed, head, DFLASH)
        for g in range(DRAFT_GROUPS):
            at = g * DRAFT_WORDS
            assert section[at + 1 : at + DRAFT_WORDS] == head[g * MAX_DRAFTS : (g + 1) * MAX_DRAFTS]
        result = unpack_result(header + section)
        for r in (5, 6):
            d = chained_draft(result, src[r], vp.positions)
            assert d.drafts == head[src[r].group * MAX_DRAFTS : (src[r].group + 1) * MAX_DRAFTS]
    # Group 1 alone: group 0's block idle, its rows at position 0.
    one = dflash_inputs(DFLASH, [(1, [(4, 77)], 123, 78, 4)])
    assert one[0:2] == [0, 0] and one[8:10] == [123, 5]
    assert one[DFLASH.BLOCK : DFLASH.BLOCK + 16] == list(range(8)) + [78 + j for j in range(8)]
    assert one[DFLASH.RING_ROWS + 4] == 1024 + 77 and one[DFLASH.RING_ROWS] == DFLASH.no_write
    assert dflash_drafts_step(one, list(range(B)), DFLASH)[: DRAFT_WORDS + 1] == [NO_SLOT] + [0xFFFFFFFF] * 3 + [4]
    assert DRAFTS_AT % 16 == 0


def test_the_backfill_fills_its_groups_ring_and_stash():
    """A backfill for draft group 1 fills that group's ring blocks (its table offset by one group's physical blocks)
    and re-plans only group 1's stash slots; group 0's stash is left as it was."""
    from models.experimental.mimo_v2_6_flash.reference.mtp_backfill import (
        HISTORY_ROWS,
        RING_BLOCKS,
        SCRATCH_BLOCK,
        TABLE_WORDS,
        History,
        backfill_plan,
        backfill_row,
        decode_writes,
    )

    import torch

    hist = History(8)
    a, slot = 700, 3
    for q in range(a - HISTORY_ROWS + 5, a + 1):
        positions = [-1] * SPEC_SLOTS
        positions[slot] = q
        tokens = [0] * SPEC_SLOTS
        tokens[slot] = (q * 7) % VOCAB
        hist.apply(decode_writes(positions, tokens), torch.zeros(SPEC_SLOTS, 8))
    stash = stash_page(B, [10, 11, 12, 13] + [-1] * 4, [1, 2, 3, 4] + [0] * 4)
    for group in range(DRAFT_GROUPS):
        plan, new = backfill_plan(backfill_row(B, slot, a, group), hist.page, MAX_DRAFTS, stash)
        base = group * TABLE_WORDS
        assert all(base <= b < base + TABLE_WORDS for b in plan["table"])
        assert plan["table"][RING_BLOCKS:] == [
            base + SCRATCH_BLOCK + k % RING_BLOCKS for k in range(RING_BLOCKS, TABLE_WORDS)
        ]
        lo = group * GROUP_SLOTS
        assert [k for k, _ in plan["stash_rows"]] == [lo + GROUP_SLOTS - 2, lo + GROUP_SLOTS - 1]
        other = [j for j in range(SPEC_SLOTS) if not lo <= j < lo + GROUP_SLOTS]
        assert all(new[j] == stash[j] and new[SPEC_SLOTS + j] == stash[SPEC_SLOTS + j] for j in other)
        assert new[lo + GROUP_SLOTS - 2 : lo + GROUP_SLOTS] == [a - 2, a - 1]
