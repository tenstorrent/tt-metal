# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive draft length (MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=2,4): one process keeps both draft-group layouts, two
requests of 3 drafts (2 groups of 4 slots) and four of 1 (4 groups of 2), and a lane's step drafts in the one with the
most drafts that holds its requests (reference/spec_decode.py ``layout_for``). No device; skipped unless the process
runs with both variables set (the layouts are fixed at import).

Checked: the layout and the plugin's per-lane cap by request count; each layout's passes, accept groups and sources;
per layout, the chain step writes each group's description (pack_mtp_commits in that layout), the drafts step, the
result page and chained_draft read each group back, the relay writes each relayed group, and the plan and tokens steps
run each group on its own slots with the layout's MTP layers (3, or 1: no re-prefill); the result header sits at the
same words in both layouts."""

import pytest

from models.experimental.mimo_v2_6_flash.reference.sample_verify import (
    GROUP_WORDS,
    control_words,
    pass_inputs,
    pass_rows,
)
from models.experimental.mimo_v2_6_flash.reference.spec_chain import (
    CHAIN_SOURCE,
    NO_SLOT,
    PAGES,
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
    LAYOUTS,
    SPEC_SLOTS,
    DraftSource,
    MtpStash,
    draft_sources,
    k_by_rows,
    layout_for,
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
    not ADAPTIVE or SPEC_SLOTS != 8 or [layout.groups for layout in LAYOUTS] != [2, 4],
    reason="adaptive draft length: run with MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=2,4",
)

B = 32
VOCAB = 152576
MAX_SEQ_LEN = 1 << 20
LIMIT = MAX_SEQ_LEN - 3


def layers(layout):
    """The MTP layers a layout drafts with: one per draft (3), or layer 0 alone for one draft."""
    return layout.max_drafts


def verify_row(r, p, n, rng):
    """A verify row: its last committed token and n drafts at positions p .. p + n."""
    return (r, [rng.randrange(VOCAB) for _ in range(n + 1)], list(range(p, p + n + 1)), n)


def full_pass(layout, rng, starts, drafts=None):
    """One pass of rows 10, 11, ... (one per group, row 10 + g in group g, at ``starts[g]`` with the layout's drafts),
    and its sources by row."""
    n = layout.max_drafts if drafts is None else drafts
    rows = [verify_row(10 + g, p, n, rng) for g, p in enumerate(starts)]
    groups = {10 + g: g for g in range(len(starts))}
    (vp,) = plan_verify_passes(rows, groups=groups, group_slots=layout.group_slots)
    return vp, {s.row: s for s in draft_sources([vp])}


def committed_positions(p, m):
    return [p + 1 + c for c in range(m)] + [-1] * (4 - m)


def test_the_layout_follows_the_lane_load():
    k3, k1 = LAYOUTS
    assert (k3.group_slots, k3.max_drafts, k1.group_slots, k1.max_drafts) == (4, 3, 2, 1)
    assert [layout_for(n) for n in range(1, 6)] == [k3, k3, k1, k1, None]
    # The plugin's cap by the lane's requests covers the launch's --max-num-seqs; past 4 the device drafts for none.
    assert k_by_rows(8) == ((2, 3), (8, 1))
    assert k_by_rows(4) == ((2, 3), (4, 1))


@pytest.mark.parametrize("groups", [2, 4])
def test_a_layouts_rows_take_their_groups(groups):
    layout = next(layout for layout in LAYOUTS if layout.groups == groups)
    gs = layout.group_slots
    rng = SeededRng(groups)
    vp, src = full_pass(layout, rng, [100 * (g + 1) for g in range(groups)])
    assert vp.group_slots == gs and len(src) == groups
    for g in range(groups):
        assert src[10 + g] == DraftSource(row=10 + g, slots=list(range(g * gs, (g + 1) * gs)), group=g)
    assert sorted(pass_rows(vp)) == [(g, 10 + g) for g in range(groups)]
    assert [g[0] for g in pass_inputs(vp, None)[1]] == [g * gs for g in range(groups)]
    # A request more than the groups starts a second pass, and the step drafts for none.
    rows = [verify_row(r, 10 * r, layout.max_drafts, rng) for r in range(groups + 1)]
    passes = plan_verify_passes(rows, group_slots=gs)
    assert len(passes) == 2 and draft_sources(passes) == []


def test_the_header_is_the_same_in_both_layouts():
    """The accept page, argmax ids, positions and status sit at the same words in both layouts, and so does the drafts
    section; only the chain blocks and the drafts section's group stride differ."""
    k3, k1 = (PAGES[layout] for layout in LAYOUTS)
    assert k3.drafts_at == k1.drafts_at and k3.result_words == k1.result_words
    assert (k3.group_words, k1.group_words, k3.draft_words, k1.draft_words) == (8, 6, 4, 2)
    assert k3.hslots + SPEC_SLOTS <= B and k1.hslots + SPEC_SLOTS <= B


@pytest.mark.parametrize("groups", [2, 4])
@pytest.mark.parametrize("bonus", [3, 255, 256, 1023, 4096, MAX_SEQ_LEN - 10])
def test_the_chain_writes_each_groups_description(groups, bonus):
    """Commits in every group of a layout: the chain step's upload is pack_mtp_commits of them in that layout, each
    group's status its own; a group past the limit declines alone; the drafts step and chained_draft read each group
    back from the layout's page."""
    layout = next(layout for layout in LAYOUTS if layout.groups == groups)
    gs, k = layout.group_slots, layout.max_drafts
    rng = SeededRng(groups * 7919 + bonus)
    for _ in range(6):
        bonuses = [bonus + 40 * g if bonus < 1000 else bonus - 40 * g for g in range(groups)]
        ms = [rng.randrange(1, min(gs, b) + 1) for b in bonuses]
        vp, src = full_pass(layout, rng, [b - m for b, m in zip(bonuses, ms)])
        cs = [[rng.randrange(VOCAB) for _ in range(m)] for m in ms]
        accept = accept_page(list(zip(ms, cs)))
        for lim in (LIMIT, max(bonuses), min(bonuses)):
            verify = upload(vp, chain_row(B, list(src.values()), lim, layout=layout))
            mtp, header = chain_step(verify, accept, [0] * B, layers=layers(layout), layout=layout)
            commits = []
            for g in range(groups):
                status = header[STATUS_AT + g * STATUS_WORDS : STATUS_AT + (g + 1) * STATUS_WORDS]
                if bonuses[g] < lim:
                    commits.append((src[10 + g], cs[g], committed_positions(bonuses[g] - ms[g], ms[g]), ms[g]))
                    assert status == [1, (g + 1) * gs - 1, bonuses[g], 0]
                else:
                    assert status == [0, NO_SLOT, 0, 0]
            assert mtp == pack_mtp_commits(B, commits, layout).tolist(), (ms, lim)
            heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(k)]
            result = unpack_result(header + drafts_step(mtp, heads, layers(layout), layout), layout)
            for g in range(groups):
                d = chained_draft(result, src[10 + g], vp.positions, layers=layers(layout), layout=layout)
                if bonuses[g] >= lim:
                    assert d is None
                    continue
                anchor = (g + 1) * gs - 1
                assert d.tokens == cs[g] and d.drafts == [h[anchor] for h in heads]


@pytest.mark.parametrize("groups", [2, 4])
@pytest.mark.parametrize("seed", range(20))
def test_the_relay_writes_each_relayed_group(groups, seed):
    """A verify chained onto a pass of a layout: each relayed group's columns take its last committed token at the
    bonus and its drafts, its control group the drafts; a group whose chain declined goes idle alone."""
    layout = next(layout for layout in LAYOUTS if layout.groups == groups)
    gs, k, pages = layout.group_slots, layout.max_drafts, PAGES[layout]
    rng = SeededRng(groups * 1000 + seed)
    bonuses = [2000 + 3000 * g for g in range(groups)]
    ms = [rng.randrange(1, gs + 1) for _ in range(groups)]
    vp, src = full_pass(layout, rng, [b - m for b, m in zip(bonuses, ms)])
    cs = [[rng.randrange(VOCAB) for _ in range(m)] for m in ms]
    limit = rng.choice([LIMIT, bonuses[-1]])  # the second: the last group declines
    verify = upload(vp, chain_row(B, list(src.values()), limit, layout=layout))
    mtp, header = chain_step(verify, accept_page(list(zip(ms, cs))), [0] * B, layers(layout), layout)
    heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(k)]
    page = header + drafts_step(mtp, heads, layers(layout), layout)
    nxt, nsrc = full_pass(layout, rng, [b + 100 for b in bonuses])
    ctrl = control_words(pass_inputs(nxt, None)[1])
    nverify = upload(nxt, chain_row(B, list(nsrc.values()), LIMIT, relay=True, layout=layout))
    out, ctrl = relay_step(nverify, page, ctrl, layout)
    for g in range(groups):
        lo, anchor = g * gs, (g + 1) * gs - 1
        drafts = [h[anchor] for h in heads]
        if bonuses[g] < limit:
            assert out[lo : lo + gs] == [cs[g][-1]] + drafts
            assert out[B + lo : B + lo + gs] == [bonuses[g] + j for j in range(gs)]
            assert ctrl[g * GROUP_WORDS + 4 : g * GROUP_WORDS + 4 + k] == drafts
        else:
            assert out[lo : lo + gs] == [0] * gs and out[B + lo : B + lo + gs] == [-1] * gs
            assert out[2 * B + g * pages.group_words + CHAIN_SOURCE] == 0


@pytest.mark.parametrize("groups", [2, 4])
@pytest.mark.parametrize("seed", range(30))
def test_the_plan_and_tokens_steps_run_each_group(groups, seed):
    """The plan step on a layout's descriptions writes plan_mtp_layers' passes of each group in its own slots with the
    layout's layers (one layer re-runs no row), and the new stash is each group's mtp_stash; the tokens step shifts
    each group's window by the layer."""
    layout = next(layout for layout in LAYOUTS if layout.groups == groups)
    gs, n = layout.group_slots, layers(layout)
    rng = SeededRng(groups * 100 + seed)
    stash_pos, stash_tok, commits = [-1] * SPEC_SLOTS, [0] * SPEC_SLOTS, []
    for g in range(groups):
        a = rng.choice([2, 255, 256, 3000, MAX_SEQ_LEN - 10])
        m = rng.randrange(1, gs + 1)
        src = DraftSource(row=10 + g, slots=list(range(g * gs, g * gs + m)), group=g)
        commits.append((src, [rng.randrange(VOCAB) for _ in range(m)], committed_positions(a, m), m))
        rows = rng.randrange(0, gs + 1)  # the group's last draft's rows, anchor at a - 1
        stash_pos[g * gs : (g + 1) * gs] = [-1] * (gs - rows) + [a - rows + j for j in range(rows)]
    stash_tok = [rng.randrange(VOCAB) if p >= 0 else 0 for p in stash_pos]
    page = stash_page(B, stash_pos, stash_tok)
    got, new = plan_step(pack_mtp_commits(B, commits, layout).tolist(), page, B, layers=n, layout=layout)
    want_pos, want_hidden, want_tokens, want_stash = [-1] * B, [B + j for j in range(B)], [0] * B, list(page)
    for src, tokens, positions, m in commits:
        plan = plan_mtp_layers(src, positions, m, MtpStash(stash_pos), layers=n, group_slots=gs)
        lo = src.group * gs
        assert all(lo <= j < lo + gs for j in plan.rows) and plan.anchor == lo + gs - 1
        assert n > 1 or plan.reprefill == 0
        for j in plan.rows:
            kind, k = plan.sources[j]
            want_pos[j] = mtp_cache_position(plan.positions[j])
            want_hidden[j] = k if kind == "step" else B + k
            want_tokens[j] = tokens[src.slots.index(k)] if kind == "step" else stash_tok[k]
        fresh = mtp_stash(plan).positions
        for j in range(lo, lo + gs):
            want_stash[j] = fresh[j]
            want_stash[SPEC_SLOTS + j] = want_tokens[j] if fresh[j] >= 0 else 0
    assert got["pos"] == want_pos and got["hidden"] == want_hidden and got["tokens0"] == want_tokens
    assert new == want_stash
    heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(n)]
    for layer in range(1, n):
        out = tokens_step(got["tokens0"], heads, layer, layout)
        for g in range(groups):
            lo, anchor = g * gs, (g + 1) * gs - 1
            used = [j for j in range(lo, lo + gs) if got["pos"][j] >= 0]
            window = mtp_layer_tokens([got["tokens0"][j] for j in used], [h[anchor] for h in heads], layer)
            assert [out[j] for j in used] == window
