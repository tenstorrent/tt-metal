# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The pair draft: two requests of a lane, one draft each (reference/spec_decode.py ``draft_sources``,
``plan_mtp_pair``; reference/spec_inputs.py ``pack_mtp_pair``; the chain, drafts and relay steps' mirrors in
reference/spec_chain.py; ReferenceMTPDrafter's pair draft in reference/mtp.py); no device.

Checked: which lane steps draft for two requests; the plan and its upload; the MTP rings (one per request slot) never
share a physical row inside any window; the chain step writes exactly the upload the host builds from the plugin's
commits, for every pair of accepted counts, around the ring's wrap and at the context limit (each source declined on
its own); the drafts step and the result page carry both sources; the relay writes a pair's two chain rows (or one of
them, from either page group) exactly as the host would lay out the next verify, and idles a source that verifies more
drafts than its page holds; the plan step of a single draft addresses its request's ring. The pair drafter drafts
what a recomputation of layer 0 over each request's whole committed sequence drafts, after steps of every accepted
count, and speculative greedy decoding of two requests with it equals plain greedy."""

import random

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference.mtp import MtpProgramMirror, PositionalHistory, cache_ring
from models.experimental.mimo_v2_6_flash.reference.sample_verify import control_words, pass_inputs
from models.experimental.mimo_v2_6_flash.reference.spec_chain import (
    CHAIN_GROUP,
    CHAIN_NEXT,
    CHAIN_PAIR,
    CHAIN_RELAY,
    CHAIN_RING,
    CHAIN_SOURCE,
    DRAFT_WORDS,
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
    unpack_result,
)
from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    MAX_DRAFTS,
    MTP_RING_TOKENS,
    MTP_RINGS,
    PAIR_DRAFTS,
    SPEC_SLOTS,
    DraftSource,
    MtpPairInputs,
    accept_greedy,
    draft_sources,
    mtp_cache_position,
    mtp_ring_of,
    mtp_table_row,
    plan_mtp_pair,
    plan_verify_passes,
)
from models.experimental.mimo_v2_6_flash.reference.spec_inputs import (
    DESC_RING,
    MTP_ROWS,
    PAIR_ANCHORS,
    derive_pair,
    pack_mtp,
    pack_mtp_layers,
    pack_mtp_pair,
    slot_caps,
)
from models.experimental.mimo_v2_6_flash.tests.test_spec_chain_host import (
    B,
    BONUS,
    MAX_SEQ_LEN,
    VOCAB,
    accept_page,
    upload,
)
from models.experimental.mimo_v2_6_flash.tests.test_spec_decode_host import (
    TinyMiMo,
    mtp_drafter,
    recompute_drafts,
    tiny_config,
)

LIMIT = MAX_SEQ_LEN - MAX_DRAFTS
X, Y = 1, 3  # the pair's lane slots (their rings)


# ---------------------------------------------------------------------- planning
def pair_pass(px, nx, py, ny, rng=None):
    """A verify pass of X (its last committed token at px and nx drafts) and Y (at py, ny drafts), rows 5 and 7."""
    rng = rng or random.Random(px * 7 + py)
    rows = []
    for row, p, n in ((5, px, nx), (7, py, ny)):
        tokens = [rng.randrange(VOCAB) for _ in range(n + 1)] + [0] * (MAX_DRAFTS - n)
        rows.append((row, tokens, list(range(p, p + n + 1)) + [-1] * (MAX_DRAFTS - n), n))
    (vp,) = plan_verify_passes(rows)
    return vp


def test_draft_sources():
    vp = pair_pass(40, 1, 90, 1)
    assert draft_sources([vp]) == [DraftSource(row=5, slots=[0, 1]), DraftSource(row=7, slots=[2, 3])]
    assert draft_sources([pair_pass(40, 0, 90, 1)]) == [DraftSource(row=5, slots=[0]), DraftSource(row=7, slots=[1, 2])]
    assert draft_sources([vp], pair=False) == []
    # A request with two drafts beside another: no pair (one draft each at most), and so none at all.
    rows = [(5, [1, 2, 3, 0], [40, 41, 42, -1], 2), (7, [4, 0, 0, 0], [90, -1, -1, -1], 0)]
    assert draft_sources(plan_verify_passes(rows)) == []
    # Three requests (a rider among them): none.
    rows = [(r, [1, 0, 0, 0], [10 * r, -1, -1, -1], 0) for r in (1, 2, 3)]
    assert draft_sources(plan_verify_passes(rows)) == []
    # One request drafts alone, whatever its drafts.
    (one,) = plan_verify_passes([(5, [1, 2, 3, 4], [40, 41, 42, 43], 3)])
    assert draft_sources([one]) == [DraftSource(row=5, slots=[0, 1, 2, 3])]


def test_plan_mtp_pair(expect_error):
    sources = [DraftSource(row=5, slots=[0, 1]), DraftSource(row=7, slots=[2, 3])]
    plan = plan_mtp_pair(sources, [[41, 42, -1, -1], [91, -1, -1, -1]], [2, 1], [X, Y])
    assert plan.fill == [40, 41, 90, -1] and plan.rings == [X, X, Y, -1] and plan.anchors == [1, 2]
    assert plan.rows == [0, 1, 2]
    with expect_error(ValueError, "one MTP ring"):
        plan_mtp_pair(sources, [[41, -1, -1, -1], [91, -1, -1, -1]], [1, 1], [X, X])
    with expect_error(ValueError, "another source's"):
        plan_mtp_pair([DraftSource(5, [0, 1]), DraftSource(7, [1])], [[41, 42], [91]], [2, 1], [X, Y])
    # Its upload: pack_mtp's rows, each slot's cache position in its ring, the anchors in row PAIR_ANCHORS.
    tokens = [11, 12, 13, 0]
    packed = pack_mtp_pair(B, tokens, plan).reshape(MTP_ROWS, B)
    assert packed[0, :4].tolist() == tokens
    assert packed[1, :4].tolist() == [
        mtp_cache_position(40, X),
        mtp_cache_position(41, X),
        mtp_cache_position(90, Y),
        -1,
    ]
    assert packed[PAIR_ANCHORS, :3].tolist() == [1, 2, -1] and packed[4, :4].tolist() == [40, 41, 90, 0]
    assert torch.equal(packed[[3, 5, 6]], pack_mtp(B).reshape(MTP_ROWS, B)[[3, 5, 6]])
    assert torch.equal(pack_mtp_pair(B), pack_mtp(B)), "the idle upload is pack_mtp's"
    # The pair trace's derive program: tokens, fill, its serial positions and RoPE indices.
    got = derive_pair(pack_mtp_pair(B, tokens, plan), slot_caps(B))
    assert got["tokens"][0, :4].tolist() == tokens and got["rope_idx"][0, :4].tolist() == [40, 41, 90, 0]
    assert [int(s[j]) for j, s in enumerate(got["serial"])] == packed[1, :4].tolist()


@pytest.mark.parametrize("t", [0, 1, 127, 128, 255, 256, 1000, 1 << 20])
def test_the_rings_never_share_a_row(t):
    """Every ring addresses each window position of its own (spec_decode.mtp_cache_position) through one table row
    shared by every slot, and no two rings ever reach the same physical row."""
    row, blocks = mtp_table_row(64)
    assert len(row) == blocks and max(row) < MTP_RINGS * MTP_RING_TOKENS // 64

    def physical(cache_pos):
        assert 0 <= cache_pos < len(row) * 64
        return row[cache_pos // 64], cache_pos % 64

    owner = {}
    for ring in range(MTP_RINGS):
        for u in range(t, t + 128):
            for d in range(128):
                if u - d < 0:
                    break
                key = physical(mtp_cache_position(u, ring) - d)
                assert key == physical(mtp_cache_position(u - d, ring))
                assert owner.setdefault(key, ring) == ring
        assert cache_ring(mtp_cache_position(t, ring)) == ring
    assert [mtp_ring_of(s) for s in range(6)] == [0, 1, 2, 3, 0, 1]


# ---------------------------------------------------------------------- the chain, drafts and relay steps
def committed(p, m):
    return [p + 1 + c for c in range(m)] + [-1] * (1 + MAX_DRAFTS - m)


def chain_of(vp, limit=LIMIT, relay=False, group=None):
    return chain_row(B, draft_sources([vp]), limit, relay=relay, ring=[X, Y], group=group, pair=True)


@pytest.mark.parametrize("bonus", BONUS)
def test_the_chain_step_writes_the_host_pair_upload(bonus):
    rng = random.Random(bonus)
    for nx in range(PAIR_DRAFTS + 1):
        for ny in range(PAIR_DRAFTS + 1):
            for mx in range(1, nx + 2):
                for my in range(1, ny + 2):
                    px, py = bonus - mx, bonus + 300 - my
                    if px < 0:
                        continue
                    vp = pair_pass(px, nx, py, ny, rng)
                    sources = draft_sources([vp])
                    tx = [rng.randrange(VOCAB) for _ in range(mx)]
                    ty = [rng.randrange(VOCAB) for _ in range(my)]
                    argmax = [rng.randrange(VOCAB) for _ in range(B)]
                    mtp, header = chain_step(upload(vp, chain_of(vp)), accept_page([(mx, tx), (my, ty)]), argmax)
                    plan = plan_mtp_pair(sources, [committed(px, mx), committed(py, my)], [mx, my], [X, Y])
                    slot_tokens = [0] * SPEC_SLOTS
                    for src, toks in zip(sources, (tx, ty)):
                        for c, tok in enumerate(toks):
                            slot_tokens[src.slots[c]] = tok
                    assert mtp == pack_mtp_pair(B, slot_tokens, plan).tolist(), (bonus, nx, ny, mx, my)
                    for k, (src, m, p) in enumerate(zip(sources, (mx, my), (px, py))):
                        status = header[STATUS_AT + k * STATUS_WORDS : STATUS_AT + (k + 1) * STATUS_WORDS]
                        assert status == [1, src.slots[m - 1], p + m, 1]


def test_each_pair_source_is_declined_on_its_own():
    """At the context limit only the source whose bonus reaches it drafts nothing."""
    vp = pair_pass(1995, 1, 400, 1)
    mtp, header = chain_step(upload(vp, chain_of(vp, limit=1997)), accept_page([(2, [3, 4]), (1, [5])]), [0] * B)
    assert header[STATUS_AT : STATUS_AT + 2 * STATUS_WORDS] == [0, NO_SLOT, 0, 0, 1, 2, 401, 1]
    assert mtp[PAIR_ANCHORS * B : PAIR_ANCHORS * B + 2] == [-1, 2] and mtp[B : B + 2] == [-1, -1]
    # A single chain row (no pair word) after a pair layout: only source 0, the description's ring its own.
    one = chain_row(B, draft_sources([vp])[:1], 5000, ring=2)
    mtp, header = chain_step(upload(vp, one), accept_page([(1, [3]), (1, [5])]), [0] * B, layers=MAX_DRAFTS)
    assert mtp[DESC_RING] == 2 and header[STATUS_AT + STATUS_WORDS] == 0


def page_of(vp, mx, my, rng, heads=None, limit=LIMIT):
    """The result page a pair pass leaves: its chain step, then the pair draft's drafts step."""
    tx = [rng.randrange(VOCAB) for _ in range(mx)]
    ty = [rng.randrange(VOCAB) for _ in range(my)]
    heads = heads or [[rng.randrange(VOCAB) for _ in range(B)]]
    mtp, header = chain_step(upload(vp, chain_of(vp, limit)), accept_page([(mx, tx), (my, ty)]), list(range(B)))
    return header + drafts_step(mtp, heads, pair=True), tx, ty, heads


def test_the_drafts_step_and_the_page_carry_both_sources():
    rng = random.Random(3)
    vp = pair_pass(700, 1, 900, 1)
    page, tx, ty, heads = page_of(vp, 2, 1, rng)
    result = unpack_result(page)
    x, y = result.sources
    assert (x.anchor, x.draft_anchor, x.bonus, x.pair) == (1, 1, 702, True)
    assert (y.anchor, y.draft_anchor, y.bonus, y.pair) == (2, 2, 901, True)
    assert x.drafts == [heads[0][1], 0xFFFFFFFF, 0xFFFFFFFF] and y.drafts[0] == heads[0][2]
    sources = draft_sources([vp])
    dx = chained_draft(result, sources[0], vp.positions, layers=MAX_DRAFTS, group=0)
    dy = chained_draft(result, sources[1], vp.positions, layers=MAX_DRAFTS, group=1)
    assert (dx.row, dx.tokens, dx.positions, dx.drafts) == (5, tx, [701, 702], [heads[0][1]])
    assert (dy.row, dy.tokens, dy.positions, dy.drafts) == (7, ty, [901], [heads[0][2]])
    # The single drafters' pages carry no second source.
    assert drafts_step(pack_mtp(B), [[0] * B] * MAX_DRAFTS)[DRAFT_WORDS] == NO_SLOT
    assert drafts_step(pack_mtp_layers(B), [[0] * B] * MAX_DRAFTS, layers=MAX_DRAFTS)[DRAFT_WORDS] == NO_SLOT
    # An idle pair upload: no anchors.
    assert drafts_step(pack_mtp_pair(B), [[0] * B], pair=True)[::DRAFT_WORDS][:2] == [NO_SLOT, NO_SLOT]


def host_pair_block(lasts, bonuses, drafts):
    """The upload and control page the host builds for the next verify of the pair, having read the page."""
    rows = []
    for row, last, bonus, d in zip((5, 7), lasts, bonuses, drafts):
        n = len(d)
        rows.append((row, [last] + d + [0] * (MAX_DRAFTS - n), list(range(bonus, bonus + n + 1)) + [-1] * (3 - n), n))
    (vp,) = plan_verify_passes(rows)
    return upload(vp, chain_of(vp)), control_words(pass_inputs(vp, None)[1]), vp


def chained_pair_block(bounds, hints):
    rows = []
    for row, bound, n in zip((5, 7), bounds, hints):
        rows.append((row, [0] * 4, list(range(bound, bound + n + 1)) + [-1] * (3 - n), n))
    (vp,) = plan_verify_passes(rows)
    return upload(vp, chain_of(vp, relay=True)), control_words(pass_inputs(vp, None)[1])


@pytest.mark.parametrize("bonus", BONUS)
def test_the_relay_writes_the_pair_block_the_host_would_build(bonus):
    rng = random.Random(bonus + 1)
    for mx in (1, 2):
        for my in (1, 2):
            px, py = bonus - mx, bonus + 500 - my
            if px < 0:
                continue
            vp = pair_pass(px, 1, py, 1, rng)
            page, tx, ty, heads = page_of(vp, mx, my, rng)
            result = unpack_result(page)
            bx, by = result.sources[0].bonus, result.sources[1].bonus
            assert (bx, by) == (bonus, bonus + 500)
            ax, ay = result.sources[0].anchor, result.sources[1].anchor
            for hx in (0, 1):
                for hy in (0, 1):
                    dx, dy = [heads[0][ax]][:hx], [heads[0][ay]][:hy]
                    want, want_ctrl, _ = host_pair_block((tx[-1], ty[-1]), (bx, by), (dx, dy))
                    if hx == 0 or hy == 0:
                        continue  # a chain row verifies at least one draft
                    for bounds in ((bx, by), (bx + 3, by + 1)):
                        overlay, ctrl = chained_pair_block(bounds, (hx, hy))
                        got, got_ctrl = relay_step(overlay, page, ctrl)
                        assert got[: 2 * B] == want[: 2 * B] and got_ctrl == want_ctrl, (bonus, mx, my, bounds)


def test_a_relay_of_one_pair_source_and_of_too_many_drafts():
    """One request of the pair goes on alone (the other finished): its chain row reads its own page group; a source
    verifying more drafts than its pair page section holds idles."""
    rng = random.Random(5)
    vp = pair_pass(300, 1, 600, 1)
    page, tx, ty, heads = page_of(vp, 1, 2, rng)
    result = unpack_result(page)
    by, ay = result.sources[1].bonus, result.sources[1].anchor
    (alone,) = plan_verify_passes([(7, [0, 0, -1, -1], [by + 2, by + 3, -1, -1], 1)])
    overlay = upload(alone, chain_row(B, draft_sources([alone]), LIMIT, relay=True, ring=Y, group=1))
    ctrl = control_words(pass_inputs(alone, None)[1])
    got, got_ctrl = relay_step(overlay, page, ctrl)
    assert got[:2] == [ty[-1], heads[0][ay]] and got[B : B + 2] == [by, by + 1] and got_ctrl[4] == heads[0][ay]
    (wide,) = plan_verify_passes([(7, [0, 0, 0, -1], [by + 2, by + 3, by + 4, -1], 2)])
    overlay = upload(wide, chain_row(B, draft_sources([wide]), LIMIT, relay=True, ring=Y, group=1))
    got, _ = relay_step(overlay, page, control_words(pass_inputs(wide, None)[1]))
    assert got[B : B + 3] == [-1, -1, -1] and got[2 * B + CHAIN_SOURCE] == 0
    # A group past the pair: idle too.
    overlay = upload(alone, chain_row(B, draft_sources([alone]), LIMIT, relay=True, ring=Y, group=2))
    got, _ = relay_step(overlay, page, ctrl)
    assert got[B : B + 2] == [-1, -1]


def test_the_chain_row_layout():
    vp = pair_pass(40, 1, 90, 0)
    row = chain_of(vp, relay=[True, False], group=[1, 0])
    assert row[CHAIN_PAIR] == 1 and row[CHAIN_RING] == X and row[CHAIN_NEXT + CHAIN_RING] == Y
    assert row[CHAIN_RELAY] == 1 and row[CHAIN_NEXT + CHAIN_RELAY] == 0
    assert row[CHAIN_GROUP] == 1 and row[CHAIN_NEXT + CHAIN_GROUP] == 0
    assert row[CHAIN_NEXT + CHAIN_SOURCE] == 1 and row[CHAIN_NEXT + 1] == 1 and row[CHAIN_NEXT + 3] == 2


def test_the_plan_step_addresses_the_request_ring():
    source = DraftSource(row=0, slots=[0, 1, 2, 3])
    for ring in range(MTP_RINGS):
        desc = pack_mtp_layers(B, source, [5, 6], [301, 302, -1, -1], 2, ring).tolist()
        out, _ = plan_step(desc, stash_page(B, [-1, -1, -1, 299], [0, 0, 0, 7]), B)
        assert [out["pos"][j] for j in (1, 2, 3)] == [mtp_cache_position(q, ring) for q in (299, 300, 301)]


# ---------------------------------------------------------------------- the pair drafter against a recomputation
PAIR_SCHEDULE = [(1, 1), (2, 1), (1, 2), (2, 2), (1, 1), (2, 2), (2, 1), (1, 2), (2, 2), (1, 1), (1, 2)]


def drive_pair(target, drafter, prompts, schedule):
    """Serves two requests (lane slots X and Y, their rings) with the pair draft after a plain first step; step k
    commits per ``schedule[k]`` for each: m = 2 accepts its draft and adds the target's token, m = 1 commits a token
    that is not the draft. Yields each request's draft and output and the recomputation's."""
    vocab = target.cfg.vocab_size
    seqs = [list(p) for p in prompts]
    slot_tokens, slot_hidden, sources, positions, counts = {}, {}, [], [], []
    for k, (slot, seq) in enumerate(zip((X, Y), seqs)):
        tok, h = target.greedy(seq)
        seq.append(tok)
        slot_tokens[slot], slot_hidden[slot] = tok, h
        sources.append(DraftSource(row=k, slots=[slot]))
        positions.append([len(seq) - 1, -1, -1, -1])
        counts.append(1)
    firsts = [len(s) - 2 for s in seqs]
    for step, ms in enumerate(schedule):
        plan = drafter.plan_pair(sources, positions, counts, [X, Y])
        drafts = drafter.draft(plan, slot_tokens, slot_hidden)
        for k in range(2):
            yield k, drafts[k], drafter.outputs[k], recompute_drafts(target, drafter.mtps[:1], seqs[k], firsts[k])
        rows = []
        for k, seq in enumerate(seqs):
            p = len(seq) - 1
            rows.append((k, [seq[-1], drafts[k], 0, 0], [p, p + 1, -1, -1], 1))
        (vp,) = plan_verify_passes(rows)
        sources = draft_sources([vp])
        slot_tokens, slot_hidden, positions, counts = {}, {}, [], []
        for k, (seq, m) in enumerate(zip(seqs, ms)):
            p = len(seq) - 1
            new = [drafts[k], target.greedy(seq + [drafts[k]])[0]] if m == 2 else [(drafts[k] + 1 + step) % vocab]
            hidden = target.hidden(seq + new)
            for c, s in enumerate(sources[k].slots[: len(new)]):
                slot_tokens[s], slot_hidden[s] = new[c], hidden[p + c]
            positions.append([p + 1 + c for c in range(len(new))] + [-1] * (4 - len(new)))
            counts.append(len(new))
            seq += new


@pytest.mark.parametrize("layers", [1, MAX_DRAFTS])
def test_the_pair_drafter_drafts_what_a_recomputation_drafts(layers):
    """Either drafter's layer 0, each request in its own ring: exact."""
    g = torch.Generator().manual_seed(2)
    cfg = tiny_config()
    target = TinyMiMo(cfg, g)
    drafter = mtp_drafter(cfg, target, g, layers)
    prompts = ([3, 14, 15, 92, 65, 35, 89, 79, 32, 38], [2, 71, 82, 81, 82, 84])
    seen = 0
    for k, draft, output, (want, want_outputs) in drive_pair(target, drafter, prompts, PAIR_SCHEDULE):
        assert draft == want[0], k
        torch.testing.assert_close(output, want_outputs[0], atol=1e-5, rtol=1e-5)
        seen += 1
    assert seen == 2 * len(PAIR_SCHEDULE) and set(drafter.rings) == {X, Y}


def test_a_shared_ring_would_mix_the_requests():
    """The test's power: the same pair in one ring (one cache for both) drafts differently."""
    g = torch.Generator().manual_seed(2)
    cfg = tiny_config()
    target = TinyMiMo(cfg, g)
    drafter = mtp_drafter(cfg, target, g, 1)
    drafter.ring_histories = lambda ring, _shared=[PositionalHistory()]: _shared  # noqa: B006
    prompts = ([3, 14, 15, 92, 65, 35, 89, 79, 32, 38], [2, 71, 82, 81, 82, 84])
    worst = 0.0
    for _, _, output, (_, want_outputs) in drive_pair(target, drafter, prompts, PAIR_SCHEDULE):
        worst = max(worst, float((output - want_outputs[0]).abs().max()))
    assert worst > 0.1


def test_the_pair_program_mirror_is_the_reference_pair_drafter():
    """MtpProgramMirror.run_pair (the device program's mirror, on the packed upload and the step's norm rows) drafts
    what ReferenceMTPDrafter's pair draft drafts."""
    g = torch.Generator().manual_seed(4)
    cfg = tiny_config()
    target = TinyMiMo(cfg, g)
    reference = mtp_drafter(cfg, target, g, MAX_DRAFTS)
    mirror = MtpProgramMirror(reference.mtps, target.embedding, target.lm_head, table_rows=B)
    prompts = ([3, 14, 15, 92, 65, 35, 89, 79, 32, 38], [2, 71, 82, 81, 82, 84])
    original = reference.draft

    def draft(plan, slot_tokens, slot_hidden):
        want = original(plan, slot_tokens, slot_hidden)
        tokens = [slot_tokens.get(s, 0) for s in range(SPEC_SLOTS)]
        rows = torch.zeros(B, cfg.hidden_size)
        for s, h in slot_hidden.items():
            rows[s] = h
        out = mirror.run_pair(pack_mtp_pair(B, tokens, plan).tolist(), rows)
        for k, anchor in enumerate(plan.anchors):
            assert out[k * DRAFT_WORDS] == anchor and out[k * DRAFT_WORDS + 1] == want[k]
        return want

    reference.draft = draft
    assert sum(1 for _ in drive_pair(target, reference, prompts, PAIR_SCHEDULE[:6])) == 12
    assert isinstance(reference.plan_pair([DraftSource(0, [X])], [[5, -1]], [1], [X]), MtpPairInputs)


def test_speculative_greedy_with_pair_drafts_equals_plain_greedy():
    """Two requests verified in one pass with one draft each (the MTP's, or every other step the target's own token)
    commit exactly the plain greedy tokens."""
    g = torch.Generator().manual_seed(0)
    cfg = tiny_config()
    target = TinyMiMo(cfg, g)
    drafter = mtp_drafter(cfg, target, g, 1)
    prompts = ([3, 14, 15, 92, 65, 35, 89, 79, 32, 38], [5, 6, 7, 8, 9])
    steps = 20
    plain = [list(p) for p in prompts]
    for seq in plain:
        for _ in range(steps):
            seq.append(target.greedy(seq)[0])
    seqs = [list(p) for p in prompts]
    slot_tokens, slot_hidden, sources, positions, counts = {}, {}, [], [], []
    for k, (slot, seq) in enumerate(zip((X, Y), seqs)):
        tok, h = target.greedy(seq)
        seq.append(tok)
        slot_tokens[slot], slot_hidden[slot] = tok, h
        sources.append(DraftSource(row=k, slots=[slot]))
        positions.append([len(seq) - 1, -1, -1, -1])
        counts.append(1)
    accepted = []
    while min(len(s) - len(p) for s, p in zip(seqs, prompts)) < steps:
        drafts = drafter.draft(drafter.plan_pair(sources, positions, counts, [X, Y]), slot_tokens, slot_hidden)
        if len(accepted) % 4 == 0:  # the random MTP rarely guesses right: every other step the target's own token
            drafts = [target.greedy(seq)[0] for seq in seqs]
        rows = [(k, [seq[-1], drafts[k], 0, 0], [len(seq) - 1, len(seq), -1, -1], 1) for k, seq in enumerate(seqs)]
        (vp,) = plan_verify_passes(rows)
        argmax = [[0, 0] for _ in seqs]
        slot_tokens, slot_hidden = {}, {}
        for s, row, c in vp.answers:
            seq = seqs[row]
            argmax[row][c], slot_hidden[s] = target.greedy(seq + [drafts[row]][:c])
            slot_tokens[s] = argmax[row][c]
        sources, positions, counts = draft_sources([vp]), [], []
        for k, seq in enumerate(seqs):
            new = accept_greedy(argmax[k], [drafts[k]], 1)
            positions.append([len(seq) + c for c in range(len(new))] + [-1] * (4 - len(new)))
            counts.append(len(new))
            accepted.append(len(new))
            seq += new
    assert all(s[: len(p) + steps] == q[: len(p) + steps] for s, p, q in zip(seqs, prompts, plain))
    assert set(accepted) == {1, 2}
