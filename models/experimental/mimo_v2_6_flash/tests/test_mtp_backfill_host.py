# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The target-hidden history and the MTP ring backfill (reference/mtp_backfill.py), on a small fp32 MiMo; no device.

A lane serves a request X the way the adapter does, with the device program's mirrors as the drafter (one layer:
ReferenceMTPDrafter, which the one-layer program runs; every layer: reference.mtp.MtpProgramMirror): its prompt in
chunks, a plain step, then speculative steps, then a second request Y joins the lane (its prompt resets the rings and
the stash, as the adapter's prefill does), both decode plainly for a while (no drafting), Y leaves, and X drafts
again. The history is written as the decode, verify and prefill programs write it. Before the first draft after the
prompt and after Y leaves the lane runs the backfill. Every draft must equal the drafts of an MTP that saw every
position: a recomputation of every layer over X's whole sequence (one layer: its fill over the whole sequence, then
its two steps on its own output). Without the backfill they differ (the test's power), and with a history shorter
than the window or after a gap the backfill reaches, it still drafts what the reference drafts. A backfill right
before the draft behind a verify, which it plans without that verify's commit, leaves every layer's ring over the
window the draft reads equal to the recomputation's, for every accepted count.

The history is sized as production sizes it for its MTP window (history_rows), here for the tiny model's window.
"""

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference.mtp import (
    MtpProgramMirror,
    PositionalHistory,
    ReferenceMTPDrafter,
    backfill_rings,
)
from models.experimental.mimo_v2_6_flash.reference.mtp_backfill import (
    BACKFILL_ROWS,
    RING_BLOCKS,
    SCRATCH_BLOCK,
    History,
    backfill_plan,
    backfill_row,
    backfill_start,
    decode_writes,
    entry,
    history_rows,
    pack_writes,
    prefill_writes,
    verify_writes,
)
from models.experimental.mimo_v2_6_flash.reference.spec_chain import CHAIN_HSLOTS, chain_row, stash_page
from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    MAX_DRAFTS,
    SPEC_SLOTS,
    DraftSource,
    accept_greedy,
    draft_source,
    plan_verify_passes,
)
from models.experimental.mimo_v2_6_flash.reference.spec_inputs import pack_mtp_layers, pack_verify
from models.experimental.mimo_v2_6_flash.tests.test_spec_decode_host import (
    TinyMiMo,
    mtp_drafter,
    recompute_drafts,
    tiny_config,
)
from models.experimental.mimo_v2_6_flash.reference import spec_decode as spec_layout

# The MTP drafter's 4-slot verify (the draft groups: test_spec_groups_host.py, test_spec_groups_adapter_host.py).
pytestmark = pytest.mark.skipif(
    (spec_layout.SPEC_SLOTS, spec_layout.DRAFT_GROUPS) != (4, 1),
    reason="MTP's 4-slot layouts: run without MIMO_SPEC_SLOTS / MIMO_SPEC_GROUPS",
)


B = 32
WINDOW = tiny_config().sliding_window  # the tiny model's MTP window (8)
ROWS = history_rows(WINDOW)  # history rows per slot
X, Y = 1, 2  # the requests' lane slots


def recompute_one_layer(target, mtp, seq):
    """One-layer drafting over the whole committed ``seq``: the fill over every row q (h_q, x[q + 1]), then the two
    steps on the layer's own output at B and B + 1."""
    b = len(seq) - 1
    cache = PositionalHistory()
    h = mtp(target.hidden(seq)[:b], target.embedding[torch.tensor(seq[1:])], torch.arange(b), cache)[-1]
    drafts = [int(torch.argmax(h @ target.lm_head.T))]
    for k in range(2):
        h = mtp(h[None], target.embedding[torch.tensor([drafts[-1]])], torch.tensor([b + k]), cache)[0]
        drafts.append(int(torch.argmax(h @ target.lm_head.T)))
    return drafts


class Lane:
    """One lane: X's drafter (the device program's mirror), the history, and whether the rings continue X."""

    def __init__(self, target, reference, layers, backfill=True):
        self.target, self.layers, self.backfill = target, layers, backfill
        mtps = reference.mtps[:layers]
        self.mtps = mtps
        if layers == 1:
            self.drafter = ReferenceMTPDrafter(mtps, target.embedding, target.lm_head)
        else:
            self.program = MtpProgramMirror(mtps, target.embedding, target.lm_head, table_rows=B)
        self.history = History(target.cfg.hidden_size, ROWS)
        self.ring = None  # (slot, the last row the rings hold for it)
        self.backfills = 0

    def reset(self):
        """A prompt in the lane: the rings and the stash start empty (TtMiMoMTP.reset)."""
        if self.layers == 1:
            self.drafter.reset()
        else:
            self.program.reset()
        self.ring = None

    def prefill(self, slot, prompt, chunk):
        self.reset()
        hidden = self.target.hidden(prompt)
        for start in range(0, len(prompt), chunk):
            tokens = prompt[start : start + chunk]
            self.history.apply(prefill_writes(slot, start, tokens, start, start > 0, ROWS), hidden)

    def plain(self, seqs):
        """A plain decode step of the requests {slot: committed sequence}: each slot's input is its last token."""
        positions, tokens, rows = [-1] * B, [0] * B, torch.zeros(B, self.target.cfg.hidden_size)
        for s, seq in seqs.items():
            positions[s], tokens[s] = len(seq) - 1, seq[-1]
            tok, rows[s] = self.target.greedy(seq)
            seq.append(tok)
        self.history.apply(decode_writes(positions, tokens, SPEC_SLOTS, ROWS), rows)
        return rows

    def draft(self, source, committed, positions, m, rows):
        """Propose: the backfill when the rings do not continue the request's rows, then the draft."""
        slot, a = source.slots[0] if len(source.slots) == 1 else X, positions[0] - 1
        if self.backfill and self.ring != (slot, a - 1):
            self.backfills += 1
            upload = backfill_row(B, slot, a)
            if self.layers == 1:
                backfill_rings(self.mtps, self.drafter.histories, self.target.embedding, self.history, upload)
            else:
                p = self.program
                p.stash = backfill_rings(
                    self.mtps, p.histories, self.target.embedding, self.history, upload, p.stash, p.stash_hidden
                )
        self.ring = (slot, positions[m - 1] - 1)
        if self.layers == 1:
            plan = self.drafter.plan(source, positions, m)
            slot_tokens = {source.slots[c]: committed[c] for c in range(m)}
            return self.drafter.draft(plan, slot_tokens, {s: rows[s] for s in source.slots[:m]})
        mtp = pack_mtp_layers(B, source, committed, positions, m).tolist()
        out = self.program.run(mtp, rows)
        return out[1 : 1 + MAX_DRAFTS]

    def verify(self, seq, drafts, accept):
        """A verify pass of X (slots 0 .. 3): the history takes every slot's row; the target commits the first
        ``accept`` drafts it agrees with, then its own token. Returns (committed, source, committed positions, the
        pass's final norm rows)."""
        p = len(seq) - 1
        block = [seq[-1]] + drafts
        (vp,) = plan_verify_passes([(4 + X, block, list(range(p, p + 1 + MAX_DRAFTS)), MAX_DRAFTS)])
        rows, argmax = torch.zeros(B, self.target.cfg.hidden_size), [0] * (1 + MAX_DRAFTS)
        hidden = self.target.hidden(seq[:p] + block)
        for s, _, c in vp.answers:
            rows[s] = hidden[p + c]
            argmax[c] = int(torch.argmax(hidden[p + c] @ self.target.lm_head.T))
        # The target as the test wants it: it takes ``accept`` drafts, whatever they are.
        for c in range(MAX_DRAFTS):
            if c < accept:
                argmax[c] = drafts[c]
            elif argmax[c] == drafts[c]:
                argmax[c] = (drafts[c] + 1) % self.target.cfg.vocab_size
        committed = accept_greedy(argmax, drafts, MAX_DRAFTS)
        tok = torch.zeros(B, dtype=torch.int32)
        pos = torch.full((B,), -1, dtype=torch.int32)
        tok[:SPEC_SLOTS], pos[:SPEC_SLOTS] = torch.tensor(vp.tokens), torch.tensor(vp.positions)
        chain = chain_row(B, None, 0, hslots=[X] * SPEC_SLOTS)
        verify = pack_verify(tok, pos, torch.tensor(chain, dtype=torch.int32)).tolist()
        self.history.apply(verify_writes(verify, chain[CHAIN_HSLOTS : CHAIN_HSLOTS + SPEC_SLOTS], ROWS), rows)
        positions = [p + 1 + c for c in range(len(committed))] + [-1] * (1 + MAX_DRAFTS - len(committed))
        return committed, draft_source([vp]), positions, rows


def want_drafts(lane, reference, seq):
    if lane.layers == 1:
        return recompute_one_layer(lane.target, reference.mtps[0], seq)
    return recompute_drafts(lane.target, reference.mtps, seq, 0)[0]


def recompute_rings(lane, seq, drafts):
    """Each of the lane's MTP layers over every row q of the committed ``seq`` (the last at B) but B, row q taking h_q
    and x~[q + i + 1] (``seq`` then ``drafts``; one layer: x[q + 1]): the K / V of an MTP that saw every position,
    one PositionalHistory per layer."""
    b = len(seq) - 1
    hidden = lane.target.hidden(seq)[:b]
    caches = []
    for i, mtp in enumerate(lane.mtps):
        window = list(seq) + list(drafts[:i])
        cache = PositionalHistory()
        mtp(hidden, lane.target.embedding[torch.tensor([window[q + i + 1] for q in range(b)])], torch.arange(b), cache)
        caches.append(cache)
    return caches


def serve(target, reference, layers, backfill=True, prompt_len=40, chunk=16, gap=12, accepts=(3, 0, 1, 2, 3, 1)):
    """X's prompt, a plain step, speculative steps, Y joins for ``gap`` plain steps, Y leaves, X speculates again.
    Yields (step name, X's drafts, the drafts of an MTP that saw every position)."""
    g = torch.Generator().manual_seed(11)
    vocab = target.cfg.vocab_size
    lane = Lane(target, reference, layers, backfill)
    x = torch.randint(0, vocab, (prompt_len,), generator=g).tolist()
    lane.prefill(X, x, chunk)
    x.append(target.greedy(x)[0])  # the prompt's sampled token

    def speculate(rows, steps, name):
        nonlocal x
        drafts = lane.draft(DraftSource(row=4 + X, slots=[X]), [x[-1]], [len(x) - 1, -1, -1, -1], 1, rows)
        yield name, drafts, want_drafts(lane, reference, x)
        for k in range(steps):
            committed, source, positions, rows = lane.verify(x, drafts, accepts[k % len(accepts)])
            x += committed
            drafts = lane.draft(source, committed, positions, len(committed), rows)
            yield f"{name} verify {k} (m {len(committed)})", drafts, want_drafts(lane, reference, x)

    rows = lane.plain({X: x})
    yield from speculate(rows, 8, "after the prompt")
    # Y joins: its prompt resets the lane's rings; X's last drafts are verified beside Y's first step, then both
    # decode plainly.
    y = torch.randint(0, vocab, (23,), generator=g).tolist()
    lane.prefill(Y, y, chunk)
    y.append(target.greedy(y)[0])
    committed, _, _, _ = lane.verify(x, lane_last_drafts(lane, x, target), 1)
    x += committed
    for _ in range(gap):
        lane.plain({X: x, Y: y})
    rows = lane.plain({X: x})  # Y left
    yield from speculate(rows, 8, f"after {gap} steps beside Y")
    assert lane.backfills == (2 if backfill else 0)


def lane_last_drafts(lane, x, target):
    """X's drafts proposed before Y joined (any tokens: the verify takes one of them)."""
    return [(x[-1] + 1 + k) % target.cfg.vocab_size for k in range(MAX_DRAFTS)]


@pytest.fixture(scope="module")
def tiny():
    g = torch.Generator().manual_seed(3)
    cfg = tiny_config()
    target = TinyMiMo(cfg, g)
    return target, mtp_drafter(cfg, target, g, MAX_DRAFTS)


@pytest.mark.parametrize("layers", [MAX_DRAFTS, 1])
@pytest.mark.parametrize("gap", [3, 12, 40])
def test_drafts_after_the_backfill_are_an_mtp_that_saw_every_position(tiny, layers, gap):
    target, reference = tiny
    n = 0
    for name, got, want in serve(target, reference, layers, gap=gap):
        assert got == want, name
        n += 1
    assert n == 18


@pytest.mark.parametrize("layers", [MAX_DRAFTS, 1])
@pytest.mark.parametrize("prompt_len, chunk", [(3, 16), (6, 4)])
def test_short_prompts(tiny, layers, prompt_len, chunk):
    """Prompts shorter than the window (and than the history): the entries before position 0 are invalid."""
    target, reference = tiny
    for name, got, want in serve(target, reference, layers, prompt_len=prompt_len, chunk=chunk, gap=2):
        assert got == want, name


@pytest.mark.parametrize("layers", [MAX_DRAFTS, 1])
@pytest.mark.parametrize("accept", range(MAX_DRAFTS + 1))
def test_a_backfill_right_after_a_verify_fills_the_rings_of_an_mtp_that_saw_every_position(tiny, layers, accept):
    """X speculates, a prompt resets the lane's rings (another request's, over X's deferred verify), and X's next
    verify at A commits m = accept + 1 tokens: the draft behind it backfills first, enqueued behind the verify before
    its commit is known, from a history that also holds the verify's rows A .. A + 3. After the draft, every layer's
    K / V at A - WINDOW + 1 .. B - 1 (the windows its rows read, B the last committed position) equals a
    recomputation's over X's whole sequence, and so do the drafts and (every layer) the layers' outputs."""
    target, reference = tiny
    g = torch.Generator().manual_seed(11)
    lane = Lane(target, reference, layers)
    x = torch.randint(0, target.cfg.vocab_size, (40,), generator=g).tolist()
    lane.prefill(X, x, 16)
    x.append(target.greedy(x)[0])
    rows = lane.plain({X: x})
    drafts = lane.draft(DraftSource(row=4 + X, slots=[X]), [x[-1]], [len(x) - 1, -1, -1, -1], 1, rows)
    for k in range(5):
        committed, source, positions, rows = lane.verify(x, drafts, (3, 0, 1, 2, 3)[k])
        x += committed
        drafts = lane.draft(source, committed, positions, len(committed), rows)
    lane.reset()
    a = len(x) - 1
    committed, source, positions, rows = lane.verify(x, drafts, accept)
    assert len(committed) == accept + 1 and positions[0] - 1 == a
    x += committed
    drafts = lane.draft(source, committed, positions, len(committed), rows)
    assert lane.backfills == 2, "after the prompt, after the reset"
    assert drafts == want_drafts(lane, reference, x)
    if layers > 1:
        for got, want in zip(lane.program.outputs, recompute_drafts(target, reference.mtps, x, 0)[1]):
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    rings = lane.drafter.histories if layers == 1 else lane.program.histories
    for i, (got, want) in enumerate(zip(rings, recompute_rings(lane, x, drafts))):
        for q in range(max(0, a - WINDOW + 1), len(x) - 1):
            for t, w in zip(got.entries[q], want.entries[q]):
                torch.testing.assert_close(t, w, atol=1e-5, rtol=1e-5, msg=f"layer {i} position {q}")


@pytest.mark.parametrize("layers", [MAX_DRAFTS, 1])
def test_without_the_backfill_the_first_drafts_differ(tiny, layers):
    target, reference = tiny
    differ = [name for name, got, want in serve(target, reference, layers, backfill=False) if got != want]
    assert differ, "the rings would hold every position without the backfill"


# ---------------------------------------------------------------------- the writes and the plan
def test_prefill_writes_cover_the_ring_once():
    for start, n, continues in ((0, 5, False), (0, 40, False), (32, 7, True), (32, 7, False), (64, 64, True)):
        writes = prefill_writes(2, start, list(range(100, 100 + n)), 3, continues, ROWS)
        entries = [e for _, e, _, _ in writes]
        assert len(set(entries)) == len(entries) and all(2 * ROWS <= e < 3 * ROWS for e in entries)
        end = start + n
        for s, e, p, t in writes:
            if s >= 0:
                assert start <= p < end and e == entry(2, p, ROWS) and t == 100 + p - start and s == 3 + p - start
            else:
                assert p == -1
        own = {p for s, _, p, _ in writes if s >= 0}
        assert own == set(range(max(start, end - ROWS), end))
        assert len(writes) == (ROWS if not continues else len(own) + sum(q < 0 for q in range(end - ROWS, end)))


def test_verify_writes_go_to_each_slots_request():
    tok = torch.arange(10, 10 + B, dtype=torch.int32)
    pos = torch.full((B,), -1, dtype=torch.int32)
    pos[:3] = torch.tensor([50, 51, 90])
    chain = chain_row(B, None, 0, hslots=[1, 1, 3, None])
    assert chain[:CHAIN_HSLOTS] == [0] * CHAIN_HSLOTS
    verify = pack_verify(tok, pos, torch.tensor(chain, dtype=torch.int32)).tolist()
    writes = verify_writes(verify, chain[CHAIN_HSLOTS : CHAIN_HSLOTS + SPEC_SLOTS], ROWS)
    assert writes == [(0, entry(1, 50, ROWS), 50, 10), (1, entry(1, 51, ROWS), 51, 11), (2, entry(3, 90, ROWS), 90, 12)]
    # A zero chain row (no history slots) writes nothing.
    assert verify_writes(pack_verify(tok, pos).tolist(), [0] * SPEC_SLOTS, ROWS) == []


@pytest.mark.parametrize("a", [0, 1, 2, 5, 15, 16, 17, 63, 64, 200, 255, 256, 300, 1 << 20])
def test_backfill_plan(a):
    """The plan fills the whole ring from S0 (a block boundary, the window's start in it), real rows exactly where the
    history holds the row, each layer's token wherever the history holds its position (past a, a verify pass's
    rows), and stashes rows a - 2, a - 1. The ring holds the window and the verify pass at a together."""
    hist = History(4, ROWS)
    held = [q for q in range(max(0, a - WINDOW + 1), a + 1) if q % 5 != 3]  # holes
    hist.apply([(0, entry(X, q, ROWS), q, 1000 + q) for q in held], torch.zeros(1, 4))
    verified = list(range(a + 1, a + SPEC_SLOTS))  # the rows of a verify pass at a past a (rejected or committed)
    hist.apply([(0, entry(X, q, ROWS), q, 2000 + q) for q in verified], torch.zeros(1, 4))
    assert all(hist.position(X, q) == q for q in held + verified)
    for layers in (1, MAX_DRAFTS):
        stash = stash_page(B, [3, 4, 5, 6], [1, 2, 3, 4]) if layers > 1 else None
        plan, new = backfill_plan(backfill_row(B, X, a), hist.page, layers, stash, ROWS, WINDOW)
        s0 = backfill_start(a, WINDOW)
        assert s0 % 64 == 0 and s0 <= max(0, a - WINDOW + 1) < s0 + 64
        assert plan["table"][:RING_BLOCKS] == [(s0 // 64 + k) % RING_BLOCKS for k in range(RING_BLOCKS)]
        assert plan["table"][RING_BLOCKS:] == [SCRATCH_BLOCK + k for k in range(RING_BLOCKS)]
        for r in range(BACKFILL_ROWS):
            q = s0 + r
            real = max(0, a - WINDOW + 1) <= q <= a - 1 and q in held
            assert plan["hidden"][r] == (entry(X, q, ROWS) if real else 0)
            assert plan["rope"][r] == min(q, a)
            for i in range(layers):
                x = q + i + 1
                want = (1000 + x if x in held else 2000 + x if x in verified else 0) if real else 0
                assert plan["tokens"][i][r] == want, (q, i)
        if layers > 1:
            positions = [-1, -1] + [q if q in held and q + 1 in held and q <= a - 1 else -1 for q in (a - 2, a - 1)]
            assert new == stash_page(
                B, positions, [0, 0] + [1000 + q + 1 if p >= 0 else 0 for q, p in zip((a - 2, a - 1), positions[2:])]
            )
            assert plan["stash_rows"] == [
                (2 + k, entry(X, q, ROWS)) for k, q in enumerate((a - 2, a - 1)) if positions[2 + k] >= 0
            ]
    idle, same = backfill_plan(backfill_row(B), hist.page, MAX_DRAFTS, [9] * B, ROWS, WINDOW)
    assert idle["table"] == [SCRATCH_BLOCK + k % RING_BLOCKS for k in range(2 * RING_BLOCKS)] and same == [9] * B
    assert pack_writes([(1, 2, 3, 4)], 2) == [1, 2, 3, 4, -2, 0, -1, 0]
