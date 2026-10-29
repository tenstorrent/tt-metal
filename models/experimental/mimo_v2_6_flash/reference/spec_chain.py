# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MTP draft chained behind a verify on device (tt/spec_chain.py, tt/kernels/spec_chain.cpp): the host layouts and
an exact integer mirror of the five kernel steps (no ttnn), shared by the vLLM adapter and the host tests.

A lane that speculates runs one request in one verify pass (reference/spec_decode.py, ``draft_source``): its column c
in slot ``slots[c]`` at position P[slots[c]]. Before stage B the host read the pass's acceptance, then planned the MTP
draft (``plan_mtp``: which slots the fill pass writes, the anchor slot whose greedy tokens are the drafts, the ring
cache positions and RoPE indices) and uploaded it (``pack_mtp``). All of that depends on the host only through the
accepted count m and the committed tokens, which the accept step leaves on device. So the queue of a speculating
lane now runs, without the host in between:

1. the verify trace (its upload carries the chain row below);
2. the accept trace: SampleVerify (reference/sample_verify.py), every row greedy on the argmax ids in an
   ``argmax_ids`` step, then the chain step: from the chain row, the verify's positions and the accept page it writes
   the draft's packed upload (``mtp_packed``, pack_mtp's layout) exactly as pack_mtp(plan_mtp(...)) would, and the
   result page's header;
3. the MTP draft trace (its derive program reads ``mtp_packed`` as before), ending with the drafts step, which writes
   the anchor slot's drafts into the result page.

With every MTP layer (MIMO_MTP_LAYERS=3) the chain step writes the commit's description instead
(reference.spec_inputs.pack_mtp_layers: what propose_draft_tokens would upload from the plugin's commit), and the draft
trace starts with two more steps of the same kernel:

* ``plan_step`` (mode 3): plan_mtp_layers on device, from the description and the stash the last draft left (the
  ``stash`` page: each slot's position and layer-0 token; the slots' target hidden rows in a buffer beside it). It
  writes the passes' inputs: per slot the ring cache position, the SPEC_SLOTS serial positions, the RoPE index, the row
  of the hidden table [the step's final norm rows, the stash rows] to gather, and the layer-0 token; and the new stash.
  The anchor is the last slot.
* ``tokens_step`` (mode 4, before passes 1 and 2): layer i's tokens, the layer-0 tokens shifted by i, the drafts so far
  (the anchor's greedy token of each pass before it) appended (spec_decode.mtp_layer_tokens).

The host then reads the result page, one 320-byte read per pass: what the verify answers (the accept page, or the
argmax ids) and the drafts the next propose returns. Every other lane's verify passes run the accept trace too, with
an idle chain row (no draft): their result page carries the accept page and the argmax ids.

Under async scheduling the plugin chains verifies (its section 4j): the next verify of such a lane is built and
enqueued before the host has read this one. Its upload marks the pass's source as relayed, and the relay step (mode
2), the first program of the verify trace, writes the source's tokens and positions from the result page the lane's
last pass left (the last committed token at the bonus, then the drafts at the positions that follow) and the drafts
into the accept step's control page (``relay_step``); SpecInputs.derive_verify and the decode step then run as on an
upload the host built. The host never sees those tokens before it reads the next page, whose positions section says
where the device put them.

The chain row only holds what the host knows before the verify: whether a draft chains behind the pass, the source's
columns and slots, the position limit, and whether the source's tokens and positions are relayed. The draft is
declined on device when the bonus (the position after the last committed token) is at or past the limit,
max_seq_len - K: the next verify writes positions bonus .. bonus + K, and the draft's RoPE indices must stay inside
the device tables (they end at max_seq_len).

Layouts:

- chain row (row CHAIN_ROW of the verify upload, int32, B wide): source (1: a draft chains behind this pass), the
  source's columns n + 1, limit, then the source's slots [SPEC_SLOTS] (-1 past its columns), relay (1: the relay step
  writes the source's tokens and positions), each verify slot's request slot + 1 [SPEC_SLOTS] (the MTP history's,
  reference/mtp_backfill.py; 0: none), zeros. The source's request is the pass's only one: group 0 of the accept page
  and of the control page.
- result page (uint32 [RESULT_WORDS], sections on 64-byte boundaries, each written by one step):
  [ACCEPT_AT, +GROUPS * OUT_WORDS) the accept page (unpack_out); [ARGMAX_AT, +SPEC_SLOTS) the verify's argmax ids of
  slots 0 .. SPEC_SLOTS - 1 and [POSITIONS_AT, +SPEC_SLOTS) their positions, as the verify ran them (a relayed pass's
  are the device's); [STATUS_AT, +4) the chain: drafting (0 / 1), anchor slot (NO_SLOT: none), bonus position,
  unused; [DRAFTS_AT, +1 + MAX_DRAFTS) the drafts step: the anchor slot it found (the slot with a step2 position;
  NO_SLOT: an idle draft) and that slot's greedy token of each MTP pass. With every layer the chain's anchor is the
  draft's (the last slot), and the drafts step finds it from the description.
- stash page (int32, B wide; every layer): per slot of the last draft's pass its true position (-1 none)
  [STASH_POSITIONS, +SPEC_SLOTS), and its layer-0 token [STASH_TOKENS, +SPEC_SLOTS).

The relay step reads the result page of the lane's last pass, which must be the pass whose draft chained (the
adapter checks that on the host); it relays only a page whose chain drafted (drafting 1, the drafts step's anchor the
chain's). Otherwise the relayed slots go idle (position -1: no K / V written) and the pass drafts nothing: the host
finds position -1 in the positions section and raises. The page reads the same with one MTP layer or every layer
(there the chain's and the drafts step's anchor are both the last slot, and the bonus is still the position after
the last committed token), so the relay does not depend on the drafter.
"""

from dataclasses import dataclass

from .sample_verify import GROUP_WORDS, GROUPS, NO_TOKEN, OUT_WORDS, unpack_out
from .spec_decode import (
    DRAFT_GROUPS,
    LAYOUT,
    LAYOUTS,
    MAX_DRAFTS,
    SPEC_SLOTS,
    SpecLayout,
    mtp_cache_position,
    planned_drafts,
)
from .spec_inputs import (
    DESC_A,
    DESC_COUNT,
    DESC_DRAFTING,
    DESC_SLOTS,
    MTP_ROWS,
    VERIFY_ROWS,
    desc_tokens,
)

CHAIN_ROW = 2  # the verify upload's row (reference.spec_inputs.pack_verify)
# Per draft group g, a block of the chain row at g * CHAIN_GROUP_WORDS (one block, the whole pass, without groups).
CHAIN_SOURCE, CHAIN_COLUMNS, CHAIN_LIMIT, CHAIN_SLOTS = 0, 1, 2, 3
NO_SLOT = 0xFFFFFFFF
ACCEPT_WORDS = GROUPS * OUT_WORDS
STATUS_WORDS = 4  # a draft group's status: drafting, anchor, bonus, unused


def _align16(words):
    return (words + 15) // 16 * 16


# The result page's sections (0, 32, 40, 48, 64, 80 with 4 slots; 0, 128, 136, 144, 160, 176 with 8; 0, 64, 72, 80,
# 96, 112 with 8 slots in 2 draft groups). The header up to the status is the same in every layout.
ACCEPT_AT = 0
ARGMAX_AT = _align16(ACCEPT_WORDS)
POSITIONS_AT = ARGMAX_AT + 8 * ((SPEC_SLOTS + 7) // 8)
STATUS_AT = _align16(POSITIONS_AT + SPEC_SLOTS)
CTRL_DRAFTS = 4  # a control page group's first draft word (reference.sample_verify.group_words)


@dataclass(frozen=True)
class ChainPages:
    """The chain row's and the result page's group sections in a SpecLayout: a chain block per draft group (its
    source's slots ``group_slots`` words, then the relay word), the history's words after the blocks, and a status and
    a drafts section (the drafts step's anchor and ``max_drafts`` drafts) per draft group."""

    layout: SpecLayout

    @property
    def relay(self):
        return CHAIN_SLOTS + self.layout.group_slots

    @property
    def group_words(self):
        return self.relay + 1

    @property
    def words(self):
        return self.layout.groups * self.group_words

    @property
    def hslots(self):
        """The history's words: each verify slot's request slot + 1 (0: none; reference/mtp_backfill.py), after the
        blocks of every layout of the launch, so that the history's verify write reads them in either."""
        return max(layout.groups * (CHAIN_SLOTS + layout.group_slots + 1) for layout in LAYOUTS)

    @property
    def draft_words(self):
        return 1 + self.layout.max_drafts

    @property
    def drafts_at(self):
        return _align16(STATUS_AT + self.layout.groups * STATUS_WORDS)

    @property
    def result_words(self):
        return _align16(self.drafts_at + self.layout.groups * self.draft_words)


PAGES = {layout: ChainPages(layout) for layout in LAYOUTS}
for _p in PAGES.values():
    assert _p.drafts_at % 16 == 0 and STATUS_AT + _p.layout.groups * STATUS_WORDS <= _p.drafts_at
    assert CTRL_DRAFTS + _p.layout.max_drafts <= GROUP_WORDS and _p.layout.groups <= GROUPS
assert ACCEPT_AT + ACCEPT_WORDS <= ARGMAX_AT and ARGMAX_AT + SPEC_SLOTS <= POSITIONS_AT
assert POSITIONS_AT + SPEC_SLOTS <= STATUS_AT and all(w % 16 == 0 for w in (ARGMAX_AT, STATUS_AT))
assert CHAIN_ROW < VERIFY_ROWS
# LAYOUT's (the only layout of a launch with one).
CHAIN_RELAY = PAGES[LAYOUT].relay
CHAIN_GROUP_WORDS = PAGES[LAYOUT].group_words
CHAIN_WORDS = PAGES[LAYOUT].words
CHAIN_HSLOTS = PAGES[LAYOUT].hslots
DRAFT_WORDS = PAGES[LAYOUT].draft_words  # a draft group's drafts section: the drafts step's anchor, the drafts
DRAFTS_AT = PAGES[LAYOUT].drafts_at
RESULT_WORDS = PAGES[LAYOUT].result_words
HEADER_WORDS = DRAFTS_AT  # the chain step's sections; the drafts step writes the rest
STASH_POSITIONS, STASH_TOKENS = 0, SPEC_SLOTS
STASH_WORDS = STASH_TOKENS + SPEC_SLOTS


def _sources(source):
    """``source``: None, a DraftSource, or a list of them."""
    if source is None:
        return []
    return list(source) if isinstance(source, (list, tuple)) else [source]


def chain_row(max_batch, source=None, limit=0, relay=False, hslots=None, layout=LAYOUT):
    """The verify upload's chain row, int32 values [max_batch]: ``source`` the pass's DraftSource when a draft chains
    behind it, or a list of them, one per draft group (None: an idle row), ``limit`` the first bonus position a draft
    is declined at, ``relay`` the relay step writes the sources' tokens and positions from the lane's last result page
    (a chained verify), ``hslots`` each verify slot's request slot in the lane (None: no history write) for the history
    (reference/mtp_backfill.py); in ``layout``'s chain blocks."""
    pages = PAGES[layout]
    assert max_batch >= pages.hslots + SPEC_SLOTS
    sources = _sources(source)
    assert sources or not relay, "the relay writes a source's columns"
    row = [0] * max_batch
    for j, h in enumerate(hslots or []):
        row[pages.hslots + j] = 0 if h is None else int(h) + 1
    for src in sources:
        assert 0 <= src.group < layout.groups and 1 <= len(src.slots) <= layout.group_slots, src
        at = src.group * pages.group_words
        assert row[at + CHAIN_SOURCE] == 0, f"two sources in draft group {src.group}"
        row[at + CHAIN_SOURCE], row[at + CHAIN_COLUMNS], row[at + CHAIN_LIMIT] = 1, len(src.slots), int(limit)
        for c in range(layout.group_slots):
            row[at + CHAIN_SLOTS + c] = src.slots[c] if c < len(src.slots) else -1
        row[at + pages.relay] = int(bool(relay))
    return row


# ---------------------------------------------------------------------- the kernel (spec_chain.cpp)
def _header(verify, accept, argmax, layout):
    """The chain step's result header with every group of ``layout`` idle (the accept page, the argmax ids and
    positions of the pass's slots), the verify's positions and its chain row."""
    b = len(verify) // VERIFY_ROWS
    pos = verify[b : 2 * b]
    header = [0] * PAGES[layout].drafts_at
    header[ACCEPT_AT : ACCEPT_AT + ACCEPT_WORDS] = [int(w) & 0xFFFFFFFF for w in accept]
    header[ARGMAX_AT : ARGMAX_AT + SPEC_SLOTS] = [int(t) & 0xFFFFFFFF for t in argmax[:SPEC_SLOTS]]
    header[POSITIONS_AT : POSITIONS_AT + SPEC_SLOTS] = [int(p) & 0xFFFFFFFF for p in pos[:SPEC_SLOTS]]
    for g in range(layout.groups):
        header[STATUS_AT + g * STATUS_WORDS : STATUS_AT + (g + 1) * STATUS_WORDS] = [0, NO_SLOT, 0, 0]
    return header, pos, verify[CHAIN_ROW * b : (CHAIN_ROW + 1) * b]


def _commits(header, pos, chain, accept, layout):
    """The draft groups whose draft the chain step writes: [(group, m, slots, committed tokens, bonus)], each with a
    source, 1 <= m <= its columns, and the bonus P[slots[m - 1]] + 1 below the limit."""
    out, words = [], PAGES[layout].group_words
    for g in range(layout.groups):
        blk = chain[g * words : (g + 1) * words]
        m, columns = int(accept[g * OUT_WORDS]), blk[CHAIN_COLUMNS]
        if blk[CHAIN_SOURCE] == 0 or not 1 <= m <= columns <= layout.group_slots:
            continue
        slots = blk[CHAIN_SLOTS : CHAIN_SLOTS + m]
        bonus = pos[slots[m - 1]] + 1
        if bonus < blk[CHAIN_LIMIT]:
            out.append((g, m, slots, [int(t) for t in accept[g * OUT_WORDS + 1 : g * OUT_WORDS + 1 + m]], bonus))
    return out


def _status(header, g, anchor, bonus):
    header[STATUS_AT + g * STATUS_WORDS : STATUS_AT + (g + 1) * STATUS_WORDS] = [1, anchor, bonus, 0]


def chain_step(verify, accept, argmax, layers=1, layout=LAYOUT):
    """The chain step (mode 0): ``verify`` the verify upload's int32 values [VERIFY_ROWS * B], ``accept`` the accept
    page's uint32 words [ACCEPT_WORDS], ``argmax`` the verify's greedy tokens [B]. Returns (the draft's packed upload,
    int32 values [MTP_ROWS * B]; the result header, uint32 words [HEADER_WORDS]). Draft group g's request commits accept
    group g (its source in chain block g). Planned drafting (planned_drafts(layers)): the upload holds each group's
    commit description (pack_mtp_layers, row g) and a group's anchor is its last slot; the one-layer program drafts
    one group. ``layout``: the pass's draft groups."""
    header, pos, chain = _header(verify, accept, argmax, layout)
    b = len(pos)
    planned = planned_drafts(layers, layout)
    # pack_mtp's idle draft: tokens 0, cache positions -1, RoPE indices 0 (pack_mtp_layers': zeros).
    mtp = [0] * b + [0 if planned else -1] * (3 * b) + [0] * ((MTP_ROWS - 4) * b)
    assert planned or layout.groups == 1, "the one-layer program drafts one request per lane"
    gs = layout.group_slots
    for g, m, slots, tokens, bonus in _commits(header, pos, chain, accept, layout):
        if planned:
            # pack_mtp_layers of the commit: committed position c at P[slots[c]] + 1, so A = P[slots[0]].
            d = g * b
            mtp[d + DESC_DRAFTING], mtp[d + DESC_COUNT], mtp[d + DESC_A] = 1, m, pos[slots[0]]
            for c in range(gs):
                mtp[d + DESC_SLOTS + c] = slots[c] if c < m else -1
                mtp[d + desc_tokens(gs) + c] = tokens[c] if c < m else 0
            _status(header, g, (g + 1) * gs - 1, bonus)
        else:
            # plan_mtp with committed position c at P[slots[c]] + 1: the fill writes column c's hidden at P[slots[c]].
            anchor = slots[m - 1]
            for c, s in enumerate(slots):
                mtp[s] = tokens[c]
                mtp[b + s] = mtp_cache_position(pos[s])
                mtp[4 * b + s] = max(pos[s], 0)
            mtp[2 * b + anchor], mtp[3 * b + anchor] = mtp_cache_position(bonus), mtp_cache_position(bonus + 1)
            mtp[5 * b + anchor], mtp[6 * b + anchor] = bonus, bonus + 1
            _status(header, g, anchor, bonus)
    return mtp, header


def relay_step(verify, result, ctrl, layout=LAYOUT):
    """The relay step (mode 2, first in the verify trace): ``verify`` the verify upload's int32 values
    [VERIFY_ROWS * B] as the host copied it, ``result`` the lane's last result page (uint32 words [RESULT_WORDS]),
    ``ctrl`` the accept step's control page as the host staged it (uint32 words [GROUPS * GROUP_WORDS]). Returns the
    upload and the control page the verify and accept steps then read. Per draft group whose chain block says relay,
    the source's columns c = 0 .. n take, at slot slots[c], the last committed token (c = 0) or draft c at position
    bonus + c, and the control page's group g the drafts, from the page's group g; a group whose chain did not draft
    idles them instead (position -1, token 0, draft 0) and clears its source. Groups without the relay word are left
    unchanged. ``layout``: the draft groups of the upload and of the page (one layout: a lane's chain breaks when its
    layout changes)."""
    b = len(verify) // VERIFY_ROWS
    verify, ctrl = list(verify), [int(w) & 0xFFFFFFFF for w in ctrl]
    page = [int(w) & 0xFFFFFFFF for w in result]
    pages, k = PAGES[layout], layout.max_drafts
    for g in range(layout.groups):
        blk = CHAIN_ROW * b + g * pages.group_words
        if verify[blk + pages.relay] == 0:
            continue
        status = STATUS_AT + g * STATUS_WORDS
        drafts, accept = pages.drafts_at + g * pages.draft_words, ACCEPT_AT + g * OUT_WORDS
        columns, m = verify[blk + CHAIN_COLUMNS], page[accept]
        ok = page[status] == 1 and page[drafts] == page[status + 1]
        ok = ok and 1 <= columns <= 1 + k and 1 <= m <= 1 + k
        bonus = page[status + 2]
        for c in range(min(max(columns, 0), layout.group_slots)):
            s = verify[blk + CHAIN_SLOTS + c]
            if not 0 <= s < SPEC_SLOTS:
                continue
            token = (page[accept + m] if c == 0 else page[drafts + c]) if ok else 0
            verify[s] = token - (1 << 32) if token >= 1 << 31 else token
            verify[b + s] = bonus + c if ok else -1
            if c > 0:
                ctrl[g * GROUP_WORDS + CTRL_DRAFTS + c - 1] = page[drafts + c] if ok else 0
        if not ok:
            verify[blk + CHAIN_SOURCE] = 0
    return verify, ctrl


def drafts_step(mtp, heads, layers=1, layout=LAYOUT):
    """The drafts step (mode 1): ``mtp`` the draft's packed upload [MTP_ROWS * B] (int32 values), ``heads`` each MTP
    pass's greedy tokens [B]. Returns the result's drafts section, uint32 words [RESULT_WORDS - DRAFTS_AT]: per draft
    group its anchor and the anchor's token of each pass. Planned drafting (planned_drafts(layers)): the upload holds
    descriptions, and a drafting group's anchor is its last slot."""
    b = len(mtp) // MTP_ROWS
    pages = PAGES[layout]
    out = [0] * (pages.result_words - pages.drafts_at)
    for g in range(layout.groups):
        if planned_drafts(layers, layout):
            anchor = (g + 1) * layout.group_slots - 1 if mtp[g * b + DESC_DRAFTING] != 0 else None
        else:
            anchor = next((s for s in range(SPEC_SLOTS) if mtp[2 * b + s] >= 0), None)
        at = g * pages.draft_words
        out[at] = NO_SLOT if anchor is None else anchor
        for k, head in enumerate(heads[: layout.max_drafts]):
            out[at + 1 + k] = NO_TOKEN if anchor is None else int(head[anchor]) & 0xFFFFFFFF
    return out


def stash_page(max_batch, positions=None, tokens=None):
    """The stash page's int32 values [max_batch]: ``positions`` / ``tokens`` per slot (None: an empty stash)."""
    assert max_batch >= STASH_WORDS
    page = [0] * max_batch
    page[STASH_POSITIONS : STASH_POSITIONS + SPEC_SLOTS] = list(positions) if positions else [-1] * SPEC_SLOTS
    page[STASH_TOKENS : STASH_TOKENS + SPEC_SLOTS] = list(tokens) if tokens else [0] * SPEC_SLOTS
    return page


def plan_step(mtp, stash, table_rows, layers=MAX_DRAFTS, layout=LAYOUT):
    """The plan step (mode 3): ``mtp`` the descriptions (pack_mtp_layers: draft group g's in row g, int32 values
    [MTP_ROWS * B]), ``stash`` the stash page [B], ``table_rows`` the row count T of the step's final norm buffer (the
    hidden table's first part; the stash rows follow it). plan_mtp_layers on device for each drafting group, in its
    slots: returns ({"pos": ring cache positions [B], "serial": SPEC_SLOTS lists [B] each keeping slot j's, "rope":
    RoPE indices [B], "hidden": hidden table rows [B], "tokens0": layer-0 tokens [B]}, the new stash page). Idle slots
    keep position -1, RoPE index 0, token 0 and gather their own stash row (T + j), so the stash copy rewrites it
    unchanged; an idle description (or a count out of range) leaves its group's stash as it was."""
    b = len(mtp) // MTP_ROWS
    assert b <= table_rows
    pos, rope, tokens0 = [-1] * b, [0] * b, [0] * b
    hidden = [table_rows + j for j in range(b)]
    new = list(stash)
    gs = layout.group_slots
    for g in range(layout.groups):
        desc = mtp[g * b : (g + 1) * b]
        m = desc[DESC_COUNT]
        if desc[DESC_DRAFTING] == 0 or not 1 <= m <= gs:
            continue
        a, lo = desc[DESC_A], g * gs

        def find(q, lo=lo):
            stashed = stash[STASH_POSITIONS + lo : STASH_POSITIONS + lo + gs]
            return next((lo + k for k in range(gs) if q >= 0 and stashed[k] == q), None)

        found = []
        while len(found) < max(0, layers - m) and find(a - 1 - len(found)) is not None:
            found.append(find(a - 1 - len(found)))
        r = len(found)
        first = lo + gs - (r + m)
        for j in range(lo, lo + gs):
            new[STASH_POSITIONS + j], new[STASH_TOKENS + j] = -1, 0
        for t in range(r + m):
            j, q = first + t, a - r + t
            if t < r:
                k = found[r - 1 - t]
                hidden[j], tokens0[j] = table_rows + k, stash[STASH_TOKENS + k]
            else:
                hidden[j], tokens0[j] = desc[DESC_SLOTS + t - r], desc[desc_tokens(gs) + t - r]
            pos[j], rope[j] = mtp_cache_position(q), max(q, 0)
            new[STASH_POSITIONS + j], new[STASH_TOKENS + j] = q, tokens0[j]
    serial = [[pos[j] if i == j else -1 for i in range(b)] for j in range(SPEC_SLOTS)]
    return {"pos": pos, "rope": rope, "hidden": hidden, "tokens0": tokens0, "serial": serial}, new


def tokens_step(tokens0, heads, layer, layout=LAYOUT):
    """The tokens step (mode 4) before pass ``layer``: its tokens [B], slot j of a draft group taking the layer-0
    token of the group's slot j + layer (past its last slot: the group anchor's greedy token of pass
    j + layer - GROUP_SLOTS, a draft); 0 past the slots."""
    b = len(tokens0)
    out = [0] * b
    gs = layout.group_slots
    for g in range(layout.groups):
        lo, anchor = g * gs, (g + 1) * gs - 1
        for j in range(gs):
            x = j + layer
            out[lo + j] = tokens0[lo + x] if x < gs else heads[x - gs][anchor]
    return out


# ---------------------------------------------------------------------- DFlash (tt/dflash.py)
DFLASH_TILE = 32  # TtDFlash's block tile: the rows of its token, block, context and ring-row sections


@dataclass(frozen=True)
class DFlashLayout:
    """TtDFlash's packed upload (uint32 words, ``words`` of them) for ``groups`` draft groups, group g's block in rows
    g * block .. g * block + block - 1 and its context ring in ring rows g * ring .. g * ring + ring - 1: the tokens
    [TILE] at TOKENS (word g * block group g's anchor token, word g * block + 1 its anchor slot + 1, which the drafts
    step reads, 0 for an idle group; embedded, only the anchor rows kept), the queries' RoPE rows
    [groups * block * heads] at QUERY (group g's query r * heads + h: its block row r), the block rows' [TILE] at
    BLOCK (a padding row past the groups': the last group's last row), the context rows' [TILE] at CONTEXT and their
    ring rows [TILE] at RING_ROWS (``no_write``: none). A RoPE row past ``last``, the tables' last row, reads it."""

    ring: int
    block: int
    heads: int
    last: int
    groups: int = 1

    QUERY = DFLASH_TILE

    @property
    def BLOCK(self):
        return self.QUERY + self.groups * self.block * self.heads

    @property
    def CONTEXT(self):
        return self.BLOCK + DFLASH_TILE

    @property
    def RING_ROWS(self):
        return self.CONTEXT + DFLASH_TILE

    @property
    def words(self):
        return self.RING_ROWS + DFLASH_TILE

    @property
    def no_write(self):
        return self.groups * self.ring


def dflash_inputs(layout, drafts=()):
    """The packed upload of a DFlash draft program (uint32 words): ``drafts`` [(group, context, anchor, start,
    anchor_slot)], per drafting group ``context`` [(tap row, position)] the committed rows to write into its ring,
    ``anchor`` the last committed token, at position ``start``, from slot ``anchor_slot`` of the pass (None: its drafts
    are not reported). Groups not listed are idle (start 0)."""
    t, last, n = DFLASH_TILE, layout.last, layout.block
    starts, tokens = [0] * layout.groups, [0] * t
    ctx, rows = [0] * t, [layout.no_write] * t
    for g, context, anchor, start, anchor_slot in drafts:
        assert 0 <= g < layout.groups
        starts[g] = int(start)
        tokens[g * n], tokens[g * n + 1] = int(anchor), 0 if anchor_slot is None else int(anchor_slot) + 1
        for row, pos in context:
            ctx[row], rows[row] = min(int(pos), last), g * layout.ring + int(pos) % layout.ring
    block = []
    for j in range(t):
        g = min(j // n, layout.groups - 1)
        block.append(min(starts[g] + min(j - g * n, n - 1), last))
    query = [block[g * n + r] for g in range(layout.groups) for r in range(n) for _ in range(layout.heads)]
    return [w & 0xFFFFFFFF for w in tokens + query + block + ctx + rows]


def dflash_chain_step(verify, accept, argmax, layout):
    """The chain step with DFlash drafting (mode 0, ``drafter`` 1): as chain_step, but the upload it writes is the
    DFlash draft's (dflash_inputs) of each group's commit: the committed columns' slots as context rows at
    P[slots[c]], the last committed token at the bonus as the anchor, from slot slots[m - 1]; idle groups otherwise."""
    header, pos, chain = _header(verify, accept, argmax, LAYOUT)
    drafts = []
    for g, m, slots, tokens, bonus in _commits(header, pos, chain, accept, LAYOUT):
        context = [(s, pos[s]) for s in slots if 0 <= s < DFLASH_TILE and pos[s] >= 0]
        drafts.append((g, context, tokens[m - 1], bonus, slots[m - 1]))
        _status(header, g, slots[m - 1], bonus)
    return dflash_inputs(layout, drafts), header


def dflash_drafts_step(packed, head, layout):
    """The drafts step with DFlash drafting (mode 1, ``drafter`` 1): ``packed`` the draft's upload (dflash_inputs),
    ``head`` its GreedyHead's tokens [B] (group g's draft k at row g * MAX_DRAFTS + k). Returns the result's drafts
    section, as drafts_step."""
    out = [0] * (RESULT_WORDS - DRAFTS_AT)
    for g in range(DRAFT_GROUPS):
        flag = int(packed[g * layout.block + 1])
        anchor = flag - 1 if flag != 0 else None
        at = g * DRAFT_WORDS
        out[at] = NO_SLOT if anchor is None else anchor
        for k in range(MAX_DRAFTS):
            out[at + 1 + k] = NO_TOKEN if anchor is None else int(head[g * MAX_DRAFTS + k]) & 0xFFFFFFFF
    return out


# ---------------------------------------------------------------------- the host side
@dataclass
class GroupChain:
    """One draft group of a result page: the chain's ``drafting`` / ``anchor`` / ``bonus``, and the drafts step's
    ``draft_anchor`` and ``drafts`` [MAX_DRAFTS] (None when it found no anchor)."""

    drafting: bool
    anchor: int
    bonus: int
    draft_anchor: int
    drafts: list


@dataclass
class ChainResult:
    """One verify pass's result page: ``accept`` per group (count, committed tokens, truncated) (unpack_out),
    ``argmax`` the greedy tokens of slots 0 .. SPEC_SLOTS - 1 and ``positions`` their positions (-1: idle), and per
    draft group its GroupChain (``groups``; the attributes below read group 0's)."""

    accept: list
    argmax: list
    positions: list
    groups: list

    @property
    def drafting(self):
        return self.groups[0].drafting

    @property
    def anchor(self):
        return self.groups[0].anchor

    @property
    def bonus(self):
        return self.groups[0].bonus

    @property
    def draft_anchor(self):
        return self.groups[0].draft_anchor

    @property
    def drafts(self):
        return self.groups[0].drafts


def unpack_result(words, layout=LAYOUT):
    """The ChainResult of a result page written in ``layout``'s draft groups."""
    pages = PAGES[layout]
    w = [int(x) & 0xFFFFFFFF for x in words]
    assert len(w) >= pages.result_words

    def slot(x):
        return None if x == NO_SLOT else x

    groups = []
    for g in range(layout.groups):
        status, drafts = STATUS_AT + g * STATUS_WORDS, pages.drafts_at + g * pages.draft_words
        groups.append(
            GroupChain(
                drafting=bool(w[status]),
                anchor=slot(w[status + 1]),
                bonus=w[status + 2],
                draft_anchor=slot(w[drafts]),
                drafts=w[drafts + 1 : drafts + 1 + layout.max_drafts],
            )
        )
    return ChainResult(
        accept=unpack_out(w[ACCEPT_AT : ACCEPT_AT + ACCEPT_WORDS]),
        argmax=w[ARGMAX_AT : ARGMAX_AT + SPEC_SLOTS],
        positions=[p - (1 << 32) if p >= 1 << 31 else p for p in w[POSITIONS_AT : POSITIONS_AT + SPEC_SLOTS]],
        groups=groups,
    )


@dataclass
class ChainedDraft:
    """The draft a verify chained for its source row: from the ``tokens`` the device committed at ``positions``."""

    row: int
    tokens: list
    positions: list
    drafts: list

    def matches(self, committed_tokens, committed_positions, count):
        """The plugin committed what the device drafted from (its row of propose_draft_tokens' arguments)."""
        m = int(count)
        return (
            m == len(self.tokens)
            and [int(t) for t in committed_tokens[:m]] == self.tokens
            and [int(p) for p in committed_positions[:m]] == self.positions
        )


def chained_draft(result, source, positions=None, layers=1, layout=LAYOUT):
    """The ChainedDraft of a source pass's ChainResult (its draft group's), or None when the chain declined or its
    drafts are not the anchor's. ``positions``: the pass's slot positions (VerifyPass.positions; None: the page's, as
    the verify ran them, which a relayed pass needs). Planned drafting (planned_drafts(layers)): the anchor is the
    group's last slot."""
    positions = result.positions if positions is None else positions
    chain = result.groups[source.group]
    if not chain.drafting or chain.draft_anchor != chain.anchor:
        return None
    count, tokens, _ = result.accept[source.group]
    last = (source.group + 1) * layout.group_slots - 1
    anchor = (
        last
        if planned_drafts(layers, layout)
        else (source.slots[count - 1] if 1 <= count <= len(source.slots) else None)
    )
    if not 1 <= count <= len(source.slots) or anchor != chain.anchor:
        raise RuntimeError(f"row {source.row}: the chain drafted at slot {chain.anchor} after {count} tokens")
    committed = [int(positions[s]) + 1 for s in source.slots[:count]]
    if committed[-1] != chain.bonus:
        raise RuntimeError(f"row {source.row}: the chain's bonus {chain.bonus} != {committed[-1]}")
    return ChainedDraft(row=source.row, tokens=list(tokens), positions=committed, drafts=list(chain.drafts))
