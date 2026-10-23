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

A lane that ran two requests in one pass, each with at most one draft (spec_decode.py ``draft_sources``), drafts one
token for each with the pair draft trace (MTP layer 0's fill over both requests' committed rows, each in its own ring:
``pack_mtp_pair``), the chain row naming both sources (``pair``): the chain step writes that upload, the pair trace's
drafts step each source's anchor and draft, and a chained verify of the pair relays both requests from the page (each
source k from its group of the page the lane's last pass left, ``group``). Source k of a pass is the pass's request k
(group k of its accept page and of its control page): a pass with sources runs no other request.

The chain row only holds what the host knows before the verify: whether a draft chains behind the pass, the sources'
columns, slots and MTP rings, the position limit, and whether the sources' tokens and positions are relayed. A draft is
declined on device when the bonus (the position after the last committed token) is at or past the limit,
max_seq_len - K: the next verify writes positions bonus .. bonus + K, and the draft's RoPE indices must stay inside
the device tables (they end at max_seq_len).

Layouts:

- chain row (row CHAIN_ROW of the verify upload, int32, B wide): source 0's words: source (1: a draft chains behind
  this pass), its columns n + 1, limit (every source's), then its slots [SPEC_SLOTS] (-1 past its columns), relay (1:
  the relay step writes the source's tokens and positions); each verify slot's request slot + 1 [SPEC_SLOTS] (the MTP
  history's, reference/mtp_backfill.py; 0: none); pair (1: the pair draft chains behind the pass); source 0's MTP ring
  and its group in the page it relays from. Source k's words sit CHAIN_NEXT * k further (source 1: source, columns,
  slots, relay, ring, group; its other words unused), then zeros.
- result page (uint32 [RESULT_WORDS], sections on 64-byte boundaries, each written by one step):
  [ACCEPT_AT, +GROUPS * OUT_WORDS) the accept page (unpack_out); [ARGMAX_AT, +SPEC_SLOTS) the verify's argmax ids of
  slots 0 .. SPEC_SLOTS - 1 and [POSITIONS_AT, +SPEC_SLOTS) their positions, as the verify ran them (a relayed pass's
  are the device's); [STATUS_AT + STATUS_WORDS * k, +STATUS_WORDS) the chain's source k: drafting (0 / 1), anchor slot
  (NO_SLOT: none), bonus position, pair (1: its drafts are the pair draft's, PAIR_DRAFTS of them; 0: MAX_DRAFTS);
  [DRAFTS_AT + DRAFT_WORDS * k, +DRAFT_WORDS) the drafts step's source k: the anchor slot it found (one layer: the slot
  with a step2 position; the pair draft: the upload's anchor word k; NO_SLOT: an idle draft) and that slot's greedy
  token of each MTP pass (NO_TOKEN past the pair draft's one). With every layer the chain's anchor is the draft's (the
  last slot), and the drafts step finds it from the description.
- stash page (int32, B wide; every layer): per slot of the last draft's pass its true position (-1 none)
  [STASH_POSITIONS, +SPEC_SLOTS), and its layer-0 token [STASH_TOKENS, +SPEC_SLOTS).

The relay step reads the result page of the lane's last pass, which must be the pass whose draft chained (the
adapter checks that on the host); it relays a source only from a page section whose chain drafted (drafting 1, the
drafts step's anchor the chain's) and verifies no more drafts than that section holds. Otherwise that source's slots
go idle (position -1: no K / V written) and it drafts nothing: the host finds position -1 in the positions section and
raises. The page reads the same with one MTP layer or every layer
(there the chain's and the drafts step's anchor are both the last slot, and the bonus is still the position after
the last committed token), so the relay does not depend on the drafter.
"""

from dataclasses import dataclass

from .sample_verify import GROUP_WORDS, GROUPS, NO_TOKEN, OUT_WORDS, unpack_out
from .spec_decode import MAX_DRAFTS, PAIR_DRAFTS, PAIR_SOURCES, SPEC_SLOTS, DraftSource, mtp_cache_position
from .spec_inputs import (
    DESC_A,
    DESC_COUNT,
    DESC_DRAFTING,
    DESC_RING,
    DESC_SLOTS,
    DESC_TOKENS,
    MTP_ROWS,
    PAIR_ANCHORS,
    VERIFY_ROWS,
)

CHAIN_ROW = 2  # the verify upload's row (reference.spec_inputs.pack_verify)
CHAIN_SOURCE, CHAIN_COLUMNS, CHAIN_LIMIT, CHAIN_SLOTS = 0, 1, 2, 3
CHAIN_RELAY = CHAIN_SLOTS + SPEC_SLOTS
CHAIN_WORDS = CHAIN_RELAY + 1
# The history's words: each verify slot's request slot + 1 (0: none; reference/mtp_backfill.py).
CHAIN_HSLOTS = CHAIN_WORDS
CHAIN_PAIR = CHAIN_HSLOTS + SPEC_SLOTS  # 1: the pair draft chains behind the pass
CHAIN_RING = CHAIN_PAIR + 1  # source 0's MTP ring
CHAIN_GROUP = CHAIN_RING + 1  # source 0's group in the page it relays from
# Source k's words: CHAIN_NEXT * k + (CHAIN_SOURCE, CHAIN_COLUMNS, CHAIN_SLOTS .., CHAIN_RELAY, CHAIN_RING, CHAIN_GROUP).
CHAIN_NEXT = CHAIN_GROUP + 1
CHAIN_ROW_WORDS = (PAIR_SOURCES - 1) * CHAIN_NEXT + CHAIN_GROUP + 1
NO_SLOT = 0xFFFFFFFF
ACCEPT_WORDS = GROUPS * OUT_WORDS
ACCEPT_AT, ARGMAX_AT, POSITIONS_AT, STATUS_AT, DRAFTS_AT, RESULT_WORDS = 0, 32, 40, 48, 64, 80
STATUS_WORDS = 4  # per source: drafting, anchor, bonus, pair
DRAFT_WORDS = 1 + MAX_DRAFTS  # per source: the anchor the drafts step found, the drafts
HEADER_WORDS = DRAFTS_AT  # the chain step's sections; the drafts step writes the rest
CTRL_DRAFTS = 4  # a control page group's first draft word (reference.sample_verify.group_words)
assert ACCEPT_AT + ACCEPT_WORDS <= ARGMAX_AT and ARGMAX_AT + SPEC_SLOTS <= POSITIONS_AT
assert POSITIONS_AT + SPEC_SLOTS <= STATUS_AT and STATUS_AT + PAIR_SOURCES * STATUS_WORDS <= DRAFTS_AT
assert CTRL_DRAFTS + MAX_DRAFTS <= GROUP_WORDS and PAIR_SOURCES <= GROUPS
assert DRAFTS_AT + PAIR_SOURCES * DRAFT_WORDS <= RESULT_WORDS
assert all(w % 16 == 0 for w in (ARGMAX_AT, STATUS_AT, DRAFTS_AT))
assert CHAIN_ROW < VERIFY_ROWS and CHAIN_GROUP < CHAIN_NEXT
STASH_POSITIONS, STASH_TOKENS = 0, SPEC_SLOTS
STASH_WORDS = STASH_TOKENS + SPEC_SLOTS


def _per_source(value, n):
    """A per-source chain row argument: one value for every source, or one per source."""
    if isinstance(value, (list, tuple)):
        assert len(value) == n, (value, n)
        return list(value)
    return [value] * n


def chain_row(max_batch, source=None, limit=0, relay=False, hslots=None, ring=0, group=None, pair=False):
    """The verify upload's chain row, int32 values [max_batch]: ``source`` the pass's DraftSource when a draft chains
    behind it, or its sources (two with ``pair``: the pair draft; None or none: an idle row), ``limit`` the first
    bonus position a draft is declined at, ``relay`` the relay step writes a source's tokens and positions from the
    lane's last result page (a chained verify), ``ring`` its MTP ring, ``group`` the page group it relays from
    (default its index), each one value or one per source; ``hslots`` each verify slot's request slot in the lane
    (None: no history write) for the history (reference/mtp_backfill.py)."""
    assert max_batch >= CHAIN_ROW_WORDS
    sources = [] if source is None else [source] if isinstance(source, DraftSource) else list(source)
    n = len(sources)
    assert n <= (PAIR_SOURCES if pair else 1), f"{n} sources chained behind one pass (pair {pair})"
    relays, rings = _per_source(relay, n), _per_source(ring, n)
    groups = list(range(n)) if group is None else _per_source(group, n)
    assert sources or not any(relays), "the relay writes a source's columns"
    row = [0] * max_batch
    for j, h in enumerate(hslots or []):
        row[CHAIN_HSLOTS + j] = 0 if h is None else int(h) + 1
    if sources:
        row[CHAIN_LIMIT], row[CHAIN_PAIR] = int(limit), int(bool(pair))
    for k, src in enumerate(sources):
        f = k * CHAIN_NEXT
        assert 1 <= len(src.slots) <= SPEC_SLOTS
        row[f + CHAIN_SOURCE], row[f + CHAIN_COLUMNS] = 1, len(src.slots)
        for c in range(SPEC_SLOTS):
            row[f + CHAIN_SLOTS + c] = src.slots[c] if c < len(src.slots) else -1
        row[f + CHAIN_RELAY] = int(bool(relays[k]))
        row[f + CHAIN_RING], row[f + CHAIN_GROUP] = int(rings[k]), int(groups[k])
    return row


# ---------------------------------------------------------------------- the kernel (spec_chain.cpp)
def chain_step(verify, accept, argmax, layers=1):
    """The chain step (mode 0): ``verify`` the verify upload's int32 values [VERIFY_ROWS * B], ``accept`` the accept
    page's uint32 words [ACCEPT_WORDS], ``argmax`` the verify's greedy tokens [B]. Returns (the draft's packed upload,
    int32 values [MTP_ROWS * B]; the result header, uint32 words [HEADER_WORDS]). The chain row's pair word: the pair
    draft's upload (pack_mtp_pair) for each source k from group k of the accept page; else ``layers`` > 1: the
    commit's description (pack_mtp_layers) and the anchor the last slot; else pack_mtp's."""
    b = len(verify) // VERIFY_ROWS
    pos = verify[b : 2 * b]
    chain = verify[CHAIN_ROW * b : (CHAIN_ROW + 1) * b]
    pair = chain[CHAIN_PAIR] != 0
    # pack_mtp's idle draft: tokens 0, cache positions -1, RoPE indices 0 (pack_mtp_layers': zeros; the pair's:
    # pack_mtp's, which idles its anchor words too).
    idle = -1 if layers == 1 or pair else 0
    mtp = [0] * b + [idle] * (3 * b) + [0] * ((MTP_ROWS - 4) * b)
    header = [0] * HEADER_WORDS
    header[ACCEPT_AT : ACCEPT_AT + ACCEPT_WORDS] = [int(w) & 0xFFFFFFFF for w in accept]
    header[ARGMAX_AT : ARGMAX_AT + SPEC_SLOTS] = [int(t) & 0xFFFFFFFF for t in argmax[:SPEC_SLOTS]]
    header[POSITIONS_AT : POSITIONS_AT + SPEC_SLOTS] = [int(p) & 0xFFFFFFFF for p in pos[:SPEC_SLOTS]]
    for k in range(PAIR_SOURCES):
        header[STATUS_AT + k * STATUS_WORDS : STATUS_AT + (k + 1) * STATUS_WORDS] = [0, NO_SLOT, 0, 0]
    limit = chain[CHAIN_LIMIT]
    for k in range(PAIR_SOURCES if pair else 1):
        f = k * CHAIN_NEXT
        group = accept[k * OUT_WORDS : (k + 1) * OUT_WORDS]
        m, columns, ring = int(group[0]), chain[f + CHAIN_COLUMNS], chain[f + CHAIN_RING]
        if chain[f + CHAIN_SOURCE] == 0 or not 1 <= m <= columns <= SPEC_SLOTS:
            continue
        anchor = chain[f + CHAIN_SLOTS + m - 1]
        bonus = pos[anchor] + 1
        if bonus >= limit:
            continue
        if pair:
            # pack_mtp_pair: the source's committed slots fill its ring; its anchor in word k of the anchors row.
            for c in range(m):
                s = chain[f + CHAIN_SLOTS + c]
                mtp[s], mtp[b + s], mtp[4 * b + s] = int(group[1 + c]), mtp_cache_position(pos[s], ring), max(pos[s], 0)
            mtp[PAIR_ANCHORS * b + k] = anchor
            status = [1, anchor, bonus, 1]
        elif layers > 1:
            # pack_mtp_layers of the commit: committed position c at P[slots[c]] + 1, so A = P[slots[0]].
            mtp[DESC_DRAFTING], mtp[DESC_COUNT], mtp[DESC_A] = 1, m, pos[chain[CHAIN_SLOTS]]
            for c in range(SPEC_SLOTS):
                mtp[DESC_SLOTS + c] = chain[CHAIN_SLOTS + c] if c < m else -1
                mtp[DESC_TOKENS + c] = int(group[1 + c]) if c < m else 0
            mtp[DESC_RING] = ring
            status = [1, SPEC_SLOTS - 1, bonus, 0]
        else:
            # plan_mtp with committed position c at P[slots[c]] + 1: the fill writes column c's hidden at P[slots[c]].
            for c in range(m):
                s = chain[CHAIN_SLOTS + c]
                mtp[s] = int(group[1 + c])
                mtp[b + s] = mtp_cache_position(pos[s], ring)
                mtp[4 * b + s] = max(pos[s], 0)
            mtp[2 * b + anchor] = mtp_cache_position(bonus, ring)
            mtp[3 * b + anchor] = mtp_cache_position(bonus + 1, ring)
            mtp[5 * b + anchor], mtp[6 * b + anchor] = bonus, bonus + 1
            status = [1, anchor, bonus, 0]
        header[STATUS_AT + k * STATUS_WORDS : STATUS_AT + (k + 1) * STATUS_WORDS] = status
    return mtp, header


def relay_step(verify, result, ctrl):
    """The relay step (mode 2, first in the verify trace): ``verify`` the verify upload's int32 values
    [VERIFY_ROWS * B] as the host copied it, ``result`` the lane's last result page (uint32 words [RESULT_WORDS]),
    ``ctrl`` the accept step's control page as the host staged it (uint32 words [GROUPS * GROUP_WORDS]). Returns the
    upload and the control page the verify and accept steps then read. Each source k whose relay word is set reads its
    page group g (its group word): its columns c = 0 .. n take, at slot slots[c], g's last committed token (c = 0) or
    g's draft c at position g's bonus + c, and control group k's drafts are those; a page section whose chain did not
    draft, or that holds fewer drafts than the source verifies, idles them instead (position -1, token 0, draft 0) and
    clears the source. Without a relay word both are unchanged."""
    b = len(verify) // VERIFY_ROWS
    verify, ctrl = list(verify), [int(w) & 0xFFFFFFFF for w in ctrl]
    chain = CHAIN_ROW * b
    page = [int(w) & 0xFFFFFFFF for w in result]
    for k in range(PAIR_SOURCES):
        f = chain + k * CHAIN_NEXT
        if verify[f + CHAIN_RELAY] == 0:
            continue
        columns, g = verify[f + CHAIN_COLUMNS], verify[f + CHAIN_GROUP]
        ok = 0 <= g < PAIR_SOURCES
        st, dr = STATUS_AT + g * STATUS_WORDS, DRAFTS_AT + g * DRAFT_WORDS
        m = page[ACCEPT_AT + g * OUT_WORDS] if ok else 0
        drafted = (PAIR_DRAFTS if page[st + 3] else MAX_DRAFTS) if ok else 0
        ok = ok and page[st] == 1 and page[dr] == page[st + 1]
        ok = ok and 1 <= columns <= 1 + drafted and 1 <= m <= 1 + MAX_DRAFTS
        bonus = page[st + 2] if ok else 0
        for c in range(min(max(columns, 0), SPEC_SLOTS)):
            s = verify[f + CHAIN_SLOTS + c]
            if not 0 <= s < SPEC_SLOTS:
                continue
            token = (page[ACCEPT_AT + g * OUT_WORDS + m] if c == 0 else page[dr + c]) if ok else 0
            verify[s] = token - (1 << 32) if token >= 1 << 31 else token
            verify[b + s] = bonus + c if ok else -1
            if c > 0:
                ctrl[k * GROUP_WORDS + CTRL_DRAFTS + c - 1] = page[dr + c] if ok else 0
        if not ok:
            verify[f + CHAIN_SOURCE] = 0
    return verify, ctrl


def drafts_step(mtp, heads, layers=1, pair=False):
    """The drafts step (mode 1): ``mtp`` the draft's packed upload [MTP_ROWS * B] (int32 values), ``heads`` each MTP
    pass's greedy tokens [B] (the pair draft: one pass). Returns the result's drafts section, uint32 words
    [RESULT_WORDS - DRAFTS_AT]: per source its anchor and drafts (NO_SLOT, NO_TOKEN: none). ``pair``: the pair
    draft's, source k's anchor in the upload's anchors row; else source 0's, with ``layers`` > 1 from a description
    (the last slot), else the slot with a step2 position."""
    b = len(mtp) // MTP_ROWS
    out = [0] * (RESULT_WORDS - DRAFTS_AT)
    anchors = [None] * PAIR_SOURCES
    if pair:
        for k in range(PAIR_SOURCES):
            a = mtp[PAIR_ANCHORS * b + k]
            anchors[k] = a if 0 <= a < SPEC_SLOTS else None
    elif layers > 1:
        anchors[0] = SPEC_SLOTS - 1 if mtp[DESC_DRAFTING] != 0 else None
    else:
        anchors[0] = next((s for s in range(SPEC_SLOTS) if mtp[2 * b + s] >= 0), None)
    for k, anchor in enumerate(anchors):
        out[k * DRAFT_WORDS] = NO_SLOT if anchor is None else anchor
        for d in range(MAX_DRAFTS):
            drafted = anchor is not None and d < (PAIR_DRAFTS if pair else len(heads[:MAX_DRAFTS]))
            out[k * DRAFT_WORDS + 1 + d] = int(heads[d][anchor]) & 0xFFFFFFFF if drafted else NO_TOKEN
    return out


def stash_page(max_batch, positions=None, tokens=None):
    """The stash page's int32 values [max_batch]: ``positions`` / ``tokens`` per slot (None: an empty stash)."""
    assert max_batch >= STASH_WORDS
    page = [0] * max_batch
    page[STASH_POSITIONS : STASH_POSITIONS + SPEC_SLOTS] = list(positions) if positions else [-1] * SPEC_SLOTS
    page[STASH_TOKENS : STASH_TOKENS + SPEC_SLOTS] = list(tokens) if tokens else [0] * SPEC_SLOTS
    return page


def plan_step(mtp, stash, table_rows, layers=MAX_DRAFTS):
    """The plan step (mode 3): ``mtp`` the description (pack_mtp_layers, int32 values [MTP_ROWS * B]), ``stash`` the
    stash page [B], ``table_rows`` the row count T of the step's final norm buffer (the hidden table's first part; the
    stash rows follow it). plan_mtp_layers on device: returns ({"pos": cache positions in the description's ring [B],
    "serial": SPEC_SLOTS
    lists [B] each keeping slot j's, "rope": RoPE indices [B], "hidden": hidden table rows [B], "tokens0": layer-0
    tokens [B]}, the new stash page). Idle slots keep position -1, RoPE index 0, token 0 and gather their own stash
    row (T + j), so the stash copy rewrites it unchanged; an idle description (or a count out of range) leaves the
    stash page as it was."""
    b = len(mtp) // MTP_ROWS
    assert b <= table_rows
    pos, rope, tokens0 = [-1] * b, [0] * b, [0] * b
    hidden = [table_rows + j for j in range(b)]
    out = {"pos": pos, "rope": rope, "hidden": hidden, "tokens0": tokens0}
    m = mtp[DESC_COUNT]
    if mtp[DESC_DRAFTING] == 0 or not 1 <= m <= SPEC_SLOTS:
        out["serial"] = [[-1] * b for _ in range(SPEC_SLOTS)]
        return out, list(stash)
    a = mtp[DESC_A]
    stashed = stash[STASH_POSITIONS : STASH_POSITIONS + SPEC_SLOTS]

    def find(q):
        return next((k for k in range(SPEC_SLOTS) if q >= 0 and stashed[k] == q), None)

    found = []
    while len(found) < max(0, layers - m) and find(a - 1 - len(found)) is not None:
        found.append(find(a - 1 - len(found)))
    r = len(found)
    first = SPEC_SLOTS - (r + m)
    new = list(stash)
    for j in range(SPEC_SLOTS):
        new[STASH_POSITIONS + j], new[STASH_TOKENS + j] = -1, 0
    for t in range(r + m):
        j, q = first + t, a - r + t
        if t < r:
            k = found[r - 1 - t]
            hidden[j], tokens0[j] = table_rows + k, stash[STASH_TOKENS + k]
        else:
            hidden[j], tokens0[j] = mtp[DESC_SLOTS + t - r], mtp[DESC_TOKENS + t - r]
        pos[j], rope[j] = mtp_cache_position(q, mtp[DESC_RING]), max(q, 0)
        new[STASH_POSITIONS + j], new[STASH_TOKENS + j] = q, tokens0[j]
    out["serial"] = [[pos[j] if i == j else -1 for i in range(b)] for j in range(SPEC_SLOTS)]
    return out, new


def tokens_step(tokens0, heads, layer):
    """The tokens step (mode 4) before pass ``layer``: its tokens [B], slot j taking the layer-0 token of slot j + layer
    (the last slot's beyond it: the anchor's greedy token of pass j + layer - SPEC_SLOTS, a draft); 0 past the slots."""
    b = len(tokens0)
    out = [0] * b
    for j in range(SPEC_SLOTS):
        x = j + layer
        out[j] = tokens0[x] if x < SPEC_SLOTS else heads[x - SPEC_SLOTS][SPEC_SLOTS - 1]
    return out


# ---------------------------------------------------------------------- the host side
@dataclass
class ChainSource:
    """A result page's source section: the chain's ``drafting`` / ``anchor`` / ``bonus`` / ``pair`` (the pair draft's
    one draft) and the drafts step's ``draft_anchor`` (None: none) and ``drafts`` [MAX_DRAFTS] (NO_TOKEN past the
    ones drafted)."""

    drafting: bool
    anchor: int
    bonus: int
    pair: bool
    draft_anchor: int
    drafts: list


@dataclass
class ChainResult:
    """One verify pass's result page: ``accept`` per group (count, committed tokens, truncated) (unpack_out),
    ``argmax`` the greedy tokens of slots 0 .. SPEC_SLOTS - 1 and ``positions`` their positions (-1: idle), and per
    source k its ChainSource ``sources[k]``; ``drafting`` .. ``drafts`` are source 0's."""

    accept: list
    argmax: list
    positions: list
    drafting: bool
    anchor: int
    bonus: int
    draft_anchor: int
    drafts: list
    sources: list


def unpack_result(words):
    w = [int(x) & 0xFFFFFFFF for x in words]
    assert len(w) >= RESULT_WORDS

    def slot(x):
        return None if x == NO_SLOT else x

    sources = []
    for k in range(PAIR_SOURCES):
        st, dr = STATUS_AT + k * STATUS_WORDS, DRAFTS_AT + k * DRAFT_WORDS
        sources.append(
            ChainSource(
                drafting=bool(w[st]),
                anchor=slot(w[st + 1]),
                bonus=w[st + 2],
                pair=bool(w[st + 3]),
                draft_anchor=slot(w[dr]),
                drafts=w[dr + 1 : dr + DRAFT_WORDS],
            )
        )
    first = sources[0]
    return ChainResult(
        accept=unpack_out(w[ACCEPT_AT : ACCEPT_AT + ACCEPT_WORDS]),
        argmax=w[ARGMAX_AT : ARGMAX_AT + SPEC_SLOTS],
        positions=[p - (1 << 32) if p >= 1 << 31 else p for p in w[POSITIONS_AT : POSITIONS_AT + SPEC_SLOTS]],
        drafting=first.drafting,
        anchor=first.anchor,
        bonus=first.bonus,
        draft_anchor=first.draft_anchor,
        drafts=first.drafts,
        sources=sources,
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


def chained_draft(result, source, positions=None, layers=1, group=0):
    """The ChainedDraft of a source pass's ChainResult for its source ``group`` (its index among the pass's sources,
    its group in the accept page), or None when the chain declined or its drafts are not the anchor's. ``positions``:
    the pass's slot positions (VerifyPass.positions; None: the page's, as the verify ran them, which a relayed pass
    needs). ``layers`` > 1: a single source's anchor is the last slot; a pair source's is always its last committed
    slot, and it carries PAIR_DRAFTS drafts."""
    positions = result.positions if positions is None else positions
    st = result.sources[group]
    if not st.drafting or st.draft_anchor != st.anchor:
        return None
    count, tokens, _ = result.accept[group]
    last = source.slots[count - 1] if 1 <= count <= len(source.slots) else None
    anchor = SPEC_SLOTS - 1 if layers > 1 and not st.pair else last
    if last is None or anchor != st.anchor:
        raise RuntimeError(f"row {source.row}: the chain drafted at slot {st.anchor} after {count} tokens")
    committed = [int(positions[s]) + 1 for s in source.slots[:count]]
    if committed[-1] != st.bonus:
        raise RuntimeError(f"row {source.row}: the chain's bonus {st.bonus} != {committed[-1]}")
    drafts = list(st.drafts[: PAIR_DRAFTS if st.pair else MAX_DRAFTS])
    return ChainedDraft(row=source.row, tokens=list(tokens), positions=committed, drafts=drafts)
