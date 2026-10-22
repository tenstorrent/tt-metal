# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MTP draft chained behind a verify on device (tt/spec_chain.py, tt/kernels/spec_chain.cpp): the host layouts and
an exact integer mirror of the two kernel steps (no ttnn), shared by the vLLM adapter and the host tests.

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

* ``plan_step`` (mode 2): plan_mtp_layers on device, from the description and the stash the last draft left (the
  ``stash`` page: each slot's position and layer-0 token; the slots' target hidden rows in a buffer beside it). It
  writes the passes' inputs: per slot the ring cache position, the SPEC_SLOTS serial positions, the RoPE index, the row
  of the hidden table [the step's final norm rows, the stash rows] to gather, and the layer-0 token; and the new stash.
  The anchor is the last slot.
* ``tokens_step`` (mode 3, before passes 1 and 2): layer i's tokens, the layer-0 tokens shifted by i, the drafts so far
  (the anchor's greedy token of each pass before it) appended (spec_decode.mtp_layer_tokens).

The host then reads the result page, one 320-byte read per pass: what the verify answers (the accept page, or the
argmax ids) and the drafts the next propose returns. Every other lane's verify passes run the accept trace too, with
an idle chain row (no draft): their result page carries the accept page and the argmax ids.

The chain row only holds what the host knows before the verify: whether a draft chains behind the pass, the source's
columns and slots, and the position limit. The draft is declined on device when the bonus (the position after the
last committed token) is at or past the limit, max_seq_len - K: the next verify writes positions bonus .. bonus + K,
and the draft's RoPE indices must stay inside the device tables (they end at max_seq_len).

Layouts:

- chain row (row CHAIN_ROW of the verify upload, int32, B wide): source (1: a draft chains behind this pass), the
  source's columns n + 1, limit, then the source's slots [SPEC_SLOTS] (-1 past its columns), zeros. The source's
  request is the pass's only one: group 0 of the accept page.
- result page (uint32 [RESULT_WORDS], sections on 64-byte boundaries, each written by one step):
  [ACCEPT_AT, +GROUPS * OUT_WORDS) the accept page (unpack_out); [ARGMAX_AT, +SPEC_SLOTS) the verify's argmax ids of
  slots 0 .. SPEC_SLOTS - 1; [STATUS_AT, +4) the chain: drafting (0 / 1), anchor slot (NO_SLOT: none), bonus position,
  unused; [DRAFTS_AT, +1 + MAX_DRAFTS) the drafts step: the anchor slot it found (the slot with a step2 position;
  NO_SLOT: an idle draft) and that slot's greedy token of each MTP pass. With every layer the chain's anchor is the
  draft's (the last slot), and the drafts step finds it from the description.
- stash page (int32, B wide; every layer): per slot of the last draft's pass its true position (-1 none)
  [STASH_POSITIONS, +SPEC_SLOTS), and its layer-0 token [STASH_TOKENS, +SPEC_SLOTS).
"""

from dataclasses import dataclass

from .sample_verify import GROUPS, NO_TOKEN, OUT_WORDS, unpack_out
from .spec_decode import MAX_DRAFTS, SPEC_SLOTS, mtp_cache_position
from .spec_inputs import (
    DESC_A,
    DESC_COUNT,
    DESC_DRAFTING,
    DESC_SLOTS,
    DESC_TOKENS,
    MTP_ROWS,
    VERIFY_ROWS,
)

CHAIN_ROW = 2  # the verify upload's row (reference.spec_inputs.pack_verify)
CHAIN_SOURCE, CHAIN_COLUMNS, CHAIN_LIMIT, CHAIN_SLOTS = 0, 1, 2, 3
CHAIN_WORDS = CHAIN_SLOTS + SPEC_SLOTS
NO_SLOT = 0xFFFFFFFF
ACCEPT_WORDS = GROUPS * OUT_WORDS
ACCEPT_AT, ARGMAX_AT, STATUS_AT, DRAFTS_AT, RESULT_WORDS = 0, 32, 48, 64, 80
HEADER_WORDS = DRAFTS_AT  # the chain step's sections; the drafts step writes the rest
assert ACCEPT_AT + ACCEPT_WORDS <= ARGMAX_AT and ARGMAX_AT + SPEC_SLOTS <= STATUS_AT and STATUS_AT + 4 <= DRAFTS_AT
assert DRAFTS_AT + 1 + MAX_DRAFTS <= RESULT_WORDS and all(w % 16 == 0 for w in (ARGMAX_AT, STATUS_AT, DRAFTS_AT))
assert CHAIN_ROW < VERIFY_ROWS
STASH_POSITIONS, STASH_TOKENS = 0, SPEC_SLOTS
STASH_WORDS = STASH_TOKENS + SPEC_SLOTS


def chain_row(max_batch, source=None, limit=0):
    """The verify upload's chain row, int32 values [max_batch]: ``source`` the pass's DraftSource when a draft chains
    behind it (None: an idle row), ``limit`` the first bonus position the draft is declined at."""
    assert max_batch >= CHAIN_WORDS
    row = [0] * max_batch
    if source is not None:
        assert 1 <= len(source.slots) <= SPEC_SLOTS
        row[CHAIN_SOURCE], row[CHAIN_COLUMNS], row[CHAIN_LIMIT] = 1, len(source.slots), int(limit)
        for c in range(SPEC_SLOTS):
            row[CHAIN_SLOTS + c] = source.slots[c] if c < len(source.slots) else -1
    return row


# ---------------------------------------------------------------------- the kernel (spec_chain.cpp)
def chain_step(verify, accept, argmax, layers=1):
    """The chain step (mode 0): ``verify`` the verify upload's int32 values [VERIFY_ROWS * B], ``accept`` the accept
    page's uint32 words [ACCEPT_WORDS], ``argmax`` the verify's greedy tokens [B]. Returns (the draft's packed upload,
    int32 values [MTP_ROWS * B]; the result header, uint32 words [HEADER_WORDS]). ``layers`` > 1: the upload is the
    commit's description (pack_mtp_layers) and the anchor the last slot."""
    b = len(verify) // VERIFY_ROWS
    pos = verify[b : 2 * b]
    chain = verify[CHAIN_ROW * b : (CHAIN_ROW + 1) * b]
    # pack_mtp's idle draft: tokens 0, cache positions -1, RoPE indices 0 (pack_mtp_layers': zeros).
    mtp = [0] * b + [-1 if layers == 1 else 0] * (3 * b) + [0] * ((MTP_ROWS - 4) * b)
    header = [0] * HEADER_WORDS
    header[ACCEPT_AT : ACCEPT_AT + ACCEPT_WORDS] = [int(w) & 0xFFFFFFFF for w in accept]
    header[ARGMAX_AT : ARGMAX_AT + SPEC_SLOTS] = [int(t) & 0xFFFFFFFF for t in argmax[:SPEC_SLOTS]]
    status = [0, NO_SLOT, 0, 0]
    m, columns, limit = int(accept[0]), chain[CHAIN_COLUMNS], chain[CHAIN_LIMIT]
    if chain[CHAIN_SOURCE] != 0 and 1 <= m <= columns <= SPEC_SLOTS:
        anchor = chain[CHAIN_SLOTS + m - 1]
        bonus = pos[anchor] + 1
        if bonus < limit and layers > 1:
            # pack_mtp_layers of the commit: committed position c at P[slots[c]] + 1, so A = P[slots[0]].
            mtp[DESC_DRAFTING], mtp[DESC_COUNT], mtp[DESC_A] = 1, m, pos[chain[CHAIN_SLOTS]]
            for c in range(SPEC_SLOTS):
                mtp[DESC_SLOTS + c] = chain[CHAIN_SLOTS + c] if c < m else -1
                mtp[DESC_TOKENS + c] = int(accept[1 + c]) if c < m else 0
            status = [1, SPEC_SLOTS - 1, bonus, 0]
        elif bonus < limit:
            # plan_mtp with committed position c at P[slots[c]] + 1: the fill writes column c's hidden at P[slots[c]].
            for c in range(m):
                s = chain[CHAIN_SLOTS + c]
                mtp[s] = int(accept[1 + c])
                mtp[b + s] = mtp_cache_position(pos[s])
                mtp[4 * b + s] = max(pos[s], 0)
            mtp[2 * b + anchor], mtp[3 * b + anchor] = mtp_cache_position(bonus), mtp_cache_position(bonus + 1)
            mtp[5 * b + anchor], mtp[6 * b + anchor] = bonus, bonus + 1
            status = [1, anchor, bonus, 0]
    header[STATUS_AT : STATUS_AT + 4] = status
    return mtp, header


def drafts_step(mtp, heads, layers=1):
    """The drafts step (mode 1): ``mtp`` the draft's packed upload [MTP_ROWS * B] (int32 values), ``heads`` each MTP
    pass's greedy tokens [B]. Returns the result's drafts section, uint32 words [RESULT_WORDS - DRAFTS_AT]. With
    ``layers`` > 1 the upload is a description, and a draft's anchor the last slot."""
    b = len(mtp) // MTP_ROWS
    if layers > 1:
        anchor = SPEC_SLOTS - 1 if mtp[DESC_DRAFTING] != 0 else None
    else:
        anchor = next((s for s in range(SPEC_SLOTS) if mtp[2 * b + s] >= 0), None)
    out = [0] * (RESULT_WORDS - DRAFTS_AT)
    out[0] = NO_SLOT if anchor is None else anchor
    for k, head in enumerate(heads[:MAX_DRAFTS]):
        out[1 + k] = NO_TOKEN if anchor is None else int(head[anchor]) & 0xFFFFFFFF
    return out


def stash_page(max_batch, positions=None, tokens=None):
    """The stash page's int32 values [max_batch]: ``positions`` / ``tokens`` per slot (None: an empty stash)."""
    assert max_batch >= STASH_WORDS
    page = [0] * max_batch
    page[STASH_POSITIONS : STASH_POSITIONS + SPEC_SLOTS] = list(positions) if positions else [-1] * SPEC_SLOTS
    page[STASH_TOKENS : STASH_TOKENS + SPEC_SLOTS] = list(tokens) if tokens else [0] * SPEC_SLOTS
    return page


def plan_step(mtp, stash, table_rows, layers=MAX_DRAFTS):
    """The plan step (mode 2): ``mtp`` the description (pack_mtp_layers, int32 values [MTP_ROWS * B]), ``stash`` the
    stash page [B], ``table_rows`` the row count T of the step's final norm buffer (the hidden table's first part; the
    stash rows follow it). plan_mtp_layers on device: returns ({"pos": ring cache positions [B], "serial": SPEC_SLOTS
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
        pos[j], rope[j] = mtp_cache_position(q), max(q, 0)
        new[STASH_POSITIONS + j], new[STASH_TOKENS + j] = q, tokens0[j]
    out["serial"] = [[pos[j] if i == j else -1 for i in range(b)] for j in range(SPEC_SLOTS)]
    return out, new


def tokens_step(tokens0, heads, layer):
    """The tokens step (mode 3) before pass ``layer``: its tokens [B], slot j taking the layer-0 token of slot j + layer
    (the last slot's beyond it: the anchor's greedy token of pass j + layer - SPEC_SLOTS, a draft); 0 past the slots."""
    b = len(tokens0)
    out = [0] * b
    for j in range(SPEC_SLOTS):
        x = j + layer
        out[j] = tokens0[x] if x < SPEC_SLOTS else heads[x - SPEC_SLOTS][SPEC_SLOTS - 1]
    return out


# ---------------------------------------------------------------------- the host side
@dataclass
class ChainResult:
    """One verify pass's result page: ``accept`` per group (count, committed tokens, truncated) (unpack_out),
    ``argmax`` the greedy tokens of slots 0 .. SPEC_SLOTS - 1, the chain's ``drafting`` / ``anchor`` / ``bonus``,
    and the drafts step's ``draft_anchor`` and ``drafts`` [MAX_DRAFTS] (None when it found no anchor)."""

    accept: list
    argmax: list
    drafting: bool
    anchor: int
    bonus: int
    draft_anchor: int
    drafts: list


def unpack_result(words):
    w = [int(x) & 0xFFFFFFFF for x in words]
    assert len(w) >= RESULT_WORDS

    def slot(x):
        return None if x == NO_SLOT else x

    return ChainResult(
        accept=unpack_out(w[ACCEPT_AT : ACCEPT_AT + ACCEPT_WORDS]),
        argmax=w[ARGMAX_AT : ARGMAX_AT + SPEC_SLOTS],
        drafting=bool(w[STATUS_AT]),
        anchor=slot(w[STATUS_AT + 1]),
        bonus=w[STATUS_AT + 2],
        draft_anchor=slot(w[DRAFTS_AT]),
        drafts=w[DRAFTS_AT + 1 : DRAFTS_AT + 1 + MAX_DRAFTS],
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


def chained_draft(result, source, positions, layers=1):
    """The ChainedDraft of a source pass's ChainResult, or None when the chain declined or its drafts are not the
    anchor's. ``positions``: the pass's slot positions (VerifyPass.positions). ``layers`` > 1: the anchor is the last
    slot."""
    if not result.drafting or result.draft_anchor != result.anchor:
        return None
    count, tokens, _ = result.accept[0]
    anchor = SPEC_SLOTS - 1 if layers > 1 else (source.slots[count - 1] if 1 <= count <= len(source.slots) else None)
    if not 1 <= count <= len(source.slots) or anchor != result.anchor:
        raise RuntimeError(f"row {source.row}: the chain drafted at slot {result.anchor} after {count} tokens")
    committed = [int(positions[s]) + 1 for s in source.slots[:count]]
    if committed[-1] != result.bonus:
        raise RuntimeError(f"row {source.row}: the chain's bonus {result.bonus} != {committed[-1]}")
    return ChainedDraft(row=source.row, tokens=list(tokens), positions=committed, drafts=list(result.drafts))
