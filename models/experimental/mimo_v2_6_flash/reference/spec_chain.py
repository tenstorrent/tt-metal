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
  NO_SLOT: an idle draft) and that slot's greedy token of each MTP pass.
"""

from dataclasses import dataclass

from .sample_verify import GROUPS, NO_TOKEN, OUT_WORDS, unpack_out
from .spec_decode import MAX_DRAFTS, SPEC_SLOTS, mtp_cache_position
from .spec_inputs import MTP_ROWS, VERIFY_ROWS

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
def chain_step(verify, accept, argmax):
    """The chain step (mode 0): ``verify`` the verify upload's int32 values [VERIFY_ROWS * B], ``accept`` the accept
    page's uint32 words [ACCEPT_WORDS], ``argmax`` the verify's greedy tokens [B]. Returns (the draft's packed upload,
    int32 values [MTP_ROWS * B]; the result header, uint32 words [HEADER_WORDS])."""
    b = len(verify) // VERIFY_ROWS
    pos = verify[b : 2 * b]
    chain = verify[CHAIN_ROW * b : (CHAIN_ROW + 1) * b]
    # pack_mtp's idle draft: tokens 0, cache positions -1, RoPE indices 0.
    mtp = [0] * b + [-1] * (3 * b) + [0] * ((MTP_ROWS - 4) * b)
    header = [0] * HEADER_WORDS
    header[ACCEPT_AT : ACCEPT_AT + ACCEPT_WORDS] = [int(w) & 0xFFFFFFFF for w in accept]
    header[ARGMAX_AT : ARGMAX_AT + SPEC_SLOTS] = [int(t) & 0xFFFFFFFF for t in argmax[:SPEC_SLOTS]]
    status = [0, NO_SLOT, 0, 0]
    m, columns, limit = int(accept[0]), chain[CHAIN_COLUMNS], chain[CHAIN_LIMIT]
    if chain[CHAIN_SOURCE] != 0 and 1 <= m <= columns <= SPEC_SLOTS:
        anchor = chain[CHAIN_SLOTS + m - 1]
        bonus = pos[anchor] + 1
        if bonus < limit:
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


def drafts_step(mtp, heads):
    """The drafts step (mode 1): ``mtp`` the draft's packed upload [MTP_ROWS * B] (int32 values), ``heads`` each MTP
    pass's greedy tokens [B]. Returns the result's drafts section, uint32 words [RESULT_WORDS - DRAFTS_AT]."""
    b = len(mtp) // MTP_ROWS
    anchor = next((s for s in range(SPEC_SLOTS) if mtp[2 * b + s] >= 0), None)
    out = [0] * (RESULT_WORDS - DRAFTS_AT)
    out[0] = NO_SLOT if anchor is None else anchor
    for k, head in enumerate(heads[:MAX_DRAFTS]):
        out[1 + k] = NO_TOKEN if anchor is None else int(head[anchor]) & 0xFFFFFFFF
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


def chained_draft(result, source, positions):
    """The ChainedDraft of a source pass's ChainResult, or None when the chain declined or its drafts are not the
    anchor's. ``positions``: the pass's slot positions (VerifyPass.positions)."""
    if not result.drafting or result.draft_anchor != result.anchor:
        return None
    count, tokens, _ = result.accept[0]
    if not 1 <= count <= len(source.slots) or source.slots[count - 1] != result.anchor:
        raise RuntimeError(f"row {source.row}: the chain drafted at slot {result.anchor} after {count} tokens")
    committed = [int(positions[s]) + 1 for s in source.slots[:count]]
    if committed[-1] != result.bonus:
        raise RuntimeError(f"row {source.row}: the chain's bonus {result.bonus} != {committed[-1]}")
    return ChainedDraft(row=source.row, tokens=list(tokens), positions=committed, drafts=list(result.drafts))
