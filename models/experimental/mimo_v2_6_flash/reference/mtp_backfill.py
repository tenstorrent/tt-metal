# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The target-hidden history and the MTP ring backfill (tt/mtp_backfill.py, kernels/mtp_history.cpp): the layouts and
an exact mirror of the kernel steps (no ttnn), shared by the vLLM adapter and the host tests.

The MTP rings only hold what drafts wrote. A lane drafts for its one request; while it serves several it decodes
without drafting, and a new request's ring starts empty, so the first draft after either would attend a window
(MTP_WINDOW positions) whose entries are missing or stale. vLLM has none of that: its MTP layers see every position,
the prompt's too. The history closes the gap:

* every decode slot that may speculate (SPEC_SLOTS per lane) keeps a ring of HISTORY_ROWS entries, indexed by
  position % HISTORY_ROWS: the target's final normed hidden at that position (the row the step's final norm wrote,
  bf16), the position, and the token the step took there (its input: committed). HISTORY_ROWS = MTP_WINDOW +
  SPEC_SLOTS (``history_rows``): a backfill right after a verify pass at a reads the window before a while the pass's
  rows a .. a + SPEC_SLOTS - 1 are in the ring too;
* every plain decode step writes its slots' rows (``decode_writes``, in the decode trace), every verify pass the rows
  of its slots into their requests' rings (``verify_writes``, in the verify trace; the chain row carries each slot's
  request slot): a verify's rows past the committed ones are wrong, but the next step starts at the first of them and
  rewrites it, so every entry before the latest step's first position is the committed one;
* every prefill chunk writes the final normed hidden of its last HISTORY_ROWS rows and invalidates the slot's
  entries no chunk of this request wrote (``prefill_writes``, eager prefill and the traced batch passes).

Before the first draft that does not continue the rings (the lane's rings last ran another request, or rows they
never drafted: after a prompt, after the lane served several requests), the adapter runs the backfill trace: every
MTP layer over the history of the window before the draft's first row a (a plain step's position, or the column-0
position of the verify the draft follows). Layer i's entry at q takes h_q and x[q + i + 1]. The plan
(``backfill_plan``) writes the whole ring, BACKFILL_ROWS = MTP_RING_TOKENS positions from a block boundary S0, so that
one fill per layer covers it (paged_fill_cache, no read-modify-write): row r at position S0 + r is real when the
history holds q inside the window [a - MTP_WINDOW + 1, a - 1], and layer i takes the token the history holds at
q + i + 1 (0 when it holds none); the other rows write garbage at positions no query reads before a draft rewrites
them (below the window, or at a and past it). The draft commits x[a + 1 .. a + m] (m = 1 after a plain step) and,
with every layer, re-runs the max(0, layers - m) rows before a in every layer (plan_mtp_layers' re-prefill, from the
stash): the plan stashes the rows a - 2, a - 1 (their hidden from the history, their layer-0 tokens, positions). Every
row it does not re-run takes tokens at positions <= a + m - 1, which the history holds committed: the rows before a,
and after a verify at a the pass's rows a .. a + m - 1 (its later rows hold rejected drafts, which only rows the draft
re-runs take, as does the bonus x[a + m]). The commit itself is not needed: the plan reads the history alone.

Layouts (int32, one page each):

- history page [2 * SPEC_SLOTS * HISTORY_ROWS]: entry e = slot * HISTORY_ROWS + q % HISTORY_ROWS, its position at
  [HIST_POSITIONS + e] (-1 none), its token at [HIST_TOKENS + e]; the hidden rows in a bf16 row-major buffer
  [SPEC_SLOTS * HISTORY_ROWS, hidden], row e.
- write list [4 * n] (prefill): per entry the source row (-1: invalidate the entry, -2: skip), the entry, position,
  token.
- backfill upload [B]: drafting (0 / 1), the request's slot, A (the draft's first row, a plain step's position), the
  draft group whose ring and stash slots it fills (draft group g's ring: physical blocks g * TABLE_WORDS onward, its
  first RING_BLOCKS the ring, the rest scratch).
- chain row (reference/spec_chain.py): words [CHAIN_HSLOTS, +SPEC_SLOTS), each verify slot's request slot + 1 (0:
  none), so that a zero row writes nothing.
"""

import torch

from .spec_decode import GROUP_SLOTS, MTP_RING_TOKENS, SPEC_SLOTS

MTP_WINDOW = 128  # the MTP layers' SWA window


def history_rows(window):
    """A history slot's ring for MTP layers of SWA ``window``: the window before a verify pass's first row, the pass's
    SPEC_SLOTS rows and one spare."""
    return window + SPEC_SLOTS


HISTORY_ROWS = history_rows(MTP_WINDOW)
BLOCK = 64  # the ring cache's block size (the vLLM block size)
BACKFILL_ROWS = MTP_RING_TOKENS
RING_BLOCKS = MTP_RING_TOKENS // BLOCK
SCRATCH_BLOCK = RING_BLOCKS  # the ring cache's blocks RING_BLOCKS .. 2 RING_BLOCKS - 1 are never addressed
TABLE_WORDS = 2 * RING_BLOCKS  # the fill's page table row: as wide as the cache (paged_update_cache's limit)
HIST_POSITIONS, HIST_TOKENS = 0, SPEC_SLOTS * HISTORY_ROWS
HIST_WORDS = 2 * SPEC_SLOTS * HISTORY_ROWS
BACKFILL_DRAFTING, BACKFILL_SLOT, BACKFILL_A, BACKFILL_GROUP, BACKFILL_KEEP_STASH = 0, 1, 2, 3, 4
# S0 is at most a block below the window's start, and BACKFILL_ROWS rows from it reach a - 1.
assert MTP_WINDOW + BLOCK <= BACKFILL_ROWS


def entry(slot, q, rows=HISTORY_ROWS):
    return slot * rows + q % rows


class History:
    """A lane's history on the host: the page (positions, tokens) and the hidden rows [SPEC_SLOTS * rows, hidden]."""

    def __init__(self, hidden_size, rows=HISTORY_ROWS):
        self.rows = rows
        self.page = [-1] * (SPEC_SLOTS * rows) + [0] * (SPEC_SLOTS * rows)
        self.hidden = torch.zeros(SPEC_SLOTS * rows, hidden_size)

    def apply(self, writes, src):
        """The kernel's copy: each (source row, entry, position, token) of ``writes``; source -1 invalidates the
        entry, -2 skips."""
        n = SPEC_SLOTS * self.rows
        for s, e, p, t in writes:
            if s == -2:
                continue
            self.page[e], self.page[n + e] = (p, t) if s >= 0 else (-1, 0)
            if s >= 0:
                self.hidden[e] = src[s]

    def position(self, slot, q):
        return self.page[entry(slot, q, self.rows)]

    def token(self, slot, q):
        return self.page[SPEC_SLOTS * self.rows + entry(slot, q, self.rows)]


# ---------------------------------------------------------------------- writes (the kernel's lists)
def decode_writes(positions, tokens, users=SPEC_SLOTS, rows=HISTORY_ROWS):
    """A plain decode step: slot s < ``users`` at ``positions[s]`` (>= 0) with input ``tokens[s]``, source row s."""
    return [(s, entry(s, positions[s], rows), positions[s], tokens[s]) for s in range(users) if positions[s] >= 0]


def verify_writes(verify, chain_hslots, rows=HISTORY_ROWS):
    """A verify pass (the upload's int32 values: tokens, positions, chain row): slot j at P[j] >= 0 into the ring of
    its request's slot (``chain_hslots``: the chain row's words, slot + 1, 0 none), source row j."""
    b = len(verify) // 3
    tokens, positions = verify[:b], verify[b : 2 * b]
    out = []
    for j in range(SPEC_SLOTS):
        h = chain_hslots[j]
        if positions[j] >= 0 and h > 0:
            out.append((j, entry(h - 1, positions[j], rows), positions[j], tokens[j]))
    return out


def prefill_writes(slot, start, tokens, src_first, continues, rows=HISTORY_ROWS):
    """A prefill chunk of ``slot`` at positions start .. start + n - 1 (``tokens`` its n tokens), its row q being
    source row src_first + q - start: one write per ring entry, positions end - rows .. end - 1: the chunk's own
    rows, then (``continues`` False: the chunk is the request's first) the entries before ``start`` invalidated
    (positions < 0 always). A continuing chunk keeps the earlier chunks' entries."""
    end = start + len(tokens)
    out = []
    for q in range(end - rows, end):
        if q >= start:
            out.append((src_first + q - start, entry(slot, q, rows), q, int(tokens[q - start])))
        elif q < 0 or not continues:
            out.append((-1, entry(slot, q, rows), -1, 0))
    return out


def pack_writes(writes, entries):
    """The write list's int32 values [4 * entries], skipped entries (source -2) past the writes."""
    words = [-2, 0, -1, 0] * entries
    assert len(writes) <= entries
    for k, w in enumerate(writes):
        words[4 * k : 4 * k + 4] = list(w)
    return words


# ---------------------------------------------------------------------- the backfill
def backfill_row(max_batch, slot=None, a=None, group=0, keep_stash=False):
    """The backfill upload's int32 values [max_batch]: the request in lane slot ``slot``, the draft's first row at
    ``a``, into draft group ``group``'s ring and stash slots (``keep_stash``: the ring only, for a draft in groups of
    one draft, which re-runs no row: adaptive draft length's K = 1 layout); None: idle."""
    row = [0] * max_batch
    if slot is not None:
        row[BACKFILL_DRAFTING], row[BACKFILL_SLOT], row[BACKFILL_A] = 1, int(slot), int(a)
        row[BACKFILL_GROUP], row[BACKFILL_KEEP_STASH] = int(group), int(bool(keep_stash))
    return row


def backfill_start(a, window=MTP_WINDOW):
    """S0: the block the window [a - window + 1, a - 1] starts in."""
    return max(0, a - window + 1) // BLOCK * BLOCK


def backfill_plan(upload, history_page, layers, stash, rows=HISTORY_ROWS, window=MTP_WINDOW, group_slots=GROUP_SLOTS):
    """The backfill's plan step (kernels/mtp_history.cpp mode 3): ``upload`` the backfill upload, ``history_page``
    the history page (``rows`` per slot), ``layers`` the MTP layers (SWA ``window``), ``stash`` the draft's stash
    page (every layer; None with one). Returns ({"hidden": history rows [BACKFILL_ROWS], "rope": RoPE indices
    [BACKFILL_ROWS], "tokens": per layer [BACKFILL_ROWS], "table": the fill's page table row [TABLE_WORDS] (the
    ring's blocks, then scratch blocks), "stash_rows": (stash slot, history entry) to copy}, the new stash page). Row
    r writes position S0 + r. An idle upload fills draft group 0's scratch blocks and leaves the stash; a drafting
    one fills its group's ring and re-plans that group's stash slots only (draft groups of ``group_slots`` slots)."""
    assert window + SPEC_SLOTS - 1 <= rows and window + BLOCK <= BACKFILL_ROWS
    n = SPEC_SLOTS * rows
    out = {
        "hidden": [0] * BACKFILL_ROWS,
        "rope": list(range(BACKFILL_ROWS)),
        "tokens": [[0] * BACKFILL_ROWS for _ in range(layers)],
        "table": [SCRATCH_BLOCK + k % RING_BLOCKS for k in range(TABLE_WORDS)],
        "stash_rows": [],
    }
    if upload[BACKFILL_DRAFTING] == 0:
        return out, stash
    slot, a, group = upload[BACKFILL_SLOT], upload[BACKFILL_A], upload[BACKFILL_GROUP]
    lo, s0 = max(0, a - window + 1), backfill_start(a, window)
    base = group * TABLE_WORDS

    def held(q):  # the history holds position q
        return q >= 0 and history_page[entry(slot, q, rows)] == q

    def token(q):
        return history_page[n + entry(slot, q, rows)]

    out["table"] = [base + (s0 % MTP_RING_TOKENS // BLOCK + k) % RING_BLOCKS for k in range(RING_BLOCKS)]
    out["table"] += [base + SCRATCH_BLOCK + k % RING_BLOCKS for k in range(RING_BLOCKS, TABLE_WORDS)]
    for r in range(BACKFILL_ROWS):
        q = s0 + r
        out["rope"][r] = min(q, a)  # the garbage rows past a stay inside the RoPE tables
        if lo <= q <= a - 1 and held(q):
            out["hidden"][r] = entry(slot, q, rows)
            for i in range(layers):
                if held(q + i + 1):
                    out["tokens"][i][r] = token(q + i + 1)
    if stash is None or layers == 1 or upload[BACKFILL_KEEP_STASH]:  # no row re-run: the stash is never read
        return out, stash
    from .spec_chain import STASH_POSITIONS, STASH_TOKENS

    new = list(stash)
    first, last = group * group_slots, (group + 1) * group_slots - 1
    for k in range(first, last + 1):
        new[STASH_POSITIONS + k], new[STASH_TOKENS + k] = -1, 0
    for k, q in ((last - 1, a - 2), (last, a - 1)):
        if held(q) and held(q + 1):
            new[STASH_POSITIONS + k], new[STASH_TOKENS + k] = q, token(q + 1)
            out["stash_rows"].append((k, entry(slot, q, rows)))
    return out, new
