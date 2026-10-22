# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host bookkeeping of MiMo's MTP speculative decoding (no ttnn): shared by the vLLM adapter (tt/generator_vllm.py)
and the host tests.

Verify. A lane's decode step serves SPEC_SLOTS decode slots, the users the fast decode path takes (the streaming MoE,
SwaDecodeQkv and GreedyHead serve rows 0-3). A verify gives each active row of the lane one slot per candidate
column, columns 0 .. n (the last committed token and its n drafts), consecutive slots at consecutive positions with
the row's page tables. The rows are packed into passes of at most SPEC_SLOTS slots, run one after another. Slots of
one request write K / V into the same cache tiles, so a verify pass writes one slot at a time (serial_positions);
attention then reads every slot's own prefix. Slot j's greedy token is the target's choice after column j: the
row's argmax_ids column j.

Propose. After a lane step that ran one row in one pass, MTP drafts for it from the device: the step's greedy tokens
and the target's final normed hidden rows are still resident. The fill pass runs the MTP layer over the m committed
tokens, row c taking the hidden at the position before committed token c and that token's embedding, and writes the
MTP layer's K / V there; its output at the last committed token gives draft 1. Two single-row steps continue from
the MTP layer's own output (vLLM's MTP proposer with one layer) and give drafts 2 and 3 (plan_mtp).

Propose with every MTP layer (plan_mtp_layers; vLLM 0.31's MultiModuleMTPSpeculator, which MiMo's three layers run
with the num_nextn_predict_layers patch). Draft i + 1 comes from MTP layer i, and the layers are not chained: each
takes the target's hidden. With x the committed tokens (the last at position B), d_1 .. d_3 the drafts and
x~ = x followed by d_1, d_2, layer i at position q takes the target's final normed hidden h_q and the embedding of
x~[q + i + 1], with RoPE at q, writes its K / V at q in its own cache, and its output at q predicts x~[q + i + 2].
Draft i + 1 is layer i's greedy token at the last row, B - 1: d_{i+1} = argmax LM(layer_i(h_{B-1}, x~[B + i])),
attending layer i's cache. That is MTP training's computation (module i over the sequence, the target hidden at q and
the token i + 1 positions later), the drafts standing in for tokens not yet committed.

Each layer's cache at every position q <= B - 1 holds the entry of (h_q, x~[q + i + 1]): the passes run the
committed rows (q = A .. B - 1, A the position of the step's first column) in every layer, the token window shifted
by one per layer (layer i's rows q >= B - i take drafts). Layer i's rows A - i .. A - 1 took the last draft's d_1 ..
d_i; once the verify rejects some of them they are stale, so the passes also re-run the rows before A (re-prefill):
max(0, layers - m) of them for m committed tokens, the most any layer needs (layer i needs max(0, i + 1 - m)), from
the target hidden rows and layer-0 tokens the last draft cached (MtpStash: every row of its pass, found by position).
With 3 layers a draft runs 3 rows, 4 when all 3 drafts were accepted (fewer for a request's first drafts, while the
stash holds fewer rows), so the rows fit one pass of SPEC_SLOTS slots: right-aligned, the anchor (the row at B - 1)
in the last slot. vLLM re-runs these rows in every layer, as here; re-running a row whose entry is current rewrites
the same entry. The rows a draft re-runs are the last draft's last ones (its anchor at A - 1); the stash keeps all of
that draft's rows so that a draft planned again from the same commit (the adapter's MIMO_SPEC_CHAIN_CHECK, or a host
plan after the device chained a draft from a different commit) finds the same ones: re-planning is idempotent.
"""

from dataclasses import dataclass, field

SPEC_SLOTS = 4  # decode slots of a verify pass
MAX_DRAFTS = SPEC_SLOTS - 1  # drafts per verify: the last committed token and its drafts fill one pass


@dataclass
class VerifyPass:
    """One decode step of a lane: per slot the token, position (-1 idle) and the batch row whose page tables it reads
    (None idle), and the (row, column) whose argmax each used slot answers."""

    tokens: list = field(default_factory=lambda: [0] * SPEC_SLOTS)
    positions: list = field(default_factory=lambda: [-1] * SPEC_SLOTS)
    table_rows: list = field(default_factory=lambda: [None] * SPEC_SLOTS)
    answers: list = field(default_factory=list)  # (slot, row, column)

    @property
    def used(self):
        return sum(p >= 0 for p in self.positions)


def plan_verify_passes(rows, slots=SPEC_SLOTS):
    """rows: (row, tokens [1 + K], positions [1 + K], num_valid_drafts) of a lane's active rows, in batch order.
    Returns the lane's passes. A row takes columns 0 .. num_valid_drafts, in consecutive slots of one pass."""
    passes = []
    for row, tokens, positions, n in rows:
        width = int(n) + 1
        if width > slots:
            raise ValueError(f"row {row}: {n} drafts need {width} slots, a pass has {slots}")
        for c in range(width):
            if int(positions[c]) < 0:
                raise ValueError(f"row {row}: column {c} of {width} has no position")
        if not passes or passes[-1].used + width > slots:
            passes.append(VerifyPass())
        p = passes[-1]
        first = p.used
        for c in range(width):
            s = first + c
            p.tokens[s], p.positions[s], p.table_rows[s] = int(tokens[c]), int(positions[c]), row
            p.answers.append((s, row, c))
    return passes


def serial_positions(positions, slots=SPEC_SLOTS):
    """One position list per slot with only that slot's position kept: a verify pass writes K / V slot by slot."""
    out = []
    for j in range(slots):
        one = [-1] * len(positions)
        one[j] = positions[j]
        out.append(one)
    return out


@dataclass
class DraftSource:
    """Where a lane's single row of the last step left its committed tokens on the device: ``slots[c]`` holds the
    greedy token committed at column c (and the target hidden that chose it)."""

    row: int
    slots: list


def draft_source(passes):
    """The DraftSource of a lane step, or None unless it ran one row in one pass (MTP drafts for one request per
    lane)."""
    if len(passes) != 1:
        return None
    rows = {row for _, row, _ in passes[0].answers}
    if len(rows) != 1:
        return None
    by_column = sorted((c, s) for s, _, c in passes[0].answers)
    return DraftSource(row=rows.pop(), slots=[s for _, s in by_column])


@dataclass
class MtpInputs:
    """Positions of the MTP fill pass and its two continuation steps (-1 idle), each list SPEC_SLOTS long, and the
    slot whose greedy tokens are the drafts."""

    fill: list
    step2: list
    step3: list
    anchor: int


def plan_mtp(source, committed_positions, accepted, slots=SPEC_SLOTS):
    """source: the lane's DraftSource; committed_positions: that row's [1 + K] committed positions; accepted: its
    committed count m. The committed token c sits at committed_positions[c]; the hidden that chose it, and the MTP
    entry it fills, at committed_positions[c] - 1."""
    m = int(accepted)
    if not 1 <= m <= len(source.slots):
        raise ValueError(f"row {source.row}: {m} committed tokens, the step answered {len(source.slots)} columns")
    fill = [-1] * slots
    for c in range(m):
        fill[source.slots[c]] = int(committed_positions[c]) - 1
    anchor = source.slots[m - 1]
    bonus = int(committed_positions[m - 1])
    step2, step3 = [-1] * slots, [-1] * slots
    step2[anchor], step3[anchor] = bonus, bonus + 1
    return MtpInputs(fill=fill, step2=step2, step3=step3, anchor=anchor)


@dataclass
class MtpStash:
    """The rows of a multi-layer draft's pass, which the next one re-runs from (re-prefill): per slot its true position
    (-1 idle or none). The drafter keeps each row's target hidden and layer-0 token, by slot. A re-prefill finds a row
    by its position (``row``)."""

    positions: list

    @classmethod
    def empty(cls, slots=SPEC_SLOTS):
        return cls(positions=[-1] * slots)

    def row(self, position):
        """The slot holding ``position`` (the first, as the device's plan step searches), or None."""
        if position < 0:
            return None
        return next((k for k, p in enumerate(self.positions) if p == position), None)


@dataclass
class MtpLayersInputs:
    """The passes of a multi-layer draft, one per MTP layer, all over the same slots: per slot the true position (-1
    idle), the source of its target hidden and layer-0 token, ("step", s) the last step's slot s (the hidden there and
    the token committed there) or ("stash", k) slot k of the MtpStash (None idle), the re-prefilled row count, and the
    anchor slot (the row at B - 1, whose greedy tokens are the drafts). Used slots are consecutive and in position
    order, ending at the anchor."""

    positions: list
    sources: list
    reprefill: int
    anchor: int

    @property
    def rows(self):
        """The used slots, in position order."""
        return [s for s, p in enumerate(self.positions) if p >= 0]


def plan_mtp_layers(source, committed_positions, accepted, stash, layers=MAX_DRAFTS, slots=SPEC_SLOTS):
    """source: the lane's DraftSource; committed_positions: that row's [1 + K] committed positions; accepted: its
    committed count m; stash: the MtpStash of the lane's last draft. The committed token c sits at
    committed_positions[c] (consecutive, from A + 1); the row taking it is at committed_positions[c] - 1. The stash
    rows re-run are the max(0, layers - m) positions before A, as far as the stash holds them consecutively from
    A - 1 down (it holds nothing after a step that did not follow a draft; then nothing is stale either)."""
    m = int(accepted)
    if not 1 <= m <= len(source.slots):
        raise ValueError(f"row {source.row}: {m} committed tokens, the step answered {len(source.slots)} columns")
    a = int(committed_positions[0]) - 1
    for c in range(m):
        if int(committed_positions[c]) != a + 1 + c:
            raise ValueError(f"row {source.row}: committed positions {list(committed_positions[:m])} not consecutive")
    if len(stash.positions) != slots:
        raise ValueError(f"a stash of {len(stash.positions)} rows for passes of {slots} slots")
    want, found = max(0, layers - m), []  # found[k]: the stash slot of position a - 1 - k
    while len(found) < want and stash.row(a - 1 - len(found)) is not None:
        found.append(stash.row(a - 1 - len(found)))
    r = len(found)
    rows = r + m
    if rows > slots:
        raise ValueError(f"row {source.row}: {rows} MTP rows, a pass has {slots} slots")
    positions, sources = [-1] * slots, [None] * slots
    first = slots - rows
    for j in range(rows):
        positions[first + j] = a - r + j
        sources[first + j] = ("stash", found[r - 1 - j]) if j < r else ("step", source.slots[j - r])
    return MtpLayersInputs(positions=positions, sources=sources, reprefill=r, anchor=slots - 1)


def mtp_layer_tokens(tokens0, drafts, layer):
    """The tokens of ``layer``'s pass over the used rows, in position order: ``tokens0`` the rows' layer-0 tokens
    (row q takes the token at q + 1), ``drafts`` the drafts so far (at least ``layer``). Layer i's row takes the token
    i positions further: the window tokens0 + drafts shifted by i."""
    window = list(tokens0) + list(drafts[:layer])
    return window[layer : layer + len(tokens0)]


def mtp_stash(plan):
    """The MtpStash after the draft ``plan``: every slot's position (stash slot k is the pass's slot k)."""
    return MtpStash(positions=list(plan.positions))


# The MTP layer's own KV cache: one request per lane drafts, and its attention reads its 128-token window, so a ring of
# MTP_RING_TOKENS positions holds everything it reads. The layer addresses the cache at mtp_cache_position(t) and
# takes its RoPE at the true position t (host cos / sin): the window of a cache position >= MTP_RING_TOKENS stays at
# cache positions >= MTP_RING_TOKENS - 127, inside the table's 2 * MTP_RING_TOKENS positions, and logical block b
# maps to physical block b % ring blocks, so a write at t and every later read of t meet in one row.
MTP_RING_TOKENS = 256


def mtp_cache_position(t, ring=MTP_RING_TOKENS):
    return ring + t % ring if t >= 0 else -1


def mtp_table_row(block_size, ring=MTP_RING_TOKENS):
    """The MTP cache's page table row (2 * ring positions) and its physical block count (paged_update_cache needs at
    least as many blocks as the table is wide)."""
    assert ring % block_size == 0
    blocks = ring // block_size
    return [b % blocks for b in range(2 * blocks)], 2 * blocks


def accept_greedy(argmax_ids, drafts, num_valid):
    """Committed tokens of one row: drafts while they match the target's choice, then the target's own token (the
    plugin's accept_greedy_drafts, for the host tests)."""
    out = []
    for j in range(int(num_valid)):
        out.append(int(argmax_ids[j]))
        if int(drafts[j]) != int(argmax_ids[j]):
            return out
    out.append(int(argmax_ids[int(num_valid)]))
    return out
