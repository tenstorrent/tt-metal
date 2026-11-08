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

Draft groups (DRAFT_GROUPS = 2, MIMO_SPEC_GROUPS): a pass of SPEC_SLOTS = 8 slots holds two requests, each in its own
group of GROUP_SLOTS = 4 slots (its last committed token and 3 drafts), and the drafter drafts for both in one program,
each request with its group's state (MTP rings and stash, DFlash context ring). A request keeps its group while it
runs (the adapter assigns it at the prompt), so its state stays its own. Everything below describes one group.

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

import os
from dataclasses import dataclass, field

# Decode slots of a verify pass: 4, 8 (MIMO_SPEC_SLOTS=8: DFlash's block of 7 drafts, or two requests of 3; the
# verify then runs the pass in 8 decode slots, so the lane decodes 8 users) or 16 (four requests of 3, or eight of 1:
# the lane decodes 16 users). Fixed per process: every layout below derives from it.
SPEC_SLOTS = int(os.environ.get("MIMO_SPEC_SLOTS", "4"))
assert SPEC_SLOTS in (4, 8, 16), f"MIMO_SPEC_SLOTS={SPEC_SLOTS}: 4, 8 or 16"
# Requests a pass drafts for: 1, 2 or 4 (MIMO_SPEC_GROUPS with MIMO_SPEC_SLOTS=8: a lane's requests, each in a draft
# group of GROUP_SLOTS consecutive slots with its own drafter state; 2: c=8 over 4 lanes with 3 drafts each, 4: c=16
# with 1 draft each, which one MTP layer drafts through the planned program, planned_drafts); with MIMO_SPEC_SLOTS=16
# 4 (3 drafts each) or 8 (1 draft each).
# MIMO_SPEC_GROUPS=adaptive (with MIMO_SPEC_SLOTS=8 or 16): each pass picks its group width from its requests
# (pass_width): groups of 4 slots (3 drafts each, every MTP layer) for up to ADAPTIVE_WIDE_ROWS requests (SPEC_SLOTS /
# 4), groups of 2 (1 draft) for more. DRAFT_GROUPS and GROUP_SLOTS are then the most groups (SPEC_SLOTS / 2) and the
# widest group (4) a pass may have: the drafter state (rings, stash slots, descriptions, chain blocks, result sections)
# is laid out for both, and a group's width travels with it (its chain block's source word and its description's
# drafting word hold the width).
_GROUPS = os.environ.get("MIMO_SPEC_GROUPS", "1")
ADAPTIVE = _GROUPS == "adaptive"
if ADAPTIVE:
    assert SPEC_SLOTS in (8, 16), "MIMO_SPEC_GROUPS=adaptive needs MIMO_SPEC_SLOTS=8 or 16"
    DRAFT_GROUPS, GROUP_SLOTS = SPEC_SLOTS // 2, 4
else:
    DRAFT_GROUPS = int(_GROUPS)
    assert DRAFT_GROUPS == 1 or (
        SPEC_SLOTS > 4 and DRAFT_GROUPS in (SPEC_SLOTS // 4, SPEC_SLOTS // 2)
    ), f"MIMO_SPEC_GROUPS={DRAFT_GROUPS}: 1, or groups of 4 / 2 slots / adaptive with MIMO_SPEC_SLOTS=8 or 16"
    GROUP_SLOTS = SPEC_SLOTS // DRAFT_GROUPS
MAX_DRAFTS = GROUP_SLOTS - 1  # drafts per request: its last committed token and drafts fill one draft group
WIDTHS = (2, 4) if ADAPTIVE else (GROUP_SLOTS,)  # the group widths a pass may use
ADAPTIVE_WIDE_ROWS = int(os.environ.get("MIMO_SPEC_WIDE_ROWS", str(SPEC_SLOTS // 4)))


def pass_width(live_rows):
    """The group width of a lane step with ``live_rows`` requests: GROUP_SLOTS, or with ADAPTIVE groups of 4 for up to
    ADAPTIVE_WIDE_ROWS requests and of 2 for more."""
    if not ADAPTIVE:
        return GROUP_SLOTS
    return 4 if live_rows <= ADAPTIVE_WIDE_ROWS else 2


def width_drafts(width):
    """Drafts per request in groups of ``width`` slots."""
    return width - 1


# K / V writes of a verify pass (and of an MTP pass): only one request's slots share cache tiles, and with draft groups
# a request's slots are consecutive inside its group, so write k takes every slot j with j % SERIAL_WRITES == k: one
# write per column of a group (2 or 4) instead of one per slot (serial_positions). MIMO_SERIAL_COLUMNS=0: per slot.
SERIAL_WRITES = GROUP_SLOTS if DRAFT_GROUPS > 1 and os.environ.get("MIMO_SERIAL_COLUMNS", "1") == "1" else SPEC_SLOTS


def planned_drafts(layers):
    """Whether the MTP drafter runs its planned program (per-group commit descriptions, the plan step, the hidden
    gather: tt/mtp.py ``planned``) with ``layers`` MTP layers drafting: every layer (``layers`` > 1, one per draft), or
    one layer drafting one token per request (MAX_DRAFTS == 1, groups of 2 slots), which is every-layer drafting with
    one layer (no re-prefill). Otherwise one layer runs the one-layer program (fill, then continuation steps), which
    drafts for one request per lane."""
    return layers > 1 or MAX_DRAFTS == 1


@dataclass
class VerifyPass:
    """One decode step of a lane: per slot the token, position (-1 idle) and the batch row whose page tables it reads
    (None idle), and the (row, column) whose argmax each used slot answers. ``width``: its draft groups' slots (group g
    holds slots g * width ..; SPEC_SLOTS without draft groups)."""

    tokens: list = field(default_factory=lambda: [0] * SPEC_SLOTS)
    positions: list = field(default_factory=lambda: [-1] * SPEC_SLOTS)
    table_rows: list = field(default_factory=lambda: [None] * SPEC_SLOTS)
    answers: list = field(default_factory=list)  # (slot, row, column)
    width: int = GROUP_SLOTS

    @property
    def used(self):
        return sum(p >= 0 for p in self.positions)


def plan_verify_passes(rows, slots=SPEC_SLOTS, groups=None, group_slots=GROUP_SLOTS, aligned=False):
    """rows: (row, tokens [1 + K], positions [1 + K], num_valid_drafts) of a lane's active rows, in batch order.
    Returns the lane's passes. A row takes columns 0 .. num_valid_drafts, in consecutive slots of one pass. With draft
    groups (``group_slots`` < ``slots``) each row starts a group: its own (``groups`` {row: group}, the request's
    sticky group) in the first pass where that group is free, or the first free group. ``aligned`` (the grouped
    verify attention, one KV scan per group: tt/attention.py verify_grouped): without draft groups too, each row
    starts a pass of its own, so every group of slots holds one request."""
    passes = []
    multi = group_slots < slots

    def free(p, g):
        return all(p.positions[s] < 0 for s in range(g * group_slots, (g + 1) * group_slots))

    for row, tokens, positions, n in rows:
        width = int(n) + 1
        if width > (group_slots if multi else slots):
            raise ValueError(f"row {row}: {n} drafts need {width} slots, a pass has {slots} in groups of {group_slots}")
        for c in range(width):
            if int(positions[c]) < 0:
                raise ValueError(f"row {row}: column {c} of {width} has no position")
        if multi:
            want = (groups or {}).get(row)
            options = range(slots // group_slots) if want is None else [want]
            p, g = next(((p, g) for p in passes for g in options if free(p, g)), (None, None))
            if p is None:
                p, g = VerifyPass(width=group_slots), options[0]
                passes.append(p)
            first = g * group_slots
        else:
            if not passes or passes[-1].used + width > slots or (aligned and passes[-1].used > 0):
                passes.append(VerifyPass(width=group_slots))
            p = passes[-1]
            first = p.used
        for c in range(width):
            s = first + c
            p.tokens[s], p.positions[s], p.table_rows[s] = int(tokens[c]), int(positions[c]), row
            p.answers.append((s, row, c))
    return passes


def serial_positions(positions, slots=SPEC_SLOTS, writes=SERIAL_WRITES):
    """One position list per serial write, list k keeping the positions of slots j < ``slots`` with j % ``writes`` ==
    k: a verify pass writes K / V one column of its draft groups at a time (one slot at a time when ``writes`` ==
    ``slots``)."""
    out = []
    for k in range(writes):
        one = [-1] * len(positions)
        for j in range(k, slots, writes):
            one[j] = positions[j]
        out.append(one)
    return out


@dataclass
class DraftSource:
    """Where a row of the last step left its committed tokens on the device: ``slots[c]`` holds the greedy token
    committed at column c (and the target hidden that chose it). ``group``: its draft group (DRAFT_GROUPS > 1), whose
    drafter state (rings, stash) it drafts with; a verify row's is the group its slots are in."""

    row: int
    slots: list
    group: int = 0
    width: int = GROUP_SLOTS  # its draft group's slots: group g holds slots g * width .. (g + 1) * width - 1

    @property
    def anchor(self):
        """The group's last slot: the planned MTP program's anchor (the draft's last row)."""
        return (self.group + 1) * self.width - 1


def draft_sources(passes, group_slots=GROUP_SLOTS):
    """The DraftSources of a lane step: every row of a step that ran one pass, at most DRAFT_GROUPS of them (one per
    draft group, plan_verify_passes); none otherwise."""
    if len(passes) != 1:
        return []
    by_row = {}
    for s, row, c in passes[0].answers:
        by_row.setdefault(row, []).append((c, s))
    if len(by_row) > DRAFT_GROUPS:
        return []
    out = []
    for row, cells in by_row.items():
        slots = [s for _, s in sorted(cells)]
        out.append(DraftSource(row=row, slots=slots, group=slots[0] // group_slots, width=group_slots))
    return out


def draft_source(passes):
    """The DraftSource of a lane step that ran one row in one pass, or None."""
    sources = draft_sources(passes)
    return sources[0] if len(sources) == 1 else None


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


def plan_mtp_layers(
    source, committed_positions, accepted, stash, layers=MAX_DRAFTS, slots=SPEC_SLOTS, group_slots=GROUP_SLOTS
):
    """source: a DraftSource; committed_positions: that row's [1 + K] committed positions; accepted: its committed
    count m; stash: the MtpStash of the lane's last draft. The committed token c sits at committed_positions[c]
    (consecutive, from A + 1); the row taking it is at committed_positions[c] - 1. The stash rows re-run are the
    max(0, layers - m) positions before A, as far as the stash holds them consecutively from A - 1 down (it holds
    nothing after a step that did not follow a draft; then nothing is stale either). The rows fill the source's draft
    group (slots g * group_slots .. (g + 1) * group_slots - 1 of the pass), right-aligned, and re-run that group's
    stash rows only."""
    m = int(accepted)
    if not 1 <= m <= len(source.slots):
        raise ValueError(f"row {source.row}: {m} committed tokens, the step answered {len(source.slots)} columns")
    a = int(committed_positions[0]) - 1
    for c in range(m):
        if int(committed_positions[c]) != a + 1 + c:
            raise ValueError(f"row {source.row}: committed positions {list(committed_positions[:m])} not consecutive")
    if len(stash.positions) != slots:
        raise ValueError(f"a stash of {len(stash.positions)} rows for passes of {slots} slots")
    lo, end = source.group * group_slots, (source.group + 1) * group_slots
    group = MtpStash(positions=[p if lo <= k < end else -1 for k, p in enumerate(stash.positions)])
    want, found = max(0, layers - m), []  # found[k]: the stash slot of position a - 1 - k
    while len(found) < want and group.row(a - 1 - len(found)) is not None:
        found.append(group.row(a - 1 - len(found)))
    r = len(found)
    rows = r + m
    if rows > group_slots:
        raise ValueError(f"row {source.row}: {rows} MTP rows, a draft group has {group_slots} slots")
    positions, sources = [-1] * slots, [None] * slots
    first = end - rows
    for j in range(rows):
        positions[first + j] = a - r + j
        sources[first + j] = ("stash", found[r - 1 - j]) if j < r else ("step", source.slots[j - r])
    return MtpLayersInputs(positions=positions, sources=sources, reprefill=r, anchor=end - 1)


def mtp_layer_tokens(tokens0, drafts, layer):
    """The tokens of ``layer``'s pass over the used rows, in position order: ``tokens0`` the rows' layer-0 tokens
    (row q takes the token at q + 1), ``drafts`` the drafts so far (at least ``layer``). Layer i's row takes the token
    i positions further: the window tokens0 + drafts shifted by i."""
    window = list(tokens0) + list(drafts[:layer])
    return window[layer : layer + len(tokens0)]


def mtp_stash(plan):
    """The MtpStash after the draft ``plan``: every slot's position (stash slot k is the pass's slot k)."""
    return MtpStash(positions=list(plan.positions))


# The MTP layer's own KV cache: one ring per draft group (one request), and its attention reads its 128-token window, so
# a ring of MTP_RING_TOKENS positions holds everything it reads. The layer addresses the cache at mtp_cache_position(t) and
# takes its RoPE at the true position t (host cos / sin): the window of a cache position >= MTP_RING_TOKENS stays at
# cache positions >= MTP_RING_TOKENS - 127, inside the table's 2 * MTP_RING_TOKENS positions, and logical block b
# maps to physical block b % ring blocks, so a write at t and every later read of t meet in one row.
MTP_RING_TOKENS = 256


def mtp_cache_position(t, ring=MTP_RING_TOKENS):
    return ring + t % ring if t >= 0 else -1


def mtp_table_row(block_size, ring=MTP_RING_TOKENS, group=0):
    """The MTP cache's page table row (2 * ring positions) of draft group ``group`` and the physical blocks one group
    takes (paged_update_cache needs at least as many blocks as the table is wide): group g's ring is blocks
    g * 2 * (ring / block_size) onward."""
    assert ring % block_size == 0
    blocks = ring // block_size
    return [2 * blocks * group + b % blocks for b in range(2 * blocks)], 2 * blocks


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
