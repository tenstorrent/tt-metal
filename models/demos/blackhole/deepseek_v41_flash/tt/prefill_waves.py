# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill slot bookkeeping and wave planning of ``Model.prefill_interleaved`` (no ttnn: unit-testable on the CPU).

Only used when the traced chunk holds fewer users per mesh row than the row has (``Up != U``, e.g. B=16: U=4, Up=2). A user that is being prefilled owns one slot of its row (its carried state lives
there) until it finishes or another prompt of the row evicts it. A call can carry more users of one row than Up slots: they run in waves of at most Up users per row; a wave allocates slots for its NEW
prompts (and for prefix-cache hits whose restore is still pending: they have no slot yet), a continuation keeps its slot.

Invariant: a user that continues in a later wave of the same call must not lose its slot to a user of an earlier wave. ``plan_waves`` therefore puts every real continuation of a row before the users that
need a slot (a new prompt, a pending hit): a hit has start > 0 and used to sort with the continuations, then took the slot of a continuation of the NEXT wave (``user N has no prefill slot to continue``,
AgentX c8 on the B=16 load-trace server, 2026-11-08).
"""


def protect_partial_on():
    """DSV41_PROTECT_PARTIAL=1 (default off; live switch file DSV41_PROTECT_PARTIAL_FILE holding 0 / 1): a partly prefilled prompt is protected from losing its carried state (T24, PROTECT_PARTIAL notes):
    the cheapest owner is evicted when an eviction cannot be avoided, a new prompt prefers a mesh row with a free prefill slot, a prefix hit is only reserved on a row with a free prefill slot, and a prefix
    hit whose first piece would be skipped is computed one chunk instead (its restore is not lost)."""
    import os

    f = os.environ.get("DSV41_PROTECT_PARTIAL_FILE")
    if f and os.path.exists(f):
        try:
            return open(f).read().strip() == "1"
        except OSError:
            pass
    return os.environ.get("DSV41_PROTECT_PARTIAL", "0") == "1"


def resume_record_on():
    """DSV41_RESUME_FROM_RECORD=1 (default off; live switch file DSV41_RESUME_FROM_RECORD_FILE holding 0 / 1): a continuation whose carried prefill state is gone resumes from the deepest saved key / halo record
    of its own prefix (the paged index, as a prefix hit) instead of recomputing from position 0."""
    import os

    f = os.environ.get("DSV41_RESUME_FROM_RECORD_FILE")
    if f and os.path.exists(f):
        try:
            return open(f).read().strip() == "1"
        except OSError:
            pass
    return os.environ.get("DSV41_RESUME_FROM_RECORD", "0") == "1"


def snapshot_on_evict_on():
    """DSV41_SNAPSHOT_ON_EVICT=1 (default off; live file DSV41_SNAPSHOT_ON_EVICT_FILE 0 / 1): when a prefill slot is taken from a prompt in progress, its carried halo state is saved as a record of the
    paged index at the carried position first (device-local copy, ~0.6 MiB per bank), so that DSV41_RESUME_FROM_RECORD resumes exactly there.
    """
    import os

    f = os.environ.get("DSV41_SNAPSHOT_ON_EVICT_FILE")
    if f and os.path.exists(f):
        try:
            return open(f).read().strip() == "1"
        except OSError:
            pass
    return os.environ.get("DSV41_SNAPSHOT_ON_EVICT", "0") == "1"


def needs_slot(item, pending_hits):
    """True for a user that will be allocated a slot in the wave it runs in: a new prompt (start 0) or a prefix hit whose restore has not been done (it owns no slot yet)."""
    return item[2] == 0 or item[0] in pending_hits


def plan_groups(items, pending_hits, per_row):
    """Groups of items processed one after the other (each group is split into waves). Real continuations come first, users that need a slot after them."""
    if per_row:
        return [sorted(items, key=lambda it: needs_slot(it, pending_hits))]
    groups = []
    for want_slot in (False, True):
        part = [it for it in items if needs_slot(it, pending_hits) == want_slot]
        groups += [[it for it in part if it[2] == st0] for st0 in sorted({it[2] for it in part}, reverse=True)]
    return groups


def split_waves(grp, Up, U):
    """At most Up users of a mesh row per wave, rows side by side, in the order of ``grp``."""
    by_row = {}
    for it in grp:
        by_row.setdefault(it[0] // U, []).append(it)
    return [
        [it for r in sorted(by_row) for it in by_row[r][k * Up : (k + 1) * Up]]
        for k in range(-(-max(len(v) for v in by_row.values()) // Up))
    ]


def slot_alloc(owner, of, Up, U, wave, resume, on_evict=None, cheapest=False, log=None):
    """Assign a prefill slot (global index r * Up + i) to every user of ``wave`` [(b, tokens, start, end)]; users that continue keep theirs. ``owner`` [rows * Up] / ``of`` {user: slot} are the
    persistent state (updated in place), ``resume`` {user: carried position} loses the evicted owners. -> {b: slot}."""
    out, taken = {}, set()
    for b, _, st, _ in wave:
        if st > 0:
            assert of.get(b) is not None and owner[of[b]] == b, f"user {b} has no prefill slot to continue"
            out[b] = of[b]
            taken.add(of[b])
    for b, _, st, _ in wave:
        if st > 0:
            continue
        r = b // U
        cand = [r * Up + i for i in range(Up) if r * Up + i not in taken]
        assert cand, "more users of one mesh row than prefill slots in one wave"
        if of.get(b) in cand:
            sl = of[b]
        else:  # a slot whose owner is finished / not resumable first, else evict (its owner has to recompute from position 0)
            free = [c for c in cand if owner[c] is None or owner[c] not in resume]
            if free:
                sl = free[0]
            elif (
                cheapest
            ):  # (DSV41_PROTECT_PARTIAL) every slot holds a prompt in progress: evict the one with the least carried progress (the cheapest to recompute)
                sl = min(cand, key=lambda c: (resume.get(owner[c], 0), c))
            else:
                sl = cand[0]
        o = owner[sl]
        if o is not None and o != b:
            if log is not None and o in resume:
                log(f"prefill slot {sl}: partial prompt of user {o} (carried at {resume.get(o)}) evicted by user {b}")
            if on_evict is not None:
                on_evict(o)
            resume.pop(o, None)
            of.pop(o, None)
        if b in of and of[b] != sl:
            owner[of[b]] = None
        owner[sl], of[b], out[b] = b, sl, sl
        taken.add(sl)
    return out
