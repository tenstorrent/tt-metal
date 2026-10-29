# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler-driven chunked prefill for batched TP serving: the host-side row plan (no ttnn, host-testable).

vLLM (with the plugin's chunk policy, QWEN36_CHUNKED_PREFILL=1) may split one long prompt into 2048-aligned chunks that
run in separate prefill calls, with decode steps and other prompts' prefills in between. The per-sequence GDN state of
the partial prompt (recurrent state, conv taps, cross-chunk conv carry) lives in the persistent B=1 prefill scratch
between its calls (its decode slot row is NOT safe storage: the batched decode rewrites idle rows inside the pow2 bucket
and slot remaps gather every row). Any other prompt prefilled while a partial is in flight resets that scratch, so the
partial's state is first copied to a same-shape park buffer ("park") and copied back before its next chunk ("unpark").

``ChunkedPrefillPlanner.plan`` turns one prefill call's rows into an execution plan and the scratch owner that results:

* a row is a RESUME row only when the caller flags it (``resume_mask``; the plugin sets it for a scheduler chunk
  continuation). A resume row must continue the current owner exactly: same first KV block, ``start == next_pos``,
  ``start`` a positive multiple of the chunk size. Anything else raises (never silently re-prefill with a wrong state).
* every other row re-prefills from position 0 (``start`` ignored), which is the pre-chunking behaviour for any row.
* an intermediate row (``final_mask`` False) must end on a chunk boundary; it becomes the scratch owner, writes no
  decode slot and reads back no logits.
* the resume row runs first. Before any row that resets the scratch (every non-resume row) an unparked owner is parked;
  before a resume row a parked owner is unparked. So a call that carries only short prompts between two chunks of the
  partial parks it too (review B1), and a stale owner left by an aborted/preempted partial costs one park at most.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ScratchOwner:
    """The partial prompt whose GDN state the B=1 prefill scratch (or its park buffer) holds."""

    first_block: int  # page_table[row, 0] of the request: its identity for the continuation check
    next_pos: int  # tokens [0, next_pos) are in the paged KV and in the state; the next chunk must start here
    parked: bool  # True: the state is in the park buffer (the scratch was reused since)


@dataclass(frozen=True)
class RowPlan:
    row: int  # index into the call's rows (logits are returned in call order)
    start: int  # absolute first position this call computes (0 unless resume)
    end: int  # exclusive end (the chunk end / prompt length)
    resume: bool
    final: bool  # False = intermediate chunk: no slot write, no logits readback
    park_before: bool
    unpark_before: bool


class ChunkedPrefillPlanner:
    def __init__(self, chunk_tokens):
        if int(chunk_tokens) <= 0:
            raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
        self.chunk = int(chunk_tokens)
        self.owner = None  # ScratchOwner | None

    def plan(self, starts, ends, resume_mask, final_mask, first_blocks):
        """Return (list[RowPlan] in execution order, owner after the call). Does not modify ``self.owner``."""
        n = len(ends)
        for name, seq in (("starts", starts), ("resume_mask", resume_mask), ("final_mask", final_mask)):
            if len(seq) != n:
                raise ValueError(f"chunked prefill: {name} has {len(seq)} entries for {n} rows")
        if len(first_blocks) != n:
            raise ValueError(f"chunked prefill: first_blocks has {len(first_blocks)} entries for {n} rows")
        C = self.chunk
        resume_rows = [u for u in range(n) if resume_mask[u]]
        if len(resume_rows) > 1:
            raise ValueError(f"chunked prefill: {len(resume_rows)} resume rows {resume_rows} in one call (max 1)")
        new_partials = [u for u in range(n) if not resume_mask[u] and not final_mask[u]]
        if len(new_partials) + (1 if resume_rows and not final_mask[resume_rows[0]] else 0) > 1:
            raise ValueError(
                f"chunked prefill: more than one partial prompt in one call (resume rows {resume_rows}, "
                f"new intermediate rows {new_partials}); the scheduler admits one long prompt at a time"
            )
        for u in range(n):
            if int(ends[u]) < 1:
                raise ValueError(f"chunked prefill: row {u} has end {ends[u]}")
            if not final_mask[u] and int(ends[u]) % C != 0:
                raise ValueError(
                    f"chunked prefill: intermediate row {u} ends at {ends[u]}, not a multiple of the chunk {C}"
                )
        owner = self.owner
        if resume_rows:
            u = resume_rows[0]
            s = int(starts[u])
            if s <= 0 or s % C != 0 or s >= int(ends[u]):
                raise ValueError(
                    f"chunked prefill: resume row {u} has start {s}, end {ends[u]}: the start must be a positive "
                    f"multiple of {C} below the end"
                )
            if owner is None or owner.first_block != int(first_blocks[u]) or owner.next_pos != s:
                raise ValueError(
                    f"chunked prefill: resume row {u} (first_block={int(first_blocks[u])}, start={s}) does not continue "
                    f"the scratch owner {owner}"
                )
        order = resume_rows + [u for u in range(n) if u not in resume_rows]
        plans = []
        for u in order:
            resume = bool(resume_mask[u])
            final = bool(final_mask[u])
            park = unpark = False
            if resume:
                unpark = owner.parked
                start = int(starts[u])
            else:
                park = owner is not None and not owner.parked
                if park:
                    owner = replace(owner, parked=True)
                start = 0
            plans.append(RowPlan(u, start, int(ends[u]), resume, final, park, unpark))
            if final:
                if resume:
                    owner = None
            else:
                owner = ScratchOwner(int(first_blocks[u]), int(ends[u]), parked=False)
        return plans, owner
