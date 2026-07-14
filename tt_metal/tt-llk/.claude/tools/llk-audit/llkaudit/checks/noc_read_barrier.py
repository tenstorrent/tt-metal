# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
noc-read-barrier checker — an inbound NoC read consumed before its read-barrier.

The read-side dual of noc-sync's write→credit race. `noc_async_read*` issues an
ASYNCHRONOUS read into local L1; the data is present only after `noc_async_read_
barrier()` drains it (which also invalidates the L1 cache on BH). If the buffer is
CONSUMED before that barrier — handed to compute via `cb_push_back`, or forwarded by
`tt_memmove` (whose aligned path is itself a NoC write sourcing the read buffer) —
the consumer reads stale/partial L1. The correct idiom is
`noc_async_read(...); noc_async_read_barrier(); <consume>`.

Recall: for each inbound read, find its NEXT read-barrier; if a consumer sits between
the read and that barrier (or after the read when no barrier follows), emit a
candidate at the read (READ_CONSUMED_BEFORE_BARRIER). The common BATCH-then-drain
idiom (`read; read; barrier; consume; consume`) is NOT flagged — the consumers fall
after the barrier. Empty over tt-llk; requires a KERNEL fact base.
"""

from __future__ import annotations

from collections import defaultdict

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding


class NocReadBarrier(Check):
    name = "noc-read-barrier"
    description = "Inbound NoC read consumed before its read-barrier (kernel tier)"
    blind_spots = (
        "Does NOT verify that the consumed CB / forwarded buffer is actually the "
        "read's DESTINATION — a kernel that reads buffer A and, before A's barrier, "
        "pushes an UNRELATED buffer B that was filled+drained earlier is a false "
        "candidate (no dataflow / buffer-identity tracking). The consumer set is "
        "cb_push_back (free + object API) and tt_memmove ONLY; a forward via a plain "
        "noc_async_write or an unmodeled helper is a MISS (a plain write's source may "
        "be unrelated, so including it would over-flag). A read-barrier reached only "
        "on another control-flow path than the consumer is matched textually, not by "
        "reachability. A bare noc_async_reads_flushed is not treated as the drain "
        "(it does not invalidate the BH L1 cache) — such a kernel is surfaced "
        "conservatively. Whether the consumer truly reads the not-yet-landed bytes is "
        "the /noc-sync-audit (read-side) skill's verdict. The read matcher "
        "(registry.noc_is_read) matches ANY noc_async_read* name except barrier/"
        "flush forms, so the NON-ISSUING state/counter helpers "
        "(noc_async_read_set_state, noc_async_read_inc_num_issued) are mis-read as "
        "inbound reads and can raise a SPURIOUS candidate — only the issuing forms "
        "actually fill L1 (the LLM refutes a set_state / inc-num-issued candidate). "
        "Facts are grouped by the (file, function-NAME) string, so two same-named "
        "bodies in one file (multiple lambdas — each `operator()` — or C++ "
        "overloads) merge into one bucket, and a read-barrier in ONE body can "
        "clear a read in the OTHER (a cross-scope false-clear); semaphore-handshake "
        "and cb-sync note the same grouping limitation. "
        "Requires a KERNEL fact base; empty over tt-llk."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        findings: list[Finding] = []
        byfn: dict = defaultdict(list)
        for c in fb.family("call"):
            byfn[(c["file"], c.get("function", "?"))].append(c)
        for (file, func), calls in sorted(byfn.items()):
            calls.sort(key=lambda c: c["off"])
            reads = [c for c in calls if registry.noc_is_read(c)]
            if not reads:
                continue
            barrier_offs = sorted(
                c["off"] for c in calls if registry.noc_is_read_barrier(c)
            )
            consumers = [c for c in calls if registry.is_read_consumer(c)]
            for r in reads:
                # the read's NEXT read-barrier (the drain that would make it safe)
                nb = next((b for b in barrier_offs if b > r["off"]), None)
                hi = nb if nb is not None else float("inf")
                between = [c for c in consumers if r["off"] < c["off"] < hi]
                if not between:
                    continue
                findings.append(
                    Finding(
                        file=file,
                        line=r.get("line", 0),
                        function=func,
                        kind="read_consumed_before_barrier",
                        hint="READ_CONSUMED_BEFORE_BARRIER",
                        detail=f"{r.get('name','')} is consumed by "
                        f"{between[0].get('name','')} before a noc_async_read_barrier "
                        f"drains it in {func} (read may not have landed in L1)",
                        evidence=[self._ev(r, r.get("name", "") + " (inbound read)")]
                        + [
                            self._ev(
                                between[0], between[0].get("name", "") + " (consumer)"
                            )
                        ],
                    )
                )
        return findings
