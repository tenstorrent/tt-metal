# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
noc-atomic-exit checker — a non-posted NoC ATOMIC left in flight at kernel exit.

A remote atomic credit (`noc_semaphore_inc` / `Semaphore::up(noc, ...)`) is tracked
by its own HW counter (noc_nonposted_atomics_acked), distinct from writes. Only
`noc_async_atomic_barrier()` (or the all-draining `noc_async_full_barrier()`) drains
it — a write barrier / writes_flushed does not.
The post-`kernel_main` firmware epilogue does NOT drain in-flight atomics in
release / Watcher-off builds (it only ASSERTs idle under DM_DEDICATED_NOC), so a
kernel that issues an atomic and returns without an atomic barrier leaves it in
flight: on Watcher builds the NOC-idle ASSERT can trip, and the readiness signal it
posts can be delayed past program teardown / noc_init across back-to-back launches.

Recall: for each KERNEL ENTRY (`kernel_main`), if it issues an atomic credit and NO
`noc_async_atomic_barrier` follows the LAST such atomic before the function ends,
emit one candidate at that last atomic (NO_ATOMIC_BARRIER_AT_EXIT). A barrier after
the last atomic clears it (it drains all outstanding atomics). Empty over tt-llk
(no kernel_main); requires a KERNEL fact base (the on-request capture).
"""

from __future__ import annotations

from collections import defaultdict

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding


class NocAtomicExit(Check):
    name = "noc-atomic-exit"
    description = (
        "Non-posted NoC atomic left in flight at kernel exit (no atomic barrier)"
    )
    blind_spots = (
        "Scoped to functions named `kernel_main` (a DM kernel's entry) — an atomic "
        "issued in a HELPER called near the tail of kernel_main is not attributed to "
        "the entry, and an atomic barrier supplied by a CALLER/callee is not seen "
        "(both can yield a false candidate or a miss). Only checks that SOME atomic "
        "barrier follows the LAST atomic textually — it does not model a barrier that "
        "is control-flow-unreachable from the atomic (e.g. under a branch not taken). "
        "It does not distinguish an atomic that is deliberately left for the NEXT "
        "kernel to drain (rare, and still violates the NOC-idle contract). Whether "
        "the missing drain actually matters (Watcher build vs release, program reuse) "
        "is a severity call for the /noc-sync-audit skill. Requires a KERNEL fact base; "
        "empty over tt-llk."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        findings: list[Finding] = []
        byfn: dict = defaultdict(list)
        for c in fb.family("call"):
            byfn[(c["file"], c.get("function", "?"))].append(c)
        for (file, func), calls in sorted(byfn.items()):
            if not registry.is_kernel_entry(func):
                continue
            atomic_sigs = [c for c in calls if registry.noc_signal_is_atomic(c)]
            if not atomic_sigs:
                continue
            last = max(atomic_sigs, key=lambda c: c["off"])
            # A barrier AFTER the last atomic drains all outstanding atomics -> safe.
            if any(
                registry.noc_is_atomic_barrier(c) and c["off"] > last["off"]
                for c in calls
            ):
                continue
            findings.append(
                Finding(
                    file=file,
                    line=last.get("line", 0),
                    function=func,
                    kind="atomic_no_exit_barrier",
                    hint="NO_ATOMIC_BARRIER_AT_EXIT",
                    detail=f"{last.get('name','')} (non-posted atomic) is the last atomic "
                    f"in {func} with no following noc_async_atomic_barrier — the atomic "
                    "is in flight at kernel exit (write/flush barriers do NOT drain the "
                    "atomic counter; only noc_async_atomic_barrier or noc_async_full_barrier does)",
                    evidence=[self._ev(last, last.get("name", ""))],
                )
            )
        return findings
