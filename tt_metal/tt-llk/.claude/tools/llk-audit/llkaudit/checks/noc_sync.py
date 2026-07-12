# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
noc-sync checker — NoC credit SIGNAL issued without a preceding write-flush.

The classic cross-core race: a producer writes data to a remote L1 with
`noc_async_write*`, then posts a remote credit — `noc_semaphore_inc` /
`inc_multicast` / `set_remote` / `set_multicast` (NOT the LOCAL `noc_semaphore_set`
reset) — but the signal must not overtake the data. It is safe only if a NoC write
completes before the signal. A WRITE credit (set/set_multicast/relay_*) is a write,
so any flush orders it (`noc_async_write_barrier` OR `noc_async_writes_flushed`). An
ATOMIC credit (inc/inc_multicast/remote up) is CLEARED only by a preceding ack
barrier — the CONSERVATIVE choice: `writes_flushed` only guarantees the write
DEPARTED (dataflow-API semantics), not that it LANDED, and departure-order suffices
only for a same-NoC/VC unicast credit (in-issue-order delivery, per the ISA
"<arch>/NoC/Ordering.md" doc), not for a multicast / cross-command-buffer / cross-VC
credit. So a flush-but-no-barrier atomic is still flagged (never miss a real race)
but TAGGED `safety="FLUSH_NOT_BARRIER"` (likely the same-VC-unicast-safe idiom — the
skill adjudicates via the NoC Ordering doc). Flushes are recognized in BOTH the
free-function and the modern `Noc`-method form (`noc.async_write_barrier()`) via
registry.noc_op_of/noc_flush_kind.
Signals are recognized in both forms too: the free functions AND the `Semaphore`
object methods (`sem.set_multicast(...)`, remote `sem.up(noc, x, y, v)`); the LOCAL
`up(value)`/`set(value)` forms are excluded (no NoC, no flush needed). This surface
lives in JIT-compiled kernels OUTSIDE tt-llk, so
the checker is committed and deterministic but only yields findings when fed a
KERNEL fact base (the on-request capture — see run.sh --full-jit); empty over
tt-llk.

Pure AUGMENTOR: the mechanical, in-one-function signal it recalls is a credit
signal (`inc`/`set`/multicast) with NO write flush before it in the same function
→ a data-before-credit candidate. The VERDICTS stay with the /noc-sync skill (see
blind_spots): which buffer the flush actually covers, cross-core/cross-kernel
signal↔wait pairing, and multicast fan-out count vs destination count.
"""
from __future__ import annotations

from collections import defaultdict

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding


class NocSync(Check):
    name = "noc-sync"
    description = "NoC credit signal missing a preceding write-flush (kernel tier)"
    blind_spots = (
        "Checks that a suitable flush precedes the signal in the SAME function — an "
        "ACK barrier for an atomic credit (inc), any write flush for a write credit "
        "(set/mcast) — but does NOT verify the flush covers the specific buffer/"
        "transaction the signal credits, nor a flush supplied by a CALLER (shows as "
        "a candidate). It also does not model same-VC transit ordering, which can "
        "make even a missing flush safe for a write credit — so a set/mcast candidate "
        "may be a false positive there (the /noc-sync skill checks it against the ISA "
        "NoC Ordering doc). An atomic credit that HAS a preceding writes_flushed (but "
        "no barrier) is flagged with safety=FLUSH_NOT_BARRIER — safe for a same-NoC/VC "
        "unicast credit, needs the barrier only for multicast/cross-cmd-buffer/cross-VC. "
        "Cross-core / cross-kernel signal↔wait pairing (does a consumer actually "
        "wait on this semaphore?) and multicast fan-out (inc count == number of "
        "destinations) are NOT decided here — the /noc-sync skill owns them. "
        "File-scope signals outside any function are not attributed. A flush at "
        "the TOP of a loop precedes a later same-iteration signal textually but "
        "drains the PRIOR iteration's write, not this one's — the byte-offset "
        "heuristic can't see that (LLM must widen). A credit posted via "
        "noc_inline_dw_write (a direct dword write to a remote semaphore) is NOT "
        "recognized as a signal (it is also used for non-credit register writes, "
        "so including it would over-flag). Requires a KERNEL fact base (the "
        "on-request capture); empty over tt-llk."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        findings: list[Finding] = []
        # Group by the fact's recorded (innermost) `function` field — NOT by
        # iterating fb.functions + facts_in(), which selects by offset range and so
        # returns a call nested in a lambda for BOTH the lambda and the outer
        # function (double emission, and a cross-scope false NO_FLUSH). cb_sync uses
        # the same per-`function` grouping.
        byfn: dict = defaultdict(list)
        for c in fb.family("call"):
            byfn[(c["file"], c.get("function", "?"))].append(c)
        for (_file, func), calls in sorted(byfn.items()):
            calls.sort(key=lambda c: c["off"])
            # noc_op_of is fact-aware: free-function signals/flushes AND the
            # Noc-method write-flush (noc.async_write_barrier()). Distinguish the
            # ACK barrier (data LANDED) from a bare writes_flushed (data LEFT).
            barrier_offs = [
                c["off"] for c in calls if registry.noc_flush_kind(c) == "barrier"
            ]
            flush_offs = [c["off"] for c in calls if registry.noc_op_of(c) == "flush"]
            for c in calls:
                op = registry.noc_op_of(c)
                if op not in ("inc", "set", "mcast"):
                    continue
                # An ATOMIC credit (inc / inc_multicast / remote up) is cleared only
                # by a preceding ack BARRIER — the conservative choice (writes_flushed
                # guarantees departure, not landing; departure-order suffices only for
                # a same-NoC/VC unicast credit, not multicast/cross-cmd-buf/cross-VC).
                # A write credit (set / set_multicast / relay_*) is ordered by ANY
                # flush. The op label (mcast) doesn't decide this — inc_multicast is
                # atomic.
                atomic = registry.noc_signal_is_atomic(c)
                if atomic:
                    clearing, need = barrier_offs, "noc_async_write_barrier"
                else:
                    clearing, need = (
                        flush_offs,
                        "noc_async_write_barrier/writes_flushed",
                    )
                if any(fo < c["off"] for fo in clearing):
                    continue  # a suitable flush precedes this signal — candidate-safe
                # A flagged ATOMIC credit that DOES have a preceding writes_flushed
                # (just not a barrier) is likely the same-NoC/VC-unicast-safe idiom —
                # surface it, but tag low-confidence so it's triaged, not read as a
                # hard race. The skill confirms the path against the NoC Ordering doc.
                safety = ""
                if atomic and any(fo < c["off"] for fo in flush_offs):
                    safety = "FLUSH_NOT_BARRIER"
                findings.append(
                    Finding(
                        file=c["file"],
                        line=c.get("line", 0),
                        function=func,
                        kind=f"noc_signal:{op}",
                        hint="NOC_SIGNAL_NO_FLUSH",
                        detail=f"{c.get('name','')} with no preceding {need} in {func} "
                        "(data-before-credit: signal may overtake the write"
                        + (
                            "; a writes_flushed precedes it — safe for a same-NoC/VC "
                            "unicast credit, needs the ack barrier for multicast/"
                            "cross-cmd-buffer/cross-VC)"
                            if safety
                            else (
                                "; atomic credit needs the ACK barrier)"
                                if atomic
                                else ")"
                            )
                        ),
                        evidence=[self._ev(c, c.get("name", ""))],
                        safety=safety,
                    )
                )
        return findings
