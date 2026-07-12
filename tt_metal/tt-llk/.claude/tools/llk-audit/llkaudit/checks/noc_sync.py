# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
noc-sync checker — NoC credit SIGNAL issued without a preceding write-flush.

The classic cross-core race: a producer writes data to a remote L1 with
`noc_async_write*`, then posts a remote credit — `noc_semaphore_inc` /
`inc_multicast` / `set_remote` / `set_multicast` (NOT the LOCAL `noc_semaphore_set`
reset) — but the signal must not overtake the data. It is safe only if a NoC
write flush/barrier (`noc_async_write_barrier` / `noc_async_writes_flushed`)
drains the write BEFORE the signal — recognized in BOTH the free-function form and
the modern `Noc`-method form (`noc.async_write_barrier()`) via registry.noc_op_of.
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
        "Only checks that SOME flush precedes the signal in the SAME function — it "
        "does NOT verify the flush covers the specific buffer/transaction the "
        "signal credits, nor a flush supplied by a CALLER (shows as a candidate). "
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
            # Noc-method write-flush (noc.async_write_barrier()).
            flush_offs = [c["off"] for c in calls if registry.noc_op_of(c) == "flush"]
            for c in calls:
                op = registry.noc_op_of(c)
                if op not in ("inc", "set", "mcast"):
                    continue
                if any(fo < c["off"] for fo in flush_offs):
                    continue  # a write flush precedes this signal — ok (candidate-safe)
                findings.append(
                    Finding(
                        file=c["file"],
                        line=c.get("line", 0),
                        function=func,
                        kind=f"noc_signal:{op}",
                        hint="NOC_SIGNAL_NO_FLUSH",
                        detail=f"{c.get('name','')} with no preceding "
                        f"noc_async_write_barrier/writes_flushed in {func} "
                        "(data-before-credit: signal may overtake the write)",
                        evidence=[self._ev(c, c.get("name", ""))],
                    )
                )
        return findings
