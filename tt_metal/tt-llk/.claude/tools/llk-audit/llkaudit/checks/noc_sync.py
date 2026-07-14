# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
noc-sync checker — NoC credit SIGNAL issued without a preceding write-flush.

The classic cross-core race: a producer writes data to a remote L1 with
`noc_async_write*`, then posts a remote credit — `noc_semaphore_inc` /
`inc_multicast` / `set_remote` / `set_multicast` (NOT the LOCAL `noc_semaphore_set`
reset) — but the signal must not overtake the data. It is safe only if a NoC write
completes before the signal. A WRITE credit (set/set_multicast/relay_*) same-NoC/VC
to the same dest is ordered by issue-order (any flush, or none, orders it). An
ATOMIC credit (inc/inc_multicast/remote up) is CLEARED only by a preceding ack
barrier — the CONSERVATIVE, doc-grounded choice: the data-before-credit race needs
the payload write COMMITTED (landed), which `noc_async_write_barrier` gives (ACK)
but `noc_async_writes_flushed` does NOT (it guarantees only DEPARTURE) — see the
data-movement doc `data_movement_doc/general/posted_writes.md`. So a flush-but-no-
barrier atomic is still flagged (never miss a real race) and TAGGED
`safety="FLUSH_NOT_BARRIER"`; whether a same-VC unicast atomic is actually safe with
only a flush is a VERDICT the /noc-sync-audit skill grounds in `<arch>/NoC/Ordering.md` +
`posted_writes.md`, NOT an assumption the tool bakes. Flushes are recognized in BOTH the
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
→ a data-before-credit candidate. The VERDICTS stay with the /noc-sync-audit skill (see
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
        "may be a false positive there (the /noc-sync-audit skill checks it against the ISA "
        "NoC Ordering doc). An atomic credit that HAS a preceding writes_flushed (but "
        "no barrier) is flagged with safety=FLUSH_NOT_BARRIER — surfaced for the skill "
        "to confirm against the NoC Ordering + data-movement docs, NOT pre-cleared as "
        "safe (writes_flushed = departure, not commit). "
        "A credit whose ONLY preceding flush is noc_async_posted_writes_flushed is "
        "flagged safety=POSTED_FLUSH_ONLY: that flush drains only the posted-writes HW "
        "counter, so it is a no-op (no ordering) when the credited write/inc was issued "
        "non-posted — the tool cannot see a write's posted-ness (a constexpr/template "
        "property), so the /noc-sync-audit skill confirms it. NOTE this fires only when the "
        "credit SIGNAL is visible in the kernel; a credit posted INSIDE a primitive "
        "(e.g. remote_cb_push_back_and_write_pages, whose internal write+inc are not "
        "kernel-level facts) is NOT surfaced — that posted/non-posted-flush pairing is "
        "left to the LLM. "
        "The checker models a NoC-WRITE-before-credit race: it treats a NoC "
        "flush/barrier as what orders the data, so it CLEARS a signal that a barrier "
        "precedes. It does NOT model the case where the credited data is a RISC L1 "
        "*store* (a plain/volatile store, then a remote inc/up or mailbox push that a "
        "consumer reads) — a different ordering domain (BabyRISC MemoryOrdering.md "
        "'Cross-core signalling') where a NoC barrier/flush is IRRELEVANT and the fix "
        "is load-back+consume, not a barrier. So a barrier-preceded signal that is "
        "really guarding a RISC store is a false-clear here — the /noc-sync-audit skill "
        "must check whether the producer is a RISC store. "
        "Cross-core / cross-kernel signal↔wait pairing (does a consumer actually "
        "wait on this semaphore?) and multicast fan-out (inc count == number of "
        "destinations) are NOT decided here — the /noc-sync-audit skill owns them. "
        "File-scope signals outside any function are not attributed. A flush at "
        "the TOP of a loop precedes a later same-iteration signal textually but "
        "drains the PRIOR iteration's write, not this one's — the byte-offset "
        "heuristic can't see that (LLM must widen). A credit posted via "
        "noc_inline_dw_write (a direct dword write to a remote semaphore) is NOT "
        "recognized as a signal (it is also used for non-credit register writes, "
        "so including it would over-flag). "
        "Facts are grouped by the (file, function-NAME) string, so two "
        "lexically-distinct bodies that share a name in one file — multiple "
        "lambdas (each `operator()`) or C++ overloads — merge into one bucket, and "
        "a flush/barrier in ONE body then clears a signal in the OTHER (a "
        "cross-scope false-clear); the tool has no per-lambda / per-overload "
        "identity (semaphore-handshake and cb-sync note the same grouping "
        "limitation). Requires a KERNEL fact base (the "
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
            # NON-posted departure-or-stronger flushes clear a WRITE credit. A
            # posted-only flush (noc_flush_kind == "posted") is DELIBERATELY excluded:
            # it drains the wrong HW counter and is a no-op for non-posted writes/incs,
            # so it must never clear a credit — it is only tracked (posted_flush_offs)
            # to TAG a would-be-flagged credit (POSTED_FLUSH_ONLY) for the skill.
            flush_offs = [
                c["off"]
                for c in calls
                if registry.noc_flush_kind(c) in ("barrier", "flushed")
            ]
            posted_flush_offs = [
                c["off"] for c in calls if registry.noc_flush_kind(c) == "posted"
            ]
            for c in calls:
                op = registry.noc_op_of(c)
                if op not in ("inc", "set", "mcast"):
                    continue
                # An ATOMIC credit (inc / inc_multicast / remote up) is cleared only
                # by a preceding ack BARRIER — the conservative choice: the credit
                # needs the write COMMITTED, and writes_flushed guarantees only
                # departure, not commit (posted_writes.md). Whether same-VC unicast is
                # safe with just a flush is the skill's doc-grounded verdict, not ours.
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
                # (just not a barrier) — surface it and tag it so the skill confirms
                # (against NoC Ordering + posted_writes docs) rather than reads it as a
                # hard race; do NOT pre-declare it safe (departure != commit).
                safety = ""
                if atomic and any(fo < c["off"] for fo in flush_offs):
                    safety = "FLUSH_NOT_BARRIER"
                elif any(fo < c["off"] for fo in posted_flush_offs):
                    # The ONLY preceding flush is a posted-writes flush — it drains the
                    # posted counter, a no-op if the credited write/inc was issued
                    # non-posted (the common default). Surface it so the skill confirms
                    # the write's posted-ness (dataflow-API posted/non-posted counters);
                    # do NOT treat the posted-flush as ordering (silent false-clear).
                    safety = "POSTED_FLUSH_ONLY"
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
                            "; only a POSTED-writes flush precedes it — a no-op if the "
                            "write/inc is non-posted (wrong HW counter); confirm the "
                            "write's posted-ness, don't assume ordered)"
                            if safety == "POSTED_FLUSH_ONLY"
                            else (
                                "; a writes_flushed precedes it (departure, not commit) — "
                                "confirm vs NoC/Ordering.md + posted_writes.md, don't assume safe)"
                                if safety == "FLUSH_NOT_BARRIER"
                                else (
                                    "; atomic credit needs the write COMMITTED (ack barrier))"
                                    if atomic
                                    else ")"
                                )
                            )
                        ),
                        evidence=[self._ev(c, c.get("name", ""))],
                        safety=safety,
                    )
                )
        return findings
