# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
noc-sync checker — NoC credit SIGNAL issued without a preceding write-flush.

The classic cross-core race: a producer writes data to a remote L1 with
`noc_async_write*`, then signals the consumer via `noc_semaphore_inc/set` — but
the signal must not overtake the data. It is safe only if a NoC write flush/
barrier (`noc_async_write_barrier` / `noc_async_writes_flushed`) drains the write
BEFORE the signal. This surface lives in JIT-compiled kernels OUTSIDE tt-llk, so
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
        "File-scope signals outside any function are not attributed. Requires a "
        "KERNEL fact base (the on-request capture); empty over tt-llk."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        findings: list[Finding] = []
        for fn in fb.functions:
            calls = fb.facts_in(fn, ("call",))
            flush_offs = [
                c["off"] for c in calls if registry.noc_op(c.get("name", "")) == "flush"
            ]
            for c in calls:
                op = registry.noc_op(c.get("name", ""))
                if op not in ("inc", "set", "mcast"):
                    continue
                if any(fo < c["off"] for fo in flush_offs):
                    continue  # a write flush precedes this signal — ok (candidate-safe)
                findings.append(
                    Finding(
                        file=c["file"],
                        line=c["line"],
                        function=fn.name,
                        kind=f"noc_signal:{op}",
                        hint="NOC_SIGNAL_NO_FLUSH",
                        detail=f"{c.get('name','')} with no preceding "
                        f"noc_async_write_barrier/writes_flushed in {fn.name} "
                        "(data-before-credit: signal may overtake the write)",
                        evidence=[self._ev(c, c.get("name", ""))],
                    )
                )
        return findings
