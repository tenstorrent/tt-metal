# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
cb-sync checker — circular-buffer producer/consumer credit balance.

CB flow control: a producer `cb_reserve_back(cb, n)` claims space and
`cb_push_back(cb, n)` commits it (hands the consumer a credit); a consumer
`cb_wait_front(cb, n)` waits and `cb_pop_front(cb, n)` releases. Modern ttnn
kernels use the OBJECT form (`CircularBuffer cb(id); cb.reserve_back(n)`) — both
are handled via registry.cb_classify (method form is grouped by the RECEIVER
object and gated on the CircularBuffer type). The CB surface
lives in JIT-compiled kernels OUTSIDE tt-llk, so this checker is committed and
deterministic but only yields findings when fed a KERNEL fact base (the
on-request capture — see run.sh --full-jit / the race-audit-all runbook); over
the tt-llk fact base there are no cb_* calls, so it is trivially empty.

Pure AUGMENTOR: the mechanical, in-one-function signal it recalls is a
reserve/push or wait/pop COUNT imbalance per CB — a leaked or missing credit.
The load-bearing VERDICTS stay with the /dataflow-cb-sync skill (see
blind_spots): cross-kernel producer↔consumer balance, the data-before-credit
NoC-flush ordering, capacity vs num_pages, and single-producer/consumer.
"""
from __future__ import annotations

from collections import defaultdict

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding


class CbSync(Check):
    name = "cb-sync"
    description = "Circular-buffer reserve/push & wait/pop credit balance (kernel tier)"
    blind_spots = (
        "Balance is checked WITHIN one function only; the load-bearing CROSS-KERNEL "
        "balance (producer total push == consumer total pop over the program) is "
        "NOT decided here. The data-before-credit ordering (a NoC write flush/"
        "barrier before cb_push_back) is NOT checked — that is noc-sync's join. "
        "Capacity (pages reserved <= CB depth / num_pages), single-producer/"
        "single-consumer, the page COUNT argument (reserve n vs push n), and "
        "branch-conditional reserve/push (which can show a false imbalance) are "
        "deferred. The remote/sharded CB family IS balance-checked (remote_cb_* in "
        "CB_CALLS; its push is the fused remote_cb_push_back_and_write_pages) — but "
        "its data-before-credit ordering is noc-sync's job. Requires a KERNEL fact "
        "base (the on-request capture); empty over tt-llk."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        # Group cb_* calls per (file, function, cb-id). The CB id is arg0 (a
        # literal index, a CBIndex enum, or a variable) — grouped by its text.
        groups: dict = defaultdict(lambda: defaultdict(list))
        for c in fb.family("call"):
            # cb_classify handles BOTH the free-function API (cb id = arg0) and the
            # object/method API (cb id = the receiver object), gated on the CB type.
            op, cbid = registry.cb_classify(c)
            if not op:
                continue
            groups[(c["file"], c.get("function", "?"))][cbid].append((op, c))

        findings: list[Finding] = []
        for (file, func), bycb in sorted(groups.items()):
            for cbid, ops in sorted(bycb.items()):
                sites = {
                    r: [f for o, f in ops if o == r]
                    for r in ("reserve", "push", "wait", "pop")
                }
                # producer side: reserve must match push
                self._imbalance(
                    findings,
                    file,
                    func,
                    cbid,
                    sites,
                    "reserve",
                    "push",
                    "CB_RESERVE_PUSH_IMBALANCE",
                    "reserve_back",
                    "push_back",
                )
                # consumer side: wait must match pop
                self._imbalance(
                    findings,
                    file,
                    func,
                    cbid,
                    sites,
                    "wait",
                    "pop",
                    "CB_WAIT_POP_IMBALANCE",
                    "wait_front",
                    "pop_front",
                )
        return findings

    def _imbalance(self, findings, file, func, cbid, sites, a, b, hint, na, nb):
        fa, fb_ = sites[a], sites[b]
        if len(fa) != len(fb_) and (fa or fb_):
            anchor = (fa or fb_)[0]
            findings.append(
                Finding(
                    file=file,
                    line=anchor.get("line", 0),
                    function=func,
                    kind=f"{hint.lower()}@{cbid}",
                    hint=hint,
                    detail=f"CB {cbid}: {len(fa)} {na} vs {len(fb_)} {nb} in {func} "
                    "(within-function credit imbalance)",
                    evidence=[self._ev(f, na) for f in fa]
                    + [self._ev(f, nb) for f in fb_],
                )
            )
