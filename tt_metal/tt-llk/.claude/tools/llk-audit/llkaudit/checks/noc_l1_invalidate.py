# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
noc-l1-invalidate checker (Blackhole) — a hand-rolled poll of a remotely-written L1
flag with no L1-cache invalidate.

On Blackhole the baby-RISC core caches L1 reads (write-through cache), so a NoC write
into L1 does NOT invalidate the core's cached copy: a repeated plain `*ptr` load can
return a stale cached value indefinitely (spin forever / read stale data). The
framework's `noc_semaphore_wait` places `invalidate_l1_cache()` inside its spin, so a
poll routed through it is safe; a HAND-ROLLED poll loop must call `invalidate_l1_cache()`
itself. On Wormhole this is a no-op (no such read cache), so the check is BH-only.

The extractor emits a `pointer_read` fact for a VOLATILE-pointer read inside a LOOP —
the busy-poll shape. Recall: on Blackhole, a function that has such a poll, does NOT
call `invalidate_l1_cache`, and is a dataflow-kernel poll of a remotely-written flag
(the poll pointer comes from `get_semaphore`, or the function issues NoC ops) → a
stale-cache candidate (MISSING_L1_INVALIDATE). The dataflow-kernel scoping excludes
LLK spin-waits on LOCAL status registers (no get_semaphore, no noc_*), so this is
empty over tt-llk. Requires a KERNEL fact base (the on-request capture).
"""

from __future__ import annotations

from collections import defaultdict

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding


class NocL1Invalidate(Check):
    name = "noc-l1-invalidate"
    description = (
        "Blackhole hand-rolled L1 poll missing invalidate_l1_cache (kernel tier)"
    )
    blind_spots = (
        "Blackhole-ONLY (the write-through L1 read cache is BH-specific; empty on WH/"
        "Quasar/sim). A `pointer_read` fact is emitted only for a VOLATILE-pointer read "
        "inside a loop — a non-volatile poll, or a poll spread across a helper call, is "
        "not seen. Clears a function that calls invalidate_l1_cache ANYWHERE (does not "
        "verify it is inside THIS poll's loop, nor before the specific re-read — a "
        "wrongly-placed invalidate is a false clear). Scopes to dataflow-kernel polls "
        "(poll pointer from get_semaphore, or the function issues NoC ops) to exclude "
        "LLK spin-waits on local status registers — so a hand-rolled poll of a remote "
        "flag whose address provenance did NOT resolve to get_semaphore AND whose "
        "function has no NoC call is MISSED. Whether the polled location is truly "
        "remotely-written (vs a local flag that needs no invalidate) is the skill's "
        "verdict. Requires a KERNEL fact base; empty over tt-llk."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        # The hazard is the Blackhole baby-RISC write-through L1 read cache; on other
        # archs invalidate_l1_cache is a no-op and the poll is safe.
        if fb.arch != "blackhole":
            return []
        reads_by_fn: dict = defaultdict(list)
        for r in fb.family("pointer_read"):
            reads_by_fn[(r["file"], r.get("function", "?"))].append(r)
        calls_by_fn: dict = defaultdict(list)
        for c in fb.family("call"):
            calls_by_fn[(c["file"], c.get("function", "?"))].append(c)

        findings: list[Finding] = []
        for (file, func), reads in sorted(reads_by_fn.items()):
            calls = calls_by_fn.get((file, func), [])
            names = [c.get("name", "") for c in calls]
            # Safe if the function invalidates the L1 cache (coarse — see blind_spots).
            if any("invalidate_l1_cache" in n for n in names):
                continue
            # Scope to a dataflow-kernel poll of a remotely-written flag: the poll
            # pointer is a get_semaphore result, OR the function issues NoC ops. This
            # excludes LLK spin-waits on local status regs (which are not L1 and need
            # no invalidate), keeping the check empty over tt-llk.
            sem_polls = [
                r for r in reads if "get_semaphore" in (r.get("producer", "") or "")
            ]
            has_noc = any(n.startswith("noc_") for n in names) or any(
                registry.noc_op_of(c) for c in calls
            )
            if not (sem_polls or has_noc):
                continue
            anchor = sem_polls[0] if sem_polls else reads[0]
            findings.append(
                Finding(
                    file=file,
                    line=anchor.get("line", 0),
                    function=func,
                    kind="missing_l1_invalidate",
                    hint="MISSING_L1_INVALIDATE",
                    detail=f"{func} hand-rolls a volatile L1 poll (via "
                    f"{anchor.get('producer','?')}) with no invalidate_l1_cache — on "
                    "Blackhole the baby-RISC L1 read cache can return a stale value "
                    "(use noc_semaphore_wait, or invalidate_l1_cache each iteration)",
                    evidence=[
                        self._ev(r, "volatile L1 poll (" + r.get("producer", "?") + ")")
                        for r in reads[:4]
                    ],
                )
            )
        return findings
