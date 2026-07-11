# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
semaphore-handshake checker — Tensix semaphores + ATGETM/ATRELM mutexes.

Recall of two mechanical signals (the reliable ones per the audit):
  1. MUTEX_IMBALANCE — a function whose t6_mutex_acquire/ATGETM count does not
     equal its t6_mutex_release/ATRELM count (on the straight-line body). Strong
     signal; unbalanced acquire/release is a deadlock/leak risk.
  2. WAIT_WITHOUT_INIT — a semaphore is waited on but no SEMINIT/t6_semaphore_init
     appears anywhere in the parsed fact base. Candidate only: the BRISC boot
     firmware inits Max out-of-tree, so this is a flag for the LLM, not a verdict.

Everything else (post/get direction across threads, cross-layer producers,
deadlock cycles) is deferred — see blind_spots.
"""
from __future__ import annotations

from collections import defaultdict

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding


class SemaphoreHandshake(Check):
    name = "semaphore-handshake"
    description = "Semaphore/mutex balance + wait-without-init candidates"
    blind_spots = (
        "Cross-thread post/get direction, cross-layer producers (in ttnn/models, "
        "not tt-llk), and deadlock cycles are NOT decided here. INIT presence is "
        "checked only within the parsed tree — the BRISC boot firmware inits "
        "semaphore Max out-of-tree, so WAIT_WITHOUT_INIT is a candidate, not a bug. "
        "Branch-conditional acquire/release may show a false imbalance."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        # Gather semaphore ops (calls + macros), classified + attributed.
        # Skip the DEFINITIONS of the wrapper functions themselves (their body
        # contains the primitive by definition) and RAII ctor/dtor guards
        # (acquire-in-ctor / release-in-dtor is balanced at the object level).
        def skip(fact):
            fn = fact.get("function", "")
            return registry.is_semaphore_wrapper_def(fn) or registry.is_ctor_or_dtor(fn)

        ops = []  # (op, fact)
        for c in fb.family("call"):
            op = registry.classify_semaphore_call(c.get("name", ""))
            if op and not skip(c):
                ops.append((op, c))
        for m in fb.family("macro"):
            op = registry.classify_semaphore_macro(m.get("name", ""))
            if op and not skip(m):
                ops.append((op, m))

        findings: list[Finding] = []

        # 1) mutex acquire/release balance per function
        per_fn = defaultdict(lambda: {"acq": [], "rel": []})
        for op, f in ops:
            key = (f["file"], f.get("function", "?"))
            if op == "mutex_acquire":
                per_fn[key]["acq"].append(f)
            elif op == "mutex_release":
                per_fn[key]["rel"].append(f)
        for (file, fn), d in sorted(per_fn.items()):
            if len(d["acq"]) != len(d["rel"]) and (d["acq"] or d["rel"]):
                anchor = (d["acq"] or d["rel"])[0]
                findings.append(
                    Finding(
                        file=file,
                        line=anchor["line"],
                        function=fn,
                        kind="mutex_imbalance",
                        hint="MUTEX_IMBALANCE",
                        detail=f'{len(d["acq"])} acquire vs {len(d["rel"])} release in {fn}',
                        evidence=[self._ev(x, "acquire") for x in d["acq"]]
                        + [self._ev(x, "release") for x in d["rel"]],
                    )
                )

        # 2) wait-without-init, matched PER SEMAPHORE IDENTITY (not a single
        # global flag). Inits and waits use different vocabularies
        # (p_stall::SEMAPHORE_n vs semaphore::NAME) and a generic parameterized
        # init (t6_sem(index)) can init any semaphore, so:
        #   - a parameterized/generic init is a WILDCARD -> suppress all waits;
        #   - otherwise flag a wait whose identity has no matching concrete init.
        # Init identities are gathered from ALL init ops (incl. the wrapper
        # definitions), since that is where the generic init lives.
        init_ids: set[str] = set()
        wildcard = False
        for c in fb.family("call"):
            if registry.classify_semaphore_call(c.get("name", "")) == "init":
                sid, param = registry.semaphore_target(c)
                wildcard = wildcard or param
                if sid:
                    init_ids.add(sid)
        for m in fb.family("macro"):
            if registry.classify_semaphore_macro(m.get("name", "")) == "init":
                sid, param = registry.semaphore_target(m)
                wildcard = wildcard or param
                if sid:
                    init_ids.add(sid)

        if not wildcard:
            for op, f in ops:
                if op != "wait":
                    continue
                sid, param = registry.semaphore_target(f)
                if param or (sid and sid in init_ids):
                    continue
                findings.append(
                    Finding(
                        file=f["file"],
                        line=f["line"],
                        function=f.get("function", "?"),
                        kind="wait_without_init",
                        hint="WAIT_WITHOUT_INIT",
                        detail=f"wait on semaphore '{sid or '?'}' with no matching "
                        "SEMINIT in the parsed tree (BRISC boot firmware may init "
                        "out-of-tree)",
                        evidence=[self._ev(f, f.get("name", "wait"))],
                    )
                )
        return findings
