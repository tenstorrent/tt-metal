# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
tensix-risc-order checker — a Tensix config/GPR write, then a RISC MMIO store to
the SAME register on the same thread, with nothing that makes the RISC wait for
the Tensix write to land.

A Tensix write (WRCFG / REG2FLOP / RMWCIB / cfg_reg_rmw_tensix / SETDMAREG /
CFGSHIFTMASK) is queued in the thread's instruction stream; a RISC `cfg[...] =`
/ `regfile[...] =` store lands at once. If the Tensix write is still queued when
the store lands (held behind a STALLWAIT, a stalled SEMPOST/SEMGET, a busy unit),
it lands LAST and overwrites the store (write-after-write). What orders them:

  * `tensix_sync()` before the store (in its function, or in every caller
    before the call): the RISC blocks until the whole thread is idle.
  * a RISC poll (`semaphore_read`, e.g. wait_for_idle on UNPACK_SYNC) before the
    store, where the Tensix write is issued BEFORE a SEMGET of that semaphore
    (in its function or in every caller): the poll cannot return until that
    SEMGET, and so the write, has executed.

Per RISC store with at least one unordered same-register Tensix writer:
  TENSIX_WRITE_SAME_FN     — the writer is earlier in the store's own function
                             (order certain) with no drain between.
  TENSIX_WRITE_AFTER_RELEASE — the store is behind a poll, but the writer is
                             issued after the SEMGET that releases it (the poll
                             can return while the writer is still queued).
  TENSIX_WRITE_NO_RELEASE  — the writer sits in a function that never releases
                             the polled semaphore after it (init / reconfig /
                             uninit paths), or the store has no poll at all; it
                             races if a kernel can issue that function and then
                             reach the store while the write is still held.
The hint is the strongest class present; every writer is listed as evidence,
marked "(held: after <stall>)" when a stall precedes it in its own function.
"""

from __future__ import annotations

from collections import defaultdict

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding

_RANK = {
    "TENSIX_WRITE_SAME_FN": 3,
    "TENSIX_WRITE_AFTER_RELEASE": 2,
    "TENSIX_WRITE_NO_RELEASE": 1,
}


class TensixRiscOrder(Check):
    name = "tensix-risc-order"
    description = (
        "Tensix config/GPR write then RISC MMIO store to the same register, unordered"
    )
    blind_spots = (
        "Call order ACROSS API calls is not known statically: a NO_RELEASE writer is "
        "a candidate until a kernel is shown to issue it and then reach the store "
        "while it is held — the skill traces that at kernel level. Coverage by a "
        "release or a tensix_sync() is checked in the function and up to two caller "
        "levels by name (same-name overloads are merged). RISC config READS (a "
        "save/restore or a software read-modify-write that reads a word a queued "
        "Tensix write has not landed in yet) are not in the fact base and are not "
        "seen. Thread is inferred from the file name; cross-thread pairs belong to "
        "cfg-word-overlap. A RISC store with a runtime word offset "
        "(`FIELD_ADDR32 + i`) is matched by REG prefix, which can over-match. "
        "Value-invariant pairs (both sides write the same value) are reported; the "
        "skill clears them. MOP/replay-slotted writes are matched by their source "
        "line, not their execution slot. Quasar's TTSync holds a RISC config store "
        "behind older queued config-touching Tensix instructions, so the shape "
        "cannot race there."
    )

    CALLER_DEPTH = 2

    def run(self, fb: FactBase) -> list[Finding]:
        self.fb = fb
        self.callers = defaultdict(list)  # callee name -> call facts
        for c in fb.family("call"):
            self.callers[c.get("name", "")].append(c)

        # semaphores a RISC poll reads, and the functions holding a poll
        self.gate_sem: dict[str, set] = defaultdict(set)
        for c in fb.family("call"):
            if c.get("name", "") in registry.RISC_POLL_CALLS:
                sid, param = registry.semaphore_target(c)
                if sid and not param:
                    self.gate_sem[c.get("function", "")].add(sid)

        # releases (Tensix SEMGET) per function
        self.releases = defaultdict(list)  # (file, begin_off, sid) -> [off]
        for c in fb.family("call"):
            if (
                registry.classify_semaphore_call(c.get("name", ""))
                not in registry.RELEASE_SEMAPHORE_OPS
            ):
                continue
            if not c.get("name", "").startswith("t6_"):
                continue  # a RISC semaphore_get is not an in-stream release
            sid, param = registry.semaphore_target(c)
            fn = fb.enclosing(c["file"], c["off"])
            if sid and not param and fn is not None:
                self.releases[(fn.file, fn.begin_off, sid)].append(c["off"])

        writers = []  # (namespace, token, fact)
        for p in fb.facts:
            if p["family"] not in ("macro", "call"):
                continue
            role = (
                registry.classify_macro(p.get("name", ""))
                if p["family"] == "macro"
                else registry.classify_call(p.get("name", ""), p.get("text", ""))
            )
            if role != "ordered_write":
                continue
            ns, tok = registry.ordered_write_target(p)
            if ns in ("cfg", "gpr"):  # ThreadConfig (SETC16) has no RISC store path
                writers.append((ns, tok, p))

        findings: list[Finding] = []
        for pw in fb.family("pointer_write"):
            ns, tok, runtime_off = registry.risc_store_target(pw)
            if not ns:
                continue
            fn = fb.enclosing(pw["file"], pw["off"])
            if fn is None:
                continue
            # A tensix_sync() orders every Tensix write issued before it; a write
            # in this function AFTER the drain is still unordered.
            local = self._calls_before(fn, pw["off"], ("tensix_sync",))
            drain_off = max((c["off"] for c in local), default=None)
            drained = drain_off is not None or self._drained_by_callers(
                fn, self.CALLER_DEPTH
            )
            polls = self._polls(fn, pw["off"], self.CALLER_DEPTH)
            thread = registry.thread_of_fact(pw)
            ev, best = [], None
            for wns, wtok, w in writers:
                if wns != ns or not registry.same_register(
                    wtok, tok, runtime_off, fb.addr32
                ):
                    continue
                if (
                    registry.thread_of_fact(w) not in (thread, "UNKNOWN")
                    and thread != "UNKNOWN"
                ):
                    continue
                in_fn_after_drain = fn.contains(w["file"], w["off"]) and (
                    drain_off is None or w["off"] > drain_off
                )
                if drained and not in_fn_after_drain:
                    continue
                cls = self._classify(w, fn, pw, polls)
                if cls is None:
                    continue
                held = self._stall_before(w)
                ev.append(
                    (
                        -_RANK[cls],
                        not held,
                        self._ev(
                            w,
                            f"{cls}: {w.get('function', '?')}: {w.get('text', w.get('name', ''))[:80]}",
                        )
                        + (f" (held: after {held})" if held else ""),
                    )
                )
                best = cls if best is None or _RANK[cls] > _RANK[best] else best
            if best is None:
                continue
            ev.sort()
            guard = (
                f"behind a poll of {', '.join(sorted(polls))}"
                if polls
                else "with no poll or drain before it"
            )
            findings.append(
                Finding(
                    file=pw["file"],
                    line=pw.get("line", 0),
                    function=fn.name,
                    kind="tensix_then_risc_store",
                    hint=best,
                    detail=f"RISC store to {pw.get('index_text', tok)} {guard}; "
                    f"{len(ev)} same-register Tensix write(s) not ordered before it",
                    evidence=[e for _, _, e in ev],
                )
            )
        return findings

    # -- helpers ---------------------------------------------------------------
    def _calls_before(self, fn, off, names=None):
        return [
            c
            for c in self.fb.facts_in(fn, ("call",))
            if c["off"] < off and (names is None or c.get("name", "") in names)
        ]

    def _drained_by_callers(self, fn, depth) -> bool:
        """tensix_sync() before the call to fn in EVERY caller (recursively)."""
        if depth == 0:
            return False
        sites = self.callers.get(fn.name, [])
        if not sites:
            return False
        for c in sites:
            cf = self.fb.enclosing(c["file"], c["off"])
            if cf is None:
                return False
            if not self._calls_before(
                cf, c["off"], ("tensix_sync",)
            ) and not self._drained_by_callers(cf, depth - 1):
                return False
        return True

    def _polls(self, fn, off, depth) -> set:
        """Semaphores a RISC poll reads before `off`: in fn, or (when fn itself has
        none) in every caller before the call."""
        out = set()
        for c in self._calls_before(fn, off):
            nm = c.get("name", "")
            out |= self.gate_sem.get(nm, set())
            if nm in registry.RISC_POLL_CALLS:
                sid, param = registry.semaphore_target(c)
                if sid and not param:
                    out.add(sid)
        if out or depth == 0:
            return out
        common = None
        for c in self.callers.get(fn.name, []):
            cf = self.fb.enclosing(c["file"], c["off"])
            got = self._polls(cf, c["off"], depth - 1) if cf else set()
            common = got if common is None else common & got
        return common or set()

    def _release_follows(self, w, sid, depth):
        """True if a SEMGET of sid follows w in its function or in every caller."""
        fn = self.fb.enclosing(w["file"], w["off"])
        if fn is None:
            return False
        if any(
            o > w["off"] for o in self.releases.get((fn.file, fn.begin_off, sid), [])
        ):
            return True
        if depth == 0:
            return False
        sites = self.callers.get(fn.name, [])
        return bool(sites) and all(
            self._release_follows(c, sid, depth - 1) for c in sites
        )

    def _release_precedes(self, w, sid, depth):
        """True if a SEMGET of sid precedes w in its function or in some caller."""
        fn = self.fb.enclosing(w["file"], w["off"])
        if fn is None:
            return False
        if any(
            o < w["off"] for o in self.releases.get((fn.file, fn.begin_off, sid), [])
        ):
            return True
        if depth == 0:
            return False
        return any(
            self._release_precedes(c, sid, depth - 1)
            for c in self.callers.get(fn.name, [])
        )

    def _classify(self, w, fn, pw, polls):
        if fn.contains(w["file"], w["off"]):
            if w["off"] > pw["off"]:
                return None  # after the store in program order
            for s_ in polls:
                rel = self.releases.get((fn.file, fn.begin_off, s_), [])
                if any(w["off"] < o < pw["off"] for o in rel):
                    # A release between them: on one path the write is ahead of
                    # the SEMGET the poll waits for; in different branches they
                    # meet only across calls, where that same release orders them.
                    return None
            return "TENSIX_WRITE_SAME_FN"
        if any(self._release_follows(w, s, self.CALLER_DEPTH) for s in polls):
            return None  # ordered: the poll waits for a SEMGET issued after the write
        if any(self._release_precedes(w, s, self.CALLER_DEPTH) for s in polls):
            return "TENSIX_WRITE_AFTER_RELEASE"
        return "TENSIX_WRITE_NO_RELEASE"

    def _stall_before(self, w) -> str:
        """The last stall before w in its own function ('' when none)."""
        fn = self.fb.enclosing(w["file"], w["off"])
        if fn is None:
            return ""
        last = ""
        for q in self.fb.facts_in(fn, ("macro", "call")):
            if q["off"] >= w["off"]:
                break
            if (
                q["family"] == "macro"
                and registry.classify_macro(q.get("name", "")) == "stall"
            ):
                last = q.get("text", q.get("name", ""))
            elif (
                q["family"] == "call"
                and registry.classify_semaphore_call(q.get("name", "")) is not None
                and "p_stall::" in q.get("text", "")
                and "p_stall::NONE" not in q.get("text", "")
            ):
                last = q.get("text", "")
        return last[:60]
