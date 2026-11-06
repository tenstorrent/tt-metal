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

  * `tensix_sync()` issued after the write and before the store (in the store's
    function, or in every caller before the call): the RISC blocks until the
    whole thread is idle. A write issued after the drain — in the store's
    function or in a function it calls — is not ordered by it.
  * a RISC poll (`semaphore_read`, e.g. wait_for_idle on UNPACK_SYNC) before the
    store, where the Tensix write is issued BEFORE a SEMGET of that semaphore
    and that SEMGET is issued BEFORE the poll: a full poll (wait_for_idle,
    wait_for_next_context(1)) cannot return until the SEMGET, and so the write,
    has executed. wait_for_next_context(2) returns while the most recent call's
    SEMGET may still be pending, so it orders only writes to a per-context word
    (the call that last used that context has released it). A write issued after
    the poll returned is not ordered by any later release.

A writer issued AFTER the store in the store's own function is handled like one
in another function: per-tile calls run back to back, so call N's late writer
meets call N+1's store.

Per RISC store with at least one unordered same-register Tensix writer:
  TENSIX_WRITE_SAME_FN     — the store's function issues the writer before the
                             store, itself or through a function it calls
                             (order certain), with no drain or covering
                             release-then-poll between.
  TENSIX_WRITE_AFTER_RELEASE — the store is behind a poll, but the writer is
                             issued after the SEMGET that releases it, or after
                             the poll itself, or the poll is partial and the
                             word is shared by contexts (the poll can return
                             while the writer is still queued).
  TENSIX_WRITE_NO_RELEASE  — the writer sits in a function that never releases
                             the polled semaphore after it (init / reconfig /
                             uninit paths), or the store has no poll at all; it
                             races if a kernel can issue that function and then
                             reach the store while the write is still held.
  UNRESOLVED_TARGET        — a direct cfg[]/regfile[] store whose index names
                             no register (a local such as `upk0_reg`, a wrapper
                             parameter): its writers cannot be matched.
The hint is the strongest class present; every writer is listed as evidence,
marked "(held: after <stall>)" when a stall precedes it in its own function,
followed by the call sites (up to CALLER_DEPTH) of the store's and each writer's
function, where the polls, releases and drains that decide the hint sit, so a
diff-scoped run (--changed) keeps the finding when only a caller changes.
"""

from __future__ import annotations

from collections import defaultdict

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding

_PARTIAL = "@partial"


def _sid(key: str) -> str:
    return key.split("@", 1)[0]


def _full(key: str) -> bool:
    return not key.endswith(_PARTIAL)


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
        "release, a poll or a tensix_sync() is checked in the function and up to two "
        "caller levels, and a writer issued through callees up to two levels deep, "
        "by name (same-name overloads are merged); a writer reached deeper than that "
        "after a drain is treated as ordered by it. Whether a release that follows "
        "a writer in a CALLER also precedes the poll is not checked. RISC config READS (a "
        "save/restore or a software read-modify-write that reads a word a queued "
        "Tensix write has not landed in yet) are not in the fact base and are not "
        "seen. Thread is inferred from the file name; cross-thread pairs belong to "
        "cfg-word-overlap. A RISC store with a runtime word offset "
        "(`FIELD_ADDR32 + i`) is matched by REG prefix, which can over-match. "
        "Value-invariant pairs (both sides write the same value) are reported; the "
        "skill clears them. MOP/replay-slotted writes are matched by their source "
        "line, not their execution slot. A wait_for_next_context(2) poll clears only "
        "per-context words, assuming the two contexts alternate per call. A store "
        "whose index is a local or a parameter is reported UNRESOLVED_TARGET, not "
        "matched. Quasar: TTSync holds a RISC config store until older RQ-tracked "
        "config-writing Tensix instructions ISSUE (not until they land), and "
        "RESOURCEDECL, MOP_CFG, REPLAY(load=1) and the instructions loaded after it "
        "are not in the RQ; confirm against Confluence 1340276980 before clearing a "
        "Quasar pair."
    )

    CALLER_DEPTH = 2  # caller levels searched for a drain, a poll or a release
    REACH_DEPTH = 2  # callee levels searched for a writer the store's function issues

    def run(self, fb: FactBase) -> list[Finding]:
        self.fb = fb
        self._calls_cache: dict = {}
        self._reach_cache: dict = {}
        self.defs = defaultdict(list)  # function name -> definitions
        for f in fb.functions:
            self.defs[f.name].append(f)
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
            fn = fb.enclosing(pw["file"], pw["off"])
            if fn is None:
                continue
            if not ns:
                if registry.unresolved_store_ns(pw):
                    findings.append(self._unresolved(pw, fn))
                continue
            windows = self._poll_windows(fn, pw["off"], self.CALLER_DEPTH)
            self._per_ctx = ns == "cfg" and registry.is_per_context_word(tok, fb.addr32)
            polls = {_sid(k) + ("" if _full(k) else " (partial)") for k in windows}
            thread = registry.thread_of_fact(pw)
            ev, best, writer_fns = [], None, []
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
                if self._drain_covers(w, fn, pw["off"], self.CALLER_DEPTH):
                    continue
                cls = self._classify(w, fn, pw, windows)
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
                wfn = fb.enclosing(w["file"], w["off"])
                if wfn is not None:
                    writer_fns.append(wfn)
            if best is None:
                continue
            ev.sort()
            ev = [e for _, _, e in ev] + self._site_evidence(fn, writer_fns)
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
                    f"{len(writer_fns)} same-register Tensix write(s) not ordered before it",
                    evidence=ev,
                )
            )
        return findings

    # -- helpers ---------------------------------------------------------------
    def _unresolved(self, pw, fn) -> Finding:
        return Finding(
            file=pw["file"],
            line=pw.get("line", 0),
            function=fn.name,
            kind="tensix_then_risc_store",
            hint="UNRESOLVED_TARGET",
            detail=f"RISC store to [{pw.get('index_text', '')}] does not name a "
            "register; its same-thread Tensix writers cannot be matched",
            evidence=[],
        )

    def _call_sites(self, name, depth, seen) -> list:
        """Call facts that call `name`, and their callers, up to `depth` levels."""
        out = []
        for c in self.callers.get(name, []):
            key = (c["file"], c["off"])
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
            if depth > 1 and c.get("function"):
                out += self._call_sites(c["function"], depth - 1, seen)
        return out

    def _site_evidence(self, fn, writer_fns) -> list:
        seen: set = set()
        sites = self._call_sites(fn.name, self.CALLER_DEPTH, seen)
        for wfn in writer_fns:
            sites += self._call_sites(wfn.name, self.CALLER_DEPTH, seen)
        return [
            self._ev(c, f"call site: {c.get('function', '?')} -> {c.get('name', '')}")
            for c in sites
        ]

    def _calls(self, fn):
        key = (fn.file, fn.begin_off)
        got = self._calls_cache.get(key)
        if got is None:
            got = self._calls_cache[key] = self.fb.facts_in(fn, ("call",))
        return got

    def _calls_before(self, fn, off, names=None):
        return [
            c
            for c in self._calls(fn)
            if c["off"] < off and (names is None or c.get("name", "") in names)
        ]

    def _reach_offs(self, fn, target, depth) -> list:
        """Offsets of the calls in fn that reach a function named `target`, directly
        or through up to `depth` further call levels (by name)."""
        key = (fn.file, fn.begin_off, target, depth)
        got = self._reach_cache.get(key)
        if got is not None:
            return got
        self._reach_cache[key] = []  # recursion guard
        out = []
        for c in self._calls(fn):
            nm = c.get("name", "")
            if nm == target or (
                depth > 0
                and any(
                    self._reach_offs(d, target, depth - 1)
                    for d in self.defs.get(nm, [])
                    if (d.file, d.begin_off) != (fn.file, fn.begin_off)
                )
            ):
                out.append(c["off"])
        self._reach_cache[key] = out
        return out

    def _issued_at(self, w, fn, lo, hi) -> list:
        """Offsets in fn, strictly between lo and hi, at which fn issues w: w's own
        offset when w is in fn, else the calls in fn that reach w's function."""
        if fn.contains(w["file"], w["off"]):
            return [w["off"]] if lo < w["off"] < hi else []
        wfn = self.fb.enclosing(w["file"], w["off"])
        if wfn is None:
            return []
        return [
            o for o in self._reach_offs(fn, wfn.name, self.REACH_DEPTH) if lo < o < hi
        ]

    def _drain_covers(self, w, fn, upto, depth) -> bool:
        """True if a tensix_sync() executes after w is issued and before `upto` in
        fn: the last drain in fn before `upto`, with w not issued (in fn or in a
        function fn calls) between that drain and `upto`; or, when fn has no drain,
        the same in EVERY caller before its call to fn, with w not issued in fn
        itself. Not covered when that cannot be shown within the depth."""
        drains = [c["off"] for c in self._calls_before(fn, upto, ("tensix_sync",))]
        if drains:
            return not self._issued_at(w, fn, max(drains), upto)
        if self._issued_at(w, fn, fn.begin_off - 1, upto):
            return False  # issued in fn: no drain of fn's callers can follow it
        if depth == 0:
            return False
        sites = self.callers.get(fn.name, [])
        if not sites:
            return False
        for c in sites:
            cf = self.fb.enclosing(c["file"], c["off"])
            if cf is None or not self._drain_covers(w, cf, c["off"], depth - 1):
                return False
        return True

    def _poll_windows(self, fn, off, depth) -> dict:
        """{semaphore: [(function, lo, hi)]} for each semaphore a RISC poll reads
        before `off`. Code issued in function, strictly between lo and hi, is issued
        AFTER the poll returned, so a release it is followed by cannot order it.
        Polls in fn give the window from the last poll to `off`; when fn has none,
        the semaphores polled in EVERY caller before its call to fn, with the whole
        of fn up to `off` plus each caller's own windows. A partial poll
        (wait_for_next_context(n > 1)) is keyed `<sem>@partial`."""
        local = defaultdict(list)
        for c in self._calls_before(fn, off):
            nm = c.get("name", "")
            sids = set(self.gate_sem.get(nm, set()))
            if nm in registry.RISC_POLL_CALLS:
                sid, param = registry.semaphore_target(c)
                if sid and not param:
                    sids.add(sid)
            full = registry.poll_is_full(c)
            for s_ in sids:
                local[s_ if full else s_ + _PARTIAL].append(c["off"])
        if local:
            return {s_: [(fn, max(offs), off)] for s_, offs in local.items()}
        if depth == 0:
            return {}
        # Per semaphore polled in EVERY caller: full only if every caller's poll
        # is full (a caller with both keeps its full one).
        common = None
        for c in self.callers.get(fn.name, []):
            cf = self.fb.enclosing(c["file"], c["off"])
            got = self._poll_windows(cf, c["off"], depth - 1) if cf else {}
            by_sid = {}
            for k, v in got.items():
                if _sid(k) not in by_sid or _full(k):
                    by_sid[_sid(k)] = (_full(k), list(v))
            if common is None:
                common = by_sid
            else:
                common = {
                    s_: (
                        common[s_][0] and by_sid[s_][0],
                        common[s_][1] + by_sid[s_][1],
                    )
                    for s_ in common.keys() & by_sid.keys()
                }
        if not common:
            return {}
        return {
            (s_ if full else s_ + _PARTIAL): v + [(fn, fn.begin_off - 1, off)]
            for s_, (full, v) in common.items()
        }

    def _release_follows(self, w, sid, depth):
        """True if a SEMGET of sid follows w in its function or in EVERY caller.
        Every caller, because this CLEARS the writer: one caller without the
        release leaves a path on which the poll does not wait for the write."""
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
        """True if a SEMGET of sid precedes w in its function or in SOME caller.
        Some caller, because this only picks the label for a writer that is already
        reported: one path where the write follows the release is enough for
        AFTER_RELEASE; otherwise it stays NO_RELEASE, which is still a finding."""
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

    def _poll_orders(self, key) -> bool:
        """A full poll orders every write released before it; a partial one only
        writes to a per-context word of the store being checked."""
        return _full(key) or self._per_ctx

    def _after_poll(self, w, wins) -> bool:
        """w is issued inside one of the poll windows: after the poll returned."""
        return any(self._issued_at(w, f, lo, hi) for f, lo, hi in wins)

    def _classify(self, w, fn, pw, windows):
        here = self._issued_at(w, fn, fn.begin_off - 1, pw["off"])
        if here:
            # fn issues w before the store (itself or through a callee), so the
            # order is certain. A semaphore orders it only if fn issues a SEMGET
            # of it after w (or w's function does, after w) and THEN polls it.
            in_fn = fn.contains(w["file"], w["off"])
            for s_, wins in windows.items():
                if self._after_poll(w, wins) or not self._poll_orders(s_):
                    continue
                poll = max(
                    (
                        lo
                        for f, lo, _ in wins
                        if (f.file, f.begin_off) == (fn.file, fn.begin_off)
                    ),
                    default=None,
                )
                if poll is None:
                    continue
                rel = self.releases.get((fn.file, fn.begin_off, _sid(s_)), [])
                if all(
                    any(x < o < poll for o in rel)
                    or (not in_fn and self._release_follows(w, _sid(s_), 0))
                    for x in here
                ):
                    # In different if/else arms the writer and store meet only
                    # across calls, where that same release orders them.
                    return None
            return "TENSIX_WRITE_SAME_FN"
        after_poll = False
        for s_, wins in windows.items():
            if self._after_poll(w, wins):
                after_poll = True  # issued in a caller after the poll returned
            elif self._release_follows(w, _sid(s_), self.CALLER_DEPTH):
                if self._poll_orders(s_):
                    return None  # the poll waits for a SEMGET issued after the write
                after_poll = True  # a partial poll can return before that SEMGET
        if after_poll or any(
            self._release_precedes(w, _sid(s), self.CALLER_DEPTH) for s in windows
        ):
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
