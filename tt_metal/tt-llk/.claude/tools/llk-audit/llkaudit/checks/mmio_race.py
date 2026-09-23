# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
mmio-race checker — RISC MMIO write to a CONFIG/GPR register vs. a Tensix
instruction/MOP that consumes it.

Recall of KNOWN patterns only. Per MMIO write it reports whether an *applicable*
ordering primitive follows it within the enclosing function:
  - STALLWAIT whose condition mentions the TRISC_CFG token (WH: C13; BH: 0x400;
    QSR: index 21 — matched by NAME, not value) orders any cfg/GPR write
  - mop_sync / tensix_sync (RISC-blocking drains) order any write
  - sync_regfile_write orders GPR (regfile) writes only
An ordered in-stream Tensix cfg write (REG2FLOP/WRCFG/SETC16) does NOT order a
prior MMIO write, so it is never counted as a guard.

Hints: LOCALLY_ORDERED | NO_LOCAL_ORDERING (the interprocedural triage queue) |
AUTOTTSYNC_ORDERED (Quasar only — the per-RISC TTSync HW-orders the write, so an
otherwise-unguarded write is not a race candidate there).
"""

from __future__ import annotations

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding


class MmioRace(Check):
    name = "mmio-race"
    description = "RISC MMIO cfg/GPR write vs. consuming Tensix instruction/MOP"
    blind_spots = (
        "Interprocedural ordering: a write whose guard/consumer lives in a CALLER "
        "shows as NO_LOCAL_ORDERING — the LLM must follow the call graph. "
        "Writes hidden inside SFPU files that fail to parse are absent (see "
        "parse_errors). A TRISC_CFG stall (WH: C13; BH: 0x400; QSR: index 21 — "
        "matched by token name, not value) only orders instructions its BLOCK mask "
        "holds; the tool checks the TRISC_CFG condition but not that the block mask "
        "covers the consumer. On Quasar, writes are marked "
        "AUTOTTSYNC_ORDERED (TTSync HW-orders the write->consume direction); this "
        "does NOT cover an MMIO *read* that depends on a multi-cycle instruction "
        "result (needs wait_*_idle), nor the EN_SUBDIVIDED cross-unpacker corner. "
        "It also does NOT account for the TTSync RQ EXCEPTIONS: MOP_CFG / "
        "REPLAY(load=1) / RESOURCEDECL / post-load-replay instructions are not "
        "RQ-tracked (Confluence 1340276980), so a CFG/GPR write CONSUMED by one is "
        "not auto-ordered yet is still blanket-tagged AUTOTTSYNC_ORDERED — the LLM "
        "must confirm the consumer is RQ-tracked (see KNOWN_GAPS L7). (Writes to a "
        "non-CFG/GPR MMIO space TTSync doesn't track — the replay unit's "
        "replay_mmap[] via INSTRN_BUF, the PC_BUF sync FIFO — are excluded upstream "
        "by registry.classify_write's ISA-grounded region check, so they no longer "
        "reach this tag.) "
        "MOP/REPLAY are treated as OPAQUE consumers (one consumer, stall-before): "
        "the tool does not see the instructions inside a MOP, does not distinguish "
        "TTI_REPLAY record (load) vs execute mode, does not model the "
        "record->execute DECOUPLING (a buffer programmed in one function and "
        "replayed/looped in another), and only recognizes a RAW TTI_REPLAY / "
        "mop_run — a WRAPPED replay-execute call (e.g. _execute_*_replay_buffer_) "
        "is not flagged as a consumer. The LLM must trace replay record->execute "
        "and per-iteration re-runs. The matrix/FPU CONSUMER set "
        "(registry.CONSUMER_MATH_SUBSTR) is a CURATED PARTIAL — SFPU vector ops and "
        "less-common matrix issues are not all recognized; an unrecognized consumer "
        "sitting between a write and a later guard lets that guard be mis-credited, "
        "so a genuinely-raced write can read LOCALLY_ORDERED (a false negative — the "
        "LLM must check for consuming ops the tool didn't recognize)."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        findings: list[Finding] = []

        # Collect every MMIO write site.
        writes = []
        for pw in fb.family("pointer_write"):
            kind, how = registry.classify_write(pw)
            if kind:
                writes.append((pw, kind, how))
        for c in fb.family("call"):
            k = registry.write_call_kind(c.get("name", ""))
            if k:
                writes.append((c, k, "call"))

        for w, kind, how in writes:
            fn = fb.enclosing(w["file"], w["off"])
            is_gpr = kind in registry.GPR_WRITE_KINDS
            local_ordering = False
            guard = None
            consumer_after = None

            if fn:
                for p in fb.facts_in(fn, ("macro", "call")):
                    if p["off"] <= w["off"]:
                        continue  # only what follows the write can order/consume it
                    role = self._role(p)
                    if role is None:
                        continue
                    applicable = (
                        (
                            role == "stall"
                            and registry.TRISC_CFG_TOKEN in p.get("text", "")
                        )
                        or role in ("drain:mop_sync", "drain:tensix_sync")
                        or (is_gpr and role == "drain:sync_regfile_write")
                    )
                    # A guard orders the write only if it precedes the FIRST
                    # consumer: a guard AFTER a consumer cannot un-race that
                    # consumer (it only orders vs the next run). So once a
                    # consumer is seen, stop crediting guards.
                    if applicable and not local_ordering and consumer_after is None:
                        local_ordering = True
                        guard = (p.get("name") or "", p.get("line", 0))
                    # CFGSHIFTMASK is classified ordered_write but ALSO consumes config
                    # (reads its target cfg word + SCRATCH_SEC then writes back), so it
                    # marks the consumer boundary too — a guard after it can't un-race it.
                    if (
                        registry.is_consumer(role)
                        or registry.is_mmio_config_rmw_consumer(p)
                    ) and consumer_after is None:
                        consumer_after = p.get("line", 0)

            hint = "LOCALLY_ORDERED" if local_ordering else "NO_LOCAL_ORDERING"
            # Quasar ENABLES Auto TTSync (set_ttsync_enables<TRACK_ALL> in
            # llk_math_common.h), so the per-RISC TTSync HW-orders a TRACKED CFG/GPR
            # MMIO write against the consuming Tensix instruction at ISSUE (Confluence
            # "Every Conceivable TTSync Detail", page 1340276980) — the manual
            # STALLWAIT(TRISC_CFG)/REG2FLOP discipline WH/BH need is not required, so a
            # CFG/GPR write with an RQ-tracked consumer is NOT a race candidate here.
            # OVER-BROAD CAVEAT (see blind_spots + KNOWN_GAPS L7): this blanket tag also
            # clears a CFG/GPR write consumed by an RQ-EXCEPTED instruction (MOP_CFG/
            # REPLAY/RESOURCEDECL), which TTSync does NOT order — the LLM confirms those.
            # (Non-CFG/GPR MMIO like replay_mmap[]/INSTRN_BUF/PC_BUF is excluded upstream
            # by classify_write's region check, so it never reaches this tag.)
            if fb.arch == "quasar" and hint == "NO_LOCAL_ORDERING":
                hint = "AUTOTTSYNC_ORDERED"
            ev = []
            if guard:
                ev.append(f'{w["file"].split("/")[-1]}:{guard[1]}: guard {guard[0]}')
            if consumer_after:
                ev.append(
                    f'{w["file"].split("/")[-1]}:{consumer_after}: consumer after write'
                )
            detail = f"{kind} write ({how})"
            if w.get("index_text"):
                detail += f' to {w["index_text"]}'
            findings.append(
                Finding(
                    file=w["file"],
                    line=w.get("line", 0),
                    # Label from the SAME enclosing fn used for the ordering scan
                    # (facts_in(fn)), so the reported function matches the analyzed
                    # scope; fall back to the write's recorded field, then file-scope.
                    function=(
                        fn.name
                        if fn and fn.name
                        else (w.get("function") or "<file-scope>")
                    ),
                    kind=kind,
                    hint=hint,
                    detail=detail,
                    evidence=ev,
                )
            )
        return findings

    def _role(self, fact: dict):
        if fact["family"] == "macro":
            return registry.classify_macro(fact.get("name", ""))
        return registry.classify_call(fact.get("name", ""), fact.get("text", ""))
