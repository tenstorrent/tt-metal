# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
cfg-word-overlap checker — differently-named fields that share the SAME 32-bit
config word, written by more than one Tensix thread.

Recall: resolve every config write to its 32-bit word via cfg_defines.h,
attribute the writing thread by file, and report words written by >= 2 distinct
threads. The tool ANNOTATES masking safety (SAFE_BY_MASKING / POTENTIAL_CLOBBER /
UNKNOWN — see _safety; plus UNRESOLVED_COWRITER for a 1-known-thread word with an
unattributable co-writer, a low-confidence widen); semaphore/mutex ordering and
value-invariance remain the
LLM's call.

Register FILE separation (per tt-isa-docs BackendConfiguration.md): the hardware
has TWO separate arrays, both indexed from 0, so their ADDR32 numbers alias and
must NOT be merged:
  * Config[2][...]     — thread-agnostic (ALU, THCON, PACK, ...), written by
                         WRCFG / RMWCIB / REG2FLOP / cfg_reg_rmw_tensix / RISCV sw.
  * ThreadConfig[3][.] — thread-specific (CFG_STATE_ID, ADDR_MOD, FP16A_FORCE,
                         DEST_TARGET_REG_CFG_MATH_Offset, ...), written ONLY by SETC16.
So the namespace is decided by the WRITE INSTRUCTION: SETC16 -> "THREAD", every
other config write -> "CONFIG". (THCON is a sub-range of Config, NOT its own file.)
"""
from __future__ import annotations

from collections import defaultdict

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding


class CfgWordOverlap(Check):
    name = "cfg-word-overlap"
    description = "Fields sharing one 32-bit CONFIG word written by >=2 threads"
    blind_spots = (
        "Disjoint-bit masking IS annotated by the tool (safety = SAFE_BY_MASKING / "
        "POTENTIAL_CLOBBER / UNKNOWN / UNRESOLVED_COWRITER — do not redo it); what "
        "stays deferred is "
        "semaphore/mutex ordering and value-invariance — the LLM's call. Writes "
        "whose field name does not resolve to an "
        "ADDR32 in cfg_defines.h are listed as 'unresolved'. Intra-thread "
        "full-word clobber (INTRA_THREAD_CLOBBER) is a candidate: a full-word "
        "write that intentionally sets the entire word (no separate masked "
        "sibling) is benign, and a sibling written only out-of-tree is missed. "
        "Config has two banks selected by CFG_STATE_ID; the tool does not model "
        "StateID, so it may over-approximate (flag a word when the threads use "
        "different banks). Two known under-reports (LLM must widen): (a) a "
        "RUNTIME/loop index offset — cfg[FIELD_ADDR32 + i] — resolves the BASE "
        "word only; sibling words base+1.. are not enumerated; (b) the masking "
        "annotation derives a writer's bitmask from the field's own _MASK, NOT "
        "the literal 3rd MASK operand of cfg_reg_rmw_tensix<ADDR32,SHAMT,MASK> — "
        "a wider mask can under-report bits and mis-label SAFE_BY_MASKING (the "
        "CROSS_THREAD_SHARED_WORD finding is still emitted regardless). (c) a "
        "config write RECORDED into a MOP/replay buffer (e.g. a WRCFG baked into a "
        "replay buffer) is seen as an ordinary in-place write, but its effect is "
        "DEFERRED to replay time (possibly looped / in another function) — the LLM "
        "must account for buffered/replayed config writes."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        # (namespace, word) -> list of writer dicts. Namespace separates the two
        # hardware arrays (Config vs ThreadConfig); see the module docstring.
        writers: dict[tuple, list[dict]] = defaultdict(list)
        unresolved: list[Finding] = []

        def add(ns, word, field, fact, how):
            thr = registry.thread_of(fact["file"])
            writers[(ns, word)].append(
                {
                    "thread": thr,
                    "field": field,
                    "file": fact["file"],
                    "line": fact["line"],
                    "function": fact.get("function", "?"),
                    "how": how,
                }
            )

        # RISCV sw and every RMW/flop write target Config (not ThreadConfig).
        for pw in fb.family("pointer_write"):
            kind, _ = registry.classify_write(pw)
            if kind not in registry.CFG_WRITE_KINDS:
                continue
            word, field = registry.resolve_word(pw.get("index_text", ""), fb.addr32)
            if word is not None:
                add("CONFIG", word, field, pw, f"mmio:{kind}")
            else:
                # This IS a config write (kind ∈ CFG_WRITE_KINDS) we couldn't
                # resolve to a word — including a NON-LITERAL index like
                # cfg[upk0_reg]= (resolve_word returns (None, None), so `field`
                # is None). Surface it UNRESOLVED; never silently drop it (the
                # blind_spots contract promises unresolved cfg writes are shown).
                unresolved.append(
                    self._unresolved(
                        pw, field or pw.get("index_text", "") or "<runtime index>"
                    )
                )
        for c in fb.family("call"):
            # cfg_reg_rmw_tensix<FIELD> (field in the callee template text) and
            # the direct RMW helpers cfg_rmw/cfg_rmw_gpr (field in arg0). The
            # latter often take a runtime address variable -> UNRESOLVED (still
            # surfaced, since an unresolved cfg RMW may touch a shared word).
            # All of these write Config (RMWCIB / software RMW), never ThreadConfig.
            src = None
            if "cfg_reg_rmw_tensix" in c.get("text", ""):
                src = c.get("text", "")
            elif c.get("name", "") in ("cfg_rmw", "cfg_rmw_gpr"):
                # ONLY the config-RMW helpers take a FIELD in arg0. Do NOT use the
                # broader write_call_kind() (it also matches reg_write, a raw/GPR
                # register write) — that could fold a non-config write into the
                # CONFIG shared-word map if its arg0 happened to contain _ADDR32.
                src = c.get("arg0", "")
            # Only act on a non-empty source token. A cfg_rmw(FIELD_RMW, ...) call
            # has arg0 == "" here because FIELD_RMW is an object-macro that expands
            # before the AST; that write is recovered from the *_RMW macro fact
            # below, so skip the empty case rather than emit noise.
            if src:
                word, field = registry.resolve_word(src, fb.addr32)
                if word is not None:
                    add("CONFIG", word, field, c, c.get("name") or "cfg_reg_rmw_tensix")
                else:
                    unresolved.append(self._unresolved(c, field or src))
        for m in fb.family("macro"):
            name = m.get("name", "")
            if registry.classify_macro(name) == "ordered_write":
                if "SETDMAREG" in name.upper():
                    # SETDMAREG writes a GPR (a source value), NOT a sampled
                    # config word — reconfig-stall excludes it for the same
                    # reason. Skip so it can't be mis-attributed as a Config
                    # writer (it would be if its text ever carried an _ADDR32).
                    continue
                word, field = registry.resolve_word(m.get("text", ""), fb.addr32)
                # SETC16 targets ThreadConfig; all other ordered writes target Config.
                ns = "THREAD" if "SETC16" in name.upper() else "CONFIG"
                if word is not None:
                    # A WRCFG_128b overwrites 4 consecutive words (base..base+3);
                    # enumerate them all so a sibling write to base+1..+3 is seen
                    # as overlapping. (Every other write spans one word.)
                    span = registry.wrcfg_word_count(m.get("text", ""))
                    for wd in range(word, word + span):
                        fld = field if wd == word else f"{field}+{wd - word}"
                        add(ns, wd, fld, m, f"instr:{name}")
                elif field:
                    unresolved.append(self._unresolved(m, field))
            elif name.endswith("_RMW"):
                # cfg_rmw(FIELD_RMW, ...) — the composite alias captured at PP level.
                # It is a software RMW of Config (never ThreadConfig).
                word, field = registry.resolve_word(name, fb.addr32)
                if word is not None:
                    add("CONFIG", word, field, m, f"cfg_rmw:{name}")
                elif field:
                    unresolved.append(self._unresolved(m, field))

        # De-duplicate per write-site. cfg_reg_rmw_tensix<FIELD_RMW> is captured
        # BOTH as a call (how=cfg_reg_rmw_tensix, atomic RMWCIB) and as the
        # FIELD_RMW object-macro (how=cfg_rmw:..., added for Quasar). Collapse
        # them per (file,line,field), preferring the call/instr entry so the true
        # atomicity is known. On Quasar, cfg_rmw(FIELD_RMW,...) has no
        # cfg_reg_rmw_tensix call, so only the macro entry exists and it survives
        # (correctly treated as a non-atomic software RMW).
        for key in list(writers):
            seen: dict = {}
            for w in writers[key]:
                k = (w["file"], w["line"], w["field"])
                prev = seen.get(k)
                if prev is None or (
                    prev["how"].startswith("cfg_rmw:")
                    and not w["how"].startswith("cfg_rmw:")
                ):
                    seen[k] = w
            writers[key] = list(seen.values())

        findings: list[Finding] = []
        for (ns, word), ws in sorted(writers.items()):
            # ThreadConfig[3][] has an INDEPENDENT physical copy per Tensix thread
            # (that is why the namespace is split). Two threads writing the same
            # ThreadConfig word touch DIFFERENT registers — no data race — so the
            # cross-thread finding applies to Config only. (SETC16-shared words
            # would otherwise be reported as a spurious POTENTIAL_CLOBBER.)
            if ns != "CONFIG":
                continue
            threads = {w["thread"] for w in ws if w["thread"] != "UNKNOWN"}
            has_unknown = any(w["thread"] == "UNKNOWN" for w in ws)
            # >=2 known threads -> a real cross-thread share. Exactly 1 known + an
            # unattributable co-writer -> WIDEN (emit-low-confidence): a possible
            # cross-thread share the tool can't confirm, surfaced not dropped (a
            # recall tool must not narrow away an unresolved co-writer). Neither ->
            # skip.
            if len(threads) < 2 and not (len(threads) == 1 and has_unknown):
                continue
            ev = [
                f'{w["file"].split("/")[-1]}:{w["line"]} [{w["thread"]}] '
                f'{w["field"]} ({w["how"]})'
                for w in ws
            ]
            fields = sorted({w["field"] for w in ws})
            safety, bits = self._safety(ws, fb.addr32)
            if len(threads) < 2:  # 1 known + unknown co-writer: low-confidence widen
                safety = "UNRESOLVED_COWRITER"
            bits_str = ", ".join(f"{t}=0x{m:x}" for t, m in sorted(bits.items()))
            # anchor the finding at the first writer
            first = min(ws, key=lambda w: (w["file"], w["line"]))
            findings.append(
                Finding(
                    file=first["file"],
                    line=first["line"],
                    function=first["function"],
                    kind=f"shared_word@{ns}:{word}",
                    # The cross-thread ACCESS is always reported (multi-thread use
                    # of a word is itself worth seeing, even when race-safe);
                    # safety is a SUB-annotation, not a filter.
                    hint="CROSS_THREAD_SHARED_WORD",
                    detail=(
                        f"{ns} word {word} written by threads {sorted(threads)} "
                        f"via fields {fields}; per-thread bits {{{bits_str}}}"
                    ),
                    evidence=ev,
                    safety=safety,
                )
            )

        findings += self._intra_thread_clobber(writers)
        return findings + unresolved

    @staticmethod
    def _safety(ws, defines):
        """Race-safety ANNOTATION for a cross-thread shared word (never a filter).
        Returns (label, {thread: combined_bitmask}).
          SAFE_BY_MASKING   – every cross-thread writer is a byte-atomic masked
                              RMW (cfg_reg_rmw_tensix/RMWCIB) and the per-thread
                              bit masks are pairwise disjoint (RMWCIB is byte-
                              atomic, so disjoint bits compose safely).
          POTENTIAL_CLOBBER – a full-word write, a non-atomic software cfg_rmw,
                              or overlapping bits across threads.
          UNKNOWN           – a writer's field mask isn't in cfg_defines.
        A 4th value, UNRESOLVED_COWRITER, is set by run() (NOT here) for a word with
        exactly one KNOWN thread plus an unattributable co-writer — a low-confidence
        widen (the co-writer's thread couldn't be resolved, so a cross-thread share
        can be neither confirmed nor dismissed).
        The word is reported as CROSS_THREAD_SHARED_WORD regardless of the label;
        SAFE_BY_MASKING still surfaces the multi-thread access (a possible
        ownership smell) — the label just says it isn't a data race."""
        per_thread: dict = {}
        unresolved = False
        all_atomic = True
        full_word_present = False
        for w in ws:
            if w["thread"] == "UNKNOWN":
                continue
            if registry.write_is_full_word(w["how"]):
                m, all_atomic, full_word_present = 0xFFFFFFFF, False, True
            else:
                m = registry.field_bitmask(w["field"], defines)
                if m is None:
                    unresolved, m = True, 0
                if not registry.write_is_atomic_masked(w["how"]):
                    all_atomic = False
            per_thread[w["thread"]] = per_thread.get(w["thread"], 0) | m
        masks = list(per_thread.values())
        disjoint = all(
            (masks[i] & masks[j]) == 0
            for i in range(len(masks))
            for j in range(i + 1, len(masks))
        )
        # A full-word writer definitely clobbers the whole word, so it is
        # POTENTIAL_CLOBBER regardless of whether some OTHER writer's mask was
        # unresolved — the full-word fact is stronger than the missing mask, and
        # downgrading to UNKNOWN would hide a certain clobber behind a weaker label.
        if full_word_present:
            label = "POTENTIAL_CLOBBER"
        elif unresolved:
            label = "UNKNOWN"
        elif all_atomic and disjoint:
            label = "SAFE_BY_MASKING"
        else:
            label = "POTENTIAL_CLOBBER"
        return label, per_thread

    def _intra_thread_clobber(self, writers) -> list[Finding]:
        """Pattern 3: a full-word write (cfg[]=/WRCFG_32b) to a multi-field Config
        word, where the SAME thread also sets a DIFFERENT field of that word via a
        field-scoped write elsewhere (a masked RMW, or a sized TTI_REG2FLOP /
        TTI_SETADC / cfg16 set). The full-word write is built from only its own
        field and writes 0 into the sibling the thread set separately — a
        deterministic destructive overwrite (not a concurrency race; a mutex cannot
        fix it, a masked RMW can). Candidate only: a full-word write that
        intentionally sets the entire word is benign — the LLM confirms. The victim
        bucket is any resolved-field non-full-word write (matching _safety), NOT
        just the RMW helpers — a full-word write zeroes a REG2FLOP/SETADC/cfg16
        sibling just the same."""
        out: list[Finding] = []
        for (ns, word), ws in sorted(writers.items()):
            if ns != "CONFIG":
                continue
            per_thread = defaultdict(lambda: {"full": [], "masked": []})
            for w in ws:
                if w["thread"] == "UNKNOWN":
                    continue
                if registry.write_is_full_word(w["how"]):
                    per_thread[w["thread"]]["full"].append(w)
                elif w.get("field"):
                    # any resolved-field, non-full-word write (masked RMW OR a
                    # sized REG2FLOP/SETADC/cfg16 set) is a clobber victim — a
                    # full-word write zeroes it regardless of how it was written.
                    per_thread[w["thread"]]["masked"].append(w)
            for thr, d in per_thread.items():
                full_fields = {w["field"] for w in d["full"]}
                clobbered = [w for w in d["masked"] if w["field"] not in full_fields]
                if not d["full"] or not clobbered:
                    continue
                anchor = d["full"][0]
                ev = [self._w_ev(w, "full-word write") for w in d["full"]] + [
                    self._w_ev(w, "masked sibling (may be zeroed)") for w in clobbered
                ]
                out.append(
                    Finding(
                        file=anchor["file"],
                        line=anchor["line"],
                        function=anchor["function"],
                        kind=f"clobber@CONFIG:{word}",
                        hint="INTRA_THREAD_CLOBBER",
                        detail=(
                            f"[{thr}] full-word write of {sorted(full_fields)} to word "
                            f"{word} may zero sibling field(s) "
                            f"{sorted({w['field'] for w in clobbered})} the same thread "
                            f"sets via a field-scoped write (masked RMW / REG2FLOP / "
                            f"SETADC / cfg16)"
                        ),
                        evidence=ev,
                    )
                )
        return out

    @staticmethod
    def _w_ev(w: dict, what: str) -> str:
        return f'{w["file"].split("/")[-1]}:{w["line"]} [{w["thread"]}] {w["field"]} ({w["how"]}) {what}'

    def _unresolved(self, fact: dict, field: str) -> Finding:
        return Finding(
            file=fact["file"],
            line=fact["line"],
            function=fact.get("function", "?"),
            kind="unresolved_field",
            hint="UNRESOLVED",
            detail=f"could not resolve {field} to an ADDR32 word (cfg_defines.h)",
            evidence=[],
        )
