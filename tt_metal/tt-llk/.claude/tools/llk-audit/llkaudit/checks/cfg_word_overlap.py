# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
cfg-word-overlap checker — differently-named fields that share the SAME 32-bit
config word, written by more than one Tensix thread.

Recall: resolve every config write to its 32-bit word via cfg_defines.h,
attribute the writing thread by file, and report words written by >= 2 distinct
threads. Whether a given pair actually races (masking disjoint bits,
semaphore/mutex ordering) is the LLM's call.

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
        "Whether a shared word actually races is deferred: disjoint-bit masking "
        "(RMWCIB is byte-atomic), semaphore/mutex ordering, and value-invariance "
        "are the LLM's call. Writes whose field name does not resolve to an "
        "ADDR32 in cfg_defines.h are listed as 'unresolved'. Intra-thread "
        "full-word clobber (INTRA_THREAD_CLOBBER) is a candidate: a full-word "
        "write that intentionally sets the entire word (no separate masked "
        "sibling) is benign, and a sibling written only out-of-tree is missed. "
        "Config has two banks selected by CFG_STATE_ID; the tool does not model "
        "StateID, so it may over-approximate (flag a word when the threads use "
        "different banks)."
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
            elif field:
                unresolved.append(self._unresolved(pw, field))
        for c in fb.family("call"):
            # cfg_reg_rmw_tensix<FIELD> (field in the callee template text) and
            # the direct RMW helpers cfg_rmw/cfg_rmw_gpr (field in arg0). The
            # latter often take a runtime address variable -> UNRESOLVED (still
            # surfaced, since an unresolved cfg RMW may touch a shared word).
            # All of these write Config (RMWCIB / software RMW), never ThreadConfig.
            src = None
            if "cfg_reg_rmw_tensix" in c.get("text", ""):
                src = c.get("text", "")
            elif registry.write_call_kind(c.get("name", "")):  # cfg_rmw / cfg_rmw_gpr
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
                word, field = registry.resolve_word(m.get("text", ""), fb.addr32)
                # SETC16 targets ThreadConfig; all other ordered writes target Config.
                ns = "THREAD" if "SETC16" in name.upper() else "CONFIG"
                if word is not None:
                    add(ns, word, field, m, f"instr:{name}")
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

        findings: list[Finding] = []
        for (ns, word), ws in sorted(writers.items()):
            threads = {w["thread"] for w in ws if w["thread"] != "UNKNOWN"}
            if len(threads) < 2:
                continue
            ev = [
                f'{w["file"].split("/")[-1]}:{w["line"]} [{w["thread"]}] '
                f'{w["field"]} ({w["how"]})'
                for w in ws
            ]
            fields = sorted({w["field"] for w in ws})
            # anchor the finding at the first writer
            first = min(ws, key=lambda w: (w["file"], w["line"]))
            findings.append(
                Finding(
                    file=first["file"],
                    line=first["line"],
                    function=first["function"],
                    kind=f"shared_word@{ns}:{word}",
                    hint="CROSS_THREAD_SHARED_WORD",
                    detail=(
                        f"{ns} word {word} written by threads "
                        f"{sorted(threads)} via fields {fields}"
                    ),
                    evidence=ev,
                )
            )

        findings += self._intra_thread_clobber(writers)
        return findings + unresolved

    @staticmethod
    def _is_full_word(how: str) -> bool:
        # A write of all 32 bits: a raw cfg[]= store (cfg32/cfg_ptr/mmio_ptr) or
        # WRCFG (WRCFG_32b/128b). NOT a masked/sized write (RMWCIB, cfg_rmw,
        # cfg_reg_rmw_tensix, SETC16, REG2FLOP) and NOT a 16-bit cfg16 store.
        h = how.upper()
        return "CFG32" in h or "CFG_PTR" in h or "MMIO_PTR" in h or "WRCFG" in h

    @staticmethod
    def _is_masked(how: str) -> bool:
        h = how.lower()
        return "cfg_reg_rmw_tensix" in h or "cfg_rmw" in h or "rmwcib" in h

    def _intra_thread_clobber(self, writers) -> list[Finding]:
        """Pattern 3: a full-word write (cfg[]=/WRCFG_32b) to a multi-field Config
        word, where the SAME thread also sets a DIFFERENT field of that word via a
        masked RMW elsewhere. The full-word write is built from only its own field
        and writes 0 into the sibling the thread set separately — a deterministic
        destructive overwrite (not a concurrency race; a mutex cannot fix it, a
        masked RMW can). Candidate only: a full-word write that intentionally sets
        the entire word is benign — the LLM confirms."""
        from collections import defaultdict

        out: list[Finding] = []
        for (ns, word), ws in sorted(writers.items()):
            if ns != "CONFIG":
                continue
            per_thread = defaultdict(lambda: {"full": [], "masked": []})
            for w in ws:
                if w["thread"] == "UNKNOWN":
                    continue
                if self._is_full_word(w["how"]):
                    per_thread[w["thread"]]["full"].append(w)
                elif self._is_masked(w["how"]):
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
                            f"sets via masked RMW"
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
