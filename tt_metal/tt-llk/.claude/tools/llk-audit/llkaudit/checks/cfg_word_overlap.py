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
        "full-word clobber of a sibling field is only partially modeled. "
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
            if src is not None:
                word, field = registry.resolve_word(src, fb.addr32)
                if word is not None:
                    add("CONFIG", word, field, c, c.get("name") or "cfg_reg_rmw_tensix")
                else:
                    unresolved.append(self._unresolved(c, field or src or "?"))
        for m in fb.family("macro"):
            if registry.classify_macro(m.get("name", "")) == "ordered_write":
                word, field = registry.resolve_word(m.get("text", ""), fb.addr32)
                # SETC16 targets ThreadConfig; all other ordered writes target Config.
                ns = "THREAD" if "SETC16" in m.get("name", "").upper() else "CONFIG"
                if word is not None:
                    add(ns, word, field, m, f"instr:{m.get('name','')}")
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
        return findings + unresolved

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
