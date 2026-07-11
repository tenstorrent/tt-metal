# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
cfg-word-overlap checker — differently-named fields that share the SAME 32-bit
CONFIG word, written by more than one Tensix thread.

Recall: resolve every CONFIG write (MMIO cfg[]=, ordered WRCFG/REG2FLOP/SETC16,
and cfg_reg_rmw_tensix<FIELD>) to its 32-bit word via cfg_defines.h, attribute
the writing thread by file, and report words written by >= 2 distinct threads.
That set is the cross-thread shared-word candidate list; whether a given pair
actually races (masking disjoint bits, semaphore/mutex ordering) is the LLM's
call.
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
        "full-word clobber of a sibling field is only partially modeled."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        # (namespace, word) -> list of writer dicts. Namespace separates the
        # THCON register file from the main config file (same index != same word).
        writers: dict[tuple, list[dict]] = defaultdict(list)
        unresolved: list[Finding] = []

        def add(word, field, fact, how):
            thr = registry.thread_of(fact["file"])
            writers[(registry.word_namespace(field), word)].append(
                {
                    "thread": thr,
                    "field": field,
                    "file": fact["file"],
                    "line": fact["line"],
                    "function": fact.get("function", "?"),
                    "how": how,
                }
            )

        for pw in fb.family("pointer_write"):
            kind, _ = registry.classify_write(pw)
            if kind not in registry.CFG_WRITE_KINDS:
                continue
            word, field = registry.resolve_word(pw.get("index_text", ""), fb.addr32)
            if word is not None:
                add(word, field, pw, f"mmio:{kind}")
            elif field:
                unresolved.append(self._unresolved(pw, field))
        for c in fb.family("call"):
            if "cfg_reg_rmw_tensix" in c.get("text", ""):
                word, field = registry.resolve_word(c.get("text", ""), fb.addr32)
                if word is not None:
                    add(word, field, c, "cfg_reg_rmw_tensix")
                elif field:
                    unresolved.append(self._unresolved(c, field))
        for m in fb.family("macro"):
            if registry.classify_macro(m.get("name", "")) == "ordered_write":
                word, field = registry.resolve_word(m.get("text", ""), fb.addr32)
                if word is not None:
                    add(word, field, m, f"instr:{m.get('name','')}")

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
                        f"{ns} CONFIG word {word} written by threads "
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
