# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
srcreg-bank checker — the SrcA/SrcB data-valid (dvalid) handshake.

The unpacker fills a Src bank and hands it to the Matrix Unit (set-dvalid); the
FPU consumes and hands it back (clear-dvalid). The correctness *verdict* —
bank-pointer lockstep across the tile loop, dvalid placement, single-thread
ownership — is a cross-thread state-machine property that is SEMANTIC (it needs
branch-by-branch reasoning the LLM does), so this check does NOT try to render
it. It is a pure AUGMENTOR:

  * it RECALLS every dvalid handshake control point (SETDVALID / CLEARDVALID) —
    a small, clean worklist the LLM adjudicates for placement/lockstep; and
  * it flags the one concrete, ISA-grounded, mechanical pattern:
      RAW_SETDVALID_BH — a *raw* TTI_SETDVALID on Blackhole, which is
      ISA-unsupported (it corrupts ImpliedSrcBFmt to an unpredictable value);
      the supported form is UNPACR_NOP(...,SET_DVALID,...).

Everything the tool cannot see (bank-flip lockstep, the MOV*2D consume side, the
DISABLE_IMPLIED_SRC?_FMT bit, the Quasar SrcS third lane, Dst/LReg sharing) is
declared in blind_spots and left to the /srcreg-bank-sync-audit skill.
"""
from __future__ import annotations

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding


class SrcRegBank(Check):
    name = "srcreg-bank"
    description = "SrcA/SrcB dvalid handshake sites; raw SETDVALID on Blackhole"
    blind_spots = (
        "Only the dvalid handshake CONTROL POINTS (SETDVALID/CLEARDVALID) are "
        "recalled; the bank-flip LOCKSTEP verdict (unpacker SrcBank vs FPU "
        "SrcABank/SrcBBank incrementing 1:1 across every branch of the tile "
        "loop) is a semantic state-machine property NOT modeled here — the "
        "MOV*2D consume side is not counted. dvalid PLACEMENT (set only after "
        "the fill completes / cleared only after the FPU drains) is not checked. "
        "Single-thread ownership of the bank-pointer bits is not checked. The "
        "Blackhole DISABLE_IMPLIED_SRCA/ B_FMT_Base bit on the consuming MOV is "
        "not verified. Quasar's third unpacker / SrcS lane (llk_srcs.h, UNPACR2/"
        "PACR1, *_SRCS_RDY interlocks) is not modeled. Dst/LReg shared-once "
        "overwrite (rides MATH_PACK / mutex::SFPU) is out of scope here. "
        "IMPORTANT: the SET side is recalled ONLY as a RAW SETDVALID; the SUPPORTED "
        "set path — UNPACR_NOP(...,SET_DVALID,...) — is DELIBERATELY excluded (so it "
        "is never mis-flagged as the raw anti-pattern), so on correct code the SET "
        "control points are largely absent from the worklist — the LLM must find the "
        "UNPACR_NOP set sites itself when pairing set↔clear."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        findings: list[Finding] = []
        seen: set = set()  # dedup nested macro expansions at one source line
        for m in fb.family("macro"):
            name = m.get("name", "")
            op, role = registry.classify_srcreg_macro(name)
            if op is None:
                continue
            key = (m["file"], m.get("line"), name)
            if key in seen:
                continue
            seen.add(key)

            fn = fb.enclosing(m["file"], m["off"])
            thr = registry.thread_of(m["file"])

            # The concrete ISA-grounded flag: a raw SETDVALID on Blackhole.
            if op == "SETDVALID" and fb.arch == "blackhole":
                hint = "RAW_SETDVALID_BH"
                detail = (
                    f"raw {name} is ISA-unsupported on Blackhole "
                    "(corrupts ImpliedSrcBFmt); use UNPACR_NOP(...,SET_DVALID,...)"
                )
            else:
                # Recall candidate: a dvalid control point for the LLM to place-
                # and lockstep-check (DVALID_SET / DVALID_CLEAR).
                hint = role
                detail = f"{thr} {op} — SrcA/SrcB dvalid handshake control point"

            findings.append(
                Finding(
                    file=m["file"],
                    line=m.get("line", 0),
                    function=fn.name if fn else m.get("function", ""),
                    kind=f"dvalid:{op}",
                    hint=hint,
                    detail=detail,
                    evidence=[self._ev(m, m.get("text", "") or name)],
                )
            )
        return findings
