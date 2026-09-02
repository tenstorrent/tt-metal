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
        "The Dest->Src gate (DEST2SRC_*) reads the wait mask of the nearest TEXTUALLY "
        "preceding STALLWAIT and does NOT model control flow: a stall inside an "
        "`if`/`if constexpr` branch is credited to a move outside it (so a move whose "
        "only stall is conditional can read as gated), and conversely a move inside a "
        "loop is judged on the stall preceding the loop body in source order. It reads "
        "the SAME function only: a gate that lives in the caller, or a "
        "move programmed into a MOP/replay whose stall sits elsewhere, lands as the "
        "DEST2SRC_WAIT_UNSEEN recall candidate, not a verdict — and a MOP-resident "
        "MOVD2A/MOVD2B whose STALLWAIT is in a DIFFERENT function is not correlated at "
        "all. The BLOCK mask (1st operand) is not verified to actually block the move "
        "(p_stall::STALL_MATH), only the wait condition. "
        "The UNPACK-SIDE half of the same handshake is NOT modeled: the dummy "
        "publication (UNPACR_NOP doing ZEROSRC/SET_DVALID) must wait on "
        "Unpackers[i].SrcBank — the bank it clears — rather than MatrixUnit.Src?Bank, "
        "via the wait-like-UNPACR control bit or a preceding STALLWAIT on "
        "SRCA_CLR/SRCB_CLR. Pairing a publisher to its Dest->Src consumer is "
        "cross-thread and usually cross-file, so it is left to the skill; a publication "
        "missing that wait will NOT appear here even when the math side is flagged. "
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
        findings.extend(self._dest_to_src_waits(fb))
        return findings

    def _dest_to_src_waits(self, fb: FactBase) -> list[Finding]:
        """The Dest->Src move gate: MOVD2A/MOVD2B WRITE a Src bank, so the Wait
        Gate does not cover them and software must STALLWAIT. Waiting on the
        bank-valid condition ALONE is still wrong: it indexes MatrixUnit.Src?Bank
        live, and that pointer advances in the EPILOGUE of the preceding Matrix
        Unit instruction, so with a bank-flipping op in flight it tests the
        pre-flip bank, passes vacuously, and the move then writes the post-flip
        bank the unpacker owns. The wait must ALSO drain the FPU pipe (MATH).

        Mechanical and low-false-positive, so it is a FLAG. Only the first
        offending move per function is emitted (the rest are the same defect)."""
        out: list[Finding] = []
        for fn in fb.functions:
            last_stall = None
            for f in fb.facts_in(fn, ("macro",)):
                name = f.get("name", "")
                if any(t in name.upper() for t in registry.STALL_MACRO_SUBSTR):
                    last_stall = f
                    continue
                if not registry.is_dest_to_src_move(name):
                    continue

                if last_stall is None:
                    # The gate may legitimately live in the CALLER, or the move may
                    # be programmed into a MOP/replay whose stall sits elsewhere —
                    # so this is a RECALL candidate, never a flag.
                    hint, detail = (
                        "DEST2SRC_WAIT_UNSEEN",
                        f"{name} with no STALLWAIT before it in this function — "
                        "confirm the caller (or the MOP that replays it) drains the "
                        "FPU pipe and gates on the target bank",
                    )
                else:
                    cond = registry.stallwait_wait_operand(last_stall.get("text", ""))
                    has_vld = registry.condition_drains_unit(
                        cond, registry.SRC_BANK_VLD_TOKENS
                    )
                    has_math = registry.condition_drains_unit(
                        cond, registry.MATH_FPU_TOKENS
                    )
                    if has_math:
                        continue  # correctly gated
                    if has_vld:
                        hint, detail = (
                            "DEST2SRC_NO_MATH_DRAIN",
                            f"{name} is gated on the Src bank-valid condition but "
                            "NOT on the FPU-pipeline drain (p_stall::MATH), so the "
                            "wait can observe the pre-flip bank pointer and the move "
                            "writes a bank the unpacker still owns — silent wrong "
                            "values, never a hang",
                        )
                    else:
                        hint, detail = (
                            "DEST2SRC_WAIT_UNRELATED",
                            f"{name} preceded by a STALLWAIT that gates on neither "
                            "the target bank nor the FPU pipe — confirm what orders "
                            "this move",
                        )

                out.append(
                    Finding(
                        file=f["file"],
                        line=f.get("line", 0),
                        function=fn.name,
                        kind="dvalid:DEST_TO_SRC",
                        hint=hint,
                        detail=detail,
                        evidence=[self._ev(f, f.get("text", "") or name)]
                        + (
                            [self._ev(last_stall, last_stall.get("text", ""))]
                            if last_stall is not None
                            else []
                        ),
                    )
                )
                break  # one finding per function
        return out
