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
        "On QUASAR the same mask shape is emitted as DEST2SRC_NO_MATH_DRAIN_UNCONFIRMED, "
        "not as a flag: the pre-flip-pointer mechanism is grounded on the WH/BH bank "
        "model, and Quasar's own unpack->dest semaphores / AutoTTSync / SrcS lane are "
        "not confirmed to need the drain — it must be grounded, never inferred by "
        "cross-arch analogy. "
        "The Dest->Src gate (DEST2SRC_*) reads the wait mask of the nearest TEXTUALLY "
        "preceding STALLWAIT and does NOT model control flow: a stall inside an "
        "`if`/`if constexpr` branch is credited to a move outside it (so a move whose "
        "only stall is conditional can read as gated), and conversely a move inside a "
        "loop is judged on the stall preceding the loop body in source order. Only the "
        "FIRST offending move per function is emitted, so a second, DIFFERENT defect "
        "later in the same function is not listed until the first is fixed. Bank-flip "
        "re-arm (DEST2SRC_DRAIN_REARMED) is detected textually, by a CLR_A/B/AB/SRC "
        "token on an intervening macro; a flip encoded in a numeric operand, or issued "
        "inside a MOP/replay between the stall and the move, is NOT seen. It reads "
        "the SAME function only: a gate that lives in the caller, or a "
        "move programmed into a MOP/replay whose stall sits elsewhere, lands as the "
        "DEST2SRC_WAIT_UNSEEN recall candidate, not a verdict — and a MOP-resident "
        "MOVD2A/MOVD2B whose STALLWAIT is in a DIFFERENT function is not correlated at "
        "all. The BLOCK mask (1st operand) is not verified to actually block the move "
        "(p_stall::STALL_MATH), only the wait condition. "
        "The UNPACK-SIDE half of the same handshake is modeled only as an ENCODING "
        "check on the dummy publication (UNPACR_NOP doing ZEROSRC / SET_DVALID / "
        "CLR_SRC). The wait-like-UNPACR control bit selects WHICH bank the "
        "publication waits on: set = Unpackers[i].SrcBank, the bank it clears, which "
        "lets unpack prepare the next bank while math consumes the current one; clear "
        "= MatrixUnit.Src?Bank, which under bank-pointer lockstep holds until NO bank "
        "is outstanding and is therefore a strictly STRONGER, serializing wait. So the "
        "default form is emitted as DUMMY_PUBLISH_SERIALIZING — a throughput/parity "
        "observation, NEVER corruption. The corruption risk on this handshake is the "
        "math-side drain (DEST2SRC_NO_MATH_DRAIN), which this bit does not fix. Only "
        "two encoding DEFECTS are flagged: the wait bit set together with a both-banks "
        "clear (DUMMY_PUBLISH_BOTH_BANKS_WAITLIKE), and a Wormhole-shaped packed "
        "UNP_ZEROSRC_* constant on an arch whose UNPACR_NOP takes these controls as "
        "separate operands (DUMMY_PUBLISH_PACKED_WAIT_WRONG_ARCH). Whether a "
        "SERIALIZING publication should actually change is NOT decided here: that "
        "needs its Dest->Src consumer, which is cross-thread and usually cross-file, "
        "so the pairing is left to the skill. Ownership for DUMMY_PUBLISH is tracked "
        "PER SRC REGISTER and is SPENT by each SET_DVALID (which flips "
        "Unpackers[i].SrcBank), so a guard is credited only to publications on the "
        "same register before the next publish. The Src register is read from the "
        "macro's first operand (SrcA/SrcB, UNP0/UNP1, UNP_A/UNP_B, or a bare 0/1); a "
        "publication whose selector is a variable is not attributed and is never "
        "flagged (see KNOWN_GAPS L11 for the function-NAME scope predicate, which "
        "misses MOP-config publishers). A TT_OP_ word builder is judged on its "
        "ENCODING ONLY — it has no source-position neighbours, so nothing "
        "preceding it can be credited as a "
        "guard and it establishes no ownership for what follows; a MOP/template word "
        "whose guard genuinely lives at the issue site therefore reads as SERIALIZING. "
        "Operand values are read through inline comments, but only literal 0/1 are "
        "recognised — a control passed as a variable or an expression reads as neither "
        "set nor clear (KNOWN_GAPS L12). "
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
        findings.extend(self._dummy_publication_guard(fb))
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
            flipped_since_stall = False
            for f in fb.facts_in(fn, ("macro",)):
                name = f.get("name", "")
                text = f.get("text", "")
                if any(t in name.upper() for t in registry.STALL_MACRO_SUBSTR):
                    last_stall = f
                    flipped_since_stall = False
                    continue
                if not registry.is_dest_to_src_move(name):
                    # A Src bank flip after the stall re-arms the hazard the drain
                    # settled. Non-flipping ops (incl. the moves themselves) do not.
                    if registry.is_bank_flip_macro(text):
                        flipped_since_stall = True
                    continue

                if last_stall is None:
                    hint, detail = (
                        "DEST2SRC_WAIT_UNSEEN",
                        f"{name} with no STALLWAIT before it in this function — "
                        "confirm the caller (or the MOP that replays it) drains the "
                        "FPU pipe and gates on the target bank",
                    )
                else:
                    cond = registry.stallwait_wait_operand(last_stall.get("text", ""))
                    need = registry.required_vld_token(name)
                    has_need = bool(need) and registry.condition_drains_unit(
                        cond, (need,)
                    )
                    has_any_vld = registry.condition_drains_unit(
                        cond, registry.SRC_BANK_VLD_TOKENS
                    )
                    has_math = registry.condition_drains_unit(
                        cond, registry.MATH_FPU_TOKENS
                    )
                    if not has_any_vld:
                        hint, detail = (
                            "DEST2SRC_WAIT_UNRELATED",
                            f"{name} preceded by a STALLWAIT that gates on neither "
                            "the target bank nor the FPU pipe — confirm what orders "
                            "this move",
                        )
                    elif not has_need:
                        # Gated on the OTHER Src register: proves nothing about the
                        # bank this move writes.
                        hint, detail = (
                            "DEST2SRC_WRONG_SRC_GATE",
                            f"{name} writes {'SrcA' if need == 'SRCA_VLD' else 'SrcB'} "
                            f"but its STALLWAIT gates on the other register's "
                            f"bank-valid condition, not {need} — the wait says nothing "
                            "about the bank being written",
                        )
                    elif not has_math:
                        if fb.arch == "quasar":
                            hint, detail = (
                                "DEST2SRC_NO_MATH_DRAIN_UNCONFIRMED",
                                f"{name} is gated on the Src bank-valid condition "
                                "without the FPU-pipeline drain — the WH/BH defect "
                                "shape, but Quasar's bank model (own semaphores, "
                                "AutoTTSync, SrcS lane) is not confirmed to need it; "
                                "ground against the Quasar ISA before judging",
                            )
                        else:
                            hint, detail = (
                                "DEST2SRC_NO_MATH_DRAIN",
                                f"{name} is gated on the Src bank-valid condition but "
                                "NOT on the FPU-pipeline drain (p_stall::MATH), so the "
                                "wait can observe the pre-flip bank pointer and the "
                                "move writes a bank the unpacker still owns — silent "
                                "wrong values, never a hang",
                            )
                    elif flipped_since_stall:
                        hint, detail = (
                            "DEST2SRC_DRAIN_REARMED",
                            f"{name} is correctly gated, but a Src bank flip was "
                            "issued between that STALLWAIT and this move — the drain "
                            "proved the FPU pipe empty at the stall, and the flip "
                            "re-armed exactly the in-flight-epilogue race it settled",
                        )
                    else:
                        continue  # correctly gated

                out.append(
                    Finding(
                        file=f["file"],
                        line=f.get("line", 0),
                        function=fn.name,
                        kind="dvalid:DEST_TO_SRC",
                        hint=hint,
                        detail=detail,
                        evidence=[self._ev(f, text or name)]
                        + (
                            [self._ev(last_stall, last_stall.get("text", ""))]
                            if last_stall is not None
                            else []
                        ),
                    )
                )
                break  # one finding per function
        return out

    def _dummy_publication_guard(self, fb: FactBase) -> list[Finding]:
        """The UNPACK half of check 5 — the dummy publication's bank wait.

        A dummy publication (UNPACR_NOP doing ZEROSRC / SET_DVALID / CLR_SRC) always
        clears Unpackers[i].SrcBank. The wait-like-UNPACR control bit selects which
        bank it WAITS on, and the two settings differ in strength, not in correctness:

          bit set   -> waits on Unpackers[i].SrcBank, the bank it clears. PIPELINED:
                       unpack can prepare the next bank while math consumes the
                       current one. Blackhole's preferred operating mode.
          bit clear -> waits on MatrixUnit.Src?Bank. Under the bank-pointer lockstep
                       invariant that pointer is only back with the unpackers when NO
                       bank is outstanding, so this is a strictly STRONGER, serializing
                       wait - it implies the own-bank condition.

        So a publication in the default form is SERIALIZING, not unguarded: the
        symptom is lost overlap (or a stall), never a silent wrong value. It is
        emitted as a throughput/parity recall candidate, never as corruption. The
        Dest->Src race that IS corruption lives on the math side
        (DEST2SRC_NO_MATH_DRAIN) and is not fixed by this bit.

        Two real defects DO live here and are flagged:
          - the wait bit set together with a BOTH-BANKS clear, which waits on one
            bank and then clears the other one too, and
          - a Wormhole-shaped packed NoOp constant used on an arch that takes the
            controls as separate operands, where the value lands in the wrong field.

        Scoped to functions whose PURPOSE is publishing a dummy bank - unscoped,
        ~90% of all publications in the tree are in the default form and the bucket
        carries no signal. The tool cannot pair a publisher with its Dest->Src
        consumer (cross-thread, usually cross-file), so it never proves the
        publication feeds one."""
        out: list[Finding] = []
        for fn in fb.functions:
            if not registry.is_dest_reuse_publisher_fn(fn.name):
                continue
            # Ownership of the unpacker's own bank, tracked PER Src register. A guard
            # on SrcA says nothing about SrcB.
            owned = {"A": False, "B": False}
            for f in fb.facts_in(fn, ("macro",)):
                name = f.get("name", "")
                up = name.upper()
                text = f.get("text", "")
                src = registry.src_reg_of(text)

                if any(t in up for t in registry.STALL_MACRO_SUBSTR):
                    cond = registry.stallwait_wait_operand(text)
                    if registry.condition_drains_unit(cond, ("SRCA_CLR",)):
                        owned["A"] = True
                    if registry.condition_drains_unit(cond, ("SRCB_CLR",)):
                        owned["B"] = True
                    continue

                if "UNPACR" in up and "UNPACR_NOP" not in up:
                    # A real UNPACR fills the unpacker's own bank and waits for it.
                    if src:
                        owned[src] = True
                    continue

                if "UNPACR_NOP" not in up:
                    continue
                # A TT_OP_ form builds a WORD rather than issuing at this source
                # position - but a word handed to a MOP or ckernel_template still
                # executes, so its own encoding is fair game. What does NOT apply is
                # sequencing: the word has no neighbours here, so nothing preceding
                # it establishes bank ownership and it establishes none for anything
                # after it. Skipping these entirely hid the rmsnorm MOP publications.
                is_word_builder = name.startswith("TT_OP_")
                # CLR_SRC / UNP_CLRSRC* are publications too: they clear the
                # unpacker's bank exactly as ZEROSRC does. Both spellings occur
                # (p_unpacr_nop::CLR_SRC vs p_unpacr::UNP_CLRSRC_*), and omitting
                # them hid every both-banks site in the tree, since those are all
                # encoded as a clear with Set_Dvalid=0.
                is_publish = any(
                    t in text for t in ("ZEROSRC", "SET_DVALID", "CLR_SRC", "CLRSRC")
                ) or registry.publication_sets_dvalid(text, fb.arch)
                if not is_publish:
                    continue

                # A Wormhole packed constant on Blackhole/Quasar does not mean what it
                # says: the value lands in Bank_Clr_Ctrl instead of the wait bit, with
                # no compile-time check. Report it and do not credit it as a guard.
                if registry.publication_misuses_packed_wait(text, fb.arch):
                    out.append(
                        Finding(
                            file=f["file"],
                            line=f.get("line", 0),
                            function=fn.name,
                            kind="dvalid:DUMMY_PUBLISH",
                            hint="DUMMY_PUBLISH_PACKED_WAIT_WRONG_ARCH",
                            detail=(
                                f"{name} uses a Wormhole-shaped packed UNP_ZEROSRC_* "
                                f"constant on {fb.arch}, whose UNPACR_NOP takes these "
                                "controls as SEPARATE operands. The value silently lands "
                                "in the wrong bit field (Bank_Clr_Ctrl, an unintended "
                                "both-banks clear) and leaves the wait bit clear; "
                                "TTI_UNPACR_NOP does not call TT_UNPACR_NOP_VALID, so the "
                                "operand overflow is not caught. Pass the operand instead"
                            ),
                            evidence=[self._ev(f, text or name)],
                        )
                    )
                    continue

                waits_own, both_banks = registry.publication_bank_controls(
                    text, fb.arch
                )
                if waits_own is None:
                    # Encoding not modeled for this arch - say nothing rather than
                    # imply safety.
                    continue

                # The one genuinely unsafe combination: the own-bank wait covers only
                # the bank being prepared, so clearing BOTH banks can overwrite one the
                # Matrix Unit still owns. Clearing both banks is correct ONLY with the
                # default (drained) wait.
                if waits_own and both_banks:
                    out.append(
                        Finding(
                            file=f["file"],
                            line=f.get("line", 0),
                            function=fn.name,
                            kind="dvalid:DUMMY_PUBLISH",
                            hint="DUMMY_PUBLISH_BOTH_BANKS_WAITLIKE",
                            detail=(
                                f"{name} clears BOTH banks while waiting only on "
                                "Unpackers[i].SrcBank, so it can overwrite the other bank "
                                "while the Matrix Unit still owns it. A both-banks clear "
                                "must keep the default drained wait on MatrixUnit.Src?Bank"
                            ),
                            evidence=[self._ev(f, text or name)],
                        )
                    )

                if waits_own and src and not is_word_builder:
                    # Establishes ownership for whatever is sequenced after it — this
                    # is how a wait-like ZEROSRC guards a following bare SET_DVALID.
                    owned[src] = True

                inherited = bool(src) and owned[src] and not is_word_builder
                if not (waits_own or both_banks or inherited or (src is None)):
                    out.append(
                        Finding(
                            file=f["file"],
                            line=f.get("line", 0),
                            function=fn.name,
                            kind="dvalid:DUMMY_PUBLISH",
                            hint="DUMMY_PUBLISH_SERIALIZING",
                            detail=(
                                f"{name} publishes a dummy Src bank in the default form, "
                                "waiting on MatrixUnit.Src?Bank rather than the "
                                "Unpackers[i].SrcBank it clears. That wait is STRONGER "
                                "(it holds until no bank is outstanding), so this is a "
                                "lost-overlap/parity observation, NOT corruption: unpack "
                                "cannot prepare the next bank while math consumes the "
                                "current one. Compare the other arch's twin; the "
                                "corruption risk on this handshake is the math-side drain"
                            ),
                            evidence=[self._ev(f, text or name)],
                        )
                    )

                # SET_DVALID hands the bank over and flips Unpackers[i].SrcBank, so
                # ownership of the NEW bank is not established by anything so far.
                if "SET_DVALID" in text and src and not is_word_builder:
                    owned[src] = False
        return out
