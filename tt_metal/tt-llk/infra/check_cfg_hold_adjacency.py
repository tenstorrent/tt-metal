#!/usr/bin/env python3
"""Flag a Tensix config write that can land while an earlier math instruction is held at issue.

On Wormhole the math unit reads its NUMERIC config (the ALU_FORMAT_SPEC_REG* group, ALU_ACC_CTRL_*,
ALU_ROUNDING_MODE_*, FP16A_FORCE_Enable, DEST_OFFSET_Enable) combinationally in the ISSUE STAGE.
Several instructions impose a fixed hold on the next instruction's issue. If a reader is held there
and a config write commits during the hold, the reader uses the NEW value -- silent wrong data, for
example a move whose SrcA format flips between TF32 and FP32 and so writes 16-bit datums at 16-bit
Dst addresses instead of 32-bit ones.

Address-shaping config (ADDR_MOD_*, DEST_REGW_BASE_Base, DEST_TARGET_REG_CFG_MATH_Offset,
FIDELITY_BASE_Phase, ADDR_MOD_SET_Base) is frozen at arbitration and is deliberately NOT flagged.

The guard is an ORDERING stall, not a delay:
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
Block bit B7 holds Configuration-Unit instructions until the math pipe drains (condition C7).
Padding with NOPs is NOT a valid guard: it works only while the pad exceeds the hold, and the hold
length is a hardware property the code does not control.

WORMHOLE ONLY, by design, and the exclusion is PER FIELD rather than per architecture.
On Blackhole, ALU_FORMAT_SPEC_REG* and ALU_ACC_CTRL_* are captured into the instruction as it is
accepted, so a held instruction keeps the values it was accepted with. Measured on silicon at nine
sites with a proven-live detector: seven immune outright, and the two that went red were shown to be
the prober's own side effect by a control that swapped the injected inducer and passed with its
detector still live. Those two field groups cover essentially every call site, so a Blackhole copy of
this check could never fire. That does not carry to the rest of the list.

FP16A_FORCE_Enable is unestablished there -- its one Blackhole writer sits on the integer reduce
path, whose detector was dead, so that site measured nothing. Blackhole also has math config read
when the instruction is RELEASED from the hold rather than when it is accepted, which is exposed the
same way with the opposite verdict; no Blackhole source writes any of those fields today.

A Blackhole arm would therefore need a different PREDICATE, not just a different field list, and in
particular NO hold-inducer term. This check keys on a named inducer because on Wormhole the hold is
imposed by a specific instruction on the one presented next, and is short enough to measure -- which
is what makes an adjacency test and a 3-instruction window sound here. The Blackhole exposure has
neither property: the parking that creates it follows essentially every arithmetic operation, and
the dependency, port and operand-readiness holds are longer than the post-operation one and have no
fixed length. The Blackhole predicate is "a write to one of those fields inside a stretch of math
work with no intervening math drain" -- no inducer, no adjacency, no window.

LIMITATION, and it bounds what a clean run means. A fixed post-instruction stall is not the only
way an instruction is held at issue on Wormhole: an ALU instruction is also held while the Dest
dependency scoreboard matches an in-flight overlapping write, and that match does not compare thread
ids. Such a hold has no adjacent inducer to key on, and the write it waits for may be another
thread's, so it is not decidable from the text of one file. This check therefore covers the
adjacency-decidable shape only; a clean run is not a proof that no config write can land in a hold.

Quasar has a DIFFERENT, inverted hazard (a following instruction reading a stale write, closed by
one slot of separation) which this check does not model. Pointing this check at another
architecture produces false positives and hides the real rule.
"""
import argparse
import hashlib
import re
import sys

# --- hold inducers -----------------------------------------------------------------------------
# Only rules confirmed on silicon are encoded. A broader rule ("1 cycle after any ALU instruction")
# exists in the hazard database but is contradicted by measurement, so it is deliberately omitted:
# including it flags shipped sequences that demonstrably do not fail.
MOVE_TO_SRCA = r"MOVD2A|MOVB2A"
READS_DEST = r"MOVD2A|MOVD2B|ELWMUL|ELWADD|ELWSUB|MVMUL|DOTPV|GMPOOL|GAPOOL"
ANY_MOVE = r"MOVB2D|MOVA2D|MOVB2A|MOVD2B|MOVD2A|MOVDBGA2D"

INSTR = re.compile(r"\bTTI?_([A-Z][A-Z0-9_]{2,})\b")
CFG_WRITE = re.compile(
    r"\b(cfg_reg_rmw_tensix|TTI?_SETC16|TTI?_WRCFG|TTI?_RMWCIB[0-3])\b"
)
NUMERIC_CFG = re.compile(
    r"(ALU_FORMAT_SPEC_REG\w*|ALU_ACC_CTRL_\w+|ALU_ROUNDING_MODE_\w+|FP16A_FORCE_Enable|DEST_OFFSET_Enable)"
)
# An ordering guard: blocks the Configuration Unit (B7) until the math pipe is empty (C7).
GUARD = re.compile(r"STALLWAIT\s*\(\s*p_stall::STALL_CFG")
# A reader whose numeric config is sampled at issue.
READER = re.compile(
    r"\bTTI?_(" + ANY_MOVE + r"|MVMUL|ELWADD|ELWSUB|ELWMUL|GMPOOL|GAPOOL|DOTPV)\b"
)

# How far after the reader a config write can still land inside the hold. Calibrated against a
# measured dose-response, not guessed: with the reader held, 1 and 2 filler instructions between it
# and the write STILL corrupt, 3 and 4 do not. So a write up to 3 instructions after the reader is
# in scope; 4 is out.
WINDOW = 3


def opcode(line):
    m = INSTR.search(line)
    return m.group(1) if m else None


def holds(inducer, victim):
    """Does `inducer` hold `victim` at issue? Returns the rule text, or None."""
    if inducer == "SHIFTXB":
        return "SHIFTXB holds every following instruction for one cycle"
    if inducer == "MOVD2B" and victim != "MOVD2B":
        return "MOVD2B holds a following non-MOVD2B"
    if inducer in ("MOVA2D", "MOVDBGA2D", "MOVB2D") and re.fullmatch(
        READS_DEST, victim or ""
    ):
        return f"{inducer} holds a following instruction that reads Dest"
    if re.fullmatch(MOVE_TO_SRCA, inducer or "") and not re.fullmatch(
        MOVE_TO_SRCA, victim or ""
    ):
        return f"{inducer} holds anything other than another move-to-SrcA"
    return None


def site_key(path, inducer_line, victim_line, cfg_line):
    """Content identity for the baseline.

    A line number is not a stable identity: inserting an unrelated line above a site makes that
    unchanged site look new, and replacing the sequence at an accepted line keeps it silently
    allowlisted. Key on what the site IS -- the two opcodes and the register written -- so the
    entry survives code motion and stops applying the moment the sequence changes.
    """
    reg = NUMERIC_CFG.search(cfg_line)
    ident = "|".join(
        [
            opcode(inducer_line) or "?",
            opcode(victim_line) or "?",
            reg.group(1) if reg else "?",
        ]
    )
    return f"{path}:{hashlib.sha1(ident.encode()).hexdigest()[:10]}  # {ident}"


def is_guard(line):
    return bool(GUARD.search(line) and "MATH" in line)


def logical_statements(lines):
    """Join continuation lines into whole statements, keyed by their FIRST line number.

    A call split over several lines (`cfg_reg_rmw_tensix<` with its register on the next line
    occurs in this tree) would otherwise never match both the call and the register pattern, and
    the check would silently pass over it."""
    out, buf, start = [], "", None
    for i, l in enumerate(lines):
        t = l.strip()
        if not t or t.startswith(("//", "*", "/*", "#")):
            if buf:
                out.append((start, buf))
                buf, start = "", None
            continue
        if start is None:
            start = i
        buf = (buf + " " + t).strip()
        # a statement ends at ';', or at a brace when it is not a continued call
        if t.endswith((";", "{", "}")) or len(buf) > 2000:
            out.append((start, buf))
            buf, start = "", None
    if buf:
        out.append((start, buf))
    return out


def scan(path):
    """Yield (line_no, inducer_line, victim_line, cfg_line, rule) for each unguarded adjacency."""
    lines = open(path, errors="ignore").read().split("\n")
    # instruction stream: anything that emits a Tensix instruction, guards included -- a guard that
    # is invisible to the stream makes a guarded site look unguarded.
    stream = [
        (i, l)
        for i, l in enumerate(lines)
        if (INSTR.search(l) or CFG_WRITE.search(l))
        and not l.strip().startswith(("//", "*", "/*"))
    ]
    for a, (ia, la) in enumerate(stream):
        if not (CFG_WRITE.search(la) and NUMERIC_CFG.search(la)):
            continue
        for b in range(max(0, a - WINDOW), a):
            ib, lb = stream[b]
            if not READER.search(lb):
                continue
            if any(is_guard(x) for _, x in stream[b + 1 : a]):
                continue  # already ordered
            # The inducer must IMMEDIATELY precede the victim. A hold catches only the
            # instruction presented next; anything issued in between absorbs it, and by the
            # time a later instruction is presented the window has expired.
            if b == 0:
                continue
            ic, lc = stream[b - 1]
            rule = holds(opcode(lc), opcode(lb))
            if rule:
                yield ia + 1, (ic + 1, lc), (ib + 1, lb), la, rule
                break


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*")
    ap.add_argument(
        "--baseline", help="file of 'path:line' sites that are known and accepted"
    )
    ap.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite the baseline from what is found",
    )
    args = ap.parse_args()

    accepted = set()
    if args.baseline and not args.write_baseline:
        try:
            accepted = {
                l.split("#")[0].strip()
                for l in open(args.baseline)
                if l.strip() and not l.startswith("#")
            }
        except FileNotFoundError:
            pass

    found, new = [], []
    for path in args.files:
        # Wormhole only -- see the module docstring. Other trees are skipped, not passed.
        if "tt_llk_wormhole_b0" not in path:
            continue
        for cfg_line, (iline, ltxt), (vline, vtxt), cfgtxt, rule in scan(path):
            key = site_key(path, ltxt, vtxt, cfgtxt)
            found.append(key)
            if key.split("#")[0].strip() not in accepted:
                new.append((key, iline, ltxt, vline, vtxt, cfg_line, cfgtxt, rule))

    if args.write_baseline:
        with open(args.baseline, "w") as f:
            f.write(
                "# Config writes that can land under a held math instruction (Wormhole).\n"
            )
            f.write(
                "# Pre-existing sites, accepted so the check fails only on NEW ones.\n"
            )
            f.write(
                "# Removing a line here is how you assert a site has been guarded.\n"
            )
            for k in sorted(set(found)):
                f.write(k + "\n")
        print(f"baseline written: {len(set(found))} site(s)")
        return 0

    for key, iline, ltxt, vline, vtxt, cfg_line, cfgtxt, rule in new:
        path = key.split(":")[0]
        print(
            f"{path}:{cfg_line}: config write can land while an earlier instruction is held at issue"
        )
        print(f"  L{iline:<5} {ltxt.strip()[:88]}")
        print(f"         ^ {rule}")
        print(f"  L{vline:<5} {vtxt.strip()[:88]}")
        print(f"         ^ held here; reads its numeric config at issue")
        print(f"  L{cfg_line:<5} {cfgtxt.strip()[:88]}")
        print(f"         ^ commits during the hold")
        print(
            "  fix: TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU); before the write"
        )
        print(
            "       (an ordering stall -- NOT NOP padding, which only works while it exceeds the hold)\n"
        )
    if new:
        print(
            f"{len(new)} new site(s). If a site is intentional, add it to the baseline with a reason."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
