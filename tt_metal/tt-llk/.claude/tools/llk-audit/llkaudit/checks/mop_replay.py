# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
mop-replay checker — an instruction word's SLOT, not its line.

A `TT_OP_*` value is an opcode VALUE, not an issued instruction. It gets
installed into a MOP slot or captured into a replay buffer, and an expander
issues it later, so its execution POSITION, its repeat COUNT and its
NEIGHBOURS come from the MOP/replay program and never from the source order.
Reading such a word as though it executed at its line is the dominant recall
error on this surface: a `static constexpr TT_OP_UNPACR(...)` above a site
guards nothing, and the word that ends an outer iteration lands immediately
before the NEXT outer iteration's `START_OP` — an adjacency that exists nowhere
in the text. Which word that is depends on the live END slots: `END_OP1` when
both are live, `END_OP0` when `END_OP1` is a plain NOP, and the last inner-loop
word (possibly a `*_LAST` override) when `END_OP0` is a NOP.

Per the ISA, `ckernel_template` programs MOP template 1 (loop-counted from
MopCfg: `START_OP`, inner loop, `END_OP0`/`END_OP1`, with `LOOP0_LAST` /
`LOOP1_LAST` overriding the LAST inner iteration) and `ckernel_unpack_template`
programs template 0 (each iteration selected by a runtime mask, so WHICH words
execute is runtime data). The Replay Expander sits after the MOP Expander and
before the Wait Gate: a MOP may emit `REPLAY`, but a `REPLAY` expansion may not
contain `MOP`.

This check is a pure AUGMENTOR. It does not model either expander; it recalls
the sites where a word's execution point is NOT its line, so the skill resolves
the word at its slot / at the `run()` or `replay()` site.
"""

from __future__ import annotations

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding

#: Slots whose executed neighbourhood a textual reading gets wrong, each with the
#: neighbourhood that replaces it. Per outer iteration the expander issues
#: START_OP, the inner loop, then END_OP0, then END_OP1 — and END_OP1 is issued
#: only when END_OP0 is not a plain NOP (see ExpandTemplate1 in MOPExpander.md),
#: so which word ends an iteration depends on which END slots are live.
_SLOT_NEIGHBOUR_NOTE = {
    "END_OP0": (
        "set_end_op leaves END_OP1 a plain NOP, so this word ends the outer "
        "iteration and immediately precedes the next outer iteration's START_OP"
    ),
    "END_OP0/END_OP1": (
        "only the captured END_OP0 is reported; END_OP1 still follows it, and "
        "that word — not this one — precedes the next outer iteration's START_OP"
    ),
    "LOOP0_LAST": (
        "this replaces the last inner-loop op, so any live END op issues after "
        "it and the next outer iteration's START_OP follows that"
    ),
    "LOOP1_LAST": (
        "this replaces the last inner-loop op, so any live END op issues after "
        "it and the next outer iteration's START_OP follows that"
    ),
}


def _is_literal(text: str) -> bool:
    """True for an integer literal index (decimal or hex), which pins the slot."""
    try:
        int(text, 0)
        return True
    except ValueError:
        return False


class MopReplay(Check):
    name = "mop-replay"
    description = "MOP/replay-slotted instruction words: execution point is the slot, not the line"
    blind_spots = (
        "Neither expander is modeled: no iteration COUNTS are resolved (OuterCount/"
        "InnerCount come from MopCfg at runtime, and a non-NOP LoopOp1 DOUBLES the "
        "inner count while alternating the two ops), and template 0's per-iteration "
        "mask is runtime data — so this check never says how many times a word runs, "
        "only that the count is not one-per-line. Slot attribution comes ONLY from the "
        "`set_*_op` / `set_last_*_loop_instr` setters; a loop op passed POSITIONALLY to "
        "the `ckernel_template` / `ckernel_unpack_template` CONSTRUCTOR is invisible "
        "(the extractor does not visit CXXConstructExpr, so the constructor emits no "
        "call fact) — those words surface only via the unattributed-word hint, without "
        "a slot, and nearly every loop op is constructor-passed (KNOWN_GAPS L14). "
        "`set_end_ops(a, b)` reports only its FIRST argument (the extractor captures "
        "arg0 only), so a flip installed as END_OP1 is missed. For the same reason a "
        "replay record/expand reports its start INDEX but not its COUNT, so the "
        "buffer-occupancy checks the ISA would allow — `Count == 0` meaning 64, and "
        "`Index + Count` wrapping `% 32` over another user's slots — are NOT performed "
        "(KNOWN_GAPS L13). Record/expand pairing across functions is not tracked, nor "
        "is which MOP program is live at a given `run()`: a replay whose record ran on "
        "another path, or not at all, issues whatever words the buffer last held "
        "(KNOWN_GAPS L15), and that is the skill's verdict. The unattributed-word "
        "hint reads the extractor's `opcode_value` family, so a word the extractor "
        "never filed there is not recalled: the NESTED TT_OP_* expansions from inside "
        "a TTI_* macro body are dropped (they spell at their ckernel_ops.h `#define` "
        "site, not at a use site), as is anything in an unparsed header. The hint is "
        "further scoped to Src flips, inter-thread sync ops and waits — "
        "an arithmetic/SFPU opcode value is not recalled here (instruction-latency "
        "widens its own enumeration to `TT_OP_*` instead), so this check is NOT an "
        "enumeration of every MOP-resident instruction."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        findings: list[Finding] = []
        findings += self._slotted_words(fb)
        findings += self._replay_calls(fb)
        findings += self._unattributed_words(fb)
        return findings

    # --- words installed into a known MOP slot --------------------------------
    def _slotted_words(self, fb: FactBase) -> list[Finding]:
        out: list[Finding] = []
        for f in fb.family("call"):
            slot, word = registry.mop_slot_of(f)
            if not slot:
                continue
            flips = registry.mop_word_flips_src(word)
            if flips:
                hint = "MOP_SLOTTED_SRC_FLIP"
                detail = (
                    f"a Src bank flip is installed as the MOP's {slot}; it hands the "
                    "bank back where the MOP issues it, not at this line"
                )
                note = _SLOT_NEIGHBOUR_NOTE.get(slot)
                if note:
                    detail += f" — {note}"
            else:
                hint = "MOP_SLOTTED_WORD"
                detail = (
                    f"instruction word installed as the MOP's {slot}; its execution "
                    "position, repeat count and neighbours are the MOP's, not this line's"
                )
            out.append(
                Finding(
                    file=f["file"],
                    line=f.get("line", 0),
                    function=f.get("function", ""),
                    kind=slot,
                    hint=hint,
                    detail=detail,
                    evidence=[self._ev(f, f"{f.get('name','')}({word})")],
                )
            )
        return out

    # --- replay record / expand ----------------------------------------------
    def _replay_calls(self, fb: FactBase) -> list[Finding]:
        out: list[Finding] = []
        for f in fb.family("call"):
            op, exec_mode = registry.replay_op_of(f)
            if not op:
                continue
            index = (f.get("arg0") or "").strip()
            if op == "record" and exec_mode == "NOEXEC":
                out.append(
                    Finding(
                        file=f["file"],
                        line=f.get("line", 0),
                        function=f.get("function", ""),
                        kind="record",
                        hint="REPLAY_RECORD_NOEXEC",
                        detail=(
                            "record with Exec off (explicit NoExec, or the lltt/"
                            "ckernel default): the instructions written after this "
                            "call are captured into the replay buffer and do NOT "
                            "execute here — they issue at each later replay"
                        ),
                        evidence=[self._ev(f, f"{f.get('text','')}({index}, …)")],
                    )
                )
            elif op == "record" and exec_mode == "UNRESOLVED":
                out.append(
                    Finding(
                        file=f["file"],
                        line=f.get("line", 0),
                        function=f.get("function", ""),
                        kind="record",
                        hint="REPLAY_RECORD_EXEC_UNRESOLVED",
                        detail=(
                            "record whose Exec is a dependent template argument, so "
                            "whether the following instructions also execute here is "
                            "decided by the caller's instantiation"
                        ),
                        evidence=[self._ev(f, f"{f.get('text','')}({index}, …)")],
                    )
                )
            if index and not _is_literal(index):
                out.append(
                    Finding(
                        file=f["file"],
                        line=f.get("line", 0),
                        function=f.get("function", ""),
                        kind=op,
                        hint="REPLAY_INDEX_UNRESOLVED",
                        detail=(
                            f"replay {op} start index is not a literal ({index}), so "
                            "which of the "
                            f"{registry.REPLAY_BUF_SIZE} per-thread buffer slots this "
                            "occupies — and whether it overlaps another user's range — "
                            "is not statically known"
                        ),
                        evidence=[self._ev(f, f"{f.get('text','')}({index}, …)")],
                    )
                )
        return out

    # --- opcode values whose slot is not attributable ------------------------
    def _unattributed_words(self, fb: FactBase) -> list[Finding]:
        """A sync/ordering-relevant `TT_OP_*` word with no resolvable slot.

        This is the shape a textual reading misreads as an issued instruction:
        it may be a constructor-passed loop op, a named constant consumed
        elsewhere, or a word captured into a replay buffer. The extractor files
        these under their own `opcode_value` family (never `macro`), so no
        instruction-consuming check can read one as issued at its line.
        """
        out: list[Finding] = []
        for f in fb.family(registry.OPCODE_VALUE_FAMILY):
            name = f.get("name", "") or ""
            if not registry.is_mop_word(name):
                continue
            text = f.get("text", "") or name
            up = name.upper()
            is_flip = registry.mop_word_flips_src(text)
            is_sync = any(t in up for t in registry.MOP_WORD_SYNC_SUBSTR)
            if not (is_flip or is_sync):
                continue
            what = "a Src bank flip" if is_flip else "an inter-thread sync/wait op"
            out.append(
                Finding(
                    file=f["file"],
                    line=f.get("line", 0),
                    function=f.get("function", ""),
                    kind=name,
                    hint="MOP_WORD_SLOT_UNATTRIBUTED",
                    detail=(
                        f"{what} written as an opcode VALUE, not an issued "
                        "instruction, and its slot did not resolve here — it does not "
                        "execute at this line, and its count is per MOP iteration; "
                        "resolve it where the MOP/replay is issued"
                    ),
                    evidence=[self._ev(f, name)],
                )
            )
        return out
