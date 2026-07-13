# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
reconfig-stall checker — a config rewrite that a running execution unit samples
must be preceded by a STALLWAIT whose condition DRAINS that unit (packer->PACK
[PACK0..PACK3], unpacker->UNPACK[0-2], math->MATH|WAIT_SFPU). A THCON-only stall
orders the GPR->cfg write but does NOT drain the unit — the classic miss.

Recall: for each candidate reconfig/uninit/config-writer function, walk ALL its
CONFIG writes in order (not just the first) and, for each, check that a
unit-draining STALLWAIT precedes it AND the unit was not re-armed (by a
UNPACR/PACR/matrix issue for this thread) between that drain and the write. Emit
one candidate per function, at the first offending write:
  NO_UNIT_DRAIN      — a cfg write with no preceding STALLWAIT at all
  THCON_ONLY         — a STALLWAIT precedes it, but its condition drains no unit
  DRAIN_REARMED      — a draining STALLWAIT precedes it, but the unit was re-issued
                       between the drain and the write (the drain no longer holds)
  PARTIAL_MATH_DRAIN — a MATH reconfig whose stall drains only ONE of the FPU/SFPU
                       engines (insufficient if the other engine reads the field)
Latched registers (L1_Dest_addr) are an expected THCON-only case: such a write
is SKIPPED (not the whole function) so a later sampled write is still checked.
"""

from __future__ import annotations

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding


class ReconfigStall(Check):
    name = "reconfig-stall"
    description = "Reconfig/uninit config write missing a unit-draining stall"
    blind_spots = (
        "The latched-vs-sampled register distinction is only modeled by a small "
        "allowlist (L1_Dest_addr); other latched registers would false-positive. "
        "Whether the running unit is provably idle by a handshake instead of a "
        "stall is not modeled. Cross-arch (WH self-guards vs BH relies-on-caller) "
        "asymmetry is surfaced as data, not judged. The STALLWAIT BLOCK mask "
        "(1st operand) is not verified to actually block the config-write "
        "instruction — only the drain condition (2nd operand) is checked. "
        "The unpacker/packer sub-units (UNPACK0/UNPACK1[/UNPACK2], PACK0..PACK3) are "
        "DISTINCT drainable p_stall bits, but this checker does not resolve which "
        "sub-unit a reconfig targets, so ANY sub-unit drain token clears the write — a "
        "stall on the WRONG sub-unit (e.g. draining unpacker 0 before an unpacker-1 "
        "reconfig) is NOT flagged (the LLM must confirm the drained sub-unit is the one "
        "that samples the reconfig'd field). Only MATH models a partial drain (FPU vs "
        "SFPU). A STALLWAIT whose wait-condition is a COMPOUND SUPERSET mask (e.g. "
        "`p_stall::ALL_THREAD_RES` = THCON|UNPACK|PACK|MATH|XMOV on WH/BH) is not "
        "decomposed: it reads as THCON_ONLY/NO_UNIT_DRAIN even though it drains the unit "
        "(latent — no LLK reconfig uses it today; and it omits the SFPU, so even "
        "decomposed it is only a partial MATH drain). The LLM must treat a superset "
        "wait-mask as draining the named unit. "
        "Unit re-arm detection covers UNPACR/PACR and the curated matrix consumer set "
        "(see registry.CONSUMER_MATH_SUBSTR); a re-issue via an unmodeled macro "
        "would not invalidate an earlier drain. In particular a MOP / mop_run between "
        "the drain and the cfg write is OPAQUE and is NOT counted as a re-arm — the "
        "drain is credited across it even though the MOP may re-issue the unit (the "
        "LLM must check whether a MOP between the drain and the sampled write re-armed "
        "it). Only the FIRST offending write in "
        "a function is emitted (one candidate/function). SCOPE GATE: only functions "
        "whose NAME matches registry.RECONFIG_FN_SUBSTR are scanned at all — a "
        "mid-stream config rewrite in a function named OUTSIDE that list (e.g. a "
        "bespoke hw_configure / tile-dims setter) is never checked, a false negative "
        "the LLM must widen for (widen the fn-name list, not the per-write scan)."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        findings: list[Finding] = []
        for fn in fb.functions:
            if not registry.is_reconfig_fn(fn.name):
                continue
            # file's thread, else fall back to the function NAME (thread_of_fact) —
            # consistent with cfg_word_overlap. A reconfig fn in a token-less file
            # would otherwise resolve UNKNOWN -> empty DRAIN_UNIT_TOKENS -> a
            # correctly-drained write falsely flagged NO_UNIT_DRAIN (latent today:
            # all reconfig fns live in pack/unpack/math files).
            thr = registry.thread_of_fact({"file": fn.file, "function": fn.name})
            drain_tokens = registry.DRAIN_UNIT_TOKENS.get(thr, ())

            facts = fb.facts_in(fn, ("macro", "call", "pointer_write"))

            # A config write to the UNIT-SAMPLED CONFIG file. Note: SETDMAREG
            # writes a GPR (a source value), NOT a sampled config reg, so it is
            # excluded (registry.RECONFIG_WRITE_MACRO_SUBSTR) — otherwise a GPR
            # setup line would be mistaken for the reconfig write.
            def is_cfg_write(f) -> bool:
                if f["family"] == "pointer_write":
                    kind, _ = registry.classify_write(f)
                    return kind in registry.CFG_WRITE_KINDS  # cfg[] MMIO (not GPR)
                if f["family"] == "macro":
                    up = f.get("name", "").upper()
                    return f.get("name", "").startswith(
                        registry.INSTR_PREFIXES
                    ) and any(s in up for s in registry.RECONFIG_WRITE_MACRO_SUBSTR)
                if f["family"] == "call":
                    # cfg_rmw / cfg_rmw_gpr — Quasar's dominant config-write idiom;
                    # without these a Quasar reconfig looks like it writes nothing.
                    # Gate on those two names ONLY (not write_call_kind, which also
                    # matches reg_write — a raw/GPR register write that needs no
                    # unit drain and must not get a spurious NO_UNIT_DRAIN).
                    return registry.is_cfg_rmw_helper(f)
                return False

            # Instructions that RE-ARM this thread's unit (so a drain issued
            # before them no longer holds for a config write after them).
            def reissues_unit(f) -> bool:
                if f["family"] != "macro":
                    return False
                role = registry.classify_macro(f.get("name", ""))
                return (
                    (role == "consumer_unpack" and thr == "UNPACK")
                    or (role == "consumer_pack" and thr == "PACK")
                    or (role == "consumer_math" and thr == "MATH")
                )

            reissue_offs = [f["off"] for f in facts if reissues_unit(f)]
            stalls = [
                f
                for f in facts
                if f["family"] == "macro"
                and registry.classify_macro(f.get("name", "")) == "stall"
            ]

            emitted = False
            for w in facts:
                if emitted or not is_cfg_write(w):
                    continue
                # latched-register exception: skip THIS write, keep scanning.
                idx = w.get("index_text", "") + w.get("text", "")
                if any(lf in idx for lf in registry.LATCHED_FIELDS):
                    continue

                stalls_before = [s for s in stalls if s["off"] < w["off"]]

                # (drains_any, drains_full) per stall for this thread's unit. For MATH
                # a FULL drain needs both the FPU and SFPU engines; a partial one
                # drains only one (see registry.unit_drain_state). Word-boundary
                # matched so 'PACK' does not match inside 'UNPACK'.
                def _drain(s):
                    return registry.unit_drain_state(
                        registry.stallwait_wait_operand(s.get("text", "")), thr
                    )

                def _held(s):  # the drain still holds iff the unit wasn't re-armed
                    return not any(s["off"] < r < w["off"] for r in reissue_offs)

                states = [(s, *_drain(s)) for s in stalls_before]
                if any(full and _held(s) for (s, _any, full) in states):
                    continue  # a full, still-valid drain — check later writes

                partial_held = any(
                    a and not full and _held(s) for (s, a, full) in states
                )
                any_drain = any(a for (s, a, full) in states)
                if partial_held:
                    # MATH only: a stall drains ONE engine (FPU or SFPU) but not both.
                    # Whether that is sufficient is code-dependent (does the OTHER
                    # engine read the reconfig'd field?) — surface low-confidence, do
                    # NOT silently clear (a false negative) nor hard-flag NO_UNIT_DRAIN.
                    hint, detail = "PARTIAL_MATH_DRAIN", (
                        f"{thr} reconfig: the preceding STALLWAIT drains only ONE of the "
                        f"FPU(MATH)/SFPU engines — insufficient IF the reconfig'd field is "
                        f"also sampled by the other engine (confirm which engine reads it)"
                    )
                elif not stalls_before:
                    hint, detail = "NO_UNIT_DRAIN", (
                        f"{thr} reconfig writes config with no preceding STALLWAIT"
                    )
                elif any_drain:
                    hint, detail = "DRAIN_REARMED", (
                        f"{thr} unit re-issued (UNPACR/PACR/matrix) between the "
                        f"draining STALLWAIT and this config write"
                    )
                else:
                    hint, detail = "THCON_ONLY", (
                        f"STALLWAIT before write does not drain {thr} "
                        f"(needs one of {list(drain_tokens)})"
                    )

                findings.append(
                    Finding(
                        file=fn.file,
                        line=w.get("line", 0),
                        function=fn.name,
                        kind="reconfig_stall",
                        hint=hint,
                        detail=detail,
                        evidence=[self._ev(w, "config write")]
                        + [
                            self._ev(s, f'stall {s.get("text","")[:50]}')
                            for s in stalls_before
                        ],
                    )
                )
                emitted = True
        return findings
