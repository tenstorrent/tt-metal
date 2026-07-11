# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
reconfig-stall checker — a config rewrite that a running execution unit samples
must be preceded by a STALLWAIT whose condition DRAINS that unit (packer->PACK,
unpacker->UNPACK[01], math->MATH|WAIT_SFPU). A THCON-only stall orders the
GPR->cfg write but does NOT drain the unit — the classic miss.

Recall: for each candidate reconfig/uninit/config-writer function, walk ALL its
CONFIG writes in order (not just the first) and, for each, check that a
unit-draining STALLWAIT precedes it AND the unit was not re-armed (by a
UNPACR/PACR/matrix issue for this thread) between that drain and the write. Emit
one candidate per function, at the first offending write:
  NO_UNIT_DRAIN  — a cfg write with no preceding STALLWAIT at all
  THCON_ONLY     — a STALLWAIT precedes it, but its condition drains no unit
  DRAIN_REARMED  — a draining STALLWAIT precedes it, but the unit was re-issued
                   between the drain and the write (the drain no longer holds)
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
        "instruction — only the drain condition (2nd operand) is checked. Unit "
        "re-arm detection covers UNPACR/PACR and the curated matrix consumer set "
        "(see registry.CONSUMER_MATH_SUBSTR); a re-issue via an unmodeled macro "
        "would not invalidate an earlier drain. Only the FIRST offending write in "
        "a function is emitted (one candidate/function)."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        findings: list[Finding] = []
        for fn in fb.functions:
            if not registry.is_reconfig_fn(fn.name):
                continue
            thr = registry.thread_of(fn.file)
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
                    return f.get("name", "").startswith(("TTI_", "TT_")) and any(
                        s in up for s in registry.RECONFIG_WRITE_MACRO_SUBSTR
                    )
                if f["family"] == "call":
                    return bool(
                        "cfg_reg_rmw_tensix" in f.get("text", "")
                        # cfg_rmw / cfg_rmw_gpr — Quasar's dominant config-write
                        # idiom; without this a Quasar reconfig looks like it
                        # writes nothing.
                        or registry.write_call_kind(f.get("name", ""))
                    )
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
                # A stall drains the unit only if its 2nd operand (wait_res =
                # ConditionMask) names a drain token; word-boundary matched so
                # 'PACK' does not match inside 'UNPACK'.
                draining = [
                    s
                    for s in stalls_before
                    if registry.condition_drains_unit(
                        registry.stallwait_wait_operand(s.get("text", "")), drain_tokens
                    )
                ]
                # A draining stall still holds only if the unit was NOT re-armed
                # between it and the write.
                valid_drain = any(
                    not any(s["off"] < r < w["off"] for r in reissue_offs)
                    for s in draining
                )
                if valid_drain:
                    continue  # this write is properly drained — check later writes

                if not stalls_before:
                    hint, detail = "NO_UNIT_DRAIN", (
                        f"{thr} reconfig writes config with no preceding STALLWAIT"
                    )
                elif draining:
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
                        line=w["line"],
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
