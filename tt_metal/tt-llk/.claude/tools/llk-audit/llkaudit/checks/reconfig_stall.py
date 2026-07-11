# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
reconfig-stall checker — a config rewrite that a running execution unit samples
must be preceded by a STALLWAIT whose condition DRAINS that unit (packer->PACK,
unpacker->UNPACK[01], math->MATH|WAIT_SFPU). A THCON-only stall orders the
GPR->cfg write but does NOT drain the unit — the classic miss.

Recall: for each candidate reconfig/uninit/config-writer function, find its first
CONFIG write and whether a unit-draining STALLWAIT precedes it. Emit:
  NO_UNIT_DRAIN  — a cfg write with no draining STALLWAIT before it
  THCON_ONLY     — a STALLWAIT before the write, but condition drains no unit
Latched registers (L1_Dest_addr) are annotated as an expected THCON-only case,
not flagged — see registry.LATCHED_FIELDS.
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
        "instruction — only the drain condition (2nd operand) is checked."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        findings: list[Finding] = []
        for fn in fb.functions:
            if not registry.is_reconfig_fn(fn.name):
                continue
            thr = registry.thread_of(fn.file)
            drain_tokens = registry.DRAIN_UNIT_TOKENS.get(thr, ())

            facts = fb.facts_in(fn, ("macro", "call", "pointer_write"))

            # First write to the UNIT-SAMPLED CONFIG file. Note: SETDMAREG writes
            # a GPR (a source value), NOT a sampled config reg, so it is excluded
            # (registry.RECONFIG_WRITE_MACRO_SUBSTR) — otherwise a GPR setup line
            # would be mistaken for the reconfig write.
            def is_cfg_write_macro(name):
                up = name.upper()
                return name.startswith(("TTI_", "TT_")) and any(
                    s in up for s in registry.RECONFIG_WRITE_MACRO_SUBSTR
                )

            first_write = None
            for f in facts:
                if f["family"] == "pointer_write":
                    kind, _ = registry.classify_write(f)
                    if kind in registry.CFG_WRITE_KINDS:  # cfg[] MMIO (not GPR)
                        first_write = f
                        break
                elif f["family"] == "macro" and is_cfg_write_macro(f.get("name", "")):
                    first_write = f
                    break
                elif f["family"] == "call" and "cfg_reg_rmw_tensix" in f.get(
                    "text", ""
                ):
                    first_write = f
                    break
            if first_write is None:
                continue

            # latched-register exception: annotate, don't flag
            idx = first_write.get("index_text", "") + first_write.get("text", "")
            if any(lf in idx for lf in registry.LATCHED_FIELDS):
                continue

            # any STALLWAIT before the first write, and does it drain the unit?
            stalls_before = [
                f
                for f in facts
                if f["family"] == "macro"
                and registry.classify_macro(f.get("name", "")) == "stall"
                and f["off"] < first_write["off"]
            ]
            # Only the STALLWAIT's SECOND operand (wait_res = ConditionMask)
            # drains a unit; the first (stall_res = BlockMask, e.g. STALL_UNPACK)
            # names the block being held. Word-boundary match so 'PACK' does not
            # match inside 'UNPACK' (see registry.condition_drains_unit).
            drains = any(
                registry.condition_drains_unit(
                    registry.stallwait_wait_operand(s.get("text", "")), drain_tokens
                )
                for s in stalls_before
            )

            if not stalls_before:
                hint, detail = "NO_UNIT_DRAIN", (
                    f"{thr} reconfig writes config with no preceding STALLWAIT"
                )
            elif not drains:
                hint, detail = "THCON_ONLY", (
                    f"STALLWAIT before write does not drain {thr} "
                    f"(needs one of {list(drain_tokens)})"
                )
            else:
                continue  # a unit-draining stall precedes the write — ok

            findings.append(
                Finding(
                    file=fn.file,
                    line=first_write["line"],
                    function=fn.name,
                    kind="reconfig_stall",
                    hint=hint,
                    detail=detail,
                    evidence=[self._ev(first_write, "first config write")]
                    + [
                        self._ev(s, f'stall {s.get("text","")[:50]}')
                        for s in stalls_before
                    ],
                )
            )
        return findings
