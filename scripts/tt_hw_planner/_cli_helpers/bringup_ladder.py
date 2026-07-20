# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure, module-level bring-up attempt-cap logic — extracted verbatim from the closures inside
``auto_iterate._run_auto_iterate_loop`` so BOTH the legacy loop and a future Claude-Code bring-up
gate can call the IDENTICAL rule (no second copy, no rewrite).

These functions contain no gate policy change: they are the exact same arithmetic the loop's
``_effective_attempt_cap`` / ``_is_at_cap`` closures perform, with the loop's config constants and
per-component state dicts passed in explicitly instead of captured. The loop keeps computing the
complexity bonus (it reads the op-synth manifest) and passes the result in.
"""
from __future__ import annotations


def effective_attempt_cap(
    comp: str,
    *,
    max_attempts_per_component: int,
    hard_total_attempt_cap: int,
    complexity_bonus: int,
    last_failure_class: dict,
    last_pcc: dict,
    pcc_stuck_threshold: float = 0.5,
    pcc_stuck_extra_attempts: int = 2,
) -> int:
    """The per-component cap on the consecutive-same-class counter: base + complexity bonus, plus the
    PCC-stuck bonus when the last failure was PCC_ONLY and numerically close (PCC >= threshold),
    clamped at the hard ceiling. Identical to the loop's ``_effective_attempt_cap`` closure."""
    base = max_attempts_per_component + complexity_bonus
    last_class = last_failure_class.get(comp, "")
    lp = last_pcc.get(comp)
    if last_class == "PCC_ONLY" and lp is not None and lp >= pcc_stuck_threshold:
        return min(base + pcc_stuck_extra_attempts, hard_total_attempt_cap)
    return min(base, hard_total_attempt_cap)


def is_at_cap(
    comp: str,
    *,
    attempts_per_component: dict,
    consecutive_same_class_attempts: dict,
    effective_cap: int,
    hard_total_attempt_cap: int,
) -> bool:
    """A component is "at cap" when total attempts hit the hard ceiling OR the consecutive-same-class
    counter reaches the effective cap. Identical to the loop's ``_is_at_cap`` closure."""
    if attempts_per_component.get(comp, 0) >= hard_total_attempt_cap:
        return True
    return consecutive_same_class_attempts.get(comp, 0) >= effective_cap
