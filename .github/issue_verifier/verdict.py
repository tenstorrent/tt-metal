"""Verdict rules for the issue verifier.

The classification lives here, in plain Python, and not in a prompt. An agent
that both gathers evidence and grades it can talk itself into either answer;
these rules only ever read numbers that a probe printed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

# A reporter who rounds their own printout produces a row that misses the probe's
# 1e-6 comparison by a hair. A reporter who never called the reference at all
# produces a row that is finite where the reference is infinite, or wrong by
# percent. Only the second kind says anything about the report's soundness, so
# the two are separated before any verdict is assigned.
CATEGORICAL_REL_TOL = 1e-3


class Verdict(str, Enum):
    CLAIM_UNSOUND = "CLAIM_UNSOUND"
    NOT_REPRODUCED = "NOT_REPRODUCED"
    LIKELY_VALID = "LIKELY_VALID"
    NEEDS_HARDWARE = "NEEDS_HARDWARE"
    NEEDS_HUMAN = "NEEDS_HUMAN"


HEADLINES = {
    Verdict.CLAIM_UNSOUND: "The report's own expected values are not what the reference produces.",
    Verdict.NOT_REPRODUCED: "The op matches the reference on every case the report gives.",
    Verdict.LIKELY_VALID: "The op disagrees with the reference on hardware. Reproduced.",
    Verdict.NEEDS_HARDWARE: "Survived the host-side checks. A device run is needed to settle it.",
    Verdict.NEEDS_HUMAN: "Could not be settled mechanically.",
}


@dataclass
class Assessment:
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def headline(self) -> str:
        return HEADLINES[self.verdict]

    @property
    def is_actionable_bug(self) -> bool:
        return self.verdict is Verdict.LIKELY_VALID


def _disagreements(gate: dict, key: str = "rows") -> list[dict]:
    return [row for row in gate.get(key) or [] if row.get("agree") is False]


def is_categorical(row: dict) -> bool:
    """Is this mismatch too large to be the reporter rounding their own output?"""
    try:
        claimed = float(row.get("claimed_expected"))
        reference = float(row.get("reference"))
    except (TypeError, ValueError):
        return True

    claimed_finite = math.isfinite(claimed)
    reference_finite = math.isfinite(reference)
    if not (claimed_finite and reference_finite):
        # finite-vs-infinite, inf-vs-nan, or +inf-vs--inf: never a rounding artifact
        return True

    return not math.isclose(claimed, reference, rel_tol=CATEGORICAL_REL_TOL, abs_tol=0.0)


def assess(plan: dict, measurements: dict) -> Assessment:
    if not plan.get("verifiable", False):
        reason = plan.get("reason_unverifiable") or "the planning pass could not derive a runnable experiment"
        return Assessment(Verdict.NEEDS_HUMAN, [f"Not mechanically checkable: {reason}"])

    gate_a = measurements.get("gate_a") or {}
    gate_b = measurements.get("gate_b") or {}
    gate_c = measurements.get("gate_c") or {}
    gate_d = measurements.get("gate_d") or {}

    flags: list[str] = []

    # Advisory only. A prior deliberate decision does not by itself make a report
    # wrong, but it is the context a triager most often lacks. Commits that
    # merely introduced the code are recorded in the report and skipped here —
    # surfacing those as "prior decision" would imply an intent nobody had.
    seen: set[str] = set()
    for finding in gate_b.get("findings") or []:
        commit = finding.get("commit")
        if not commit or not finding.get("deliberate") or commit in seen:
            continue
        seen.add(commit)
        flags.append(f"Behavior was set deliberately in {commit} — {finding.get('subject', '?')}")

    mirror_regressions = _disagreements(gate_d, "mirror_rows")
    if mirror_regressions:
        flags.append(
            f"The proposed change breaks {len(mirror_regressions)} mirror case(s) that currently pass — "
            "it relocates the failure rather than removing it."
        )

    # Gate A is decisive and cheapest: if the report's reference column was never
    # produced by the reference, nothing downstream can rescue the claim.
    if gate_a.get("ran"):
        bad = _disagreements(gate_a)
        categorical = [row for row in bad if is_categorical(row)]
        marginal = [row for row in bad if not is_categorical(row)]

        if marginal:
            names = ", ".join(str(row.get("name")) for row in marginal)
            flags.append(
                f"{len(marginal)} case(s) miss the reference by less than {CATEGORICAL_REL_TOL:g} "
                f"relative ({names}) — treated as the reporter rounding their own output, not as "
                "evidence against the report."
            )

        if categorical:
            names = ", ".join(str(row.get("name")) for row in categorical[:5])
            reasons = [
                f"{len(categorical)} of {len(gate_a.get('rows') or [])} case(s) claim an expected value "
                f"the reference does not return ({names}).",
                "The reference was re-executed directly, so this is a property of the report, " "not of tt-metal.",
            ]
            if gate_c.get("ran") and not _disagreements(gate_c):
                reasons.append(
                    "Corroborating: every case also ran on device and matched the reference, so the "
                    "op is behaving as the reference says it should."
                )
            return Assessment(Verdict.CLAIM_UNSOUND, reasons, flags)
    else:
        flags.append(f"Gate A did not run: {gate_a.get('error') or 'unknown reason'}")

    if gate_c.get("ran"):
        bad = _disagreements(gate_c)
        if bad:
            names = ", ".join(str(row.get("name")) for row in bad[:5])
            return Assessment(
                Verdict.LIKELY_VALID,
                [f"The op disagrees with the reference on hardware for: {names}."],
                flags,
            )
        return Assessment(
            Verdict.NOT_REPRODUCED,
            [
                "Every case ran on device and matched the reference, and the report's expected "
                "values matched it too.",
            ],
            flags,
        )

    if gate_a.get("ran"):
        return Assessment(
            Verdict.NEEDS_HARDWARE,
            [
                "The report's expected values agree with the reference, so the claim is coherent — "
                "but it was not executed on a device.",
                f"Gate C skipped: {gate_c.get('skipped_reason') or 'no device on this runner'}",
            ],
            flags,
        )

    return Assessment(Verdict.NEEDS_HUMAN, ["No gate produced usable measurements."], flags)
