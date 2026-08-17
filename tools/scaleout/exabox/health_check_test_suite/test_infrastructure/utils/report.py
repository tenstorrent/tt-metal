# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Health-check report post-processing: normalization and verdict logic.

Kept stdlib-only so both the runner (``run_health_check.py``) and the CSV
analyzer (``analyze_health_check_results.py``) can import it as the single source
of truth for the post-reset verdict without pulling in the runner's runtime deps
(docker/paramiko/requests/prometheus).
"""

from __future__ import annotations


# Capture/precondition steps that aren't a hardware verdict; a FAIL here alone
# (e.g. tt-smi couldn't read a board) is a tooling hiccup, not a fleet fault.
EXCLUDED_CHECKS = frozenset({"snapshot_capture"})


def normalize_health_report(report: dict) -> dict:
    """Judge health on post-reset state: drop the pre-reset ``snapshot`` and promote
    the last post-reset re-snapshot in its place."""
    phases = report.get("phases", {})
    post = [p for p in phases if p.startswith("snapshot_after_")]
    if not post:
        return report

    final = post[-1]
    new_phases = {}
    for name, phase in phases.items():
        if name == "snapshot":
            continue
        if name.startswith("snapshot_after_"):
            if name == final:
                new_phases["snapshot"] = phase
            continue
        new_phases[name] = phase

    out = dict(report)
    out["phases"] = new_phases
    return out


def has_actionable_failure(report: dict) -> bool:
    """True if any non-excluded check FAILs in the (normalized) report."""
    for phase in report.get("phases", {}).values():
        for check in phase.get("checks", []):
            if check.get("name") in EXCLUDED_CHECKS:
                continue
            if check.get("status") == "FAIL":
                return True
    return False
