# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Retest phase: test -> reset -> snapshot -> test. The GDDR ACT alias is boot-dependent."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .common import print_phase_summary
from .device_tests import run_tests
from .model import FAIL, PASS, SKIP, Check, Phase
from .resets import reset_loop
from .snapshot import resolve_tt_smi, take_snapshot
from .snapshot_checks import validate_snapshot

# Keeps the first run's checks and logs.
RETEST_SUFFIX = "_after_reset"


def run_retest(
    tt_smi_path: str | None,
    tt_metal: Path,
    names: Sequence[str],
    reset_flag: str,
    first_run: Phase,
    phase: Phase,
    dry_run: bool,
    logs_dir: Path,
    snapshot_out: Path | None,
    gate: str | None = None,
    hook: str | None = None,
) -> Phase | None:
    """Returns the post-reset snapshot phase, or None. Named snapshot_after_* so
    normalize_health_report() judges the run on it."""
    first = {c.name: c.status for c in first_run.checks}
    not_passed = [n for n in names if first.get(n) != PASS]
    if not_passed:
        phase.add(
            Check(
                name="retest",
                status=SKIP,
                details=f"{', '.join(not_passed)} did not pass on the first boot; no second boot",
                ip="other",
            )
        )
        return None

    tt_smi = resolve_tt_smi(tt_smi_path)
    # Own dir: keeps the pre-test reset_1_* log.
    reset_loop(tt_smi, (reset_flag,), phase, dry_run, logs_dir=logs_dir / "retest", gate=gate, hook=hook)
    # reset_* checks don't count toward the verdict; restate under a name that does.
    broke = [c.name for c in phase.checks if c.status == FAIL]
    refused = next((c.details for c in phase.checks if c.status == SKIP), None)
    status, details = PASS, f"unit back after tt-smi {reset_flag}"
    if broke:
        status, details = FAIL, f"reset before the retest left the unit degraded: {', '.join(broke)}"
    elif refused:  # the gate refused the reset: no second boot to retest on
        status, details = SKIP, f"did not run: {refused}"
    phase.add(Check(name="retest_reset_ok", status=status, details=details, ip="other"))
    if status != PASS:
        return None

    snap_phase = None
    if not dry_run and snapshot_out is not None:
        snap_phase = Phase(name=f"snapshot_after_retest_{reset_flag.lstrip('-')}")
        t0 = time.time()
        try:
            validate_snapshot(take_snapshot(tt_smi, snapshot_out), snap_phase)
        except Exception as e:
            # snapshot_capture is not a verdict; returned, this phase would displace
            # the pre-test snapshot and leave the run with no snapshot verdict.
            phase.add(Check(name="snapshot_capture", status=FAIL, details=repr(e), ip="other"))
            snap_phase = None
        if snap_phase is not None:
            snap_phase.duration_s = time.time() - t0
            snap_phase.rollup()
            print_phase_summary(snap_phase.name, asdict(snap_phase))

    run_tests(tt_metal, names, phase, dry_run, logs_dir, name_suffix=RETEST_SUFFIX)
    return snap_phase
