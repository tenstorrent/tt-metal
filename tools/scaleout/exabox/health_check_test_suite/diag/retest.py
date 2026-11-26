# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Retest phase: test -> reset -> test, for the tier's retest tests.

The GDDR ACT alias shows on some boots of a faulty die and not on others, so one
clean run of a gddr_act_* test clears one boot. When every retest test passed
the first time, reset the unit, take and validate a fresh snapshot the way the
pre-test reset loop does, and run them again on the new boot.
"""
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

# Appended to each rerun's check and log name.
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
) -> Phase | None:
    """Run the retest into ``phase``. Returns the post-reset snapshot phase for the
    caller to store in the report, or None when no snapshot was taken.

    The snapshot is named snapshot_after_*, so normalize_health_report() judges
    the run on it when it is the last one, as it is on the pre-test loop's.
    """
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
    # A subdirectory, so its reset_1_* log does not overwrite the pre-test loop's.
    reset_loop(tt_smi, (reset_flag,), phase, dry_run, logs_dir=logs_dir / "retest")
    # reset_loop's reset_* checks stay out of the verdict (report.is_reset_op_check);
    # restate the outcome under a name that counts, as run_post_test_reset does.
    broke = [c.name for c in phase.checks if c.status == FAIL]
    phase.add(
        Check(
            name="retest_reset_ok",
            status=FAIL if broke else PASS,
            details=(
                f"reset before the retest left the unit degraded: {', '.join(broke)}"
                if broke
                else f"unit back after tt-smi {reset_flag}"
            ),
            ip="other",
        )
    )
    if broke:
        return None

    snap_phase = None
    if not dry_run and snapshot_out is not None:
        snap_phase = Phase(name=f"snapshot_after_retest_{reset_flag.lstrip('-')}")
        t0 = time.time()
        try:
            validate_snapshot(take_snapshot(tt_smi, snapshot_out), snap_phase)
        except Exception as e:
            snap_phase.error = repr(e)
            snap_phase.add(Check(name="snapshot_capture", status=FAIL, details=repr(e), ip="other"))
        snap_phase.duration_s = time.time() - t0
        snap_phase.rollup()
        print_phase_summary(snap_phase.name, asdict(snap_phase))

    run_tests(tt_metal, names, phase, dry_run, logs_dir, name_suffix=RETEST_SUFFIX)
    return snap_phase
