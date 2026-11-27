# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Retest phase: test -> reset -> snapshot -> test. The GDDR ACT alias is boot-dependent, and each reset trains GDDR
afresh, so a test can ask for more boots: one round per extra boot."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from .common import print_phase_summary
from .device_tests import run_tests
from .model import FAIL, PASS, SKIP, Check, Phase
from .resets import reset_loop
from .snapshot import resolve_tt_smi, take_snapshot
from .snapshot_checks import validate_snapshot

# Keeps the first run's checks and logs. Rounds after the first add their number: <check>_after_reset_2, ...
RETEST_SUFFIX = "_after_reset"
# Boots a retest test runs on when Tier.retest_boots doesn't say: the first run and one retest.
DEFAULT_BOOTS = 2


def _numbered(base: str, round_index: int) -> str:
    """Round 1 keeps the name; later rounds add their number."""
    return base if round_index == 1 else f"{base}_{round_index}"


def _snapshot(
    tt_smi: str, reset_flag: str, round_index: int, phase: Phase, dry_run: bool, snapshot_out: Path | None
) -> tuple[Phase | None, bool]:
    """The validated snapshot after this round's reset, or None, and whether the unit came back clean."""
    if dry_run or snapshot_out is None:
        return None, True
    snap_phase = Phase(name=_numbered(f"snapshot_after_retest_{reset_flag.lstrip('-')}", round_index))
    t0 = time.time()
    try:
        validate_snapshot(take_snapshot(tt_smi, snapshot_out), snap_phase)
    except Exception as e:
        # snapshot_capture is not a verdict; returned, this phase would displace
        # the pre-test snapshot and leave the run with no snapshot verdict.
        phase.add(Check(name="snapshot_capture", status=FAIL, details=repr(e), ip="other"))
        return None, False
    snap_phase.duration_s = time.time() - t0
    snap_phase.rollup()
    print_phase_summary(snap_phase.name, asdict(snap_phase))
    return snap_phase, snap_phase.status != FAIL


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
    boots: Mapping[str, int] | None = None,
) -> list[Phase]:
    """Reruns `names` on more boots. A test runs on boots[test] boots in all (DEFAULT_BOOTS when absent), counting
    the first run. Round k resets with reset_flag, takes and validates a snapshot, then reruns every test that wants
    more than k boots. Rounds stop at a reset that leaves the unit degraded, and after a round whose snapshot or
    tests failed.

    Returns the post-reset snapshot phases in order, named snapshot_after_* so normalize_health_report() judges the
    run on the last."""
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
        return []

    wanted = {n: (boots or {}).get(n, DEFAULT_BOOTS) for n in names}
    rounds = max(wanted.values(), default=1) - 1
    if rounds < 1:
        return []
    tt_smi = resolve_tt_smi(tt_smi_path)
    snaps: list[Phase] = []
    done = 0
    for k in range(1, rounds + 1):
        done = k
        round_names = tuple(n for n in names if wanted[n] > k)
        before = len(phase.checks)
        # Own dir per round: keeps the pre-test reset_1_* log and the earlier rounds'.
        reset_loop(tt_smi, (reset_flag,), phase, dry_run, logs_dir=logs_dir / _numbered("retest", k))
        reset_checks = phase.checks[before:]
        if k > 1:
            # reset_loop numbers its checks within one call; number them by round instead.
            for c in reset_checks:
                c.name = c.name.replace("reset_1_", f"reset_{k}_", 1)
        # reset_* checks don't count toward the verdict; restate under a name that does.
        broke = [c.name for c in reset_checks if c.status == FAIL]
        if broke:
            phase.add(
                Check(
                    name="retest_reset_ok",
                    status=FAIL,
                    details=(
                        f"reset before {'the retest' if k == 1 else f'retest round {k}'} left the unit degraded: "
                        f"{', '.join(broke)}"
                    ),
                    ip="other",
                )
            )
            return snaps

        snap_phase, snap_ok = _snapshot(tt_smi, reset_flag, k, phase, dry_run, snapshot_out)
        if snap_phase is not None:
            snaps.append(snap_phase)

        before = len(phase.checks)
        run_tests(tt_metal, round_names, phase, dry_run, logs_dir, name_suffix=_numbered(RETEST_SUFFIX, k))
        failed = [c.name for c in phase.checks[before:] if c.status == FAIL]
        if k < rounds and (failed or not snap_ok):
            left = {n: wanted[n] - 1 - k for n in names if wanted[n] - 1 > k}
            phase.add(
                Check(
                    name="retest_more_boots",
                    status=SKIP,
                    details=(
                        f"stopped after round {k} ({', '.join(failed) or 'snapshot after the reset'} failed); "
                        f"not run: {', '.join(f'{n} on {left[n]} more boot(s)' for n in left)}"
                    ),
                    ip="other",
                )
            )
            break

    phase.add(
        Check(
            name="retest_reset_ok",
            status=PASS,
            details=f"unit back after tt-smi {reset_flag}" + (f", {done} times" if done > 1 else ""),
            ip="other",
        )
    )
    return snaps
