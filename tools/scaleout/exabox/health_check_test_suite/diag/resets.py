# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 2 (pre-test reset stability loop) and phase 4 (post-test reset)."""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .common import emit_result, emit_running, log, print_phase_summary, run, write_command_log
from .galaxy import EXPECTED_CHIP_COUNT
from .model import FAIL, PASS, WARN, Check, Phase
from .snapshot import resolve_tt_smi, take_snapshot
from .snapshot_checks import validate_snapshot

# The tt-smi flags a tier's reset plan may name.
RESET_FLAGS = ("-r", "-glx_reset")

# tt-smi prints this banner before attempting `-r` on Galaxy units when the CPLD
# FW is older than v1.16. The PCIe-level reset path is unreliable on those CPLDs
# (chips re-enumerate but UMD's post-reset register reads return 0xffffffff).
# Workaround per the banner: use `-glx_reset` instead.
CPLD_OLD_BANNER_RE = re.compile(r"CPLD FW v1\.16 or higher is required to use tt-smi -r", re.IGNORECASE)


def count_pcie_chips() -> int:
    """Tenstorrent chips enumerated on PCIe, or -1 if they could not be counted."""
    try:
        lspci = subprocess.run(["bash", "-c", "lspci -d 1e52: | wc -l"], capture_output=True, text=True, check=True)
        return int(lspci.stdout.strip())
    except Exception:
        return -1


def reset_loop(
    tt_smi: str,
    plan: Sequence[str],
    phase: Phase,
    dry_run: bool,
    snapshot_out: Path | None = None,
    post_reset_phases: list | None = None,
    logs_dir: Path | None = None,
) -> None:
    """Per SYS-4365: first -r, then stick to -glx_reset for subsequent iterations.

    If `snapshot_out` and `post_reset_phases` are provided, a fresh snapshot is
    taken + validated after each reset iteration (including any CPLD auto-recover),
    and appended to `post_reset_phases` as (name, Phase) pairs for the caller to
    splice into the report.
    """

    def _post_reset_snapshot(label: str) -> None:
        if dry_run or snapshot_out is None or post_reset_phases is None:
            return
        sp = Phase(name=f"snapshot_after_{label}")
        t0 = time.time()
        try:
            snap = take_snapshot(tt_smi, snapshot_out)
            validate_snapshot(snap, sp)
        except Exception as e:
            sp.error = repr(e)
            sp.add(Check(name="snapshot_capture", status=FAIL, details=repr(e), ip="other"))
        sp.duration_s = time.time() - t0
        sp.rollup()
        post_reset_phases.append((sp.name, sp))
        print_phase_summary(sp.name, asdict(sp))

    for i, flag in enumerate(plan):
        check_name = f"reset_{i+1}_{flag.lstrip('-')}"
        emit_running(check_name)
        t0 = time.time()
        cp = run([tt_smi, flag], dry_run=dry_run, capture_output=True, text=True)
        dt = time.time() - t0

        if logs_dir is not None and not dry_run:
            log_path = logs_dir / f"{check_name}.log"
            if write_command_log(log_path, f"{tt_smi} {flag}", cp.returncode, dt, cp.stdout, cp.stderr):
                log(f"  log:  {log_path}")

        # Quick post-reset enum check
        post_count = EXPECTED_CHIP_COUNT if dry_run else count_pcie_chips()
        status = PASS if (dry_run or (cp.returncode == 0 and post_count == EXPECTED_CHIP_COUNT)) else FAIL
        emit_result(check_name, status, suffix=f"({dt:.1f}s, post_pcie={post_count})")

        # A failed reset is the one case where the console should say why
        # without the reader having to go and open the log.
        if status == FAIL and not dry_run:
            text = (cp.stderr or "").strip() or (cp.stdout or "").strip()
            for line in text.splitlines()[-5:]:
                log(f"  ! {line}")
        phase.add(
            Check(
                name=check_name,
                status=status,
                details=f"flag={flag} rc={cp.returncode if not dry_run else '(dry)'} "
                f"dur={dt:.1f}s post_pcie={post_count}",
                data={
                    "flag": flag,
                    "rc": cp.returncode,
                    "duration_s": dt,
                    "post_pcie_count": post_count,
                    "stdout_tail": cp.stdout[-2000:] if cp.stdout else "",
                    "stderr_tail": cp.stderr[-2000:] if cp.stderr else "",
                },
                ip="other",
            )
        )
        cpld_old = flag == "-r" and not dry_run and cp.stdout and CPLD_OLD_BANNER_RE.search(cp.stdout)
        if cpld_old:
            phase.add(
                Check(
                    name="cpld_fw_old",
                    status=WARN,
                    details="CPLD FW < v1.16 detected — `tt-smi -r` unreliable on this system; "
                    "use `tt-smi -glx_reset` instead and request a CPLD update from the sysadmin",
                    ip="other",
                )
            )
        # Auto-recover: -r on CPLD < v1.16 doesn't just fail, it leaves chips in a
        # bad state (UMD reads 0xffffffff on subsequent calls) until -glx_reset is
        # run. Recover so we don't strand the system.
        if cpld_old and status == FAIL:
            emit_running("cpld_auto_recover")
            t0 = time.time()
            rcp = run([tt_smi, "-glx_reset"], dry_run=False, capture_output=True, text=True)
            rdt = time.time() - t0
            rpost = count_pcie_chips()
            recovered = rcp.returncode == 0 and rpost == EXPECTED_CHIP_COUNT
            rstatus = PASS if recovered else FAIL
            emit_result("cpld_auto_recover", rstatus, suffix=f"({rdt:.1f}s, post_pcie={rpost})")
            phase.add(
                Check(
                    name="cpld_auto_recover",
                    status=rstatus,
                    details=(
                        f"ran `tt-smi -glx_reset` after -r failed; "
                        f"rc={rcp.returncode} post_pcie={rpost} "
                        f"{'recovered' if recovered else 'still broken — manual intervention needed'}"
                    ),
                    data={
                        "rc": rcp.returncode,
                        "duration_s": rdt,
                        "post_pcie_count": rpost,
                        "stdout_tail": rcp.stdout[-2000:] if rcp.stdout else "",
                        "stderr_tail": rcp.stderr[-2000:] if rcp.stderr else "",
                    },
                    ip="other",
                )
            )
            # If recovery worked, refresh post_count so the enumeration-break
            # bailout below doesn't trip on the pre-recovery state.
            if recovered:
                post_count = rpost
                # Downgrade the original -r FAIL to WARN now that we know it's
                # CPLD-attributable and the unit is back up. Hard FAIL is reserved
                # for -r failures we *can't* explain or recover from.
                for c in phase.checks:
                    if c.name == check_name and c.status == FAIL:
                        c.status = WARN
                        c.details += " (downgraded to WARN: CPLD-attributable, auto-recovered)"
                        break
        # Post-reset snapshot: validate the unit's state after this iteration
        # (including any auto-recovery). Skipped on dry-run / offline.
        _post_reset_snapshot(f"r{i+1}_{flag.lstrip('-')}")
        # Bail if a reset broke enumeration — no point chasing the rest
        if not dry_run and post_count != EXPECTED_CHIP_COUNT:
            log(f"Reset {i+1} broke enumeration ({post_count}/{EXPECTED_CHIP_COUNT}); stopping reset loop")
            break


def fold_post_reset_snapshots(report: dict, post_reset_phases: list[tuple[str, Phase]]) -> None:
    """Splice the reset loop's revalidation snapshots into ``report``.

    Dedupe per-reset snapshots: splice the differing phases into the report
    (matching intermediates are noise) but ALWAYS keep the final post-reset
    snapshot, even when it matches initial. normalize_health_report judges the
    fleet on the last snapshot_after_* phase (post[-1]); if the unit regressed
    on an early reset but recovered by the final one, dropping the (matching)
    final snapshot would strand the stale regression as post[-1] and falsely
    fail a node that actually came back healthy.

    Must run right after the reset_loop phase is stored, so the spliced phases
    land between it and the tests in the report's phase order.
    """
    if not post_reset_phases:
        return

    def _check_sig(checks: list) -> tuple:
        # Compare on (name, status) only. `details` and `data` carry
        # telemetry (temps, voltages, error counters) that fluctuates
        # between snapshots without being a status concern. Status
        # changes are the signal — a chip dropping out, GDDR un-training,
        # FW versions going inconsistent — those all flip a check's status.
        return tuple(sorted((c.get("name", ""), c.get("status", "")) for c in checks))

    initial_sig = _check_sig(report["phases"]["snapshot"]["checks"])
    matched, differed = [], []
    for name, p in post_reset_phases:
        (matched if _check_sig([asdict(c) for c in p.checks]) == initial_sig else differed).append((name, p))
    for name, p in differed:
        report["phases"][name] = asdict(p)
    # Preserve the actual final snapshot so the verdict reflects the true end
    # state. If it differed it is already spliced above (in chronological
    # order, so it stays post[-1]); if it matched initial, add it back here.
    final_name, final_phase = post_reset_phases[-1]
    if final_name not in report["phases"]:
        report["phases"][final_name] = asdict(final_phase)
    summary_status = PASS if not differed else WARN
    summary_details = f"{len(matched)}/{len(post_reset_phases)} post-reset snapshots matched initial" + (
        f"; {len(differed)} differed (see {', '.join(n for n,_ in differed)})" if differed else ""
    )
    report["phases"]["snapshot"]["checks"].append(
        asdict(
            Check(
                name="post_reset_state_stable",
                status=summary_status,
                details=summary_details,
                ip="other",
            )
        )
    )
    # Re-roll the initial snapshot phase status since we just appended a check.
    statuses = [c["status"] for c in report["phases"]["snapshot"]["checks"]]
    if FAIL in statuses:
        report["phases"]["snapshot"]["status"] = FAIL
    elif WARN in statuses:
        report["phases"]["snapshot"]["status"] = WARN
    elif statuses:
        report["phases"]["snapshot"]["status"] = PASS


def run_post_test_reset(tt_smi_path: str | None, plan: Sequence[str], phase: Phase, dry_run: bool) -> None:
    """Reset the unit between the tests and the triage phase.

    Deliberately bare: reset_loop() is called without snapshot_out /
    post_reset_phases, so it takes no revalidation snapshot and stays out of
    fold_post_reset_snapshots(). Adding revalidation here means reworking that
    function, which assumes a single batch of snapshot_after_* phases judged by
    normalize_health_report() on post[-1].
    """
    try:
        tt_smi = resolve_tt_smi(tt_smi_path)
        reset_loop(tt_smi, plan, phase, dry_run)
    except Exception as e:
        phase.error = repr(e)
        phase.add(Check(name="post_test_reset", status=FAIL, details=repr(e), ip="other"))
    # reset_loop names every check reset_*, which report.py classifies as a
    # reset op and drops from the verdict. That is right for the pre-test
    # loop, where a bad reset still surfaces in the tests and the final
    # snapshot — but nothing runs after this one, so a reset that broke
    # enumeration here would leave no trace on the verdict. Restate the
    # outcome under a name that counts.
    broke = [c.name for c in phase.checks if c.status == FAIL]
    phase.add(
        Check(
            name="post_test_reset_ok",
            status=FAIL if broke else PASS,
            details=(
                f"post-test reset left the unit degraded: {', '.join(broke)}"
                if broke
                else f"unit healthy after {' '.join(plan)}"
            ),
            ip="other",
        )
    )
