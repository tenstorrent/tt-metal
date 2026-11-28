# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 2 (pre-test reset stability loop) and phase 4 (post-test reset)."""
from __future__ import annotations

import shlex
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .common import (
    call_hook,
    emit_result,
    emit_running,
    log,
    print_phase_summary,
    run,
    run_with_timeout,
    write_command_log,
)
from .eth_status import read_pcie_functions
from .galaxy import EXPECTED_BDFS, EXPECTED_CHIP_COUNT, format_bdf
from .model import FAIL, PASS, SKIP, WARN, Check, Phase
from .snapshot import resolve_tt_smi, take_snapshot
from .snapshot_checks import validate_snapshot

# The tt-smi flags a tier's reset plan may name. Only the Galaxy reset: `tt-smi -r`
# is not used on Galaxy units, it can leave chips that re-enumerate but read
# 0xffffffff until a -glx_reset.
RESET_FLAGS = ("-glx_reset",)


def chip_root_ports(sys_root: Path = Path("/sys")) -> dict[str, str]:
    """Each chip's PCIe root port, by chip BDF. Read before a reset: a chip the
    reset drops leaves sysfs, and with it the only record of its root port."""
    devices = (sys_root / "devices").resolve()
    ports = {}
    for f in read_pcie_functions(sys_root):
        try:
            parts = (sys_root / "bus" / "pci" / "devices" / f.bdf).resolve().relative_to(devices).parts
        except ValueError:
            continue
        # ("pci0000:40", "0000:40:01.1", ..., "0000:41:00.0"): the root port is second.
        if len(parts) >= 3:
            ports[f.bdf] = parts[1]
    return ports


def rescan_root_ports(ports: set[str], sys_root: Path = Path("/sys")) -> list[str]:
    """Rescan each root port, or the whole bus when none is known, so a chip that
    came back from the reset is enumerated again. Returns what could not be
    rescanned; writing needs root, and a run without it is one line, not one per port."""
    targets = [sys_root / "bus" / "pci" / "devices" / p / "rescan" for p in sorted(ports)]
    errors, denied = [], 0
    for target in targets or [sys_root / "bus" / "pci" / "rescan"]:
        try:
            target.write_text("1")
        except PermissionError:
            denied += 1
        except OSError as e:
            errors.append(f"rescan failed: {target}: {e.strerror or e}")
    if denied:
        errors.insert(0, f"root-port rescan needs root ({denied} port(s) skipped)")
    return errors


def recount_chips(sys_root: Path = Path("/sys")) -> tuple[int, list[str]]:
    """Chips on PCIe that answer, and the expected BDFs missing, each named by tray and ASIC."""
    answering = {f.bdf for f in read_pcie_functions(sys_root) if not f.unverified and not f.dropped}
    return len(answering), [format_bdf(b) for b in EXPECTED_BDFS if b not in answering]


def reset_gate(cmd: str | None) -> str:
    """Why the --reset-gate command refuses a reset now (its last output line), or "" when it allows it."""
    if not cmd:
        return ""
    try:
        rc, out, err, _ = run_with_timeout(shlex.split(cmd), 300)
    except OSError as e:
        return f"reset gate did not run: {e}"
    last = (out + err).strip().splitlines()
    return "" if rc == 0 else f"reset gate: {last[-1] if last else f'rc {rc}'}"


def reset_loop(
    tt_smi: str,
    plan: Sequence[str],
    phase: Phase,
    dry_run: bool,
    snapshot_out: Path | None = None,
    post_reset_phases: list | None = None,
    logs_dir: Path | None = None,
    sys_root: Path = Path("/sys"),
    gate: str | None = None,
    hook: str | None = None,
) -> None:
    """Run each reset of the plan, then rescan the chips' root ports and recount
    the chips by BDF. Before each reset the *gate* command runs (a refusal skips
    it and the rest); chips still missing get a second reset, then the *hook*'s
    "after" event and one more recount. A reset that lost chips stays FAIL and
    ends the loop, even when the retry or the hook brings them back.

    If `snapshot_out` and `post_reset_phases` are provided, a fresh snapshot is
    taken + validated after each reset iteration, and appended to
    `post_reset_phases` as (name, Phase) pairs for the caller to splice into the
    report.
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

    ports = set() if dry_run else set(chip_root_ports(sys_root).values())

    def _glx(check_name: str, flag: str) -> list[str] | None:
        """One reset as *check_name*, then the rescan and recount: the chips still missing, or None when the reset gate
        refused it (a SKIP that says why)."""
        why = "" if dry_run else reset_gate(gate)
        if why:
            log(f"{check_name}: reset cannot run: {why}")
            phase.add(Check(name=check_name, status=SKIP, details=f"reset cannot run: {why}", ip="other"))
            return None
        emit_running(check_name)
        t0 = time.time()
        cp = run([tt_smi, flag], dry_run=dry_run, capture_output=True, text=True)
        dt = time.time() - t0

        if logs_dir is not None and not dry_run:
            log_path = logs_dir / f"{check_name}.log"
            if write_command_log(log_path, f"{tt_smi} {flag}", cp.returncode, dt, cp.stdout, cp.stderr):
                log(f"  log:  {log_path}")

        rescan_errors = [] if dry_run else rescan_root_ports(ports, sys_root)
        post_count, missing = (EXPECTED_CHIP_COUNT, []) if dry_run else recount_chips(sys_root)
        status = PASS if (dry_run or (cp.returncode == 0 and not missing)) else FAIL
        emit_result(check_name, status, suffix=f"({dt:.1f}s, post_pcie={post_count})")

        # A failed reset is the one case where the console should say why
        # without the reader having to go and open the log.
        if status == FAIL and not dry_run:
            text = (cp.stderr or "").strip() or (cp.stdout or "").strip()
            for line in text.splitlines()[-5:]:
                log(f"  ! {line}")
            if missing:
                log(f"  ! missing after rescan: {', '.join(missing)}")
        details = f"flag={flag} rc={cp.returncode if not dry_run else '(dry)'} dur={dt:.1f}s post_pcie={post_count}"
        if missing:
            details += f" missing: {', '.join(missing)}"
        if rescan_errors:
            details += f" ({'; '.join(rescan_errors)})"
        phase.add(
            Check(
                name=check_name,
                status=status,
                details=details,
                data={
                    "flag": flag,
                    "rc": cp.returncode,
                    "duration_s": dt,
                    "post_pcie_count": post_count,
                    "rescan_errors": rescan_errors,
                    "stdout_tail": cp.stdout[-2000:] if cp.stdout else "",
                    "stderr_tail": cp.stderr[-2000:] if cp.stderr else "",
                },
                ip="other",
                parts=missing,
            )
        )
        return missing

    for i, flag in enumerate(plan):
        check_name = f"reset_{i+1}_{flag.lstrip('-')}"
        missing = _glx(check_name, flag)
        if missing is None:
            break
        retried = bool(missing)
        if missing:  # a second round for chips still missing
            missing = _glx(f"{check_name}_round2", flag)
            if missing is None:
                break
        if missing:  # still missing: the hook's turn (a caller's site power cycle), then one more rescan and recount
            outdir = (logs_dir or Path(".")) / check_name
            rc = call_hook(hook, "after", check_name, FAIL, outdir, dry_run)
            if hook and not dry_run:
                rescan_root_ports(ports, sys_root)
                _, missing = recount_chips(sys_root)
                said = f"after the hook (rc {rc}): " + (
                    f"still missing {', '.join(missing)}" if missing else "all back"
                )
                phase.add(
                    Check(name=f"recount_{check_name}", status=FAIL if missing else PASS, details=said, parts=missing)
                )
        # Post-reset snapshot: validate the unit's state after this iteration.
        # Skipped on dry-run / offline.
        _post_reset_snapshot(f"r{i+1}_{flag.lstrip('-')}")
        # Bail if a reset broke enumeration — no point chasing the rest
        if retried:
            log(f"Reset {i+1} broke enumeration; stopping reset loop")
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


def run_post_test_reset(
    tt_smi_path: str | None,
    plan: Sequence[str],
    phase: Phase,
    dry_run: bool,
    gate: str | None = None,
    hook: str | None = None,
) -> None:
    """Reset the unit between the tests and the triage phase.

    Deliberately bare: reset_loop() is called without snapshot_out /
    post_reset_phases, so it takes no revalidation snapshot and stays out of
    fold_post_reset_snapshots(). A snapshot_after_* phase here would become
    post[-1], the one snapshot normalize_health_report() judges, and displace
    the snapshot taken before the tests.
    """
    try:
        tt_smi = resolve_tt_smi(tt_smi_path)
        reset_loop(tt_smi, plan, phase, dry_run, gate=gate, hook=hook)
    except Exception as e:
        phase.error = repr(e)
        phase.add(Check(name="post_test_reset", status=FAIL, details=repr(e), ip="other"))
    # reset_loop names every check reset_*, which report.py classifies as a
    # reset op and drops from the verdict. That is right for the pre-test
    # loop, where a bad reset still surfaces in the tests and the final
    # snapshot — but nothing runs after this one, so a reset that broke
    # enumeration here would leave no trace on the verdict. Restate the
    # outcome under a name that counts.
    add_reset_outcome(
        phase, "post_test_reset_ok", f"unit healthy after {' '.join(plan)}", "post-test reset left the unit degraded"
    )


def add_reset_outcome(phase: Phase, name: str, ok_text: str, fail_prefix: str) -> str:
    """Restate *phase*'s reset checks as one check *name* that counts toward the verdict, and return its status:
    FAIL if any check failed, else SKIP if the reset gate refused a reset (its SKIP says why), else PASS."""
    broke = [c.name for c in phase.checks if c.status == FAIL]
    refused = next((c.details for c in phase.checks if c.status == SKIP), None)
    status, details = PASS, ok_text
    if broke:
        status, details = FAIL, f"{fail_prefix}: {', '.join(broke)}"
    elif refused:
        status, details = SKIP, f"did not run: {refused}"
    phase.add(Check(name=name, status=status, details=details, ip="other"))
    return status
