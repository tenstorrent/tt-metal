#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Blackhole Galaxy system diagnostic tool (DRAFT).

Orchestrates: tt-smi snapshot validation, direct FW-telemetry-table reads,
reset stability loop, tt-metal deployment-test gtest invocation, the kmd_triage
first-step tools, the tt-triage ETH status read, and — where the host has the
package — the QSFP tests and the sys-triage bundle.
 Emits a single JSON pass/fail report.

This file holds the CLI and the order the phases run in. What each tier runs is
in tiers.py; the phases themselves are in diag/.

Run via run_diag.sh which sets TT_METAL_HOME / PYTHONPATH / LD_LIBRARY_PATH.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# Run as a script this directory is already sys.path[0], but run_diag() is also a
# programmatic entry point, and a caller that loads this file by path need not
# have put the suite directory on the path for diag/ and tiers.py to import.
_SUITE_DIR = str(Path(__file__).resolve().parent)
if _SUITE_DIR not in sys.path:
    sys.path.insert(0, _SUITE_DIR)

from diag.aer import check_aer, read_aer  # noqa: E402
from diag.captures import capture_events  # noqa: E402
from diag.common import call_hook, default_tt_metal_path, log, print_phase_line, print_phase_summary  # noqa: E402
from diag.device_tests import run_tests  # noqa: E402
from diag.eth_status import run_eth_status  # noqa: E402
from diag.eth_summary import SUMMARY_FILE, render_eth_summary  # noqa: E402
from diag.model import FAIL, PASS, SKIP, WARN, Check, Phase  # noqa: E402
from diag.qsfp import QSFP_TOOL_BIN, run_qsfp_tests  # noqa: E402
from diag.resets import fold_post_reset_snapshots, reset_loop, run_post_test_reset  # noqa: E402
from diag.retest import run_retest  # noqa: E402
from diag.sensors import Bmc, check_tray_power, read_power  # noqa: E402
from diag.snapshot import (  # noqa: E402
    DEFAULT_TT_SMI_PROVISIONED,
    collect_fru_info,
    detect_tt_smi_version,
    resolve_tt_smi,
    take_snapshot,
)
from diag.snapshot_checks import validate_snapshot  # noqa: E402
from diag.sys_triage import SYS_TRIAGE_BIN, run_sys_triage  # noqa: E402
from diag.triage import TRIAGE_DIR_ENV, TRIAGE_SUBDIR, default_triage_dir, run_triage  # noqa: E402
from diag.verdict import attribute_power, verdict_lines  # noqa: E402
from tiers import DEPRECATED_TIER_ALIASES, TIERS, select_stages  # noqa: E402


def build_diag_arg_parser() -> argparse.ArgumentParser:
    """Build the diag CLI parser. Exposed so callers (e.g. run_health_check's
    subprocess wrapper) can introspect the accepted flags and stay in sync."""
    ap = argparse.ArgumentParser(description="SYS-4365 diag tool")
    ap.add_argument("--tier", choices=[*TIERS, *DEPRECATED_TIER_ALIASES], required=True)
    ap.add_argument("--dry-run", action="store_true", help="Print intended subprocess calls; skip destructive steps.")
    ap.add_argument("--skip-reset", action="store_true", help="Skip every reset, and so the retest.")
    ap.add_argument("--skip-tests", action="store_true", help="Skip the tests and the retest.")
    ap.add_argument(
        "--skip-triage",
        action="store_true",
        help="Skip the post-test reset and the triage tools entirely.",
    )
    ap.add_argument(
        "--triage-dir",
        type=Path,
        default=default_triage_dir(),
        help=f"Directory holding the triage scripts. Default: ${TRIAGE_DIR_ENV}, else <repo>/{TRIAGE_SUBDIR}.",
    )
    ap.add_argument(
        "--triage-gating",
        action="store_true",
        help=(
            "Let triage FAILs gate the run. Off by default: triage FAILs are held at WARN so a "
            "tool still bedding in cannot ticket the fleet. Findings are recorded either way."
        ),
    )
    ap.add_argument(
        "--skip-eth-status",
        action="store_true",
        help="Skip the ETH status read (two tt-triage check_eth_status reads) entirely.",
    )
    ap.add_argument(
        "--skip-qsfp-tests",
        action="store_true",
        help="Skip the QSFP tests entirely.",
    )
    ap.add_argument(
        "--qsfp-tool-path",
        help=f"The {QSFP_TOOL_BIN} binary. Default: whatever is on PATH (the syseng .deb installs it).",
    )
    ap.add_argument(
        "--qsfp-descriptor",
        type=Path,
        help=(
            "factory_system_descriptor.textproto, which gives the cage-attached links an expected "
            "partner. Without one only the soldered internal links are checked against a topology, "
            "and the cabling checks say so rather than reporting coverage they don't have."
        ),
    )
    ap.add_argument(
        "--qsfp-gating",
        action="store_true",
        help=(
            "Let QSFP test FAILs gate the run. Off by default for the same reason as "
            "--triage-gating: findings are recorded either way, but a tool still bedding in "
            "cannot ticket the fleet."
        ),
    )
    ap.add_argument(
        "--skip-sys-triage",
        action="store_true",
        help="Skip the sys-triage bundle collection entirely.",
    )
    ap.add_argument(
        "--sys-triage-path",
        help=f"The {SYS_TRIAGE_BIN} binary. Default: whatever is on PATH (the syseng .deb installs it).",
    )
    ap.add_argument(
        "--bmc-ip",
        help="BMC address for sys-triage. Default: $BMC_IP. Without all three BMC values the phase reports SKIP.",
    )
    ap.add_argument("--bmc-user", help="BMC user for sys-triage. Default: $BMC_USER.")
    ap.add_argument(
        "--bmc-password",
        help=(
            "BMC password for sys-triage. Default: $BMC_PASSWORD, which is preferred: a flag value is "
            "visible in the process list. Passed on to sys-triage through its environment, never argv."
        ),
    )
    ap.add_argument(
        "--input-snapshot", type=Path, help="Use pre-captured tt-smi snapshot JSON instead of live tt-smi call."
    )
    ap.add_argument(
        "--tt-smi-path", help=f"tt-smi binary or repo path. Default: {DEFAULT_TT_SMI_PROVISIONED} else PATH."
    )
    ap.add_argument(
        "--tt-metal-path",
        type=Path,
        default=default_tt_metal_path(),
        help="tt-metal repo root (contains build/test/tt_metal/unit_tests_deployment).",
    )
    ap.add_argument(
        "--no-hsc-clear",
        action="store_true",
        help="Keep the tray HSCs' under-voltage latches after a run that loaded the chips "
        "(default: read and report them, then clear them once with PMBus CLEAR_FAULTS).",
    )
    ap.add_argument(
        "--stage-order",
        help="Run only these stages, named in run order: reset, the tier's tests by name, end (the post-test "
        "reset), eth.",
    )
    ap.add_argument("--reset-gate", metavar="CMD", help="Run before each reset; a non-zero exit skips that reset.")
    ap.add_argument(
        "--hook",
        metavar="SCRIPT",
        help="Called as SCRIPT <event> <step> <status> <outdir>, synchronously: stage-start at the reset and end "
        "stages, after at a reset that leaves chips missing (then the chips are recounted).",
    )
    ap.add_argument("--output", type=Path, default=Path("diag_report.json"))
    ap.add_argument("--snapshot-out", type=Path, default=Path("/tmp/diag_snapshot.json"))
    return ap


# ─────────────────────────────────────────────────────────────────────────────
# Phase plumbing
# ─────────────────────────────────────────────────────────────────────────────


def _guarded(phase: Phase, error_check: str, error_status: str, body: Callable[[], None]) -> None:
    """Run a phase's body; an exception becomes one ``error_check`` at ``error_status``
    instead of taking the run, and the report, down with it."""
    try:
        body()
    except Exception as e:
        phase.error = repr(e)
        phase.add(Check(name=error_check, status=error_status, details=repr(e), ip="other"))


def _finish(report: dict, phase: Phase, t0: float, *, summary: bool, all_skip_stays_skip: bool = False) -> None:
    """Time, roll up, record and print a phase.

    ``summary`` prints the grouped per-check summary; without it the phase gets
    one rollup line, for phases that print their own per-step lines as they run.
    ``all_skip_stays_skip`` keeps a phase that only skipped at SKIP rather than
    rolling it up to PASS, for the optional-tool phases where no reading at all
    must not read as a clean one.
    """
    phase.duration_s = time.time() - t0
    if not all_skip_stays_skip or any(c.status != SKIP for c in phase.checks):
        phase.rollup()
    report["phases"][phase.name] = asdict(phase)
    if summary:
        print_phase_summary(phase.name, report["phases"][phase.name])
    else:
        print_phase_line(report["phases"][phase.name])


def _print_overall(report: dict) -> None:
    # Separator delineates the live per-phase output above from the final
    # rollup below, so the OVERALL summary block is visually distinct.
    print("  " + "─" * 70)
    print(f"  OVERALL        {report['overall_status']}")
    # Per-phase rollup. For non-PASS phases, list the offending checks (ip:name=status).
    # A phase that does not gate says so on its own line: otherwise a reader
    # seeing a FAIL here above an OVERALL that is not FAIL would reasonably
    # conclude the report had contradicted itself.
    for phase_name, p in report["phases"].items():
        status = p["status"]
        # Only worth saying when there is something for it to explain: a passing
        # phase that does not gate looks no different from one that does.
        note = "" if p.get("gates", True) or status == PASS else "  (does not gate)"
        if status == PASS:
            print(f"    {phase_name:14} {status}")
            continue
        bad = [c for c in p["checks"] if c["status"] in (WARN, FAIL)]
        if bad:
            items = ", ".join(f"{c.get('ip','other')}:{c['name']}={c['status']}" for c in bad)
            print(f"    {phase_name:14} {status}{note} — {items}")
        else:
            print(f"    {phase_name:14} {status}{note}")


def _print_eth_summary(report: dict, output: Path, logs_dir: Path) -> None:
    """The ETH link summary, printed last and kept beside the logs. Rendered after
    the report is written, so a fault here costs the summary and nothing else."""
    summary_path = logs_dir / SUMMARY_FILE
    try:
        text = render_eth_summary(report, report_path=output, summary_path=summary_path)
    except Exception as e:
        log(f"could not render the ETH link summary: {e!r}")
        return
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text)
    except OSError as e:
        log(f"could not write {summary_path}: {e!r}")
    print("\n" + text, end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_diag(
    tier: str,
    *,
    dry_run: bool = False,
    skip_reset: bool = False,
    skip_tests: bool = False,
    skip_triage: bool = False,
    skip_eth_status: bool = False,
    skip_qsfp_tests: bool = False,
    input_snapshot: Path | None = None,
    tt_smi_path: str | None = None,
    tt_metal_path: Path | None = None,
    triage_dir: Path | None = None,
    triage_gating: bool = False,
    qsfp_tool_path: str | None = None,
    qsfp_descriptor: Path | None = None,
    qsfp_gating: bool = False,
    skip_sys_triage: bool = False,
    sys_triage_path: str | None = None,
    bmc_ip: str | None = None,
    bmc_user: str | None = None,
    bmc_password: str | None = None,
    hsc_clear: bool = True,
    stage_order: str | None = None,
    reset_gate: str | None = None,
    hook: str | None = None,
    output: Path = Path("diag_report.json"),
    snapshot_out: Path = Path("/tmp/diag_snapshot.json"),
) -> tuple[int, dict]:
    """Run the full pipeline (snapshot → resets → gtests → retest → triage → ETH status → QSFP tests → sys-triage).

    Programmatic entry point equivalent to the CLI: writes the JSON report to
    *output* (gtest logs to ``<output_dir>/logs/``) and returns
    ``(exit_code, report_dict)``. ``exit_code`` is ``1`` on FAIL, else ``0`` —
    identical to the process exit code produced by ``main()``.
    """
    if tier in DEPRECATED_TIER_ALIASES:
        log(f"WARNING: tier '{tier}' is deprecated, running '{DEPRECATED_TIER_ALIASES[tier]}' instead")
        tier = DEPRECATED_TIER_ALIASES[tier]
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of: {', '.join(TIERS)}")
    plan = TIERS[tier]
    if stage_order is not None:
        plan = select_stages(plan, stage_order)
        log(f"--stage-order: running {stage_order}")
    if tt_metal_path is None:
        tt_metal_path = default_tt_metal_path()

    started = datetime.now(timezone.utc)
    report = {
        "tool_version": "0.3.0-draft",
        "tier": tier,
        "host": socket.gethostname(),
        "started_utc": started.isoformat(),
        "dry_run": dry_run,
        "tt_smi_version": None,
        "detected_board_rev": None,
        "phases": {},
    }

    # Phase 1: snapshot
    snap_phase = Phase(name="snapshot")
    t0 = time.time()
    try:
        if input_snapshot:
            log(f"using input snapshot: {input_snapshot}")
            text = input_snapshot.read_text().strip()
            snap = json.loads(text) if text else {}
        else:
            tt_smi = resolve_tt_smi(tt_smi_path)
            version = detect_tt_smi_version(tt_smi)
            report["tt_smi_version"] = ".".join(str(x) for x in version) if version else None
            log(f"using tt-smi: {tt_smi} (version={report['tt_smi_version']})")
            snap = take_snapshot(tt_smi, snapshot_out)
        if snap:
            report["detected_board_rev"] = validate_snapshot(snap, snap_phase)
        else:  # a caller that took its own snapshot (one stage at a time) passes an empty one: nothing to judge
            snap_phase.add(
                Check(name="snapshot_capture", status=SKIP, details=f"not judged: {input_snapshot} is empty")
            )
    except Exception as e:
        snap_phase.error = repr(e)
        snap_phase.add(Check(name="snapshot_capture", status=FAIL, details=repr(e)))
    # Host FRU inventory (BMC identity: part numbers, serials, mfg dates) —
    # store-only forensics alongside the chip-level snapshot checks; runs even
    # when the tt-smi snapshot itself failed.
    collect_fru_info(snap_phase, dry_run)
    _finish(report, snap_phase, t0, summary=True)

    # Where every phase drops its raw tool output. Derived once: the reset
    # and gtest phases both write here, and collect_run_artifacts() attaches
    # whatever it finds, so a phase that writes elsewhere silently loses its
    # logs off the ticket.
    output_dir = output.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"

    # The trays' power before anything loads or resets them: an under-voltage
    # latch already set is told apart from one this run set.
    bmc = Bmc(dry_run)
    power_before = read_power(bmc, bricks=False)  # the bricks matter after the load

    # Phase 2: reset loop + per-reset snapshot revalidations
    reset_phase = Phase(name="reset_loop")
    post_reset_phases: list = []
    # Only revalidate via fresh snapshots when running live (not --input-snapshot).
    snap_out_for_resets = None if input_snapshot else snapshot_out
    t0 = time.time()
    if skip_reset:
        reset_phase.add(Check(name="reset_loop", status=SKIP, details="--skip-reset"))
    elif not plan.resets:
        reset_phase.add(Check(name="reset_loop", status=SKIP, details=f"no pre-test resets for tier '{tier}'"))
    else:
        call_hook(hook, "stage-start", "reset", "-", logs_dir, dry_run)

        def _resets() -> None:
            reset_loop(
                resolve_tt_smi(tt_smi_path),
                plan.resets,
                reset_phase,
                dry_run,
                snapshot_out=snap_out_for_resets,
                post_reset_phases=post_reset_phases,
                logs_dir=logs_dir,
                gate=reset_gate,
                hook=hook,
            )

        _guarded(reset_phase, "reset_loop", FAIL, _resets)
    # reset_loop prints its own per-reset lines as they run; rollup line only.
    _finish(report, reset_phase, t0, summary=False, all_skip_stays_skip=True)
    fold_post_reset_snapshots(report, post_reset_phases)
    # Taken after the resets, which re-enumerate the chips and so restart their counters.
    aer_before = read_aer() if plan.aer_diff and not dry_run else {}

    # Phase 3: gtest
    test_phase = Phase(name="tests")
    t0 = time.time()
    if skip_tests:
        test_phase.add(Check(name="tests", status=SKIP, details="--skip-tests"))
    elif not plan.tests:
        test_phase.add(Check(name="tests", status=SKIP, details=f"no tests for tier '{tier}'", ip="other"))
    else:
        _guarded(test_phase, "tests", FAIL, lambda: run_tests(tt_metal_path, plan.tests, test_phase, dry_run, logs_dir))
    # tests phase prints its own per-test lines as they run — no inline summary.
    _finish(report, test_phase, t0, summary=False)

    # Phase 3b: retest (test -> reset -> test) for boot-dependent faults.
    if plan.retest:
        retest_phase = Phase(name="retest")
        t0 = time.time()
        if skip_tests or skip_reset:
            retest_phase.add(
                Check(name="retest", status=SKIP, details="--skip-tests" if skip_tests else "--skip-reset", ip="other")
            )
        else:

            def _retest() -> None:
                snap_phase = run_retest(
                    tt_smi_path,
                    tt_metal_path,
                    plan.retest,
                    plan.retest_reset,
                    test_phase,
                    retest_phase,
                    dry_run,
                    logs_dir,
                    snap_out_for_resets,
                    gate=reset_gate,
                    hook=hook,
                )
                if snap_phase is not None:
                    report["phases"][snap_phase.name] = asdict(snap_phase)

            _guarded(retest_phase, "retest", FAIL, _retest)
        _finish(report, retest_phase, t0, summary=False, all_skip_stays_skip=True)

    # Phase 3c: the PCIe AER counts the load added, before the post-test reset
    # re-enumerates the chips.
    if plan.aer_diff:
        aer_phase = Phase(name="pcie_aer")
        t0 = time.time()
        if dry_run:
            aer_phase.add(Check(name="pcie_aer_diff", status=SKIP, details="(dry) sysfs AER counters", ip="pcie"))
        else:
            _guarded(aer_phase, "pcie_aer_diff", WARN, lambda: check_aer(aer_phase, aer_before, read_aer()))
        _finish(report, aer_phase, t0, summary=True, all_skip_stays_skip=True)

    # Phase 4: post-test reset. Its job is to hand phase 5 a freshly reset unit
    # and to guarantee the triage probes, which open every chip read-write,
    # never run concurrently with the tests.
    post_test_plan = () if skip_reset or (skip_triage and stage_order is None) else plan.post_test_resets
    if post_test_plan:
        post_reset_phase = Phase(name="post_test_reset")
        t0 = time.time()
        call_hook(hook, "stage-start", "end", "-", logs_dir, dry_run)
        run_post_test_reset(tt_smi_path, post_test_plan, post_reset_phase, dry_run, reset_gate, hook)
        _finish(report, post_reset_phase, t0, summary=False, all_skip_stays_skip=True)

    # Phase 4b: the trays' power after the load and the resets, judged against the
    # reads before them; its causes name why a chip went missing. The latches are
    # cleared only after a run that loaded the chips, so the next run starts clean;
    # a run without load leaves them as found.
    power_phase = Phase(name="tray_power")
    t0 = time.time()
    loaded = bool(plan.tests) and not skip_tests
    keep = "--no-hsc-clear" if not hsc_clear else "" if loaded else "no load in this run; latches kept as found"
    power_causes: dict = {}

    def _power() -> None:
        power_causes.update(check_tray_power(power_phase, bmc, power_before, keep))

    _guarded(power_phase, "tray_power", WARN, _power)
    _finish(report, power_phase, t0, summary=True, all_skip_stays_skip=True)

    # Phase 5: triage tools, from tools/scaleout/kmd_triage in this repo.
    triage_phase = Phase(name="triage")
    t0 = time.time()
    if skip_triage:
        triage_phase.add(Check(name="triage", status=SKIP, details="--skip-triage", ip="other"))
    elif not plan.triage:
        triage_phase.add(Check(name="triage", status=SKIP, details=f"no triage tools for tier '{tier}'", ip="other"))
    else:
        # Never let the triage phase take the run down: a crash here is lost
        # coverage, not a hardware finding.
        _guarded(
            triage_phase,
            "triage",
            WARN,
            lambda: run_triage(triage_dir, plan.triage, triage_phase, dry_run, logs_dir, triage_gating, tt_metal_path),
        )
    _finish(report, triage_phase, t0, summary=True)

    # Phase 6: the ETH status read. After the post-test reset and the triage
    # tools for the reason they sit there: it opens every chip, so it must not
    # overlap the tests. It reads the links as found and never resets, so on a
    # tier with no resets the state it reads is the one the unit was left in.
    # Unlike the triage and QSFP phases it gates: it is the flash tier's verdict.
    eth_phase = Phase(name="eth_status")
    t0 = time.time()
    if skip_eth_status:
        eth_phase.add(Check(name="eth_status_read", status=SKIP, details="--skip-eth-status", ip="other"))
    elif not plan.eth_status:
        eth_phase.add(
            Check(name="eth_status_read", status=SKIP, details=f"no ETH status read for tier '{tier}'", ip="other")
        )
    else:
        # A crash here is lost coverage, not a hardware finding.
        _guarded(
            eth_phase,
            "eth_status_read",
            WARN,
            lambda: run_eth_status(tt_metal_path, eth_phase, dry_run, logs_dir),
        )
    _finish(report, eth_phase, t0, summary=True, all_skip_stays_skip=True)

    # Phase 7: the QSFP tests, when the host has the package. Last,
    # and after the post-test reset for the same two reasons the triage phase
    # sits there: it opens every chip, so it must not overlap the tests, and the
    # tool has no SIGBUS handler, so a concurrent reset would kill it outright
    # rather than being reported. Reading the links after that reset is also the
    # reading that matters — it is the state the machine is being left in.
    # The tool is still being validated against the fleet, so its findings are
    # recorded at their real severity but do not vote: --qsfp-gating is what
    # lets them, once the fleet data says they can be trusted.
    qsfp_phase = Phase(name="qsfp_tests", gates=qsfp_gating)
    t0 = time.time()
    if skip_qsfp_tests:
        qsfp_phase.add(Check(name="qsfp_collect", status=SKIP, details="--skip-qsfp-tests", ip="other"))
    elif not plan.qsfp_tests:
        qsfp_phase.add(Check(name="qsfp_collect", status=SKIP, details=f"no QSFP tests for tier '{tier}'", ip="other"))
    else:
        # Same rule as triage: a crash in an optional phase is lost
        # coverage, not a hardware finding, and must not take the run down.
        _guarded(
            qsfp_phase,
            "qsfp_collect",
            SKIP,
            lambda: run_qsfp_tests(
                tier,
                qsfp_phase,
                dry_run,
                logs_dir,
                qsfp_gating,
                binary_override=qsfp_tool_path,
                descriptor=qsfp_descriptor,
                detected_rev=report.get("detected_board_rev"),
            ),
        )
    _finish(report, qsfp_phase, t0, summary=True, all_skip_stays_skip=True)

    # The SEL and the kernel log since the run began: reads only, and evidence
    # for the verdict rather than findings, so the phase never votes.
    captures_phase = Phase(name="captures", gates=False)
    t0 = time.time()
    _guarded(captures_phase, "captures", SKIP, lambda: capture_events(captures_phase, dry_run, logs_dir, started))
    _finish(report, captures_phase, t0, summary=True, all_skip_stays_skip=True)

    # Phase 8: the sys-triage bundle, when the host has the package. After the
    # QSFP tests for the same reason they follow triage: it opens every chip
    # (tt-smi -s, tt-fw-terminal, the reset-unit scratch reads), so nothing may
    # run beside it. Forensics for a person rather than findings, so it never
    # votes.
    sys_triage_phase = Phase(name="sys_triage", gates=False)
    t0 = time.time()
    if skip_sys_triage:
        sys_triage_phase.add(Check(name="sys_triage", status=SKIP, details="--skip-sys-triage", ip="other"))
    elif not plan.sys_triage:
        sys_triage_phase.add(
            Check(name="sys_triage", status=SKIP, details=f"no sys-triage for tier '{tier}'", ip="other")
        )
    else:
        _guarded(
            sys_triage_phase,
            "sys_triage",
            SKIP,
            lambda: run_sys_triage(
                sys_triage_phase,
                dry_run,
                logs_dir,
                binary_override=sys_triage_path,
                bmc_ip=bmc_ip,
                bmc_user=bmc_user,
                bmc_password=bmc_password,
            ),
        )
    _finish(report, sys_triage_phase, t0, summary=True, all_skip_stays_skip=True)

    ended = datetime.now(timezone.utc)
    report["ended_utc"] = ended.isoformat()
    report["total_duration_s"] = (ended - started).total_seconds()
    # Only phases that gate reach the verdict. A non-gating phase still rolls up
    # and still prints, so its findings are as visible as anyone else's — it
    # simply has no vote, which is the one honest way to carry a tool whose
    # findings are worth recording before they are worth acting on.
    statuses = [p["status"] for p in report["phases"].values() if p.get("gates", True)]
    if FAIL in statuses:
        report["overall_status"] = FAIL
    elif WARN in statuses:
        report["overall_status"] = WARN
    else:
        report["overall_status"] = PASS

    attribute_power(report, power_causes)
    report["verdict"] = verdict_lines(report)

    # Only the phases that write logs create this directory, and any of them can
    # be skipped — don't lose a completed run's report to a missing parent dir.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    log(f"wrote report: {output} (overall={report['overall_status']})")

    _print_overall(report)
    # A dry run reads nothing, so there is nothing to summarise.
    if plan.eth_summary and not dry_run:
        _print_eth_summary(report, output, logs_dir)
    # Last, so the end of the log says what needs fixing.
    print("  " + "─" * 70)
    for line in report["verdict"]:
        print(f"  {line}")
    return (0 if report["overall_status"] != FAIL else 1), report


def main() -> int:
    ap = build_diag_arg_parser()
    args = ap.parse_args()
    if args.stage_order is not None:
        try:
            select_stages(TIERS[DEPRECATED_TIER_ALIASES.get(args.tier, args.tier)], args.stage_order)
        except ValueError as e:
            ap.error(str(e))
    rc, _report = run_diag(
        tier=args.tier,
        dry_run=args.dry_run,
        skip_reset=args.skip_reset,
        skip_tests=args.skip_tests,
        skip_triage=args.skip_triage,
        skip_eth_status=args.skip_eth_status,
        skip_qsfp_tests=args.skip_qsfp_tests,
        input_snapshot=args.input_snapshot,
        tt_smi_path=args.tt_smi_path,
        tt_metal_path=args.tt_metal_path,
        triage_dir=args.triage_dir,
        triage_gating=args.triage_gating,
        qsfp_tool_path=args.qsfp_tool_path,
        qsfp_descriptor=args.qsfp_descriptor,
        qsfp_gating=args.qsfp_gating,
        skip_sys_triage=args.skip_sys_triage,
        sys_triage_path=args.sys_triage_path,
        bmc_ip=args.bmc_ip,
        bmc_user=args.bmc_user,
        bmc_password=args.bmc_password,
        hsc_clear=not args.no_hsc_clear,
        stage_order=args.stage_order,
        reset_gate=args.reset_gate,
        hook=args.hook,
        output=args.output,
        snapshot_out=args.snapshot_out,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
