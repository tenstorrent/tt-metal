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
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

# Run as a script this directory is already sys.path[0], but run_diag() is also a
# programmatic entry point, and a caller that loads this file by path need not
# have put the suite directory on the path for diag/ and tiers.py to import.
_SUITE_DIR = str(Path(__file__).resolve().parent)
if _SUITE_DIR not in sys.path:
    sys.path.insert(0, _SUITE_DIR)

from diag.common import default_tt_metal_path, log, print_phase_line, print_phase_summary  # noqa: E402
from diag.galaxy import KNOWN_DROPPED  # noqa: E402
from diag.aer import AerWindow  # noqa: E402
from diag import captures  # noqa: E402
from diag.device_tests import run_tests  # noqa: E402
from diag import events  # noqa: E402
from diag.busbar import LOAD_TESTS, MODES, OVERRIDES, STEPS, link_fault, plan as busbar_plan, run_busbar  # noqa: E402
from diag.eth_status import run_eth_status  # noqa: E402
from diag import links  # noqa: E402
from diag.links import name_chips, run_links  # noqa: E402
from diag.eth_summary import SUMMARY_FILE, render_eth_summary  # noqa: E402
from diag.firmware import firmware_check  # noqa: E402
from diag.hsc import judge_hsc, read_hsc_status  # noqa: E402
from diag.model import FAIL, PASS, SKIP, WARN, Check, Phase  # noqa: E402
from diag.qsfp import QSFP_TOOL_BIN, run_qsfp_tests  # noqa: E402
from diag import report  # noqa: E402
from diag import resets  # noqa: E402
from diag.resets import END_DELAY_S, fold_post_reset_snapshots, reset_loop, run_end, run_post_test_reset  # noqa: E402
from diag.snapshot import (  # noqa: E402
    DEFAULT_TT_SMI_PROVISIONED,
    collect_fru_info,
    detect_tt_smi_version,
    resolve_tt_smi,
    take_snapshot,
)
from diag.sel import history_check, new_check, read_sel  # noqa: E402
from diag.snapshot_checks import pcie_enum_check, validate_snapshot  # noqa: E402
from diag.sys_triage import SYS_TRIAGE_BIN, run_sys_triage  # noqa: E402
from diag.triage import TRIAGE_DIR_ENV, TRIAGE_SUBDIR, default_triage_dir, run_triage  # noqa: E402
from tiers import DEPRECATED_TIER_ALIASES, TIERS, select_stages, select_tests  # noqa: E402


def build_diag_arg_parser() -> argparse.ArgumentParser:
    """Build the diag CLI parser. Exposed so callers (e.g. run_health_check's
    subprocess wrapper) can introspect the accepted flags and stay in sync."""
    ap = argparse.ArgumentParser(description="SYS-4365 diag tool")
    ap.add_argument("--tier", choices=[*TIERS, *DEPRECATED_TIER_ALIASES], required=True)
    ap.add_argument("--dry-run", action="store_true", help="Print intended subprocess calls; skip destructive steps.")
    ap.add_argument(
        "--skip-reset",
        action="store_true",
        help=(
            "Skip every reset: the reset loop, the post-test reset and the glx_reset before the links sweep. "
            "The sweep itself still runs on a link fault, on the unit as it is."
        ),
    )
    ap.add_argument(
        "--tests",
        metavar="LIST",
        help=(
            "Run only these of the tier's tests, comma-separated, in the tier's order; a bus-bar step by its check "
            "name (busbar_qb, busbar_mix, busbar_step, busbar_heavy, busbar_code), busbar for all of them. "
            "The other phases run as usual (their --skip-* flags skip them). Default: all of the tier's tests."
        ),
    )
    ap.add_argument("--stage-order", metavar="LIST", help="Run only these stages in run order: reset, tests, end, eth")
    ap.add_argument(
        "--skip-tests", action="store_true", help="Skip the tests phase entirely: gtests, pytests and the galaxy load."
    )
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
        "--load-mode", choices=MODES, help="Galaxy load: its length, in place of the tier's (with --tests pinned)."
    )
    for key, (flag, typ, what) in OVERRIDES.items():
        ap.add_argument(
            flag, dest=key, type=typ, default=None, help=f"Galaxy load: {what}. Not given: the tier's plan."
        )
    captures.add_args(ap)
    resets.add_args(ap)
    events.add_args(ap)
    links.add_args(ap)
    ap.add_argument("--output", type=Path, default=Path("diag_report.json"))
    ap.add_argument("--snapshot-out", type=Path, default=Path("/tmp/diag_snapshot.json"))
    report.add_args(ap)
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


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_TRACE_LINE_RE = re.compile(r"^(\d+\.\s|Location:)")


def _short_error(e: Exception, lines: int = 2) -> str:
    """An error's first message lines on one line, without colour codes or the stack trace under them."""
    text = [" ".join(ln.split()) for ln in _ANSI_RE.sub("", str(e)).splitlines()]
    return "; ".join([ln for ln in text if ln and not _TRACE_LINE_RE.match(ln)][:lines])


def recount_after_tests(snapshot: Callable[[], dict]) -> Check:
    """The chip count from a fresh snapshot after the tests. A snapshot that
    cannot be taken (tt-smi cannot read a chip, or is gone) is a FAIL with its reason."""
    name = "pcie_enum_count_after_tests"
    try:
        snap = snapshot()
    except Exception as e:
        return Check(name=name, status=FAIL, details=f"no snapshot after the tests: {_short_error(e)}", ip="other")
    return pcie_enum_check(snap.get("device_info") or [], name)


def _hear(stage: str, out: Path, dry_run: bool) -> None:
    """The caller's --hook hears a stage start (reset, links, end; the bus-bar stage says so itself)."""
    if events.hook and not dry_run:
        subprocess.run([events.hook, "stage-start", stage, "-", str(out)], check=False)


def _run_galaxy_load(
    tests: Sequence[str],
    phase: Phase,
    out: Path,
    mode: str,
    overrides: dict,
    dry_run: bool,
    sweep_reset: Callable[[], str],
    steps: Sequence[str] = STEPS,
    link_fault_before: bool = False,
    **links_kw,
) -> None:
    """The LOAD_TESTS among *tests* into *phase* at *mode*'s length: the bus-bar *steps*, then the links test, which
    sweeps after a hang or CCL failure in either (busbar.link_fault; never a numeric fault alone). One that raises is
    its own FAIL and does not stop the other."""
    held = len(phase.checks)
    if "busbar" in tests and KNOWN_DROPPED:
        why = f"known drop: tray {', '.join(map(str, sorted(KNOWN_DROPPED)))} is down and the load needs the 4x8 mesh"
        phase.add(Check(name="busbar", status=SKIP, details=why, ip="other"))
    elif "busbar" in tests:
        _guarded(phase, "busbar", FAIL, lambda: run_busbar(phase, out, mode, dry_run, tuple(steps), **overrides))
    fault = link_fault_before or any(link_fault(c) for c in phase.checks[held:] if c.name != "busbar")
    if "links" in tests:
        _hear("links", out, dry_run)
        _guarded(phase, "links", FAIL, lambda: run_links(phase, out, fault, sweep_reset, dry_run, **links_kw))


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
    output: Path = Path("diag_report.json"),
    snapshot_out: Path = Path("/tmp/diag_snapshot.json"),
    overrides: dict | None = None,
    tests: str | None = None,
    stage_order: str | None = None,
    reset_gate: str | None = None,
    end_delay_s: int = END_DELAY_S,
    load_mode: str | None = None,
    link_fault_before: bool = False,
    pair_s: int = links.PAIR_S,
    sweep_s: int = links.SWEEP_S,
) -> tuple[int, dict]:
    """Run the full pipeline (snapshot → resets → tests → triage → ETH status → QSFP tests → sys-triage → AER).

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
    plan, steps = TIERS[tier], STEPS
    if tests is not None:
        plan, steps = select_tests(plan, tests)
        log(f"--tests: running {', '.join(plan.tests) or 'no tests'}")
    if stage_order is not None:
        plan, steps = select_stages(plan, stage_order)
        log(f"--stage-order: running {stage_order}")
    if load_mode is not None:
        plan = replace(plan, load_mode=load_mode)
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
            snap = json.loads(input_snapshot.read_text())
        else:
            tt_smi = resolve_tt_smi(tt_smi_path)
            version = detect_tt_smi_version(tt_smi)
            report["tt_smi_version"] = ".".join(str(x) for x in version) if version else None
            log(f"using tt-smi: {tt_smi} (version={report['tt_smi_version']})")
            snap = take_snapshot(tt_smi, snapshot_out)
        if snap:
            report["detected_board_rev"] = validate_snapshot(snap, snap_phase)
        else:  # a caller that took its own snapshot (one stage at a time) passes an empty one: nothing to judge
            why = f"not judged: input snapshot {input_snapshot} is empty"
            snap_phase.add(Check(name="snapshot_capture", status=SKIP, details=why))
    except Exception as e:
        snap_phase.error = repr(e)
        snap_phase.add(Check(name="snapshot_capture", status=FAIL, details=repr(e)))
    # Host FRU inventory (BMC identity: part numbers, serials, mfg dates) —
    # store-only forensics alongside the chip-level snapshot checks; runs even
    # when the tt-smi snapshot itself failed.
    collect_fru_info(snap_phase, dry_run)
    # With the galaxy load or --stage-order, the post-test reset is followed by the end sequence, and the firmware
    # reads and the SEL are read first (the SEL again after it).
    post_test_plan = () if skip_reset or (skip_triage and stage_order is None) else plan.post_test_resets
    end_seq = bool(post_test_plan) and (plan.galaxy_load or stage_order is not None)
    live = not (input_snapshot or dry_run)
    sel_before, sel_why = read_sel() if end_seq and live else (None, "")
    if end_seq and live:
        snap_phase.add(history_check(sel_before, sel_why, time.time()))
        fw_dir = output.resolve().parent / "logs" / "firmware"
        _guarded(snap_phase, "firmware_reads", WARN, lambda: snap_phase.add(firmware_check(fw_dir)))
    _finish(report, snap_phase, t0, summary=True)

    # Where every phase drops its raw tool output. Derived once: the reset
    # and gtest phases both write here, and collect_run_artifacts() attaches
    # whatever it finds, so a phase that writes elsewhere silently loses its
    # logs off the ticket.
    output_dir = output.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"

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
        _hear("reset", logs_dir, dry_run)

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
            )

        _guarded(reset_phase, "reset_loop", FAIL, _resets)
    # reset_loop prints its own per-reset lines as they run; rollup line only.
    _finish(report, reset_phase, t0, summary=False)
    fold_post_reset_snapshots(report, post_reset_phases)

    # The AER errors over the run (pcie_aer_delta): counted from here, after the resets above, to the end of the
    # run. Every reset below goes through aer_reset(), which counts up to it and from what it left.
    aer = AerWindow() if plan.aer_diff and not dry_run else None

    def aer_reset(reset: Callable[[], bool]) -> bool:
        if aer:
            aer.mark()
        try:
            return reset()
        finally:
            if aer:
                aer.rebase()

    # Phase 3: the tests, in the tier's order: gtests and pytests, then the
    # galaxy load (the bus-bar steps, then the pair link test, its sweep on a
    # fault and the fabric open/close) at the tier's load_mode. Inside the AER window.
    test_phase = Phase(name="tests")

    def sweep_reset() -> str:
        """The links sweep's tt-smi -glx_reset: "" when it ran clean or --skip-reset skips it (the sweep then runs on the
        unit as it is), else why the sweep is skipped."""
        if skip_reset:
            log("links sweep: no glx_reset first (--skip-reset)")
            return ""
        ok = aer_reset(lambda: run_post_test_reset(tt_smi_path, ("-glx_reset",), test_phase, dry_run, reset_gate))
        return "" if ok else "reset failed"

    t0 = time.time()
    if skip_tests:
        test_phase.add(Check(name="tests", status=SKIP, details="--skip-tests"))
    elif not plan.tests:
        test_phase.add(Check(name="tests", status=SKIP, details=f"no tests for tier '{tier}'", ip="other"))
    else:
        # Around the load: the tray HSCs' input-voltage flags are read before and
        # after, and the chips are recounted after, with a fresh snapshot that
        # runs even under --skip-reset (a weak bus bar often drops a chip or
        # latches a flag without failing the test that sagged it). Not on an
        # input snapshot or a dry run: no live hardware to read.
        hsc_before = read_hsc_status() if live else None
        device_tests = [t for t in plan.tests if t not in LOAD_TESTS]
        if device_tests:  # none when --tests picks only the galaxy load
            _guarded(
                test_phase,
                "tests",
                FAIL,
                lambda: run_tests(tt_metal_path, device_tests, test_phase, dry_run, logs_dir),
            )
        if plan.galaxy_load:
            _run_galaxy_load(
                plan.tests,
                test_phase,
                logs_dir / "galaxy",
                plan.load_mode,
                overrides or {},
                dry_run,
                sweep_reset,
                steps,
                link_fault_before,
                pair_s=pair_s,
                sweep_s=sweep_s,
            )
        if live:
            _guarded(test_phase, "hsc_input_status", WARN, lambda: judge_hsc(hsc_before, read_hsc_status(), test_phase))

            after_out = snapshot_out.with_name(f"{snapshot_out.stem}_after_tests.json")
            test_phase.add(recount_after_tests(lambda: take_snapshot(resolve_tt_smi(tt_smi_path), after_out)))
    # tests phase prints its own per-test lines as they run — no inline summary.
    _finish(report, test_phase, t0, summary=False)

    # Phase 4: post-test reset. Its job is to hand phase 5 a freshly reset unit
    # and to guarantee the triage probes, which open every chip read-write,
    # never run concurrently with the tests.
    if post_test_plan:
        post_reset_phase = Phase(name="post_test_reset")
        t0 = time.time()
        _hear("end", logs_dir, dry_run)
        aer_reset(lambda: run_post_test_reset(tt_smi_path, post_test_plan, post_reset_phase, dry_run, reset_gate))
        if end_seq:
            end = lambda: run_end(tt_smi_path, post_reset_phase, logs_dir / "end", dry_run, end_delay_s, reset_gate)
            _guarded(post_reset_phase, "end_sequence", FAIL, lambda: aer_reset(end))
            events.capture("stage-end", "end")  # the final HSC read, judged against the run's start, and the bricks
        if end_seq and live:
            sel_after, why = read_sel()
            post_reset_phase.add(new_check(sel_before, sel_after, sel_why or why))
        _finish(report, post_reset_phase, t0, summary=False)

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

    # Phase 9: the AER errors over the run, on a tier with aer_diff. Last, so the window spans every phase that
    # opens the chips, and whatever was skipped. Left out of the report when there is no window (dry run).
    if aer:
        aer_phase = Phase(name="pcie_aer")
        t0 = time.time()
        _guarded(aer_phase, "pcie_aer_delta", WARN, lambda: aer_phase.add(aer.check()))
        _finish(report, aer_phase, t0, summary=True, all_skip_stays_skip=True)

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

    # Only the phases that write logs create this directory, and any of them can
    # be skipped — don't lose a completed run's report to a missing parent dir.
    output.parent.mkdir(parents=True, exist_ok=True)
    name_chips(report)
    events.check_findings(report)
    output.write_text(json.dumps(report, indent=2))
    log(f"wrote report: {output} (overall={report['overall_status']})")

    _print_overall(report)
    # A dry run reads nothing, so there is nothing to summarise.
    if plan.eth_summary and not dry_run:
        _print_eth_summary(report, output, logs_dir)
    return (0 if report["overall_status"] != FAIL else 1), report


def _exit_on_sigterm(signum, frame) -> None:
    """SIGTERM (the whole-run timeout, docker stop) as an exit, so the galaxy load stops the process groups of its
    programs on the way out: they are out of reach of a signal to this process's group."""
    raise SystemExit(128 + signum)


def main() -> int:
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    ap = build_diag_arg_parser()
    args = ap.parse_args()
    overrides = {k: getattr(args, k) for k in OVERRIDES}
    tier = TIERS[DEPRECATED_TIER_ALIASES.get(args.tier, args.tier)]
    if args.tests is not None and args.stage_order is not None:
        ap.error("--tests and --stage-order both pick the tests: give one")
    try:
        tier = select_tests(tier, args.tests)[0] if args.tests is not None else tier
        tier = select_stages(tier, args.stage_order)[0] if args.stage_order is not None else tier
    except ValueError as e:
        ap.error(str(e))
    if args.pair_s < 1 or args.sweep_s < 0:
        ap.error("--pair-s: at least 1; --sweep-s: at least 0")
    if args.load_mode and not tier.galaxy_load:
        ap.error(f"--load-mode: tier {args.tier} with these --tests runs no galaxy load")
    if any(v is not None for v in overrides.values()) and "busbar" not in tier.tests:
        loaded = ", ".join(n for n, t in TIERS.items() if t.galaxy_load)
        ap.error(
            f"the bus-bar overrides set the galaxy load's bus-bar steps, which tier {args.tier}"
            f"{' with --tests ' + args.tests if args.tests is not None else ''} does not run (tiers: {loaded})"
        )
    if tier.galaxy_load:
        try:
            busbar_plan(args.load_mode or tier.load_mode, **overrides)
        except ValueError as e:
            ap.error(str(e))
    events.setup(args)
    KNOWN_DROPPED.update([args.known_dropped_tray] if args.known_dropped_tray else [])
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
        output=args.output,
        snapshot_out=args.snapshot_out,
        overrides=overrides,
        tests=args.tests,
        stage_order=args.stage_order,
        reset_gate=args.reset_gate,
        end_delay_s=args.end_delay_s,
        load_mode=args.load_mode,
        link_fault_before=args.link_fault,
        pair_s=args.pair_s,
        sweep_s=args.sweep_s,
    )
    try:  # the run's report, last; a fault here costs the report, never the run's exit code
        print("\n".join(report.lines(args.output.resolve().parent, args)), flush=True)
    except Exception as e:
        log(f"could not print the run report: {e!r}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
