# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The resets: tt-smi -glx_reset only, never tt-smi -r. After each, the root port of every chip it left off PCIe is
rescanned and the chips are recounted, the missing ones named by BDF, tray and ASIC. Phase 2 (the pre-test reset
loop), phase 4 (the post-test reset), the recovery between bus-bar steps and the end sequence after the post-test
reset. A reset that cannot run here (no root or sudo rule for it, the reset gate refuses) is a SKIP that says why."""
from __future__ import annotations

import shlex
import subprocess
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from .aer import aer_check, aer_new, read_aer
from . import events
from .busbar import Step, run_step
from .common import emit_result, emit_running, log, print_phase_summary, root_cmd, run, write_command_log
from .galaxy import EXPECTED_CHIP_COUNT, TRAY_TO_UBB_NIBBLE, expected_bdf
from .model import FAIL, PASS, SKIP, WARN, Check, Phase
from .snapshot import resolve_tt_smi, take_snapshot
from .snapshot_checks import pcie_enum_check, validate_snapshot

# The tt-smi flags a tier's reset plan may name.
RESET_FLAGS = ("-glx_reset",)
SYS = Path("/sys")
ALL_BDFS = tuple(expected_bdf(t, a) for t in TRAY_TO_UBB_NIBBLE for a in range(1, 9))
RESCAN_WAIT_S, RESCAN_EVERY_S = 60, 5  # a missing chip's root port is rescanned every 5 s until it is back, up to 60 s
END_DELAY_S = 60  # end sequence: the wait after the post-test reset before the fabric open
END_DBI_S = 10  # the end sequence's DRAM byte/DBI readback on the open mesh, s


def add_args(ap) -> None:
    ap.add_argument(
        "--reset-gate",
        metavar="CMD",
        help="A command run before every reset; a non-zero exit skips that reset, its last output line saying why.",
    )
    ap.add_argument(
        "--end-delay-s",
        type=int,
        default=END_DELAY_S,
        help=f"End sequence: seconds between the post-test reset and the fabric open. Default {END_DELAY_S}.",
    )


def pcie_bdfs() -> list[str] | None:
    """BDFs of the Tenstorrent chips enumerated on PCIe, or None if they could not be listed."""
    try:
        lspci = subprocess.run(["lspci", "-D", "-d", "1e52:"], capture_output=True, text=True, check=True)
        return [line.split()[0] for line in lspci.stdout.splitlines() if line.strip()]
    except Exception:
        return None


def recount_check(name: str, bdfs: list[str] | None) -> Check:
    """The chip recount after a reset, from the PCIe listing: how many of the 32 are back, each missing one by BDF, tray
    and ASIC, and how many each tray is short."""
    if bdfs is None:
        return Check(name=name, status=FAIL, details="chips could not be listed (lspci failed)", ip="pcie")
    c = pcie_enum_check([{"board_info": {"bus_id": b}} for b in bdfs], name)
    short = Counter(m["tray"] for m in c.data["missing"])
    c.data["trays_short"] = dict(short)
    if short:
        c.details += "; " + ", ".join(f"tray {t}: {n} of 8 missing" for t, n in sorted(short.items()))
    return c


def root_port(bdf: str) -> str | None:
    """The bridge whose secondary bus is *bdf*'s bus: the port to rescan, found also for a chip gone from PCIe."""
    bus = str(int(bdf.split(":")[-2], 16))
    for f in sorted((SYS / "bus/pci/devices").glob("*/secondary_bus_number")):
        try:
            if f.read_text().strip() == bus:
                return f.parent.name
        except OSError:
            continue
    return None


def rescan(missing: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """Rescan the root port of each *missing* chip every RESCAN_EVERY_S s until the chip is back on PCIe, up to
    RESCAN_WAIT_S. The chips that came back, each with its root port, and why a root port could not be rescanned."""
    ports = {b: root_port(b) for b in missing}
    whys = [f"{b}: root port not found" for b, rp in ports.items() if rp is None]
    cmds = {}
    for rp in sorted({rp for rp in ports.values() if rp}):
        cmds[rp], why = root_cmd(["tee", str(SYS / "bus/pci/devices" / rp / "rescan")])
        if why:
            whys.append(why)
    left, back, t0 = {b: rp for b, rp in ports.items() if cmds.get(rp)}, {}, time.time()
    while left and time.time() - t0 < RESCAN_WAIT_S:
        for rp in sorted(set(left.values())):
            subprocess.run(cmds[rp], input="1", capture_output=True, text=True, timeout=15)
        time.sleep(RESCAN_EVERY_S)
        now = set(pcie_bdfs() or ())
        back.update({b: left.pop(b) for b in list(left) if b in now})
    return back, whys


def _gate(cmd: str | None) -> str:
    """Why the reset gate refuses a reset now; "" when there is none or it allows it."""
    if not cmd:
        return ""
    try:
        cp = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"reset gate did not run: {e}"
    last = (cp.stdout + cp.stderr).strip().splitlines()
    return "" if cp.returncode == 0 else f"reset gate: {last[-1] if last else f'rc {cp.returncode}'}"


def glx_reset(
    tt_smi: str,
    phase: Phase,
    name: str,
    recount: str,
    dry_run: bool,
    logs_dir: Path | None = None,
    gate: str | None = None,
) -> list[str] | None:
    """One tt-smi -glx_reset as check *name*, then the rescan of the root ports of the chips it left missing (check
    rescan_<name> when one came back only that way or a port could not be rescanned) and the recount as check
    *recount*. The chips on PCIe after it, or None when the reset could not run (a SKIP 'reset cannot run: <why>')."""
    cmd, why = ([tt_smi, "-glx_reset"], "") if dry_run else root_cmd([tt_smi, "-glx_reset"])
    why = why or ("" if dry_run else _gate(gate))
    if why:
        log(f"{name}: reset cannot run: {why}")
        phase.add(Check(name=name, status=SKIP, details=f"reset cannot run: {why}", ip="other"))
        return None
    # before it: the HSC latches, each tray with chips missing dumped, the AER errors (diag/captures.py reset)
    events.capture("reset", name, stage={"reset_loop": "C2", "recover": "busbar"}.get(phase.name, "C9"))
    emit_running(name)
    t0 = time.time()
    cp = run(cmd, dry_run=dry_run, capture_output=True, text=True)
    dt = time.time() - t0
    if logs_dir is not None and not dry_run:
        write_command_log(logs_dir / f"{name}.log", " ".join(cmd), cp.returncode, dt, cp.stdout, cp.stderr)
    status = PASS if cp.returncode == 0 else FAIL
    emit_result(name, status, suffix=f"({dt:.1f}s)")
    if status == FAIL:
        for line in ((cp.stderr or "").strip() or (cp.stdout or "").strip()).splitlines()[-5:]:
            log(f"  ! {line}")
    phase.add(
        Check(
            name=name,
            status=status,
            details=f"tt-smi -glx_reset rc={cp.returncode} dur={dt:.1f}s",
            data={
                "rc": cp.returncode,
                "duration_s": dt,
                "stdout_tail": (cp.stdout or "")[-2000:],
                "stderr_tail": (cp.stderr or "")[-2000:],
            },
            ip="other",
        )
    )
    if dry_run:
        return list(ALL_BDFS)
    bdfs = pcie_bdfs()
    missing = [b for b in ALL_BDFS if bdfs is not None and b not in bdfs]
    if missing:
        back, whys = rescan(missing)
        bdfs = pcie_bdfs() if back else bdfs
        if back or whys:
            phase.add(
                Check(
                    name=f"rescan_{name}",
                    status=WARN if back else SKIP,
                    details="; ".join(
                        [*(f"{b} back only after a rescan of root port {rp}" for b, rp in back.items()), *whys]
                    ),
                    data={"back": back, "not_rescanned": whys},
                    ip="pcie",
                )
            )
    phase.add(recount_check(recount, bdfs))
    return bdfs


def reset_loop(
    tt_smi: str,
    plan: Sequence[str],
    phase: Phase,
    dry_run: bool,
    snapshot_out: Path | None = None,
    post_reset_phases: list | None = None,
    logs_dir: Path | None = None,
    gate: str | None = None,
) -> None:
    """The plan's glx_resets in order (see glx_reset); it stops at one that cannot run or leaves chips missing.

    If `snapshot_out` and `post_reset_phases` are provided, a fresh snapshot is
    taken + validated after each reset and appended to `post_reset_phases` as
    (name, Phase) pairs for the caller to splice into the report.
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
        label = f"r{i+1}_{flag.lstrip('-')}"
        bdfs = glx_reset(tt_smi, phase, f"reset_{i+1}_{flag.lstrip('-')}", f"recount_{i+1}", dry_run, logs_dir, gate)
        if bdfs is None:
            return
        _post_reset_snapshot(label)
        if len(bdfs) != EXPECTED_CHIP_COUNT:
            log(f"Reset {i+1} left {len(bdfs)}/{EXPECTED_CHIP_COUNT} chips; stopping reset loop")
            return


def recover(args, outdir: Path) -> list[str]:
    """Recovery between bus-bar steps, after one that did not pass: a glx_reset, root-port rescan and recount. Its
    finding lines: why the reset could not run, the chips back only after a rescan, the chips still missing; none when
    every chip is back."""
    if args.skip_reset:
        return ["reset cannot run: --skip-reset"]
    phase = Phase(name="recover")
    try:
        glx_reset(
            resolve_tt_smi(args.tt_smi_path),
            phase,
            "reset_recover",
            "recount_recover",
            args.dry_run,
            outdir,
            args.reset_gate,
        )
    except SystemExit as e:  # resolve_tt_smi: no tt-smi
        return [f"reset cannot run: {e}"]
    return [
        c.details if c.status == SKIP else f"{c.status} {c.name}: {c.details}" for c in phase.checks if c.status != PASS
    ]


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
    tt_smi_path: str | None, plan: Sequence[str], phase: Phase, dry_run: bool, gate: str | None = None
) -> bool:
    """Reset the unit between the tests and the triage phase. False when the reset left the unit degraded; only
    the checks it adds are judged, not ones *phase* already held. A reset that cannot run leaves the unit as it is.

    Deliberately bare: reset_loop() is called without snapshot_out /
    post_reset_phases, so it takes no revalidation snapshot and stays out of
    fold_post_reset_snapshots(). Adding revalidation here means reworking that
    function, which assumes a single batch of snapshot_after_* phases judged by
    normalize_health_report() on post[-1].
    """
    held = len(phase.checks)
    try:
        tt_smi = resolve_tt_smi(tt_smi_path)
        reset_loop(tt_smi, plan, phase, dry_run, gate=gate)
    except Exception as e:
        phase.error = repr(e)
        phase.add(Check(name="post_test_reset", status=FAIL, details=repr(e), ip="other"))
    # reset_loop names every check reset_*, which report.py classifies as a
    # reset op and drops from the verdict. That is right for the pre-test
    # loop, where a bad reset still surfaces in the tests and the final
    # snapshot — but nothing runs after this one, so a reset that broke
    # enumeration here would leave no trace on the verdict. Restate the
    # outcome under a name that counts.
    broke = [c.name for c in phase.checks[held:] if c.status == FAIL]
    skipped = [c.details for c in phase.checks[held:] if c.status == SKIP and c.name.startswith("reset_")]
    phase.add(
        Check(
            name="post_test_reset_ok",
            status=FAIL if broke else SKIP if skipped else PASS,
            details=(
                f"post-test reset left the unit degraded: {', '.join(broke)}"
                if broke
                else skipped[0]
                if skipped
                else f"unit healthy after {' '.join(plan)}"
            ),
            ip="other",
        )
    )
    return not broke


def run_end(tt_smi_path: str | None, phase: Phase, out: Path, dry_run: bool, delay_s: int, gate: str | None) -> None:
    """The end sequence, after the post-test reset: *delay_s* of quiet, then one whole-box fabric open and close
    with the DRAM byte/DBI readback on the open mesh, judged with the AER errors over it. A failed or hung open, or an
    uncorrectable AER error, gets one more glx_reset, rescan and recount, so no chip is left hung. Not run when the
    post-test reset did not leave every chip back: the open would test nothing the recount has not already named."""
    reset = next((c for c in reversed(phase.checks) if c.name == "post_test_reset_ok"), None)
    if reset is not None and reset.status != PASS:
        phase.add(Check(name="end_fabric", status=SKIP, details=f"not run: {reset.details}", ip="eth"))
        return
    log(f"end sequence: {delay_s} s, then the fabric open and close")
    if not dry_run:
        time.sleep(delay_s)
    before = {} if dry_run else read_aer()
    t0 = time.time()
    argv = ("fabric_open.py", "--rows", "{out}/rows.jsonl", "--beat", "{out}/beat", "--label", "open")
    opened = run_step(
        Step(
            "fabric", "fabric open and close after the reset", (*argv, "--dbi-wall-s", str(END_DBI_S)), 60 + END_DBI_S
        ),
        out,
        dry_run,
        test="end",
    )
    phase.add(opened)
    if dry_run:
        return
    aer = replace(aer_check(aer_new(before, read_aer()), (time.time() - t0) / 3600), name="end_aer")
    phase.add(aer)
    if FAIL in (opened.status, aer.status):
        glx_reset(resolve_tt_smi(tt_smi_path), phase, "reset_end", "recount_end", dry_run, out, gate)
