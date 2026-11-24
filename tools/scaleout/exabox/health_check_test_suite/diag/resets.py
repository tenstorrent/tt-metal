# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The resets, tt-smi -glx_reset only: phase 2 (the pre-test reset loop), phase 4 (the post-test reset and the end
sequence) and the recovery between bus-bar steps."""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from contextlib import suppress
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from .aer import aer_check, aer_new, read_aer
from . import events
from .busbar import Step, run_step
from .common import emit_result, emit_running, log, print_phase_summary, root_cmd, run, write_command_log
from .galaxy import EXPECTED_CHIP_COUNT, KNOWN_DROPPED, TRAY_TO_UBB_NIBBLE, expected_bdf, parse_bdf_physical
from .model import FAIL, PASS, SKIP, WARN, Check, Phase
from .naming import blamed
from .snapshot import resolve_tt_smi, take_snapshot
from .snapshot_checks import pcie_enum_check, validate_snapshot

# The tt-smi flags a tier's reset plan may name.
RESET_FLAGS = ("-glx_reset",)
SYS = Path("/sys")
ALL_BDFS = tuple(expected_bdf(t, a) for t in TRAY_TO_UBB_NIBBLE for a in range(1, 9))
RESCAN_WAIT_S, RESCAN_EVERY_S = 60, 5  # a missing chip's root port is rescanned every 5 s until it is back, up to 60 s
END_DELAY_S, END_DBI_S = 60, 10  # end sequence: the wait before the fabric open; its DRAM byte/DBI readback, s


def add_args(ap) -> None:
    ap.add_argument("--reset-gate", metavar="CMD", help="run before each reset; a non-zero exit skips that reset")
    ap.add_argument(
        "--end-delay-s", type=int, default=END_DELAY_S, help="s from the post-test reset to the end fabric open"
    )
    ap.add_argument("--known-dropped-tray", type=int, choices=range(1, 5), help="a tray known to be down: not counted")
    ap.add_argument("--noc-wall-s", type=float, default=0, help="the end open's NOC data check, wall cap s; 0 off")
    ap.add_argument("--pcie-mb", type=int, default=0, help="the end PCIe step's MB per chip each way; 0 off")
    ap.add_argument("--pcie-s", type=int, default=240, help="the end PCIe step's cap, s")
    ap.add_argument("--pcie-floor-gbs", type=float, default=0.25, help="the end PCIe step: FAIL under this GB/s")
    ap.add_argument(
        "--pcie-warn-frac", type=float, default=0.5, help="the end PCIe step: WARN under this of the median"
    )


def pcie_bdfs() -> list[str] | None:
    """BDFs of the Tenstorrent chips enumerated on PCIe, or None if they could not be listed."""
    try:
        lspci = subprocess.run(["lspci", "-D", "-d", "1e52:"], capture_output=True, text=True, check=True)
        return [line.split()[0] for line in lspci.stdout.splitlines() if line.strip()]
    except Exception:
        return None


def recount_check(name: str, bdfs: list[str] | None) -> Check:
    """The chip recount after a reset: how many of the 32 are back, each missing one by BDF, tray and ASIC, and how
    many each tray is short."""
    if bdfs is None:
        return Check(name=name, status=FAIL, details="chips could not be listed (lspci failed)", ip="pcie")
    c = pcie_enum_check([{"board_info": {"bus_id": b}} for b in bdfs], name)
    short = Counter(m["tray"] for m in c.data["missing"])
    c.data["trays_short"] = dict(short)
    if short:
        c.details += "; " + ", ".join(f"tray {t}: {n} of 8 missing" for t, n in sorted(short.items()))
    return c


def root_port(bdf: str) -> str | None:
    """The bridge whose secondary bus is *bdf*'s bus, found also for a chip gone from PCIe."""
    bus = str(int(bdf.split(":")[-2], 16))
    for f in sorted((SYS / "bus/pci/devices").glob("*/secondary_bus_number")):
        with suppress(OSError):
            if f.read_text().strip() == bus:
                return f.parent.name
    return None


def rescan(missing: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """Rescan the root port of each *missing* chip until it is back on PCIe, up to RESCAN_WAIT_S: the chips back with
    their root port, and why a root port could not be rescanned."""
    ports = {b: root_port(b) for b in missing}
    whys = [f"{b}: root port not found" for b, rp in ports.items() if rp is None]
    rps = sorted(set(ports.values()) - {None})
    cmds = {rp: root_cmd(["tee", str(SYS / "bus/pci/devices" / rp / "rescan")]) for rp in rps}
    whys += [why for _, why in cmds.values() if why]
    left, back, t0 = {b: rp for b, rp in ports.items() if rp and cmds[rp][0]}, {}, time.time()
    while left and time.time() - t0 < RESCAN_WAIT_S:
        for rp in sorted(set(left.values())):
            subprocess.run(cmds[rp][0], input="1", capture_output=True, text=True, timeout=15)
        time.sleep(RESCAN_EVERY_S)
        now = set(pcie_bdfs() or ())
        back.update({b: left.pop(b) for b in list(left) if b in now})
    return back, whys


OPEN = """import sys, tt_umd
for n in sys.argv[1:]:
    try:
        tt_umd.PCIDevice(int(n))
    except Exception as e:
        print(n, " ".join(str(e).split())[:160])
print("done")"""


def unready(bdfs: Sequence[str]) -> dict[str, str]:
    """Why each chip is not back after a reset: missing from PCIe (a known-dropped tray's are not), on PCIe with its
    firmware not ready (tt_fw_bundle_ver 0.0.0.0), or not opening in tt-umd (not checked when tt-umd cannot run)."""
    out = {b: "missing from PCIe" for b in ALL_BDFS if b not in bdfs and parse_bdf_physical(b)[0] not in KNOWN_DROPPED}
    nodes = {}
    for b in bdfs:
        for node in (SYS / "bus/pci/devices" / b / "tenstorrent").glob("tenstorrent!*"):
            nodes[node.name.rsplit("!", 1)[1]] = b
            with suppress(OSError):
                if (node / "tt_fw_bundle_ver").read_text().strip() == "0.0.0.0":
                    out[b] = "firmware not ready (tt_fw_bundle_ver 0.0.0.0)"
    with suppress(OSError, subprocess.TimeoutExpired):
        cp = subprocess.run([sys.executable, "-c", OPEN, *nodes], capture_output=True, text=True, timeout=120)
        for n, _, why in (ln.partition(" ") for ln in cp.stdout.splitlines() if ln != "done"):
            out.setdefault(nodes.get(n, n), f"on PCIe, tt-umd cannot open it: {why}")
    return out


def _gate(cmd: str | None) -> str:  # why the reset gate refuses a reset now, or ""
    if not cmd:
        return ""
    try:
        cp = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"reset gate did not run: {e}"
    last = (cp.stdout + cp.stderr).strip().splitlines()
    return "" if cp.returncode == 0 else f"reset gate: {last[-1] if last else f'rc {cp.returncode}'}"


def glx_reset(
    tt_smi, phase: Phase, name: str, recount: str, dry_run: bool, logs_dir=None, gate=None
) -> list[str] | None:
    """One tt-smi -glx_reset as check *name*, the rescan of the chips it left missing (check rescan_<name>) and the
    recount (check *recount*): the chips on PCIe after it, or None when it could not run (a SKIP that says why)."""
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
    out, err = (cp.stdout or "")[-2000:], (cp.stderr or "")[-2000:]
    data = {"rc": cp.returncode, "duration_s": dt, "stdout_tail": out, "stderr_tail": err}
    details = f"tt-smi -glx_reset rc={cp.returncode} dur={dt:.1f}s"
    phase.add(check := Check(name=name, status=status, details=details, data=data, ip="other"))
    if dry_run:
        return list(ALL_BDFS)
    bdfs = pcie_bdfs()
    named = set(re.findall(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", out + err))
    kd = named.isdisjoint(bdfs or ()) and all((parse_bdf_physical(b) or (0,))[0] in KNOWN_DROPPED for b in named)
    if status == FAIL and named:  # the chips it names; only the known-dropped tray's, still off PCIe: that known drop
        check.data["blamed_chips"], check.status = blamed(sorted(named)), WARN if kd else FAIL
    if KNOWN_DROPPED and bdfs and all(b in bdfs for b in ALL_BDFS if parse_bdf_physical(b)[0] in KNOWN_DROPPED):
        log(f"{name}: known-dropped tray {', '.join(map(str, sorted(KNOWN_DROPPED)))} answers: the full run")
        KNOWN_DROPPED.clear()
    missing = [
        b for b in ALL_BDFS if bdfs is not None and b not in bdfs and parse_bdf_physical(b)[0] not in KNOWN_DROPPED
    ]
    if missing:
        back, whys = rescan(missing)
        bdfs = pcie_bdfs() if back else bdfs
        if back or whys:
            said = "; ".join([*(f"{b} back only after a rescan of root port {rp}" for b, rp in back.items()), *whys])
            data = {"back": back, "not_rescanned": whys}
            phase.add(Check(name=f"rescan_{name}", status=WARN if back else SKIP, details=said, data=data, ip="pcie"))
    bad = {} if bdfs is None else unready(bdfs)
    if bad and not name.endswith("_round2"):  # a second re-power round for chips missing, not ready or not opening
        log(f"{name}: {len(bad)} chip(s) not back: a second tt-smi -glx_reset")
        said = "; ".join(f"{b} {why}" for b, why in bad.items())  # a FAIL even when the second round brings them back
        ready = Check(name=f"ready_{name}", status=FAIL, details=f"not back: {said}", data={"unready": bad}, ip="pcie")
        phase.add(ready)
        return glx_reset(tt_smi, phase, f"{name}_round2", recount, dry_run, logs_dir, gate)
    if bad.keys() - {b for b, why in bad.items() if why == "missing from PCIe"}:
        said = "; ".join(f"{b} {why}" for b, why in bad.items() if why != "missing from PCIe")
        phase.add(Check(name=f"ready_{recount}", status=FAIL, details=said, data={"unready": bad}, ip="pcie"))
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
    """Recovery after a bus-bar step that did not pass: a glx_reset, rescan and recount; its finding lines, none when
    every chip is back."""
    if args.skip_reset:
        return ["reset cannot run: --skip-reset"]
    phase = Phase(name="recover")
    try:
        tt_smi = resolve_tt_smi(args.tt_smi_path)
    except SystemExit as e:  # no tt-smi
        return [f"reset cannot run: {e}"]
    glx_reset(tt_smi, phase, "reset_recover", "recount_recover", args.dry_run, outdir, args.reset_gate)
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
    the checks it adds are judged, not ones *phase* already held.

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
    """After a post-test reset that left every chip back: *delay_s* of quiet, one fabric open and close with the DRAM
    byte/DBI readback, then (--pcie-mb) each chip's PCIe bandwidth (galaxy/pcie_bw.py; its PCIE WARN lines a WARN),
    judged with the AER errors over them; a FAIL gets one more glx_reset so no chip is left hung."""
    reset = next((c for c in reversed(phase.checks) if c.name == "post_test_reset_ok"), None)
    if reset is not None and reset.status != PASS:
        phase.add(Check(name="end_fabric", status=SKIP, details=f"not run: {reset.details}", ip="eth"))
        return
    log(f"end sequence: {delay_s} s, then the fabric open and close")
    if not dry_run:
        time.sleep(delay_s)
    before = {} if dry_run else read_aer()
    t0 = time.time()
    noc = getattr(events.args, "noc_wall_s", 0)  # the NOC data check after the DRAM one: noc.txt beside rows.jsonl
    argv = ("fabric_open.py", "--rows", "{out}/rows.jsonl", "--beat", "{out}/beat", "--label", "open", "--dbi-wall-s")
    argv += (str(END_DBI_S), *(("--noc-wall-s", f"{noc:g}") if noc else ()))
    budget = 60 + END_DBI_S + (int(noc) + 5 if noc else 0)
    step = Step("fabric", "fabric open and close after the reset", argv, budget)
    opened = run_step(step, out, dry_run, test="end")
    phase.add(opened)
    a, pcie = events.args, None
    if (
        getattr(a, "pcie_mb", 0) and opened.data.get("rc", 0) == 0
    ):  # after a clean close (a data FAIL too): a 1x1 mesh per chip
        argv = ("pcie_bw.py", "--rows", "{out}/rows.jsonl", "--beat", "{out}/beat", "--mb", str(a.pcie_mb))
        argv += ("--floor-gbs", f"{a.pcie_floor_gbs:g}", "--warn-frac", f"{a.pcie_warn_frac:g}")
        pcie = run_step(
            Step("pcie", "each chip's PCIe bandwidth after the close", argv, a.pcie_s), out, dry_run, test="end"
        )
        txt = out / "pcie" / "pcie.txt"  # the judge's lines; its FAIL lines already failed the step
        warns = [ln for ln in txt.read_text().splitlines() if ln.startswith("PCIE WARN ")] if txt.is_file() else []
        st = WARN if warns and pcie.status == PASS else pcie.status
        phase.add(replace(pcie, status=st, data={**pcie.data, "fails": [*pcie.data.get("fails", []), *warns]}))
    if dry_run:
        return
    aer = replace(aer_check(aer_new(before, read_aer()), (time.time() - t0) / 3600), name="end_aer")
    phase.add(aer)
    if FAIL in (opened.status, aer.status, pcie and pcie.status):
        glx_reset(resolve_tt_smi(tt_smi_path), phase, "reset_end", "recount_end", dry_run, out, gate)
