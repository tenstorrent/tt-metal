# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The bus-bar stage: the tier's BUSBAR_STEPS (tiers.py), each a clock-checked
quiet then one of metal's loads, with the trays' HSCs polled throughout.

A weak bus bar or 12 V feed shows as an HSC trip, a chip gone from PCIe, a link
that trains down, or a chip whose matmul output differs from the other 31 (the
diag.xchip pytest plugin). After every step the chips are recounted and the links
and SEL re-read; a lost chip ends the loads, since each needs all 32.

A load that outlives its step plus bring-up is a hang: the logs are captured,
then its process group is stopped, which ends all access to the hung chip.
Each leg writes its name and boot_id to LEG_FILE, which a reboot keeps, so the
next run can say the host rebooted during that leg. The power_virus step is the
--hook script's ``load power_virus <seconds>`` call; with no hook it is a SKIP.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import sensors
from .captures import SEL_POWER_RE, _read, capture_events
from .common import SUITE_DIR, call_hook, kill_process_group, log
from .device_tests import PYTESTS, resolve_pytest
from .eth_status import sysfs_chip_bdfs
from .galaxy import format_bdf
from .model import FAIL, PASS, SKIP, WARN, Check, Phase
from .resets import recount_chips

# Checks this stage emits; a step's own check is BUSBAR + its step name.
BUSBAR = "busbar_"
BUSBAR_BRICK_VOUT = "busbar_brick_vout"
BUSBAR_CHIP_LOST = "busbar_chip_lost"
BUSBAR_HANG = "busbar_hang"
BUSBAR_HSC_POLL = "busbar_hsc_poll"
BUSBAR_HSC_TRIP = "busbar_hsc_trip"
BUSBAR_NUMERIC = "busbar_numeric"
BUSBAR_PCIE_LINK = "busbar_pcie_link"
BUSBAR_QUIET_CLOCK = "busbar_quiet_clock"
BUSBAR_REBOOT = "busbar_reboot"

QUIET_MHZ = 800  # a quiet counts only once every chip's AICLK is at or under this
QUIET_CAP_S = 300
BRINGUP_S = 600  # a load's bring-up and compile, on top of its step length
HANG_GRACE_S = 60  # past the load's own --timeout: only a chip that never answers is a hang
VOUT_LOW_V = 12.1  # a brick still at its 12.00 V default
LEG_FILE = Path(f"/var/tmp/health-check-{os.getuid()}/busbar_leg.json")
STOP_GRACE_S = 5  # TERM then KILL, inside the wrapper's 10 s TERM-to-KILL
PCC_RE = re.compile(r"Device (\d+) PCC: (?!1\.0\b)(\S+)")
XCHIP_RE = re.compile(r"Device (\d+) output differs from the other chips")
SYNC_RE = re.compile(r"(Start|End) sync device id: (\d+)")
SYS = Path("/sys")

_ALLGATHER = "tests/ttnn/unit_tests/operations/ccl/blackhole_CI/galaxy/galaxy_nightly/test_all_gather_minimal_matmul_fabric2d_bh.py"


def _didt(tt_metal: Path, s: int) -> tuple[list[str], dict]:
    # About 14 iterations a second; one determinism check at each end keeps the load a burst.
    n = max(51, s * 14)
    args = [PYTESTS["didt_matmul_galaxy"]["target"], "-k", "galaxy and bf16_HiFi2", "--didt-workload-iterations"]
    args += [str(n), "--determinism-check-interval", str(n - 1), "-s", "--timeout", str(s + BRINGUP_S)]
    path = os.pathsep.join(filter(None, [str(SUITE_DIR), os.environ.get("PYTHONPATH")]))
    return [resolve_pytest(tt_metal) or "pytest", *args, "-p", "diag.xchip"], {"PYTHONPATH": path}


def _max_util(tt_metal: Path, s: int) -> tuple[list[str], dict]:
    # 92 % FPU and DRAM, no duty cycle, about 145 iterations a second.
    env = {"FPU_UTILIZATION_PCT": 92, "DRAM_UTILIZATION_PCT": 92, "DUTY_CYCLE_PCT": 100, "SUPER_SYNC": 0}
    env |= {"NUM_WL_LOOPS": 5000, "NUM_ITERATIONS": s * 145}
    binary = tt_metal / "build_Release" / "test" / "didt" / "unit_tests_didt_max_util"
    return [str(binary), "--gtest_filter=*MaxUtilWorkload_AllDevices"], {
        f"MAX_UTIL_{k}": str(v) for k, v in env.items()
    }


def _allgather(tt_metal: Path, s: int) -> tuple[list[str], dict]:
    k = "SP8xTP4-mesh and linear_tp and kda_input_projection"
    return [resolve_pytest(tt_metal) or "pytest", _ALLGATHER, "-k", k, "-s", "--timeout", str(s + BRINGUP_S)], {}


def _pcie_bw(tt_metal: Path, s: int) -> tuple[list[str], dict]:
    binary = tt_metal / "build_Release" / "test" / "tt_metal" / "unit_tests_data_movement"
    return [str(binary), "--gtest_filter=*PCIeReadBandwidth:*PCIeWriteBandwidthSweep"], {}


# Each load: (tt_metal, seconds) -> (command, extra env). "power_virus" is the hook's.
LOADS: dict[str, Callable[[Path, int], tuple[list[str], dict]] | None] = {
    "didt": _didt,
    "max_util": _max_util,
    "allgather": _allgather,
    "pcie_bw": _pcie_bw,
    "power_virus": None,
}
IP_BY_LOAD = {"pcie_bw": "pcie", "allgather": "eth"}


def _add(phase: Phase, name: str, status: str, details: str, **kw) -> None:
    phase.add(Check(name=name, status=status, details=details, **kw))


def _read_by_chip(sys_root: Path, read: Callable[[Path], str]) -> dict[str, str]:
    """read(the chip's PCI device dir) by formatted BDF; chips whose read fails are left out."""
    out = {}
    for n in sorted((sys_root / "class" / "tenstorrent").glob("tenstorrent!*")):
        try:
            out[format_bdf((n / "device").resolve().name)] = read(n)
        except (OSError, ValueError, IndexError):
            pass
    return out


def chip_clocks(sys_root: Path = SYS) -> dict[str, int]:
    """Each chip's AICLK in MHz."""
    return _read_by_chip(sys_root, lambda n: int((n / "tt_aiclk").read_text().split()[0]))


def links(sys_root: Path = SYS) -> dict[str, str]:
    """Each chip's PCIe link as 'speed xwidth'."""
    rd = lambda d, f: (d / f).read_text().strip()
    return _read_by_chip(
        sys_root, lambda n: f"{rd(n / 'device', 'current_link_speed')} x{rd(n / 'device', 'current_link_width')}"
    )


def boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def wait_quiet(stop: threading.Event, sys_root: Path = SYS, cap_s: float | None = None) -> list[str]:
    """Wait until every chip's AICLK is at or under QUIET_MHZ; the chips still over it at the cap."""
    deadline = time.monotonic() + (QUIET_CAP_S if cap_s is None else cap_s)
    while True:
        hot = sorted(b for b, mhz in chip_clocks(sys_root).items() if mhz > QUIET_MHZ)
        if not hot or time.monotonic() >= deadline or stop.is_set():
            return hot
        stop.wait(2)


def numeric_faults(text: str, bdfs: dict[int, str]) -> dict[str, str]:
    """Chips a didt log names as wrong (a determinism mismatch or a cross-chip outlier): BDF -> first line."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = PCC_RE.search(line) or XCHIP_RE.search(line)
        if m:
            dev = int(m.group(1))
            out.setdefault(format_bdf(bdfs.get(dev, f"device {dev}")), line.strip()[-160:])
    return out


def hung_chip(text: str, bdfs: dict[int, str]) -> str:
    """The chip whose device sync started last and never ended, else ""."""
    open_ids: list[int] = []
    for kind, dev in SYNC_RE.findall(text):
        if kind == "Start":
            open_ids.append(int(dev))
        elif int(dev) in open_ids:
            open_ids.remove(int(dev))
    return format_bdf(bdfs.get(open_ids[-1], f"device {open_ids[-1]}")) if open_ids else ""


def sel_power_lines(dry_run: bool) -> list[str] | None:
    out, _ = _read(["ipmitool", "sel", "elist"], dry_run)
    return None if out is None else [ln.strip() for ln in out.splitlines() if SEL_POWER_RE.search(ln)]


def check_last_boot(phase: Phase) -> None:
    """A leg file left by another boot: the host rebooted during that leg."""
    try:
        leg = json.loads(LEG_FILE.read_text())
    except (OSError, ValueError):
        return
    if leg.get("boot_id") and leg["boot_id"] != boot_id():
        why = f"host rebooted during {leg.get('leg', '?')} -> POWER: PSU/feed, see SEL"
        _add(phase, BUSBAR_REBOOT, FAIL, why, data=leg, ip="other", cause="POWER")
    LEG_FILE.unlink(missing_ok=True)


def run_load(
    cmd: list[str], env: dict, timeout_s: float, log_path: Path, stop: threading.Event, cwd: Path
) -> tuple[str, subprocess.Popen | None]:
    """Run one load from cwd to its log: "ok", "fail", "stopped" (SIGTERM), or "hang" (left running for the capture)."""
    with open(log_path, "w") as f:
        try:
            p = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **env}, cwd=str(cwd), start_new_session=True
            )
        except OSError as e:  # e.g. the load's binary is not built
            f.write(f"{e}\n")
            return "fail", None
    deadline = time.monotonic() + timeout_s
    while p.poll() is None:
        if stop.is_set():
            kill_process_group(p, STOP_GRACE_S)
            return "stopped", p
        if time.monotonic() > deadline:
            return "hang", p
        stop.wait(1)
    return ("ok" if p.returncode == 0 else "fail"), p


def _step_line(rec: sensors.PollStep) -> dict:
    """The step's load line: per HSC, its peak current and lowest VIN."""
    line = {}
    for (t, a), amps in sorted(rec.peak_a.items()):
        vin = rec.vin_min.get((t, a))
        line[f"tray{t}/{a:#04x}"] = f"{amps:.0f} A" + ("" if vin is None else f", VIN min {vin:.2f} V")
    return line


def run_busbar(
    tt_metal: Path,
    steps: tuple,
    phase: Phase,
    bmc: sensors.Bmc,
    hook: str | None,
    dry_run: bool,
    logs_dir: Path,
    sys_root: Path = SYS,
    period: float = sensors.POLL_S,
) -> None:
    """Run the bus-bar steps into ``phase``. The bricks are read once at idle; the poll reads only HSCs."""
    if dry_run:
        for st in steps:
            _add(phase, BUSBAR + st.name, SKIP, f"(dry) {st.load} {st.load_s} s")
        return
    check_last_boot(phase)
    logs_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    old = None
    if threading.current_thread() is threading.main_thread():
        # Set from a thread: Event.set inside the handler could deadlock on stop.wait's lock.
        old = signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=stop.set).start())
    poll, poll_stop = sensors.Poll(), threading.Event()
    poller = threading.Thread(target=sensors.poll_hsc, args=(bmc, poll, poll_stop, period), daemon=True)
    try:
        bricks = sensors.read_bricks(bmc) if not bmc.why else {}
        low = [f"{n} {v:.2f} V" for n, (v, _i) in sorted(bricks.items()) if v is not None and v < VOUT_LOW_V]
        if low:
            why = f"bricks {', '.join(low)} at idle -> POWER: set brick VOUT to 12.24/12.35 V"
            _add(phase, BUSBAR_BRICK_VOUT, WARN, why, ip="other")
        poller.start()
        _run_steps(tt_metal, steps, phase, poll, hook, stop, logs_dir, sys_root)
    finally:
        poll_stop.set()
        if poller.is_alive():
            poller.join(30)
        if old is not None:
            signal.signal(signal.SIGTERM, old)
        LEG_FILE.unlink(missing_ok=True)
    _judge_poll(phase, poll, bmc)
    if stop.is_set() and old is not None:
        os.kill(os.getpid(), signal.SIGTERM)  # the run was asked to end: pass it on, no later stage starts


def _run_steps(tt_metal, steps, phase, poll, hook, stop, logs_dir, sys_root) -> None:
    bdfs = sysfs_chip_bdfs(sys_root)
    _, missing0 = recount_chips(sys_root)
    links0, boot0 = links(sys_root), boot_id()
    sel0 = sel_power_lines(False)
    LEG_FILE.parent.mkdir(parents=True, exist_ok=True)
    for st in steps:
        poll.step = f"{st.name} quiet"
        hot = wait_quiet(stop, sys_root)
        if hot:
            why = f"AICLK still over {QUIET_MHZ} MHz after {QUIET_CAP_S} s before {st.name}: {', '.join(hot)}"
            _add(phase, BUSBAR_QUIET_CLOCK, WARN, why, parts=hot, ip="asic")
        stop.wait(st.quiet_s)
        if stop.is_set():
            break
        poll.step = st.name
        LEG_FILE.write_text(json.dumps({"leg": st.name, "boot_id": boot0, "t": time.time()}))
        t0 = time.time()
        log_path = logs_dir / f"busbar_{st.name}.log"
        if st.load == "power_virus":
            if not hook:
                _add(phase, BUSBAR + st.name, SKIP, "no --hook: the power virus slot is empty")
                continue
            # The hook runs to its end (call_hook's own timeout); a SIGTERM stops the steps after it.
            status = "ok" if call_hook(hook, "load", st.name, str(st.load_s), logs_dir, False) == 0 else "fail"
        else:
            cmd, env = LOADS[st.load](tt_metal, st.load_s)
            log(f"busbar {st.name}: {st.load} {st.load_s} s")
            status, proc = run_load(cmd, env, st.load_s + BRINGUP_S + HANG_GRACE_S, log_path, stop, tt_metal)
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        data = {
            "load": st.load,
            "seconds": round(time.time() - t0),
            **_step_line(poll.steps.get(st.name, sensors.PollStep())),
        }
        if status == "hang":
            capture_events(phase, False, logs_dir, datetime.fromtimestamp(t0, timezone.utc))
            chip = hung_chip(text, bdfs)
            kill_process_group(proc, STOP_GRACE_S)
            why = f"{chip or 'a chip'} hung in {st.name}, access stopped -> CHIP/BOARD (log {log_path})"
            _add(phase, BUSBAR_HANG, FAIL, why, data=data, parts=[chip] if chip else [], ip="asic")
            break
        if status == "stopped":
            _add(phase, BUSBAR + st.name, WARN, f"stopped by SIGTERM at {st.name}")
            break
        bad = numeric_faults(text, bdfs)
        if bad:
            why = f"{', '.join(bad)} wrong output in {st.name}: {next(iter(bad.values()))}"
            _add(phase, BUSBAR_NUMERIC, FAIL, why, data=data, parts=sorted(bad), ip="asic")
        else:
            why = f"{st.load} {data['seconds']} s" + ("" if status == "ok" else f", failed (log {log_path})")
            _add(
                phase,
                BUSBAR + st.name,
                PASS if status == "ok" else FAIL,
                why,
                data=data,
                ip=IP_BY_LOAD.get(st.load, "other"),
            )
        lost, sel0 = _after_step(phase, st.name, missing0, links0, sel0, sys_root)
        if lost:
            break


def _after_step(phase, step, missing0, links0, sel0, sys_root) -> tuple[bool, list | None]:
    """Recount, re-read the links and the SEL: (a chip is gone, so the loads end; the next SEL baseline).
    A changed link becomes the next baseline."""
    lost = [b for b in recount_chips(sys_root)[1] if b not in missing0]
    if lost:
        why = f"{len(lost)} chip(s) gone from PCIe after {step}: {', '.join(lost)}; the device loads end here"
        _add(phase, BUSBAR_CHIP_LOST, FAIL, why, parts=lost, ip="pcie")
    now = links(sys_root)
    down = sorted(b for b, l in links0.items() if b in now and now[b] != l)
    if down:
        why = f"link changed after {step}: " + ", ".join(f"{b} {links0[b]} -> {now[b]}" for b in down)
        _add(phase, BUSBAR_PCIE_LINK, WARN, why, parts=down, ip="pcie")
        links0.update(now)
    sel = sel_power_lines(False)
    if sel0 is not None and sel is not None and len(sel) > len(sel0):
        why = f"host power event during {step} -> POWER: PSU/feed, see SEL: {sel[-1][:160]}"
        _add(phase, BUSBAR_REBOOT, FAIL, why, ip="other", cause="POWER")
    return bool(lost), (sel0 if sel is None else sel)


def _judge_poll(phase: Phase, poll: sensors.Poll, bmc: sensors.Bmc) -> None:
    """An HSC trip in any step is a FAIL on its tray; poll gaps are a WARN; no reads is a SKIP."""
    if not any(rec.flags for rec in poll.steps.values()):
        _add(phase, BUSBAR_HSC_POLL, SKIP, f"no HSC reads: {bmc.why or 'none answered'}")
        return
    for step, rec in poll.steps.items():
        for (t, a), flags in sorted(rec.flags.items()):
            trips = sorted(flags & sensors.TRIPS)
            if trips:
                amps = rec.peak_a.get((t, a))
                at = "" if amps is None else f" at {amps:.0f} A"
                why = f"tray {t} 12 V feed (HSC {a:#04x}) tripped{at} in {step} ({', '.join(trips)}) -> POWER"
                _add(phase, BUSBAR_HSC_TRIP, FAIL, why, parts=sensors._tray_parts(t), ip="other", cause="POWER")
    if poll.gaps:
        _add(phase, BUSBAR_HSC_POLL, WARN, f"{len(poll.gaps)} poll gap(s); {poll.gaps[-1]}", data={"gaps": poll.gaps})
