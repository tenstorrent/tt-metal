# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""didt_runner run loop: collect gate, then per step reset -> run -> tt-smi -> classify, fallbacks on FAIL/HANG."""

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

from .checks import (
    Result,
    Status,
    classify_pytest,
    collected_node_ids,
    gddr_edc,
    gddr_result,
    missing_node_ids,
    read_tail,
    sweep_skipped_windows,
    write_ack_chips,
)
from .chips import Chip, by_dev
from .config import Config, Fallback, StepCfg
from .devices import (
    core_sweep_devices,
    nd_report_devices,
    run_to_run_devices,
    triage_hang_devices,
    write_ack_devices,
)
from .status import Attempt, StepState, write

REPO = Path(__file__).resolve().parents[3]
TIMEOUT_RC = 124
# Same scripts as the gwdt hang triage: where each core is stuck (op mesh, running ops, callstacks), the ops around
# the hung one, NOC posted-write/ack counters that never drained, and firmware and link state.
HANG_TRIAGE = (
    "--run=dump_op_mesh --run=dump_running_operations --run=dump_op_window --op-window=20 "
    "--run=check_dram_health --run=dump_callstacks --run=check_noc_status --run=check_arc --run=check_eth_status"
)
# Triage on a wedged chip can hang too; the bound keeps the hook's `kill -9 $PPID` reachable so the step ends.
TRIAGE_SECS = 1200


def _kill_group(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait()


def sh(cmd: str, log: Path, env: dict, cwd: Path, timeout: float | None) -> int:
    """Run cmd with output to log. On the wall timeout or an interrupt the whole process group dies: the workload
    runs in its own session, so Ctrl-C reaches only the runner, and the mcast-ack script's per-chip children would
    otherwise keep the chips open into the next reset."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        p = subprocess.Popen(
            ["bash", "-c", cmd], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(p)
            f.write(f"\nTIMEOUT after {timeout:g} s\n")
            return TIMEOUT_RC
        except BaseException:
            _kill_group(p)
            raise


def host() -> str:
    return socket.gethostname().split(".")[0]


def repo_commit(repo: Path = REPO) -> str:
    out = subprocess.run(["git", "rev-parse", "--short=11", "HEAD"], cwd=repo, capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


# Never inherited from the caller's shell: device selection would renumber `dev` away from the whole-host chip
# snapshot every join relies on, and a leftover throttle or hang hook would change what the first attempt measures.
DROPPED_ENV = (
    "TT_VISIBLE_DEVICES",
    "TT_METAL_VISIBLE_DEVICES",
    "TT_MM_THROTTLE_PERF",
    "TT_METAL_OPERATION_TIMEOUT_SECONDS",
    "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE",
)


def base_env(repo: Path = REPO) -> dict[str, str]:
    venv = repo / "python_env"
    return {
        **{k: v for k, v in os.environ.items() if k not in DROPPED_ENV},
        "TT_METAL_HOME": str(repo),
        "PYTHONPATH": str(repo),
        "VIRTUAL_ENV": str(venv),
        "PATH": f"{venv / 'bin'}:{os.environ.get('PATH', '')}",
    }


def triage_env(cfg: Config, csv: Path, repo: Path = REPO) -> dict[str, str]:
    """On a dispatch timeout metal runs tt-triage while the workload is still up, then the hook kills the pytest."""
    return {
        "TT_METAL_OPERATION_TIMEOUT_SECONDS": str(cfg.hang_secs),
        "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE": f"timeout {TRIAGE_SECS} "
        f"{repo / 'python_env/bin/python'} {repo}/tools/tt-triage.py "
        f"--disable-progress {HANG_TRIAGE} --llm-output --llm-output-path={csv}; kill -9 $PPID",
    }


def _quote(args: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


def _device_line(d: dict, why: str | None = None) -> str:
    """why: the failure text for a device the check names without a hang or ND result (a crash, a build error)."""
    who = " ".join([f"device {d['dev']}"] + [f"{k}={d[k]}" for k in ("bdf", "tray", "asic") if k in d])
    if d.get("hung"):
        what = "hung" + (f" in {d['op']}" if "op" in d else "")
        what += f", {d['core_count']} cores" if "core_count" in d else ""
    elif "failed_checks" in d:
        what = f"{d['failed_checks']}/{d['checks']} checks, pattern {d['pattern']}"
    elif "elements" in d:
        what = f"{d['elements']} elements differ"
    elif "cores" in d:
        what = "cores " + " ".join(f"({c['x']},{c['y']})x{c.get('elements', '?')}" for c in d["cores"])
    else:
        what = why or "failed"
    return f"{who}: {what}"


class Runner:
    def __init__(self, cfg: Config, run_dir: Path, chips: list[Chip], co: bool, shell, snap=None):
        self.cfg, self.run_dir, self.chips, self.shell, self.snap = cfg, run_dir, by_dev(chips), shell, snap
        self.header = {
            "tool": "didt_runner",
            "host": host(),
            "run": run_dir.name,
            "mode": "co" if co else "real",
            "commit": repo_commit(),
            "run_dir": str(run_dir),
            "config": {"path": str(cfg.path), "sha256": cfg.sha256},
            "chips": [c.as_dict() for c in chips],
        }
        self.states: list[StepState] = []
        self.current: str | None = None

    def save(self) -> None:
        write(self.run_dir / "status.json", self.header, self.states, self.current)

    def note(self, line: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {line}"
        print(line, flush=True)
        with open(self.run_dir / "run.log", "a") as f:
            f.write(line + "\n")

    def collect(self, steps: list[StepCfg]) -> dict[str, list[str]]:
        """One pytest --collect-only for every id. pytest prints no listing at all once any id is missing, so then
        each step whose ids were not listed is collected alone, and only a genuinely stale step fails."""
        wanted = [i for s in steps for i in s.ids]
        found = set(self._collect(wanted, self.run_dir / "collect" / "collect.log"))
        missing = {}
        for s in steps:
            if not missing_node_ids(s.ids, list(found)):
                continue
            log = self.run_dir / "collect" / f"{s.key.replace(':', '_')}.log"
            if gone := missing_node_ids(s.ids, self._collect(s.ids, log)):
                missing[s.key] = gone
        return missing

    def _collect(self, ids: list[str], log: Path) -> list[str]:
        c = self.cfg.collect
        self.shell(f"{c['cmd']} {_quote(ids)}", log, base_env(), REPO, c["timeout"])
        return collected_node_ids(read_tail(log)) if log.is_file() else []

    def reset(self, d: Path) -> str | None:
        """None when a reset succeeded, else the log of the last try."""
        r = self.cfg.reset
        for n in range(1, r["tries"] + 1):
            log = d / ("reset.log" if n == 1 else f"reset.retry{n}.log")
            if self.shell(r["cmd"], log, base_env(), REPO, r["timeout"]) == 0:
                return None
        return str(log)

    def attempt(self, step: StepCfg, fb: Fallback | None, a: Attempt) -> None:
        d = self.run_dir / (step.key.replace(":", "/") + (f"_{fb.name}" if fb else ""))
        a.status, a.start = Status.RUN, time.time()
        self.save()
        self.note(f"{step.key} {fb.name if fb else 'first'} start -> {d}")
        try:
            self._attempt(step, fb, a, d)
        except Exception as e:
            a.status, a.reason, a.details, a.devices = Status.FAIL, f"{type(e).__name__}: {e}", [], []
        except BaseException:
            a.status, a.reason, a.end = Status.FAIL, "interrupted", time.time()
            self.note(f"{step.key} {fb.name if fb else 'first'} interrupted")
            raise
        a.end = time.time()
        self.note(f"{step.key} {fb.name if fb else 'first'} {a.status.value} {a.reason}")
        self.save()

    def _attempt(self, step: StepCfg, fb: Fallback | None, a: Attempt, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        if failed_log := self.reset(d):
            a.status, a.log = Status.FAIL, failed_log
            a.reason = f"reset failed {self.cfg.reset['tries']}x: {self.cfg.reset['cmd']}, see {d / 'reset.log'}"
            return
        chips, chip_warnings = self.chips_after_reset()
        log, csv, chips_dir = d / "log", d / "crash_triage.csv", d / "chips"
        a.log = str(log)
        extra_env, extra_opts = (fb.env, fb.opts) if fb else ({}, [])
        env = {**base_env(), **step.env, **extra_env}
        if step.runner == "pytest":
            cmd = "pytest " + _quote(step.ids + step.opts + extra_opts)
            env |= triage_env(self.cfg, csv)
        else:
            # The script deletes its work dir at start, so it gets a dedicated subfolder, never the attempt dir.
            cmd = step.cmd + (" " + _quote(extra_opts) if extra_opts else "")
            env["MCAST_ACK_WORK_DIR"] = str(chips_dir)
        rc = self.shell(cmd, log, env, REPO, step.timeout)
        for cache in chips_dir.glob("chip*/cache"):
            shutil.rmtree(cache, ignore_errors=True)
        snap = d / "tt-smi.json"
        self.shell(self.cfg.smi["cmd"], snap, base_env(), REPO, self.cfg.smi["timeout"])

        r, devices, warnings, whys = self.classify(step, rc, log, csv, chips_dir, chips)
        details = [_device_line(x, whys.get(x["dev"])) for x in devices] + warnings + (r.details if not devices else [])
        details += chip_warnings
        r = gddr_result(Result(r.status, r.reason, details), self.gddr(snap), {c.bdf: c.dev for c in chips.values()})
        a.status, a.reason, a.details, a.devices = r.status, r.reason, r.details, devices

    def chips_after_reset(self) -> tuple[dict[int, Chip], list[str]]:
        """A reset can leave a chip off the bus, shifting every later dev; this attempt joins on the chips it ran on."""
        now = self.snap() if self.snap else []
        if not now:
            return self.chips, []
        before, after = {c.bdf for c in self.chips.values()}, {c.bdf for c in now}
        if before == after:
            return by_dev(now), []
        change = f"chip set changed after reset: {len(now)} chips, {len(self.chips)} at run start"
        change += f"; gone {', '.join(sorted(before - after))}" if before - after else ""
        change += f"; new {', '.join(sorted(after - before))}" if after - before else ""
        return by_dev(now), [change]

    def classify(self, step: StepCfg, rc: int, log: Path, csv: Path, chips_dir: Path, chips: dict[int, Chip]):
        if csv.is_file() and csv.stat().st_size > 0:
            devices, warnings = triage_hang_devices(csv.read_text(errors="replace"), chips)
            return Result(Status.HANG, f"triage {csv}"), devices, warnings, {}
        if rc == TIMEOUT_RC:
            return Result(Status.FAIL, f"timeout after {step.timeout:g} s"), [], [], {}
        text = read_tail(log) if log.is_file() else ""
        if step.check == "write_ack":
            r, failed = write_ack_chips(text)
            logs = {
                c: read_tail(p)
                for c in failed
                if failed[c].startswith("ND") and (p := chips_dir / f"chip{c}" / "pytest.log").is_file()
            }
            devices, warnings = write_ack_devices(failed, logs, chips)
            return r, devices, warnings, failed
        r = classify_pytest(text, len(step.ids))
        if step.check == "core_sweep" and r.status == Status.PASS:
            skipped = sweep_skipped_windows(text)
            if skipped != 0:
                r = Result(Status.FAIL, f"skipped windows: {'no line' if skipped is None else skipped}")
        if r.status == Status.PASS:
            return r, [], [], {}
        if step.check == "nd_report":
            devices, warnings = nd_report_devices(text, chips)
            return r, devices, warnings, {}
        if step.check == "run_to_run":
            return r, run_to_run_devices(text, chips), [], {}
        return r, core_sweep_devices(text, chips), [], {}

    def gddr(self, snap: Path) -> list:
        try:
            return [(snap.name, e) for e in gddr_edc(json.loads(snap.read_text()))]
        except (OSError, ValueError):
            return []


def _pending() -> Attempt:
    return Attempt(None, {}, [], Status.PENDING, "", [], [], "", None, None)


def run(
    cfg: Config,
    run_dir: Path,
    chips: list[Chip],
    only: set[str] | None,
    skip: set[str],
    co: bool,
    shell=sh,
    snap=None,
) -> int:
    """Every selected step in config order; a FAIL or HANG never stops the next step. Exit code: 1 when any step's
    first attempt FAILed or HUNG, else 0."""
    r = Runner(cfg, run_dir, chips, co, shell, snap)
    selected = [s for s in cfg.steps if (only is None or s.key in only) and s.key not in skip]
    states = {s.key: StepState(s.key, _pending(), []) for s in cfg.steps}
    for s in cfg.steps:
        if s not in selected:
            states[s.key].first.status, states[s.key].first.reason = Status.SKIP, "by user"
    r.states = list(states.values())
    r.save()

    start = time.time()
    missing = r.collect(selected)
    for s in selected:
        a = states[s.key].first
        a.start, a.end = start, time.time()
        if s.key in missing:
            a.status, a.reason = Status.FAIL, "does not collect: " + ", ".join(missing[s.key])
            a.log = str(run_dir / "collect")
        elif co:
            a.status, a.reason = Status.PASS, "collects"
    r.save()

    try:
        for s in [] if co else selected:
            if s.key in missing:
                continue
            r.current = s.key
            state = states[s.key]
            r.attempt(s, None, state.first)
            if state.first.status in (Status.FAIL, Status.HANG):
                for fb in s.fallbacks:
                    f = Attempt(fb.name, dict(fb.env), list(fb.opts), Status.PENDING, "", [], [], "", None, None)
                    state.fallbacks.append(f)
                    r.attempt(s, fb, f)
    finally:
        r.current = None
        r.save()
    bad = [s.key for s in selected if states[s.key].first.status in (Status.FAIL, Status.HANG)]
    r.note(
        f"done: {len(selected) - len(bad)}/{len(selected)} steps passed" + (f"; failed {', '.join(bad)}" if bad else "")
    )
    return 1 if bad else 0
