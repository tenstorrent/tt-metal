# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""didt_runner run loop: collect gate, then per step reset -> run -> tt-smi -> classify, fallbacks on FAIL/HANG."""

import json
import math
import os
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .checks import (
    Result,
    Status,
    child_failure,
    classify_pytest,
    collected_node_ids,
    gddr_edc,
    gddr_result,
    loop_seconds,
    missing_node_ids,
    read_tail,
    sweep_skipped_windows,
)
from .chips import Chip, by_dev
from .config import Config, Fallback, StepCfg
from .devices import (
    core_sweep_devices,
    identity,
    nd_report_devices,
    run_to_run_devices,
    triage_hang_devices,
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
ITER_OPT = "--didt-workload-iterations"
ITER_ENV = ("TT_DET_SWEEP_ITERS", "TT_SWEEP_ITERS", "MATMUL_ITERS")
DEVICE_ID_ENV = "DIDT_DEVICE_ID"
THROTTLE_ENV = "TT_MM_THROTTLE_PERF"
MIN_LOOP_ENV = "DIDT_MIN_LOOP_SECONDS"
IDENTITY = ("dev", "bdf", "tray", "asic", "asic_id", "kmd")
ND_DATA = {"failed_checks", "elements", "cores"}
# Triage on a wedged chip can hang too; the bound keeps the hook's `kill -9 $PPID` reachable so the step ends.
TRIAGE_SECS = 1200
# each_chip: chip dev's process serves its inspector on this port + dev, so its triage reads that process only.
INSPECTOR_PORT_BASE = 50100


def _kill_group(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait()


def sh(cmd: str, log: Path, env: dict, cwd: Path, timeout: float | None) -> int:
    """Run cmd with output to log. On the wall timeout or an interrupt the whole process group dies: the workload
    runs in its own session, so Ctrl-C reaches only the runner, and a child it left would keep the chips open into
    the next reset."""
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


def sh_all(jobs: list[tuple[str, Path, dict]], cwd: Path, timeout: float | None) -> list[int]:
    """Run every (cmd, log, env) at once, each as sh runs one. The wall timeout is shared: whatever still runs when
    it expires is killed with its group and gets the timeout exit code; an interrupt kills them all."""
    procs = []
    try:
        for cmd, log, env in jobs:
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "w") as f:
                procs.append(
                    subprocess.Popen(
                        ["bash", "-c", cmd],
                        cwd=cwd,
                        env=env,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
        deadline = None if timeout is None else time.monotonic() + timeout
        rcs = []
        for p, (_, log, _) in zip(procs, jobs):
            try:
                rcs.append(p.wait(timeout=None if deadline is None else max(0.0, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                _kill_group(p)
                with open(log, "a") as f:
                    f.write(f"\nTIMEOUT after {timeout:g} s\n")
                rcs.append(TIMEOUT_RC)
        return rcs
    except BaseException:
        for p in procs:
            _kill_group(p)
        raise


def host() -> str:
    return socket.gethostname().split(".")[0]


def repo_commit(repo: Path = REPO) -> str:
    out = subprocess.run(["git", "rev-parse", "--short=11", "HEAD"], cwd=repo, capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


# Never inherited from the caller's shell: device selection would renumber `dev` away from the whole-host chip
# snapshot every join relies on, and a leftover throttle, hang hook or pin knob would change what the first attempt
# measures. pin_core sets its own knobs per rerun.
DROPPED_ENV = (
    "TT_VISIBLE_DEVICES",
    "TT_METAL_VISIBLE_DEVICES",
    "TT_MM_THROTTLE_PERF",
    "TT_METAL_OPERATION_TIMEOUT_SECONDS",
    "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE",
    DEVICE_ID_ENV,
    MIN_LOOP_ENV,
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


def triage_env(cfg: Config, csv: Path, repo: Path = REPO, inspector: tuple[int, Path] | None = None) -> dict[str, str]:
    """On a dispatch timeout metal runs tt-triage while the workload is still up, then the hook kills the pytest.
    inspector: (RPC port, log dir) of a process that does not use the default ones."""
    where = ""
    if inspector:
        port, logs = inspector
        where = f"--inspector-rpc-port={port} --inspector-log-path={logs} --inspector-disable-rank "
    return {
        "TT_METAL_OPERATION_TIMEOUT_SECONDS": str(cfg.hang_secs),
        "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE": f"timeout {TRIAGE_SECS} "
        f"{repo / 'python_env/bin/python'} {repo}/tools/tt-triage.py "
        f"--disable-progress {HANG_TRIAGE} {where}--llm-output --llm-output-path={csv}; kill -9 $PPID",
    }


def double_iterations(opts: list[str], env: dict[str, str]) -> tuple[list[str], dict[str, str], dict[str, int]]:
    """A pinned rerun soaks twice as long as the attempt it follows. Returns the new opts, the env overrides, and
    every knob that was doubled with its new value."""
    out, used, i = [], {}, 0
    while i < len(opts):
        o = opts[i]
        if o.startswith(ITER_OPT + "="):
            used[ITER_OPT] = 2 * int(o.split("=", 1)[1])
            out.append(f"{ITER_OPT}={used[ITER_OPT]}")
        elif o == ITER_OPT and i + 1 < len(opts):
            used[ITER_OPT] = 2 * int(opts[i + 1])
            out += [o, str(used[ITER_OPT])]
            i += 1
        else:
            out.append(o)
        i += 1
    env_out = {k: str(2 * int(env[k])) for k in ITER_ENV if k in env}
    used |= {k: int(v) for k, v in env_out.items()}
    return out, env_out, used


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


def _hang_persists(fb: Fallback) -> str:
    """A fallback run only after a hang that hangs too: its knobs do not cure it. Its triage stays in devices[]."""
    knob = f"throttle {fb.env[THROTTLE_ENV]}" if THROTTLE_ENV in fb.env else fb.name
    return f"hang persists at {knob}"


def _same_chip(d: dict, t: dict) -> bool:
    """By BDF when both name one: `dev` shifts when a reset leaves a chip off the bus."""
    return d["bdf"] == t["bdf"] if "bdf" in d and "bdf" in t else d["dev"] == t["dev"]


class Runner:
    def __init__(self, cfg: Config, run_dir: Path, chips: list[Chip], co: bool, shell, snap=None, shell_all=sh_all):
        self.cfg, self.run_dir, self.chips, self.shell, self.snap = cfg, run_dir, by_dev(chips), shell, snap
        self.shell_all = shell_all
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
        self.pin_missing: set[str] = set()
        self.aborted: str | None = None
        self.reset_failed = False

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
        wanted = [i for s in steps for i in s.ids] + [s.pin_id for s in steps if s.pin_id]
        found = set(self._collect(wanted, self.run_dir / "collect" / "collect.log"))
        missing = {}
        for s in steps:
            if not missing_node_ids(s.ids, list(found)):
                continue
            log = self.run_dir / "collect" / f"{s.key.replace(':', '_')}.log"
            if gone := missing_node_ids(s.ids, self._collect(s.ids, log)):
                missing[s.key] = gone
        # A stale pin id only disables that step's pin-core fallback; the step itself still runs.
        for s in steps:
            if s.pin_id and s.pin_id not in found:
                log = self.run_dir / "collect" / f"{s.key.replace(':', '_')}_pin.log"
                if missing_node_ids([s.pin_id], self._collect([s.pin_id], log)):
                    self.pin_missing.add(s.pin_id)
        return missing

    def _collect(self, ids: list[str], log: Path) -> list[str]:
        c = self.cfg.collect
        self.shell(f"{c['cmd']} {_quote(ids)}", log, base_env(), REPO, c["timeout"])
        return collected_node_ids(read_tail(log)) if log.is_file() else []

    def reset(self, d: Path, r: dict) -> str | None:
        """None when a reset succeeded, else the log of the last try."""
        for n in range(1, r.get("tries", 1) + 1):
            log = d / ("reset.log" if n == 1 else f"reset.retry{n}.log")
            if self.shell(r["cmd"], log, base_env(), REPO, r["timeout"]) == 0:
                return None
        return str(log)

    def attempt(
        self,
        step: StepCfg,
        a: Attempt,
        d: Path,
        label: str,
        ids: list[str] | None = None,
        env: dict | None = None,
        opts: list[str] | None = None,
        reset: dict | None = None,
    ) -> None:
        a.status, a.start = Status.RUN, time.time()
        self.reset_failed = False
        self.save()
        self.note(f"{step.key} {label} start -> {d}")
        try:
            self._attempt(step, a, d, ids or step.ids, env or {}, step.opts if opts is None else opts, reset)
        except Exception as e:
            a.status, a.reason, a.details, a.devices = Status.FAIL, f"{type(e).__name__}: {e}", [], []
        except BaseException:
            a.status, a.reason, a.end = Status.FAIL, "interrupted", time.time()
            self.note(f"{step.key} {label} interrupted")
            raise
        a.end = time.time()
        self.note(f"{step.key} {label} {a.status.value} {a.reason}")
        self.save()

    def _attempt(
        self,
        step: StepCfg,
        a: Attempt,
        d: Path,
        ids: list[str],
        extra_env: dict,
        opts: list[str],
        reset: dict | None,
    ) -> None:
        d.mkdir(parents=True, exist_ok=True)
        r = reset or self.cfg.reset
        if failed_log := self.reset(d, r):
            a.status, a.log, self.reset_failed = Status.FAIL, failed_log, True
            a.reason = f"reset failed {r.get('tries', 1)}x: {r['cmd']}, see {d / 'reset.log'}"
            return
        chips, chip_warnings = self.chips_after_reset()
        log, csv = d / "log", d / "crash_triage.csv"
        a.log = str(log)
        env = {**base_env(), **step.env, **extra_env}
        cmd = "pytest " + _quote(ids + opts)
        if step.runner == "pytest":
            rc = self.shell(cmd, log, env | triage_env(self.cfg, csv), REPO, step.timeout)
        else:
            rcs = self.each_chip(cmd, d, env, sorted(chips), step.timeout)
        snap = d / "tt-smi.json"
        self.shell(self.cfg.smi["cmd"], snap, base_env(), REPO, self.cfg.smi["timeout"])

        if step.runner == "pytest":
            r, devices, warnings, whys = self.classify(step, rc, log, csv, chips, len(ids))
        else:
            r, devices, warnings, whys = self.classify_chips(step, rcs, d, log, chips, len(ids))
        details = [_device_line(x, whys.get(x["dev"])) for x in devices] + warnings + (r.details if not devices else [])
        details += chip_warnings
        r = gddr_result(Result(r.status, r.reason, details), self.gddr(snap), {c.bdf: c.dev for c in chips.values()})
        a.status, a.reason, a.details, a.devices = r.status, r.reason, r.details, devices

    def local_root(self) -> Path:
        return Path(tempfile.gettempdir()) / f"didt_runner_{self.run_dir.name}"

    def each_chip(self, cmd: str, d: Path, env: dict, devs: list[int], timeout: float | None) -> dict[int, int]:
        """cmd once per chip, all at once, in chips/chip<dev>/. Each process gets its own inspector port and log dir,
        so a hung chip's triage reads its own process only, and its own kernel cache on local disk: 32 concurrent
        JIT builds into one NFS tree fail. Returns dev -> exit code."""
        jobs = []
        for dev in devs:
            out = d / "chips" / f"chip{dev}"
            port = INSPECTOR_PORT_BASE + dev
            chip_env = env | {
                # The BDF rank: UMD maps an integer through its BDF-sorted device list, and the test parses it as one.
                "TT_VISIBLE_DEVICES": str(dev),
                "TT_METAL_CACHE": str(self.local_root() / d.relative_to(self.run_dir) / f"chip{dev}"),
                "OUT_DIR": str(out),
                "TT_METAL_LOGS_PATH": str(out),
                "TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS": f"localhost:{port}",
            }
            chip_env |= triage_env(
                self.cfg, out / "crash_triage.csv", inspector=(port, out / "generated" / "inspector")
            )
            jobs.append((cmd, out / "pytest.log", chip_env))
        try:
            rcs = self.shell_all(jobs, REPO, timeout)
        finally:
            shutil.rmtree(self.local_root(), ignore_errors=True)
        return dict(zip(devs, rcs))

    def classify_chips(
        self, step: StepCfg, rcs: dict[int, int], d: Path, log: Path, chips: dict[int, Chip], n_ids: int
    ):
        """Each chip on its own: a non-empty triage CSV is a hang, else the child's pytest result and ND lines. The
        step is HANG when any chip hung, else FAIL when any failed. log gets one line per chip."""
        devices, warnings, whys, lines = [], [], {}, []
        hung, failed = [], []
        for dev, rc in sorted(rcs.items()):
            out = d / "chips" / f"chip{dev}"
            csv, chip_log = out / "crash_triage.csv", out / "pytest.log"
            if csv.is_file() and csv.stat().st_size > 0:
                found, warn = triage_hang_devices(csv.read_text(errors="replace"), chips, dev_override=dev)
                devices += found or [{**identity(dev, chips), "hung": True}]
                warnings += warn
                hung.append(dev)
                lines.append(f"chip {dev} HANG triage {csv}")
                continue
            text = read_tail(chip_log) if chip_log.is_file() else ""
            timed_out = rc == TIMEOUT_RC
            if not timed_out and classify_pytest(text, n_ids).status == Status.PASS:
                lines.append(f"chip {dev} PASS")
                continue
            failed.append(dev)
            nd, warn = ([], []) if timed_out else nd_report_devices(text, chips, dev_override=dev)
            warnings += warn
            if nd:
                devices += nd
                lines.append(f"chip {dev} ND {chip_log}")
            else:
                whys[dev] = f"timeout after {step.timeout:g} s" if timed_out else child_failure(text, rc)
                devices.append(identity(dev, chips))
                lines.append(f"chip {dev} FAIL {whys[dev]} {chip_log}")
        log.write_text("\n".join(lines) + "\n")
        n = len(rcs)
        if hung:
            r = Result(Status.HANG, f"hung on {len(hung)}/{n} chips" + (f", failed on {len(failed)}" if failed else ""))
        elif failed:
            r = Result(Status.FAIL, f"failed on {len(failed)}/{n} chips")
        else:
            r = Result(Status.PASS, f"{n}/{n} chips passed")
        return r, devices, warnings, whys

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

    def classify(self, step: StepCfg, rc: int, log: Path, csv: Path, chips: dict[int, Chip], n_ids: int):
        if csv.is_file() and csv.stat().st_size > 0:
            devices, warnings = triage_hang_devices(csv.read_text(errors="replace"), chips)
            return Result(Status.HANG, f"triage {csv}"), devices, warnings, {}
        if rc == TIMEOUT_RC:
            return Result(Status.FAIL, f"timeout after {step.timeout:g} s"), [], [], {}
        text = read_tail(log) if log.is_file() else ""
        r = classify_pytest(text, n_ids)
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

    def pin_core(self, step: StepCfg, fb: Fallback, state: StepState) -> None:
        """Rerun the step's per-chip test (pin_id) on each failing device of the first attempt, alone, at twice the
        iterations. A hung target gets the fallback's own reset first; a reset that fails here aborts the run."""
        first = state.first
        # Hung or ND is per device, never from the step status. ND data on the device, not only a name: a build error
        # or a killed child is no reason to pin.
        targets = [d for d in first.devices if d.get("hung") or ND_DATA & set(d)]
        if not targets:
            trigger = "hang" if first.status == Status.HANG else "nd"
            self._skip_pin(state, fb, "no failing device named by the first attempt", {"trigger": trigger})
            return
        unsupported = None
        if not step.pin_id:
            unsupported = "no per-chip test"
        elif step.pin_id in self.pin_missing:
            unsupported = f"pin id does not collect: {step.pin_id}"
        test = step.pin_id.split("::")[-1].split("[")[0] if step.pin_id else ""
        for t in targets:
            trigger = "hang" if t.get("hung") else "nd"
            # One chip iterates faster than the full mesh: the rerun loops at least as long as the original did, and
            # at least twice its iterations.
            first_log = Path(first.log) if first.log else None
            original = loop_seconds(first_log) if first_log else None
            loop_env, loop_notes = {}, []
            if original:
                loop_env[MIN_LOOP_ENV] = f"{math.ceil(original[0] * 10 - 1e-6) / 10:.1f}"
            else:
                loop_notes.append(
                    f"WARNING original loop time unknown (no iteration lines in {first_log or 'no log'}); "
                    "doubled iterations only"
                )
            extra = {
                "target": {k: t[k] for k in IDENTITY if k in t},
                "trigger": trigger,
                "cores": {"first": list(t.get("cores", [])), "rerun": []},
            }
            if unsupported:
                self._skip_pin(state, fb, f"unsupported fallback: {step.key} ({unsupported})", extra, warn=True)
                continue
            opts, env_iters, used = double_iterations(step.opts, step.env)
            extra["iterations"] = used
            extra["loop"] = {"original_s": round(original[0], 1)} if original else {}
            a = Attempt(
                fb.name,
                {**env_iters, DEVICE_ID_ENV: str(t["dev"]), **loop_env},
                opts,
                Status.PENDING,
                "",
                [],
                [],
                "",
                None,
                None,
                extra,
            )
            state.fallbacks.append(a)
            reset = fb.hang_reset if trigger == "hang" else None
            d = self.run_dir / f"{step.key.replace(':', '/')}_{fb.name}_dev{t['dev']}"
            self.attempt(step, a, d, f"{fb.name} dev {t['dev']}", [step.pin_id], a.env, opts, reset)
            if self.reset_failed:
                a.reason += "; run aborted"
                self.aborted = f"reset failed before the {fb.name} rerun of {step.key} on device {t['dev']}"
                self.note(f"ABORT {self.aborted}")
                self.save()
                return
            if original and a.log and (rerun_loop := loop_seconds(Path(a.log))):
                a.extra["loop"] |= {"rerun_s": round(rerun_loop[0], 1), "rerun_iterations": rerun_loop[1]}
            self._pin_result(step, a, t, test, used)
            a.details += loop_notes
            self.note(f"{step.key} {fb.name} dev {t['dev']} result {a.status.value} {a.reason}")
            self.save()

    def _pin_result(self, step: StepCfg, a: Attempt, t: dict, test: str, used: dict) -> None:
        rerun = [d for d in a.devices if _same_chip(d, t)]
        others = [d["dev"] for d in a.devices if not _same_chip(d, t)]
        a.devices = rerun
        a.extra["cores"]["rerun"] = list(rerun[0].get("cores", [])) if rerun else []
        notes = [f"WARNING failure on device {d}, outside the pinned chip" for d in others]
        if a.status == Status.HANG:
            pass
        elif rerun and rerun[0].get("hung"):
            a.status, a.reason = Status.FAIL, "hang reproduced on chip alone"
        elif rerun and ND_DATA & set(rerun[0]):
            nd = rerun[0]
            a.status = Status.FAIL
            a.reason = "ND reproduced on chip alone" + (
                f": {nd['failed_checks']}/{nd['checks']} checks" if "failed_checks" in nd else ""
            )
            if "failed_checks" in nd and not nd.get("cores"):
                notes.append(f"WARNING unsupported fallback: {step.key} (no ND-core attribution in {test})")
        elif a.status in (Status.PASS, Status.WARN):
            a.status, a.reason = Status.WARN, "not reproduced on chip alone"
        if not used:
            notes.append(f"WARNING iterations not doubled: no iteration knob for {step.key}")
        a.details = a.details + notes

    def _skip_pin(self, state: StepState, fb: Fallback, why: str, extra: dict, warn: bool = False) -> None:
        details = [f"WARNING {why}"] if warn else []
        state.fallbacks.append(Attempt(fb.name, {}, [], Status.SKIP, why, details, [], "", None, None, extra))
        self.note(("WARNING " if warn else "") + why)
        self.save()

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
    shell_all=sh_all,
) -> int:
    """Every selected step in config order; a FAIL or HANG never stops the next step. Exit code: 1 when any step's
    first attempt FAILed or HUNG, else 0."""
    r = Runner(cfg, run_dir, chips, co, shell, snap, shell_all)
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
            d = run_dir / s.key.replace(":", "/")
            r.attempt(s, state.first, d, "first")
            if state.first.status in (Status.FAIL, Status.HANG):
                for fb in s.fallbacks:
                    if fb.kind == "pin_core":
                        r.pin_core(s, fb, state)
                    elif state.first.status.value in fb.on:
                        f = Attempt(fb.name, dict(fb.env), list(fb.opts), Status.PENDING, "", [], [], "", None, None)
                        state.fallbacks.append(f)
                        r.attempt(s, f, d.with_name(f"{d.name}_{fb.name}"), fb.name, env=fb.env, opts=s.opts + fb.opts)
                        if f.status == Status.HANG and fb.on == ("HANG",):
                            f.reason = _hang_persists(fb)
                            r.note(f"{s.key} {fb.name} result {f.status.value} {f.reason}")
                            r.save()
                    if r.aborted:
                        break
            if r.aborted:
                for rest in selected[selected.index(s) + 1 :]:
                    if states[rest.key].first.status == Status.PENDING:
                        states[rest.key].first.status = Status.SKIP
                        states[rest.key].first.reason = f"run aborted: {r.aborted}"
                break
    finally:
        r.current = None
        r.save()
    bad = [s.key for s in selected if states[s.key].first.status in (Status.FAIL, Status.HANG)]
    r.note(
        f"done: {len(selected) - len(bad)}/{len(selected)} steps passed" + (f"; failed {', '.join(bad)}" if bad else "")
    )
    return 1 if bad else 0
