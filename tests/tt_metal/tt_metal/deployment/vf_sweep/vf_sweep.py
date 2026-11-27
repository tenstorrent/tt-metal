# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""VF sweep follow-up for the deployment test suite (test_runner.sh --run-vf-sweep).

classify  Decide from a test's log and exit code why it failed: ND (non-deterministic outputs), HANG (NOC barrier
          timeouts, stalled sources, wedged chips, pytest timeout), OTHER (setup/machine problems) or PASS.
sweep     Re-run the failing workload (DIDT minimal_matmul, or the NOC integrity test on OpTestBase) as a clock x VDD
          sweep through ARC, and report at which (AICLK, VDD) points the failure is gone.

The sweep survives hangs: OpTestBase appends one JSON line per sweep event (--vf-sweep-results). When no event arrives
within the step timeout, or pytest dies on a device timeout (a chip that stopped answering MMIO), the step is marked
HANG, pytest is killed, the boards are reset and the sweep resumes at the next step (--vf-sweep-start-step).
Reference outputs are taken at the clock and VDD of the first step each run starts from (the lowest clock at +0 mV on
a fresh sweep). A point that hangs while its reference is being taken is marked HANG and skipped like any other.

A point is FIXED when every determinism check at it was bit-exact on every chip and it did not hang. The working
(clock, Vmin) pairs are, per clock, the lowest VDD offset from which every higher tested offset also passed.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ND_COUNT = re.compile(r"Number of non-deterministic outputs on device (\d+) is (\d+)")
PYTEST_TIMEOUT = re.compile(r"\+{5,} Timeout \+{5,}|Failed: Timeout")
UNUSED = re.compile(r"unused Ethernet channel|\[unused channel\]")
NOC_HANG = re.compile(r"NOC (hang|timeout|not run):|wedged|NOC failure:.*(timeout|not run|kernel never ran)")
NOC_MISMATCH = re.compile(r"NOC mismatch:")
# A chip that stopped answering host reads: UMD gives up on the MMIO access (and aborts the process when it happens in a
# destructor). For the NOC test this is how a wedged NOC shows up on the host.
DEVICE_TIMEOUT = re.compile(r"MMIO per-op timeout|DeviceTimeoutError")

TESTS = {
    "didt": {
        "name": "DIDT minimal_matmul",
        "env": {},
        "args": [
            "tests/didt/test_minimal_matmul.py::test_minimal_matmul",
            "-k",
            "all and bf16_HiFi2",
        ],
        "iterations": 1000,
        "check_interval": 50,
    },
    "noc": {
        "name": "NOC integrity (OpTestBase)",
        # Active Ethernet cores only accept kernels in slow dispatch.
        "env": {"TT_METAL_SLOW_DISPATCH_MODE": "1"},
        "args": [
            "tests/didt/test_noc_integrity.py::test_noc_integrity",
            "-k",
            "galaxy and all",
        ],
        "iterations": 20,
        "check_interval": 5,
    },
}


# ---------------------------------------------------------------------------------------------------------------------
# classify


def classify(kind, log_text, rc):
    """Return (verdict, reason); verdict is PASS, ND, HANG or OTHER."""
    if rc == 0:
        return "PASS", "test passed"
    lines = [l for l in log_text.splitlines() if not UNUSED.search(l)]
    nd_devices = sorted({int(m.group(1)) for l in lines for m in [ND_COUNT.search(l)] if m and int(m.group(2)) > 0})
    timed_out = any(PYTEST_TIMEOUT.search(l) for l in lines)

    if kind == "didt":
        # Only a clean determinism failure counts: the test ran to its end and some chip produced different outputs.
        if timed_out:
            return "OTHER", "pytest timeout (not a determinism failure)"
        if nd_devices:
            return "ND", f"non-deterministic outputs on device(s) {', '.join(map(str, nd_devices))}"
        return "OTHER", f"failed with exit code {rc} without non-deterministic outputs (setup or machine problem)"

    hang_lines = [l for l in lines if NOC_HANG.search(l)]
    mismatch_lines = [l for l in lines if NOC_MISMATCH.search(l)]
    if kind == "noc_cpp" and rc == 2:
        return "HANG", "chips left wedged (exit code 2)"
    if hang_lines:
        return "HANG", f"{len(hang_lines)} NOC timeout/hang line(s)"
    if any(DEVICE_TIMEOUT.search(l) for l in lines):
        return "HANG", "a chip stopped answering MMIO reads (device timeout)"
    if timed_out:
        return "HANG", "pytest timeout"
    if mismatch_lines:
        return "ND", f"{len(mismatch_lines)} NOC data mismatch line(s)"
    if nd_devices:
        return "ND", f"failed checks on device(s) {', '.join(map(str, nd_devices))}"
    return "OTHER", f"failed with exit code {rc} without NOC hangs or mismatches (setup or machine problem)"


def cmd_classify(a):
    text = Path(a.log).read_text(errors="replace") if Path(a.log).exists() else ""
    verdict, reason = classify(a.kind, text, a.rc)
    print(f"{verdict} {reason}")


# ---------------------------------------------------------------------------------------------------------------------
# sweep


def log(msg, out):
    line = f"[vf-sweep {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    out.write(line + "\n")
    out.flush()


def read_events(path):
    events = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass  # a line cut short by a kill
    return events


def step_durations(events):
    """Seconds each finished step took, from its start to its end event."""
    starts = {e["step"]: e["time"] for e in events if e["event"] == "start"}
    return [e["time"] - starts[e["step"]] for e in events if e["event"] == "end" and e["step"] in starts]


def reset_boards(reset_cmd, out):
    for attempt in (1, 2):
        log(f"resetting boards ({reset_cmd}), attempt {attempt}", out)
        try:
            r = subprocess.run(reset_cmd, shell=True, stdout=out, stderr=subprocess.STDOUT, timeout=600)
            if r.returncode == 0:
                time.sleep(10)  # tt-smi needs a moment before the chips are visible again
                return True
            log(f"reset exited with {r.returncode}", out)
        except subprocess.TimeoutExpired:
            log("reset timed out", out)
    return False


def kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def attempt_hung(outcome, attempt_log):
    """Whether a run that ended early hung: killed by the watchdog, or died on a chip that stopped answering MMIO."""
    return outcome == "hang" or bool(DEVICE_TIMEOUT.search(attempt_log))


def run_invocation(a, spec, results, start_step, out):
    """Run pytest once from start_step. Returns (outcome, log): outcome is 'done', 'hang' (killed by the watchdog) or
    'exit' (ended early), log is what this run wrote to sweep.log."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *spec["args"],
        "--didt-workload-iterations",
        str(a.iterations),  # ignored by the sweep itself, kept so the fixture value is sane
        "--determinism-check-interval",
        str(a.check_interval),
        "--didt-vf-sweep",
        ",".join(map(str, a.freqs)),
        "--sweep-iterations",
        str(a.iterations),
        "--vdd-offsets",
        ",".join(map(str, a.offsets)),
        "--vf-sweep-start-step",
        str(start_step),
        "--vf-sweep-results",
        str(results),
        "-q",
        "--tb=line",
        "-p",
        "no:cacheprovider",
    ]
    env = dict(os.environ, PYTHONUNBUFFERED="1", **spec["env"])
    log(f"running from step {start_step}: {' '.join(cmd)}", out)
    seen = len(read_events(results))
    out.flush()
    log_start = out.tell()
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    phase_started = time.time()

    def attempt_log():
        with open(out.name, errors="replace") as f:
            f.seek(log_start)
            return f.read()

    while True:
        try:
            proc.wait(timeout=5)
            break
        except subprocess.TimeoutExpired:
            pass
        events = read_events(results)[seen:]
        last = events[-1] if events else None
        # A step may take up to 3x the slowest finished step (clocks differ), never less than --step-timeout.
        limit = max([a.step_timeout] + [3 * d for d in step_durations(read_events(results))])
        if last is None:
            limit, what = a.startup_timeout, "device open"
            since = phase_started
        elif last["event"] == "reference_start":  # includes the compile, so it gets the startup allowance
            limit, what = a.startup_timeout, f"reference outputs at {last['mhz']} MHz, +{last['offset']} mV"
            since = last["time"]
        elif last["event"] == "start":
            what = f"step {last['step']} ({last['mhz']} MHz, +{last['offset']} mV)"
            since = last["time"]
        else:  # reference taken or a step just ended: the next step's clock change is due shortly
            what = "the next step to start"
            since = last["time"]
        if time.time() - since > limit:
            log(f"no progress for {limit} s while waiting for {what}: treating it as a hang, killing pytest", out)
            kill_group(proc)
            return "hang", attempt_log()
    if any(e["event"] == "done" for e in read_events(results)[seen:]):
        return "done", attempt_log()
    text = attempt_log()
    log(
        f"pytest exited with {proc.returncode} before the sweep finished"
        + (" (a chip stopped answering MMIO: device timeout)" if DEVICE_TIMEOUT.search(text) else ""),
        out,
    )
    return "exit", text


def run_sweep(a, out):
    spec = TESTS[a.test]
    outdir = Path(a.output)
    results = outdir / "events.jsonl"
    if results.exists():
        results.unlink()
    steps = [(mhz, off) for mhz in a.freqs for off in a.offsets]
    status = {}  # step index -> HANG | ERROR (steps that finished are read from the events)

    if not a.no_initial_reset and not reset_boards(a.reset_cmd, out):
        return steps, status, "could not reset the boards before the sweep"

    start = 0
    startup_failures = 0
    while start < len(steps):
        before = len(read_events(results))
        outcome, attempt_log = run_invocation(a, spec, results, start, out)
        events = read_events(results)[before:]
        started = [e for e in events if e["event"] == "start"]
        ended = {e["step"] for e in events if e["event"] == "end"}
        ref_started = [e for e in events if e["event"] == "reference_start"]
        ref_done = any(e["event"] == "reference" for e in events)
        hung = attempt_hung(outcome, attempt_log)
        if outcome == "done":
            break
        if not started and ref_started and not ref_done and hung:
            # The reference run hung at this step's clock and VDD: that is the failure under test, at this point.
            current = ref_started[-1]["step"]
            status[current] = "HANG"
            startup_failures = 0
            log(f"step {current} {steps[current]} marked HANG (hung while taking the reference outputs)", out)
            start = current + 1
            if not reset_boards(a.reset_cmd, out):
                return steps, status, "could not reset the boards after a failed step"
            continue
        if not started:
            # Never reached the first step and did not hang in the reference: setup failed (or the reference run
            # failed in some other way). Retry once, then give up.
            startup_failures += 1
            reason = "hung" if hung else "failed"
            where = "while taking the reference outputs" if ref_started else "before taking the reference outputs"
            log(f"the sweep {reason} {where} (attempt {startup_failures})", out)
            if not reset_boards(a.reset_cmd, out) or startup_failures >= 2:
                return steps, status, f"the sweep {reason} {where} twice (setup problem)"
            continue
        startup_failures = 0
        current = started[-1]["step"]
        if current in ended:  # died between steps: resume after the last one that finished
            current_status = None
        else:
            current_status = "HANG" if hung else "ERROR"
            status[current] = current_status
            log(f"step {current} {steps[current]} marked {current_status}", out)
        start = current + 1
        if not reset_boards(a.reset_cmd, out):
            return steps, status, "could not reset the boards after a failed step"
    return steps, status, None


def summarize(a, steps, status, abort_reason, out):
    events = read_events(Path(a.output) / "events.jsonl")
    ends = {e["step"]: e for e in events if e["event"] == "end"}
    # Reference outputs are taken at the first step of each run; failed devices there count against that step.
    ref_failed = {}  # step index -> devices whose reference output failed
    for e in events:
        if e["event"] == "reference" and e.get("failed"):
            ref_failed.setdefault(e.get("step", 0), set()).update(e["failed"])

    points = {}  # (mhz, off) -> dict(verdict, text, vcore{dev: mV}, bad_devices)
    for idx, (mhz, off) in enumerate(steps):
        if idx in status:
            points[(mhz, off)] = {"verdict": status[idx], "text": status[idx], "vcore": {}, "bad": []}
            continue
        e = ends.get(idx)
        if e is None:
            points[(mhz, off)] = {"verdict": "NOT RUN", "text": "-", "vcore": {}, "bad": []}
            continue
        bad = sorted(set(int(d) for d, n in e["nd"].items() if n) | ref_failed.get(idx, set()))
        if e["checks"] == 0:
            verdict, text = "NO CHECKS", "no checks"
        elif bad:
            verdict = "FAIL"
            text = "FAIL " + ",".join(f"d{d}:{e['nd'].get(str(d), 'ref')}/{e['checks']}" for d in bad)
        else:
            verdict, text = "PASS", f"PASS {e['checks']}/{e['checks']}"
        points[(mhz, off)] = {
            "verdict": verdict,
            "text": text,
            "vcore": {int(d): v for d, v in e.get("vcore", {}).items()},
            "bad": bad,
        }

    problem_devices = sorted({d for p in points.values() for d in p["bad"]})
    lines = ["", "=" * 78, f"VF SWEEP RESULT: {TESTS[a.test]['name']}", "=" * 78]
    lines.append(
        f"AICLK {a.freqs} MHz x VDD offsets {a.offsets} mV over the firmware VCORE, {a.iterations} iterations "
        f"per point, determinism check every {a.check_interval}"
    )
    if abort_reason:
        lines.append(f"Sweep stopped early: {abort_reason}")

    header = ["AICLK \\ VDD"] + [f"+{o} mV" for o in a.offsets]
    rows = [[f"{mhz} MHz"] + [points[(mhz, o)]["text"] for o in a.offsets] for mhz in a.freqs]
    widths = [max(len(r[c]) for r in rows + [header]) for c in range(len(header))]
    fmt = lambda r: "  " + " | ".join(v.ljust(w) for v, w in zip(r, widths))
    lines += ["", "Points (PASS = every check bit-exact on every chip, no hang):", fmt(header)] + [fmt(r) for r in rows]
    for idx, devs in sorted(ref_failed.items()):
        if idx < len(steps):
            mhz, off = steps[idx]
            lines.append(f"  reference outputs at {mhz} MHz +{off} mV failed on device(s) {sorted(devs)}")

    def vrange(v):
        return f"{min(v.values())}-{max(v.values())} mV" if v else "?"

    lines += ["", "Working (clock, Vmin) pairs (Vmin = lowest tested VDD from which every higher offset also passed):"]
    working = []
    for mhz in a.freqs:
        ok = [points[(mhz, o)]["verdict"] == "PASS" for o in a.offsets]
        k = next((k for k in range(len(ok)) if all(ok[k:])), None)
        if k is None:
            lines.append(
                f"  {mhz:>5} MHz: none (" + ", ".join(f"+{o}: {points[(mhz, o)]['verdict']}" for o in a.offsets) + ")"
            )
            continue
        off = a.offsets[k]
        v = points[(mhz, off)]["vcore"]
        extra = ""
        if problem_devices and v:
            extra = ", problem chip(s) " + ", ".join(f"d{d} {v[d]} mV" for d in problem_devices if d in v)
        lines.append(f"  {mhz:>5} MHz: Vmin = firmware VCORE +{off} mV (VCORE {vrange(v)} across chips{extra})")
        working.append((mhz, off))

    passing = [k for k, p in points.items() if p["verdict"] == "PASS"]
    if passing:
        verdict = "FIXED"
        detail = f"failure gone at {len(passing)}/{len(points)} points"
    elif abort_reason and not any(p["verdict"] in ("FAIL", "HANG") for p in points.values()):
        verdict = "INCONCLUSIVE"
        detail = abort_reason
    else:
        verdict = "NOT FIXED"
        detail = "the failure remained at every tested point"
    lines += ["", f"VF SWEEP VERDICT: {verdict} ({detail})", "=" * 78]
    for l in lines:
        log(l, out)

    summary = {
        "test": a.test,
        "verdict": verdict,
        "detail": detail,
        "abort_reason": abort_reason,
        "points": {f"{m}MHz+{o}mV": p for (m, o), p in points.items()},
        "working_clock_vmin": [{"mhz": m, "vdd_offset_mv": o} for m, o in working],
        "problem_devices": problem_devices,
    }
    (Path(a.output) / "summary.json").write_text(json.dumps(summary, indent=2))
    # One line for the suite summary
    (Path(a.output) / "verdict.txt").write_text(
        f"{verdict}: {detail}"
        + (("; working (clock, Vmin): " + ", ".join(f"{m} MHz/+{o} mV" for m, o in working)) if working else "")
        + "\n"
    )
    return verdict


def cmd_sweep(a):
    spec = TESTS[a.test]
    a.iterations = a.iterations or spec["iterations"]
    a.check_interval = a.check_interval or spec["check_interval"]
    a.freqs = [int(f) for f in a.freqs.split(",") if f]
    a.offsets = [int(o) for o in a.offsets.split(",") if o]
    Path(a.output).mkdir(parents=True, exist_ok=True)
    with open(Path(a.output) / "sweep.log", "a") as out:
        log(f"VF sweep of {spec['name']} (log: {Path(a.output) / 'sweep.log'})", out)
        steps, status, abort_reason = run_sweep(a, out)
        verdict = summarize(a, steps, status, abort_reason, out)
    sys.exit({"FIXED": 0, "NOT FIXED": 1}.get(verdict, 2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify", help="print '<PASS|ND|HANG|OTHER> <reason>' for a test log")
    c.add_argument("--kind", choices=["didt", "noc_cpp", "noc_py"], required=True)
    c.add_argument("--log", required=True)
    c.add_argument("--rc", type=int, required=True)
    c.set_defaults(func=cmd_classify)

    s = sub.add_parser("sweep", help="run a VF sweep of a failing test; exit 0 fixed, 1 not fixed, 2 inconclusive")
    s.add_argument("--test", choices=sorted(TESTS), required=True)
    s.add_argument("--output", required=True, help="directory for sweep.log, events.jsonl, summary.json, verdict.txt")
    s.add_argument("--freqs", default="800,1000,1200,1350", help="AICLK MHz, comma separated")
    s.add_argument("--offsets", default="0,25,50", help="VDD offsets over the firmware VCORE, mV, comma separated")
    s.add_argument("--iterations", type=int, default=None, help="iterations per point (default: the suite's)")
    s.add_argument("--check-interval", type=int, default=None, help="determinism check interval (default: suite's)")
    s.add_argument("--startup-timeout", type=int, default=1200, help="s allowed for open + compile + reference")
    s.add_argument(
        "--step-timeout",
        type=int,
        default=1800,
        help="s allowed for one point before it counts as a hang (raised to 3x the slowest finished point)",
    )
    s.add_argument("--reset-cmd", default="tt-smi -glx_reset")
    s.add_argument("--no-initial-reset", action="store_true", help="do not reset the boards before the sweep")
    s.set_defaults(func=cmd_sweep)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
