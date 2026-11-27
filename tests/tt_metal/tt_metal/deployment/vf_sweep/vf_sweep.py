# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""VF sweep follow-up for the deployment test suite (test_runner.sh --run-vf-sweep).

classify  Decide from a test's log and exit code why it failed: ND (non-deterministic outputs), HANG (NOC barrier
          timeouts, stalled sources, wedged chips, chips that stopped answering MMIO, pytest timeout), OTHER
          (setup/machine problems) or PASS.
sweep     Re-run one workload (a DIDT minimal_matmul case, or a NOC integrity case on OpTestBase) as a clock x VDD
          sweep through ARC, and report at which (AICLK, VDD) points the failure is gone.
followup  What the suite runs: find every failed test in the NOC (C++), NOC integrity (py) and DIDT stage logs,
          classify each one, and sweep each test that failed on a hang or non-determinism (NOC) or on
          non-determinism (DIDT), one sweep per test. The C++ NOC tests cannot change clocks, so each one is swept
          through its Python port with the same sources: Noc00 (Ethernet sources) -> eth, Noc01 (Tensix) -> tensix.

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
        "target": "tests/didt/test_minimal_matmul.py::test_minimal_matmul",
        "k": "all and bf16_HiFi2",  # what the suite's DIDT stage runs
        "iterations": 1000,
        "check_interval": 50,
    },
    "noc": {
        "name": "NOC integrity (OpTestBase)",
        # Active Ethernet cores only accept kernels in slow dispatch.
        "env": {"TT_METAL_SLOW_DISPATCH_MODE": "1"},
        "target": "tests/didt/test_noc_integrity.py::test_noc_integrity",
        "k": "galaxy and all",  # what the suite's NOC integrity stage runs
        "iterations": 20,
        "check_interval": 5,
    },
}

# C++ NOC test cases -> the Python NOC integrity case with the same sources (the C++ test cannot change clocks).
CPP_NOC_CASES = {
    "Noc00": ("NOC (C++) Noc00, Ethernet sources", "galaxy and eth"),
    "Noc01": ("NOC (C++) Noc01, Tensix sources", "galaxy and tensix"),
}
GTEST_RUN = re.compile(r"\[ RUN\s+\] (\S+)")
GTEST_OK = re.compile(r"\[\s+OK \] (\S+)")
GTEST_FAILED = re.compile(r"\[\s+FAILED\s+\] (\S+)")
PYTEST_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+::\S+)")


def selection_args(spec, node=None, k=None):
    """pytest arguments that select the test to sweep: an exact node id, a -k expression, or the kind's default."""
    if node:
        return [node]
    return [spec["target"], "-k", k or spec["k"]]


def selection_label(spec, node=None, k=None):
    if node:
        return node.split("::")[-1]
    return f"-k '{k or spec['k']}'"


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
        *selection_args(spec, a.node, a.k),
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
        elif last["event"] in ("reference_start", "reference_set"):  # includes the compile: startup allowance
            limit, what = a.startup_timeout, f"reference outputs at {last['mhz']} MHz, +{last['offset']} mV"
            since = last["time"]
        elif last["event"] in ("start", "set"):
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
        ref_set = any(e["event"] == "reference_set" for e in events)
        ref_done = any(e["event"] == "reference" for e in events)
        set_steps = {e["step"] for e in events if e["event"] == "set"}
        hung = attempt_hung(outcome, attempt_log)
        if outcome == "done":
            break
        if not started and ref_started and not ref_done and hung:
            # The reference run hung at this step's clock and VDD: that is the failure under test, at this point.
            # HANG* = it hung while AICLK/VDD were being forced, before the workload ran there.
            current = ref_started[-1]["step"]
            status[current] = "HANG" if ref_set else "HANG*"
            startup_failures = 0
            where = "taking the reference outputs" if ref_set else "forcing AICLK/VDD for the reference outputs"
            log(f"step {current} {steps[current]} marked {status[current]} (hung while {where})", out)
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
            if hung:
                current_status = "HANG" if current in set_steps else "HANG*"
            else:
                current_status = "ERROR"
            status[current] = current_status
            where = {"HANG": " (hung while running)", "HANG*": " (hung while forcing AICLK/VDD)"}.get(
                current_status, ""
            )
            log(f"step {current} {steps[current]} marked {current_status}{where}", out)
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
            verdict = "HANG" if status[idx].startswith("HANG") else status[idx]
            points[(mhz, off)] = {"verdict": verdict, "text": status[idx], "vcore": {}, "bad": []}
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
    title = f"{TESTS[a.test]['name']} {selection_label(TESTS[a.test], a.node, a.k)}"
    lines = ["", "=" * 78, f"VF SWEEP RESULT: {title}", "=" * 78]
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
    forced_hangs = [idx for idx, s in status.items() if s == "HANG*"]
    if forced_hangs:
        lines.append(
            "  HANG* = a chip stopped answering while AICLK/VDD were being forced, before the workload ran there"
        )
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
    workload_failed = any(p["verdict"] == "FAIL" or p["text"] == "HANG" for p in points.values())
    if passing:
        verdict = "FIXED"
        detail = f"failure gone at {len(passing)}/{len(points)} points"
    elif forced_hangs and not workload_failed:
        verdict = "INCONCLUSIVE"
        detail = (
            f"forcing AICLK/VDD made a chip stop answering at {len(forced_hangs)} point(s); "
            "the workload never ran at a forced point"
        )
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
        "selection": selection_args(TESTS[a.test], a.node, a.k),
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


def sweep_one(a):
    """Run one sweep as configured in a (test, node/k, output, grid, timeouts); return its verdict."""
    spec = TESTS[a.test]
    a.iterations = a.iterations or spec["iterations"]
    a.check_interval = a.check_interval or spec["check_interval"]
    if isinstance(a.freqs, str):
        a.freqs = [int(f) for f in a.freqs.split(",") if f]
    if isinstance(a.offsets, str):
        a.offsets = [int(o) for o in a.offsets.split(",") if o]
    Path(a.output).mkdir(parents=True, exist_ok=True)
    with open(Path(a.output) / "sweep.log", "a") as out:
        title = f"{spec['name']} {selection_label(spec, a.node, a.k)}"
        log(f"VF sweep of {title} (log: {Path(a.output) / 'sweep.log'})", out)
        steps, status, abort_reason = run_sweep(a, out)
        return summarize(a, steps, status, abort_reason, out)


def cmd_sweep(a):
    verdict = sweep_one(a)
    sys.exit({"FIXED": 0, "NOT FIXED": 1}.get(verdict, 2))


# ---------------------------------------------------------------------------------------------------------------------
# followup


def failed_tests(stage, log_text, rc):
    """Each failed test of a suite stage: dicts with name, verdict, reason and what to sweep (test, node, k).

    noc_cpp: one entry per gtest case that failed or never finished, classified on its own part of the log, mapped to
             the Python NOC case with the same sources. A case that never finished in a run that exited 2 (chips left
             wedged) is the one that wedged them.
    noc_py, didt: one entry per 'FAILED/ERROR <node id>' line of pytest's summary, swept by that exact node id. When
             pytest died before printing its summary (a crash), the stage's own selection is swept. The stage log is
             classified as a whole (each stage runs a single test today).
    """
    entries = []
    if stage == "noc_cpp":
        lines = log_text.splitlines()
        runs = [(i, m.group(1)) for i, l in enumerate(lines) for m in [GTEST_RUN.search(l)] if m]
        ok = {m.group(1) for l in lines for m in [GTEST_OK.search(l)] if m}
        failed = {m.group(1) for l in lines for m in [GTEST_FAILED.search(l)] if m}
        for n, (i, case) in enumerate(runs):
            if case in ok and case not in failed:
                continue
            end = runs[n + 1][0] if n + 1 < len(runs) else len(lines)
            finished = case in failed
            verdict, reason = classify("noc_cpp", "\n".join(lines[i:end]), 1 if finished else rc)
            short = next((key for key in CPP_NOC_CASES if key in case), None)
            name, k = CPP_NOC_CASES.get(short, (f"NOC (C++) {case}", TESTS["noc"]["k"]))
            entries.append({"name": name, "verdict": verdict, "reason": reason, "test": "noc", "node": None, "k": k})
        if not entries and rc != 0:  # died before any case started, or the case lines are missing
            verdict, reason = classify("noc_cpp", log_text, rc)
            entries.append(
                {"name": "NOC (C++)", "verdict": verdict, "reason": reason, "test": "noc", "node": None, "k": None}
            )
        return entries

    kind = "noc" if stage == "noc_py" else "didt"
    stage_name = "NOC integrity (py)" if stage == "noc_py" else "DIDT"
    verdict, reason = classify("noc_py" if stage == "noc_py" else "didt", log_text, rc)
    nodes = []
    for l in log_text.splitlines():
        m = PYTEST_FAILED.search(l.strip())
        if m and m.group(1) not in nodes:
            nodes.append(m.group(1))
    for node in nodes:
        entries.append(
            {
                "name": f"{stage_name} {node.split('::')[-1]}",
                "verdict": verdict,
                "reason": reason,
                "test": kind,
                "node": node,
                "k": None,
            }
        )
    if not nodes and rc != 0:
        entries.append(
            {
                "name": f"{stage_name} (pytest died before its summary)",
                "verdict": verdict,
                "reason": reason,
                "test": kind,
                "node": None,
                "k": None,
            }
        )
    return entries


def sweep_qualifies(entry):
    """NOC tests are swept on a hang or non-determinism, DIDT only on non-determinism."""
    return entry["verdict"] in (("HANG", "ND") if entry["test"] == "noc" else ("ND",))


def slug(entry):
    sel = entry["node"].split("::")[-1] if entry["node"] else (entry["k"] or TESTS[entry["test"]]["k"])
    return entry["test"] + "_" + re.sub(r"[^A-Za-z0-9]+", "_", sel).strip("_")


def cmd_followup(a):
    entries = []
    for stage in ("noc_cpp", "noc_py", "didt"):
        given = getattr(a, stage)
        if not given:
            continue
        log_path, rc = given[0], int(given[1])
        text = Path(log_path).read_text(errors="replace") if Path(log_path).exists() else ""
        entries += failed_tests(stage, text, rc)

    Path(a.output).mkdir(parents=True, exist_ok=True)
    report = []
    swept = {}  # sweep key -> (verdict line, sweep dir), so a test failed by two stages is swept once
    for e in entries:
        print(f"{e['name']}: {e['verdict']} {e['reason']}", flush=True)
    for e in entries:
        if not sweep_qualifies(e):
            want = "a hang or non-determinism" if e["test"] == "noc" else "non-determinism"
            report.append(f"{e['name']}: not swept ({e['verdict']}: {e['reason']}; only {want} is swept)")
            continue
        key = (e["test"], e["node"] or e["k"] or TESTS[e["test"]]["k"])
        if key not in swept:
            sweep_dir = Path(a.output) / f"vf_sweep_{slug(e)}"
            print(f"\n=== VF sweep of {e['name']} ({e['verdict']}: {e['reason']}) -> {sweep_dir}", flush=True)
            s = argparse.Namespace(
                test=e["test"],
                node=e["node"],
                k=e["k"],
                output=str(sweep_dir),
                freqs=a.freqs,
                offsets=a.offsets,
                iterations=None,
                check_interval=None,
                startup_timeout=a.startup_timeout,
                step_timeout=a.step_timeout,
                reset_cmd=a.reset_cmd,
                no_initial_reset=a.no_initial_reset,
            )
            try:
                sweep_one(s)
            except Exception as ex:  # one broken sweep must not stop the others
                print(f"VF sweep of {e['name']} failed: {ex}", flush=True)
            verdict_file = sweep_dir / "verdict.txt"
            line = (
                verdict_file.read_text().strip() if verdict_file.exists() else "INCONCLUSIVE: the sweep did not finish"
            )
            swept[key] = (line, sweep_dir)
        line, sweep_dir = swept[key]
        report.append(f"{e['name']} ({e['verdict']}): {line} [{sweep_dir}/sweep.log]")
    if not entries:
        report.append("no failed tests found in the given logs")
    (Path(a.output) / "vf_followup.txt").write_text("\n".join(report) + "\n")
    print("\nVF sweep follow-up:", flush=True)
    for l in report:
        print(f"  {l}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify", help="print '<PASS|ND|HANG|OTHER> <reason>' for a test log")
    c.add_argument("--kind", choices=["didt", "noc_cpp", "noc_py"], required=True)
    c.add_argument("--log", required=True)
    c.add_argument("--rc", type=int, required=True)
    c.set_defaults(func=cmd_classify)

    def add_sweep_options(s):
        s.add_argument("--freqs", default="800,1000,1200,1350", help="AICLK MHz, comma separated")
        s.add_argument("--offsets", default="0,25,50", help="VDD offsets over the firmware VCORE, mV, comma separated")
        s.add_argument("--startup-timeout", type=int, default=1200, help="s allowed for open + compile + reference")
        s.add_argument(
            "--step-timeout",
            type=int,
            default=1800,
            help="s allowed for one point before it counts as a hang (raised to 3x the slowest finished point)",
        )
        s.add_argument("--reset-cmd", default="tt-smi -glx_reset")
        s.add_argument("--no-initial-reset", action="store_true", help="do not reset the boards before a sweep")

    s = sub.add_parser("sweep", help="run a VF sweep of one test; exit 0 fixed, 1 not fixed, 2 inconclusive")
    s.add_argument("--test", choices=sorted(TESTS), required=True)
    sel = s.add_mutually_exclusive_group()
    sel.add_argument("--node", help="exact pytest node id to sweep (default: the suite's selection for --test)")
    sel.add_argument("-k", dest="k", help="pytest -k expression within the test (e.g. 'galaxy and eth')")
    s.add_argument("--output", required=True, help="directory for sweep.log, events.jsonl, summary.json, verdict.txt")
    s.add_argument("--iterations", type=int, default=None, help="iterations per point (default: the suite's)")
    s.add_argument("--check-interval", type=int, default=None, help="determinism check interval (default: suite's)")
    add_sweep_options(s)
    s.set_defaults(func=cmd_sweep)

    f = sub.add_parser("followup", help="sweep every failed test of the suite's NOC / NOC integrity / DIDT stages")
    f.add_argument("--output", required=True, help="directory for one vf_sweep_<test> folder per sweep")
    f.add_argument("--noc-cpp", nargs=2, metavar=("LOG", "RC"), help="C++ NOC stage log and exit code")
    f.add_argument("--noc-py", nargs=2, metavar=("LOG", "RC"), help="NOC integrity (py) stage log and exit code")
    f.add_argument("--didt", nargs=2, metavar=("LOG", "RC"), help="DIDT stage log and exit code")
    add_sweep_options(f)
    f.set_defaults(func=cmd_followup)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
