# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 3: the device tests — deployment gtests and pytest stress tests.

TESTS, PYTESTS and DIDT_RUNNERS define the steps a tier can list under ``tests`` in tiers.py.
"""
from __future__ import annotations

import os
import re
import shlex
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from .common import emit_result, emit_running, log
from .model import FAIL, PASS, Check, Phase

# tt-metal deployment test binary candidate paths relative to repo root.
# build_metal.sh defaults to build_Release/; raw cmake invocations often use build/.
DEPLOYMENT_BIN_CANDIDATES = [
    "build_Release/test/tt_metal/unit_tests_deployment",
    "build/test/tt_metal/unit_tests_deployment",
    "build_Debug/test/tt_metal/unit_tests_deployment",
    "build_RelWithDebInfo/test/tt_metal/unit_tests_deployment",
]


# gtest filter -> test name table. Names match deployment_tests_dram_glx (outlogix).
# The eth deployment tests are registered as TensixDeploymentEthernet<NN><Name>
# (e.g. TensixDeploymentEthernet00LinkUp, ...Ethernet01Bandwidth), so the filters
# need a wildcard for the two-digit index; otherwise gtest matches 0 tests.
# eth_bandwidth keeps a trailing wildcard so it also picks up BandwidthBidir.
# They need slow dispatch too: fast dispatch to active ethernet cores was
# removed, and under fast dispatch each eth test skips itself and exits 0.
# Each entry is the gtest filter plus, optionally, environment set for that run
# only (recorded in the check as env_overrides and prefixed to its command line).
TESTS = {
    "eth_link_up": {"filter": "*TensixDeploymentEthernet*LinkUp", "env": {"TT_METAL_SLOW_DISPATCH_MODE": "1"}},
    "eth_bandwidth": {"filter": "*TensixDeploymentEthernet*Bandwidth*", "env": {"TT_METAL_SLOW_DISPATCH_MODE": "1"}},
    "gddr_fast": {
        "filter": "*DramDeployment_PersistentOptimalWorkersAllDramBanks",
        "env": {"DRAM_TEST_FAST": "1"},
    },
    "gddr_full": {"filter": "*DramDeployment_*"},
}

# Every variable any gtest entry overrides. A check's env_overrides reports the
# value each of them has for that run, ambient or overridden, so a variable
# inherited from the caller's environment is on the record too.
GTEST_ENV_KEYS = tuple(dict.fromkeys(k for spec in TESTS.values() for k in spec.get("env", {})))


# Pytest-based tests, keyed like TESTS but run through the repo venv's pytest
# (python_env/bin/pytest, falling back to PATH) instead of the gtest binary.
# Each entry is a pytest node id plus its extra args.
PYTESTS: dict[str, dict] = {}


# tests/didt/didt_runner.py, keyed like TESTS: every DIDT ND/hang screen of one runner config, each with its
# fallbacks (throttle, pin_core), and a status.json naming each failing chip by BDF, tray and ASIC. The runner resets
# the galaxy before each of its attempts. Measured on BH galaxies: the daily config (minimal matmul, 1000
# iterations) ~7 min when it passes; the weekly config (8 screens) ~25 min when everything passes, 30-70 min when
# steps fail and fallbacks run. timeout_s bounds a wedged run; the fleet harness's own 30-minute wall still applies.
DIDT_RUNNER = "tests/didt/didt_runner.py"
DIDT_RUNNERS = {
    "didt_runner_daily": {"config": "daily", "timeout_s": 45 * 60},
    "didt_runner_weekly": {"config": "weekly", "timeout_s": 120 * 60},
}
# After the timeout the runner gets SIGTERM, which kills its own workloads; SIGKILL follows if it is still up.
DIDT_RUNNER_TERM_GRACE_S = 120
RUNNER_RUN_DIR_RE = re.compile(r"^run dir (\S+)")


# gtest progress markers — used for per-testcase pass/fail accounting.
# Matches only on lines with a "Suite.Test" pattern, so summary lines like
# "[  FAILED  ] 1 test, listed below:" don't get counted as testcases.
GTEST_RUN_RE = re.compile(r"\[\s*RUN\s*\]\s+(\S+\.\S+)")
GTEST_OK_RE = re.compile(r"\[\s*OK\s*\]\s+(\S+\.\S+)")
GTEST_FAIL_RE = re.compile(r"\[\s*FAILED\s*\]\s+(\S+\.\S+)")
GTEST_SKIP_RE = re.compile(r"\[\s*SKIPPED\s*\]\s+(\S+\.\S+)")


# pytest progress markers. Counts come from the end-of-run summary line
# ("===== 3 passed, 1 failed, 2 deselected in 600.12s ====="); failing node ids
# from the short-summary "FAILED tests/..." lines and, for fixture/collection
# errors, the "ERROR tests/..." lines.
PYTEST_COUNT_RE = re.compile(r"(\d+) (passed|failed|error)")
PYTEST_FAILED_LINE_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)")


def parse_pytest_summary_counts(line: str) -> tuple[int, int] | None:
    """Extract (passed, failed) from a pytest end-of-run summary line, or None
    if the line isn't one. Fixture/collection `error` counts fold into failed.
    A "no tests ran" summary carries no counts and returns None — callers'
    zero-initialized counts then record the run as FAIL, not a silent PASS."""
    if not line.startswith("=") or not (" passed" in line or " failed" in line or " error" in line):
        return None
    counts = {word: int(num) for num, word in PYTEST_COUNT_RE.findall(line)}
    return counts.get("passed", 0), counts.get("failed", 0) + counts.get("error", 0)


def resolve_pytest(tt_metal: Path) -> str | None:
    """Prefer the repo venv's pytest (has ttnn + test deps); fall back to PATH."""
    venv_pytest = tt_metal / "python_env" / "bin" / "pytest"
    if venv_pytest.is_file():
        return str(venv_pytest)
    return shutil.which("pytest")


def run_pytest_test(
    name: str, spec: dict, tt_metal: Path, env: dict, phase: Phase, dry_run: bool, logs_dir: Path
) -> None:
    """Run one PYTESTS entry, mirroring the gtest flow: stream output to the
    per-test log, count testcases, and record a Check with the same
    `passed=N failed=N` details format the CSV analyzer parses."""
    pytest_bin = resolve_pytest(tt_metal)
    if pytest_bin is None:
        if dry_run:
            pytest_bin = "pytest"  # display-only; nothing is executed
        else:
            phase.add(
                Check(
                    name=name,
                    status=FAIL,
                    details=f"pytest not found ({tt_metal}/python_env/bin/pytest or PATH)",
                    ip="other",
                )
            )
            return

    # cmd is built exclusively from the hardcoded PYTESTS table and the
    # repo-resolved pytest binary — no user-controlled input — and is executed
    # as an argv list (no shell).
    target = spec["target"]
    cmd = [pytest_bin, target, *spec["args"]]
    test_env = env.copy()
    test_env.setdefault("PYTHONPATH", str(tt_metal))
    log_path = logs_dir / f"{name}.log"
    # shlex.join keeps args with spaces (-k "galaxy and bf16_HiFi2") copy-pasteable
    cmdline = shlex.join(cmd)
    log(f"--- test '{name}' (pytest) ---")
    log(f"  cmd:  {cmdline}")
    log(f"  log:  {log_path}")

    if dry_run:
        print(f"  {name:30} (dry-run)")
        phase.add(
            Check(
                name=name,
                status=PASS,
                ip="other",
                details=f"(dry) target={target} log={log_path}",
                data={
                    "target": target,
                    "command": cmdline,
                    "rc": 0,
                    "duration_s": 0.0,
                    "log_file": str(log_path),
                    "testcases": {"passed": 0, "failed": 0, "failures": []},
                },
            )
        )
        return

    emit_running(name)

    t0 = time.time()
    passed = 0
    failed = 0
    failures: list[str] = []
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            env=test_env,
            cwd=str(tt_metal),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            # All pytest output goes to the log file only — keep the console clean.
            logf.write(line)
            if m := PYTEST_FAILED_LINE_RE.match(line):
                failures.append(m.group(1))
            elif (counts := parse_pytest_summary_counts(line)) is not None:
                passed, failed = counts
        rc = proc.wait()
    # A node can appear in both a live FAILED line and the short summary.
    failures = list(dict.fromkeys(failures))
    dt = time.time() - t0

    # Require passed > 0 so an over-narrow -k expression (0 tests collected,
    # pytest rc=5) or an unparsed summary can't be recorded as a silent PASS —
    # same trap as the zero-match gtest filters documented above TESTS.
    status = PASS if rc == 0 and failed == 0 and passed > 0 else FAIL
    suffix = f"({dt:.1f}s)"
    if failed:
        suffix += f"  passed={passed} failed={failed}"
    emit_result(name, status, suffix=suffix)
    details = f"target={target} rc={rc} dur={dt:.1f}s " f"passed={passed} failed={failed} log={log_path}"
    phase.add(
        Check(
            name=name,
            status=status,
            details=details,
            data={
                "target": target,
                "command": cmdline,
                "rc": rc,
                "duration_s": dt,
                "log_file": str(log_path),
                "testcases": {"passed": passed, "failed": failed, "failures": failures},
            },
            ip="other",
        )
    )


def resolve_python(tt_metal: Path) -> str:
    """The repo venv's python when there is one; else this interpreter (the exabox-tools image's /opt/venv)."""
    venv_python = tt_metal / "python_env" / "bin" / "python"
    return str(venv_python) if venv_python.is_file() else sys.executable


def didt_runner_counts(status_json: Path) -> tuple[int, int, list[str]]:
    """(passed, failed, failures) over the runner steps that ran. A step PASSes or WARNs, or FAILs or HANGs; steps
    left out by the config or by the user (SKIP) count as neither."""
    doc = json.loads(status_json.read_text())
    passed, failed, failures = 0, 0, []
    for step in doc.get("steps", []):
        status = step.get("status")
        if status in ("PASS", "WARN"):
            passed += 1
        elif status in ("FAIL", "HANG", "PENDING", "RUNNING"):
            failed += 1
            failures.append(f"{step.get('key')}: {status} {step.get('reason', '')}".rstrip())
    return passed, failed, failures


def run_didt_runner(
    name: str, spec: dict, tt_metal: Path, env: dict, phase: Phase, dry_run: bool, logs_dir: Path
) -> None:
    """Run one DIDT_RUNNERS entry. PASS needs exit code 0, the runner's status.json, and at least one step that
    passed with none failed: a run that never wrote its status, or ran nothing, is a FAIL."""
    out_dir = logs_dir.parent / name
    cmd = [resolve_python(tt_metal), DIDT_RUNNER, "--config", spec["config"], "--out", str(out_dir)]
    test_env = env.copy()
    test_env.setdefault("PYTHONPATH", str(tt_metal))
    log_path = logs_dir / f"{name}.log"
    cmdline = shlex.join(cmd)
    log(f"--- test '{name}' (didt_runner) ---")
    log(f"  cmd:  {cmdline}")
    log(f"  log:  {log_path}")

    if dry_run:
        print(f"  {name:30} (dry-run)")
        phase.add(
            Check(
                name=name,
                status=PASS,
                ip="other",
                details=f"(dry) config={spec['config']} log={log_path}",
                data={
                    "config": spec["config"],
                    "command": cmdline,
                    "rc": 0,
                    "duration_s": 0.0,
                    "log_file": str(log_path),
                    "status_json": None,
                    "testcases": {"passed": 0, "failed": 0, "failures": []},
                },
            )
        )
        return

    emit_running(name)
    t0 = time.time()
    timed_out = False
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, env=test_env, cwd=str(tt_metal), start_new_session=True
        )
        try:
            rc = proc.wait(timeout=spec["timeout_s"])
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                rc = proc.wait(timeout=DIDT_RUNNER_TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                rc = proc.wait()
    dt = time.time() - t0

    run_dir = next(
        (Path(m.group(1)) for m in map(RUNNER_RUN_DIR_RE.match, log_path.read_text().splitlines()) if m), None
    )
    status_json = run_dir / "status.json" if run_dir else None
    passed, failed, failures = 0, 0, []
    note = ""
    if status_json is None or not status_json.is_file():
        note = f" no status.json ({'no run dir in the log' if run_dir is None else status_json})"
    else:
        try:
            passed, failed, failures = didt_runner_counts(status_json)
        except (OSError, ValueError) as e:
            note = f" unreadable status.json: {e}"
    if timed_out:
        note += f" timeout after {spec['timeout_s']} s"
    status = PASS if rc == 0 and not note and failed == 0 and passed > 0 else FAIL
    suffix = f"({dt:.1f}s)"
    if status == FAIL:
        suffix += f"  passed={passed} failed={failed}{note}"
    emit_result(name, status, suffix=suffix)
    details = (
        f"config={spec['config']} rc={rc} dur={dt:.1f}s passed={passed} failed={failed}{note} "
        f"status={status_json} log={log_path}"
    )
    phase.add(
        Check(
            name=name,
            status=status,
            details=details,
            data={
                "config": spec["config"],
                "command": cmdline,
                "rc": rc,
                "duration_s": dt,
                "log_file": str(log_path),
                "status_json": str(status_json) if status_json else None,
                "testcases": {"passed": passed, "failed": failed, "failures": failures},
            },
            ip="other",
        )
    )


def run_tests(tt_metal: Path, names: Sequence[str], phase: Phase, dry_run: bool, logs_dir: Path) -> None:
    """Run the named TESTS / PYTESTS / DIDT_RUNNERS entries, in order."""
    binary = next(
        (tt_metal / c for c in DEPLOYMENT_BIN_CANDIDATES if (tt_metal / c).is_file()),
        None,
    )
    if binary is None and any(n in TESTS for n in names):
        # Pytest and runner entries don't need the gtest binary — record the FAIL
        # for the gtest set but keep going so they still run.
        phase.add(
            Check(
                name="deployment_binary_present",
                status=FAIL,
                details=f"none of {DEPLOYMENT_BIN_CANDIDATES} found under {tt_metal}",
                ip="other",
            )
        )

    env = os.environ.copy()
    env["TT_METAL_HOME"] = str(tt_metal)
    env["TT_METAL_RUNTIME_ROOT"] = str(tt_metal)  # newer name post-rename
    if binary is not None:
        # LD_LIBRARY_PATH: match the build dir of the binary we picked
        env.setdefault("LD_LIBRARY_PATH", str(binary.parent.parent.parent / "lib"))

    logs_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        if name in PYTESTS:
            run_pytest_test(name, PYTESTS[name], tt_metal, env, phase, dry_run, logs_dir)
            continue
        if name in DIDT_RUNNERS:
            run_didt_runner(name, DIDT_RUNNERS[name], tt_metal, env, phase, dry_run, logs_dir)
            continue
        if binary is None:
            continue
        spec = TESTS[name]
        filt = spec["filter"]
        test_env = {**env, **spec.get("env", {})}
        env_overrides = {k: test_env[k] for k in GTEST_ENV_KEYS if k in test_env}
        cmd = [str(binary), f"--gtest_filter={filt}"]
        log_path = logs_dir / f"{name}.log"
        prefix = " ".join(f"{k}={v}" for k, v in env_overrides.items())
        cmdline = f"{prefix + ' ' if prefix else ''}{' '.join(cmd)}"
        log(f"--- test '{name}' ---")
        log(f"  cmd:  {cmdline}")
        log(f"  log:  {log_path}")

        if dry_run:
            print(f"  {name:30} (dry-run)")
            phase.add(
                Check(
                    name=name,
                    status=PASS,
                    ip="other",
                    details=f"(dry) filter={filt} log={log_path}",
                    data={
                        "filter": filt,
                        "command": cmdline,
                        "rc": 0,
                        "duration_s": 0.0,
                        "env_overrides": env_overrides,
                        "log_file": str(log_path),
                        "testcases": {"passed": 0, "failed": 0, "skipped": 0, "failures": []},
                    },
                )
            )
            continue

        emit_running(name)

        t0 = time.time()
        passed = 0
        failed = 0
        skipped = 0
        failures: list[str] = []
        in_flight: set[str] = set()  # testcases that have started but not yet ended
        with log_path.open("w") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                env=test_env,
                cwd=str(tt_metal),
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                # All gtest output goes to the log file only — keep the console clean.
                logf.write(line)
                # Per-testcase accounting (dedup against double-counting from gtest's
                # end-of-run summary by tracking which tests are in-flight).
                if m := GTEST_RUN_RE.search(line):
                    in_flight.add(m.group(1))
                elif m := GTEST_OK_RE.search(line):
                    if m.group(1) in in_flight:
                        in_flight.discard(m.group(1))
                        passed += 1
                elif m := GTEST_FAIL_RE.search(line):
                    if m.group(1) in in_flight:
                        in_flight.discard(m.group(1))
                        failed += 1
                        failures.append(m.group(1))
                elif m := GTEST_SKIP_RE.search(line):
                    if m.group(1) in in_flight:
                        in_flight.discard(m.group(1))
                        skipped += 1
            rc = proc.wait()
        dt = time.time() - t0

        # Require passed > 0, as for pytest: a filter that matches nothing and a
        # run whose tests all skip both exit 0 without having tested anything.
        status = PASS if rc == 0 and failed == 0 and passed > 0 else FAIL
        suffix = f"({dt:.1f}s)"
        if status == FAIL:
            suffix += f"  passed={passed} failed={failed} skipped={skipped}"
        emit_result(name, status, suffix=suffix)
        details = (
            f"filter={filt} rc={rc} dur={dt:.1f}s passed={passed} failed={failed} skipped={skipped} log={log_path}"
        )
        phase.add(
            Check(
                name=name,
                status=status,
                details=details,
                data={
                    "filter": filt,
                    "command": cmdline,
                    "rc": rc,
                    "duration_s": dt,
                    "env_overrides": env_overrides,
                    "log_file": str(log_path),
                    "testcases": {"passed": passed, "failed": failed, "skipped": skipped, "failures": failures},
                },
                ip="other",
            )
        )
