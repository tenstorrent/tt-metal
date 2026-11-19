# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 3: the device tests — deployment gtests and pytest stress tests.

TESTS and PYTESTS define the steps a tier can list under ``tests`` in tiers.py.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
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
# need a wildcard for the two-digit index; otherwise gtest matches 0 tests, exits
# 0, and the test is silently recorded as PASS without running.
# eth_bandwidth keeps a trailing wildcard so it also picks up BandwidthBidir.
# Each entry is the gtest filter plus, optionally, environment set for that run
# only (recorded in the check as env_overrides and prefixed to its command line).
TESTS = {
    "eth_link_up": {"filter": "*TensixDeploymentEthernet*LinkUp"},
    "eth_bandwidth": {"filter": "*TensixDeploymentEthernet*Bandwidth*"},
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
#
# didt_matmul_galaxy: dI/dt worst-case matmul across the full galaxy mesh (8x4),
# 1000 workload iterations with a determinism check every 50 (20 checks) —
# catches droop-induced non-determinism the short deployment gtests don't.
# Measured 545 s in-suite on a BH galaxy (bh-glx6u-28, 2026-09-01):
# iterations ~69 ms each, each determinism check ~13 s (reads the full
# 16k x 16k output back from all 32 chips), plus ~3.5 min cold 32-chip
# bring-up/compile. Check density is the dominant cost: the same test at 5000 iterations
# across all four dtype variants (-k galaxy) measures ~2 h. Note a longer
# soak catches more: a marginal chip on the bench unit failed 2/10 checks at
# 5000 iterations but passed shorter runs — bump iterations for deep triage.
PYTESTS = {
    "didt_matmul_galaxy": {
        "target": "tests/didt/test_minimal_matmul.py::test_minimal_matmul",
        "args": [
            "-k",
            "galaxy and bf16_HiFi2",
            "--didt-workload-iterations",
            "1000",
            "--determinism-check-interval",
            "50",
            "-s",
            # pytest.ini sets a repo-wide 300s pytest-timeout; this run always
            # exceeds it (~545s measured). 1200s keeps >2x headroom while still
            # bounding a standalone deploy run if a device test wedges (the
            # test-infrastructure wrapper adds its own wall-clock timeout, but
            # run_diag.sh / container entrypoints invoke this script directly).
            "--timeout",
            "1200",
        ],
    },
}


# gtest progress markers — used for per-testcase pass/fail accounting.
# Matches only on lines with a "Suite.Test" pattern, so summary lines like
# "[  FAILED  ] 1 test, listed below:" don't get counted as testcases.
GTEST_RUN_RE = re.compile(r"\[\s*RUN\s*\]\s+(\S+\.\S+)")
GTEST_OK_RE = re.compile(r"\[\s*OK\s*\]\s+(\S+\.\S+)")
GTEST_FAIL_RE = re.compile(r"\[\s*FAILED\s*\]\s+(\S+\.\S+)")


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


def run_tests(tt_metal: Path, names: Sequence[str], phase: Phase, dry_run: bool, logs_dir: Path) -> None:
    """Run the named TESTS / PYTESTS entries, in order."""
    binary = next(
        (tt_metal / c for c in DEPLOYMENT_BIN_CANDIDATES if (tt_metal / c).is_file()),
        None,
    )
    if binary is None and any(n in TESTS for n in names):
        # Pytest-based entries don't need the gtest binary — record the FAIL
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
                        "testcases": {"passed": 0, "failed": 0, "failures": []},
                    },
                )
            )
            continue

        emit_running(name)

        t0 = time.time()
        passed = 0
        failed = 0
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
            rc = proc.wait()
        dt = time.time() - t0

        status = PASS if rc == 0 and failed == 0 else FAIL
        suffix = f"({dt:.1f}s)"
        if failed:
            suffix += f"  passed={passed} failed={failed}"
        emit_result(name, status, suffix=suffix)
        details = f"filter={filt} rc={rc} dur={dt:.1f}s " f"passed={passed} failed={failed} log={log_path}"
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
                    "testcases": {"passed": passed, "failed": failed, "failures": failures},
                },
                ip="other",
            )
        )
