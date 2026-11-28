# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 3: the device tests — deployment gtests and pytest stress tests.

TESTS and PYTESTS define the steps a tier can list under ``tests`` in tiers.py.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence

from .common import emit_result, emit_running, log
from .galaxy import format_link_end, name_chips
from .model import FAIL, PASS, WARN, Check, Phase

# tt-metal deployment test binary candidate paths relative to repo root.
# build_metal.sh defaults to build_Release/; raw cmake invocations often use build/.
# Check names: each test's check is named by its key below.
ETH_LINK_UP = "eth_link_up"
ETH_BANDWIDTH = "eth_bandwidth"
ETH_DATA_INTEGRITY = "eth_data_integrity"
NOC_INTEGRITY = "noc_integrity"
PCIE_READ = "pcie_read"
PCIE_WRITE = "pcie_write"
GDDR_FAST = "gddr_fast"
GDDR_FULL = "gddr_full"
DIDT_MATMUL_GALAXY = "didt_matmul_galaxy"
GDDR_ACT_C3 = "gddr_act_c3"
GDDR_ACT_ALL = "gddr_act_all"

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
# eth_data_integrity is one way only: the both-ways test mismatches every word on healthy links.
# Each entry: gtest filter, the check's ip, and optional env for that run only (data.env_overrides).
SLOW_DISPATCH = {"TT_METAL_SLOW_DISPATCH_MODE": "1"}
TESTS = {
    ETH_LINK_UP: {"filter": "*TensixDeploymentEthernet*LinkUp", "env": SLOW_DISPATCH, "ip": "eth"},
    ETH_BANDWIDTH: {"filter": "*TensixDeploymentEthernet*Bandwidth*", "env": SLOW_DISPATCH, "ip": "eth"},
    ETH_DATA_INTEGRITY: {"filter": "*TensixDeploymentEthernet03DataIntegrityDram", "env": SLOW_DISPATCH, "ip": "eth"},
    NOC_INTEGRITY: {"filter": "*TensixDeploymentNoc*", "env": SLOW_DISPATCH, "ip": "asic"},
    GDDR_FAST: {
        "filter": "*DramDeployment_PersistentOptimalWorkersAllDramBanks",
        "env": {"DRAM_TEST_FAST": "1"},
        "ip": "gddr",
    },
    GDDR_FULL: {"filter": "*DramDeployment_*", "ip": "gddr"},
}

# Host <-> chip 0 PCIe bandwidth as the deployment test measures it: mem_bench (<build>/tools),
# one core reading 1 GB from the host's hugepage, then one writing.
MEM_BENCH_CANDIDATES = [str(Path(c).parents[2] / "tools" / "mem_bench") for c in DEPLOYMENT_BIN_CANDIDATES]
_BENCH_ARGS = "size:1073741824/page_size:32768/readers:{}/writers:{}/host_copy:0/iterations:5/manual_time"
BENCHES = {
    PCIE_READ: f"Device Reading Host/{_BENCH_ARGS.format(1, 0)}",
    PCIE_WRITE: f"Device Writing Host/{_BENCH_ARGS.format(0, 1)}",
}

# Without a mesh graph descriptor the deployment fixtures' suite setup fails on a Galaxy
# and every test skips. The deployment test's runner passes this one; a caller's own wins.
MGD_ENV = "TT_MESH_GRAPH_DESC_PATH"
MGD = "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"

# Every variable any gtest entry overrides. A check's env_overrides reports the
# value each of them has for that run, ambient or overridden, so a variable
# inherited from the caller's environment is on the record too.
GTEST_ENV_KEYS = tuple(dict.fromkeys([MGD_ENV, *(k for spec in TESTS.values() for k in spec.get("env", {}))]))


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
    DIDT_MATMUL_GALAXY: {
        "ip": "asic",
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
    # SDPA K-read mimic: ctrl 3, then all 8 ctrls, 300 s each (~10.3 min). Catches the 64 MB alias
    # (bit 26) DramDeployment misses. A hit skips the remaining passes. Timeout is a marker on the test.
    GDDR_ACT_C3: {
        "ip": "gddr",
        "target": "tests/dram/test_dram_act_patterns.py::test_dram_act_c3_screen",
        "args": ["-s"],
    },
    # Superset of gddr_act_c3: all 8 ctrls, then corner ctrls 7, 3, 4, 0 alone, 300 s each (~25.5 min).
    GDDR_ACT_ALL: {
        "ip": "gddr",
        "target": "tests/dram/test_dram_act_patterns.py::test_dram_act_all_ctrls_screen",
        "args": ["-s"],
    },
}


# gtest progress markers — used for per-testcase pass/fail accounting.
# Matches only on lines with a "Suite.Test" pattern, so summary lines like
# "[  FAILED  ] 1 test, listed below:" don't get counted as testcases.
GTEST_RUN_RE = re.compile(r"\[\s*RUN\s*\]\s+(\S+\.\S+)")
GTEST_OK_RE = re.compile(r"\[\s*OK\s*\]\s+(\S+\.\S+)")
GTEST_FAIL_RE = re.compile(r"\[\s*FAILED\s*\]\s+(\S+\.\S+)")
GTEST_SKIP_RE = re.compile(r"\[\s*SKIPPED\s*\]\s+(\S+\.\S+)")
SETUP_THROW_RE = re.compile(r'exception with description "(.*)" thrown in SetUpTestSuite')


# pytest progress markers. Counts come from the end-of-run summary line
# ("===== 3 passed, 1 failed, 2 deselected in 600.12s ====="); failing node ids
# from the short-summary "FAILED tests/..." lines and, for fixture/collection
# errors, the "ERROR tests/..." lines.
PYTEST_COUNT_RE = re.compile(r"(\d+) (passed|failed|error)")
PYTEST_FAILED_LINE_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)")

# Faults log at critical or error level. Eth tests announce each link first ("sender device
# id: 3 (0000:46:00.0, ubb: 2, chip: 6), ..." then "sender core: (x=0,y=4), ..."; the eth core's
# y is its channel). A fault line naming no chip belongs to the link last announced;
# "Failing links:" rows name theirs by device id.
FAULT_LINE_RE = re.compile(r"\|\s*(?:critical|error)\s*\|", re.IGNORECASE)
ETH_PAIR_RE = re.compile(r"sender device id: (\d+) \(([^,]+),.*receiver device id: (\d+) \(([^,]+),")
ETH_CORES_RE = re.compile(r"sender core:? \(x=\d+,y=(\d+)\), receiver core:? \(x=\d+,y=(\d+)\)")
ETH_FAILING_RE = re.compile(r"Sender device (\d+), receiver device (\d+),")
DETAIL_LINKS = 2  # links named in a check's details; data["links"] has them all


def fault_parts(lines) -> tuple[list[str], list[str]]:
    """The chips and ETH links (both ends) a test log's fault lines name."""
    bdf: dict[str, str] = {}
    parts: list[str] = []
    links: list[str] = []
    ends = link = None
    for line in lines:
        if m := ETH_PAIR_RE.search(line):
            bdf[m[1]], bdf[m[3]] = m[2], m[4]
            ends = (m[2], m[4])
        if m := ETH_CORES_RE.search(line):
            if f := ETH_FAILING_RE.search(line):
                ends = (bdf.get(f[1], f"device {f[1]}"), bdf.get(f[2], f"device {f[2]}"))
            link = (*ends, int(m[1]), int(m[2])) if ends else None
        if not FAULT_LINE_RE.search(line) or line.rstrip().endswith(":"):  # a header for the rows after it
            continue
        if chips := name_chips([line]):
            parts += chips
        elif link:
            s, r, s_ch, r_ch = link
            links.append(f"{format_link_end(s, s_ch)} <-> {format_link_end(r, r_ch)}")
            parts += name_chips([s, r])
    return list(dict.fromkeys(parts)), list(dict.fromkeys(links))


def grade(rc: int, passed: int, failed: int, log_path: Path, details: str) -> tuple[str, str]:
    """PASS, FAIL, or WARN "not run" (why, from its log) if it passed and failed nothing."""
    if passed or failed or rc not in (0, 1, 4, 5):
        return (PASS if rc == 0 and not failed else FAIL), details
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    reasons = [f"suite setup failed: {m.group(1)}" for line in lines if (m := SETUP_THROW_RE.search(line))]
    for i, line in enumerate(lines):
        if line.endswith(": Skipped"):
            msg = next((n.strip() for n in lines[i + 1 : i + 2] if n.strip() and not n.startswith("[")), "")
            reasons.append(f"skipped: {msg or line}")
        elif line.startswith("E   "):
            reasons.append(line[4:].strip())
    return WARN, f"not run: {(reasons or ['no test matched'])[0]} {details}"


def name_faults(log_path: Path, details: str, data: dict, hits=()) -> tuple[str, list[str]]:
    """A failed run's details, links its log names first, and its chips (log and pytest ``hits``)."""
    try:
        parts, links = fault_parts(log_path.read_text(errors="replace").splitlines())
    except OSError:
        parts, links = [], []
    parts = list(dict.fromkeys([*parts, *name_chips(str(h.get("bdf", "")) for h in hits if isinstance(h, dict))]))
    if links:
        data["links"] = links
        more = f" (+{len(links) - DETAIL_LINKS} more)" if len(links) > DETAIL_LINKS else ""
        details = f"link {'; link '.join(links[:DETAIL_LINKS])}{more} {details}"
    return details, parts


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
                    ip=spec["ip"],
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
    # Only the output dir survives the container.
    result_path = logs_dir / f"{name}.json"
    test_env["DIAG_RESULT_JSON"] = str(result_path)
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
                ip=spec["ip"],
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

    result_path.unlink(missing_ok=True)
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

    # Nothing passed (rc=5: 0 collected; rc=4: conftest import failed): graded not run.
    details = f"target={target} rc={rc} dur={dt:.1f}s " f"passed={passed} failed={failed} log={log_path}"
    status, details = grade(rc, passed, failed, log_path, details)
    suffix = f"({dt:.1f}s)"
    if failed:
        suffix += f"  passed={passed} failed={failed}"
    emit_result(name, status, suffix=suffix)
    data = {
        "target": target,
        "command": cmdline,
        "rc": rc,
        "duration_s": dt,
        "log_file": str(log_path),
        "testcases": {"passed": passed, "failed": failed, "failures": failures},
    }
    summary = read_result_summary(result_path)
    if summary is not None:
        data["result_file"] = str(result_path)
        data["summary"] = summary
        if summary.get("details"):
            details += f" {summary['details']}"
    parts: list[str] = []
    if status == FAIL:
        details, parts = name_faults(log_path, details, data, (summary or {}).get("hits") or ())
    phase.add(Check(name=name, status=status, details=details, data=data, ip=spec["ip"], parts=parts))


def read_result_summary(path: Path) -> dict | None:
    """``summary`` from a test's result JSON, or None. Never affects pass/fail."""
    try:
        summary = json.loads(path.read_text()).get("summary")
    except (OSError, ValueError, AttributeError):
        return None
    return summary if isinstance(summary, dict) else None


def run_tests(
    tt_metal: Path, names: Sequence[str], phase: Phase, dry_run: bool, logs_dir: Path, name_suffix: str = ""
) -> None:
    """Run the named TESTS / PYTESTS entries, in order. ``name_suffix`` keeps a rerun's checks and logs apart."""
    binary = next(
        (tt_metal / c for c in DEPLOYMENT_BIN_CANDIDATES if (tt_metal / c).is_file()),
        None,
    )
    bench = next((tt_metal / c for c in MEM_BENCH_CANDIDATES if (tt_metal / c).is_file()), None)
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
    env.setdefault(MGD_ENV, str(tt_metal / MGD))
    if binary is not None:
        # LD_LIBRARY_PATH: match the build dir of the binary we picked
        env.setdefault("LD_LIBRARY_PATH", str(binary.parent.parent.parent / "lib"))

    logs_dir.mkdir(parents=True, exist_ok=True)

    for test in names:
        name = f"{test}{name_suffix}"
        if test in PYTESTS:
            run_pytest_test(name, PYTESTS[test], tt_metal, env, phase, dry_run, logs_dir)
            continue
        if test in BENCHES:
            run_bench(test, name, bench, tt_metal, env, phase, dry_run, logs_dir)
            continue
        if binary is None:
            continue
        spec = TESTS[test]
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
                    ip=spec["ip"],
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
        failed += len(in_flight)
        failures += sorted(in_flight)
        dt = time.time() - t0

        # No test matched, all skipped, or the suite setup threw: not run, naming why. All skipped is a FAIL.
        details = (
            f"filter={filt} rc={rc} dur={dt:.1f}s passed={passed} failed={failed} skipped={skipped} log={log_path}"
        )
        status, details = grade(rc, passed, failed, log_path, details)
        if details.startswith("not run: skipped:"):
            status = FAIL  # nothing was tested; the verdict blames no part for a test that did not run
        suffix = f"({dt:.1f}s)"
        if status != PASS:
            suffix += f"  passed={passed} failed={failed} skipped={skipped}"
        emit_result(name, status, suffix=suffix)
        data = {
            "filter": filt,
            "command": cmdline,
            "rc": rc,
            "duration_s": dt,
            "env_overrides": env_overrides,
            "log_file": str(log_path),
            "testcases": {"passed": passed, "failed": failed, "skipped": skipped, "failures": failures},
        }
        parts: list[str] = []
        if status == FAIL and not details.startswith("not run:"):
            details, parts = name_faults(log_path, details, data)
        phase.add(Check(name=name, status=status, details=details, data=data, ip=spec["ip"], parts=parts))


def run_bench(
    test: str,
    name: str,
    binary: Path | None,
    tt_metal: Path,
    env: dict,
    phase: Phase,
    dry_run: bool,
    logs_dir: Path,
) -> None:
    """Run one BENCHES entry; PASS if no benchmark errored. Records GB/s per benchmark."""
    filt = BENCHES[test]
    if binary is None and not dry_run:
        details = f"none of {MEM_BENCH_CANDIDATES} found under {tt_metal}"
        phase.add(Check(name=name, status=FAIL, details=details, ip="pcie"))
        return
    log_path = logs_dir / f"{name}.log"
    out_path = logs_dir / f"{name}.json"
    cmd = [
        str(binary or "mem_bench"),
        f"--benchmark_filter={filt}",
        "--device-id=0",
        f"--benchmark_out={out_path}",
        "--benchmark_out_format=json",
    ]
    cmdline = shlex.join(cmd)
    log(f"--- test '{name}' (mem_bench) ---")
    log(f"  cmd:  {cmdline}")
    log(f"  log:  {log_path}")
    data: dict = {"filter": filt, "command": cmdline, "log_file": str(log_path)}
    if dry_run:
        print(f"  {name:30} (dry-run)")
        phase.add(Check(name=name, status=PASS, details=f"(dry) filter={filt} log={log_path}", data=data, ip="pcie"))
        return

    emit_running(name)
    out_path.unlink(missing_ok=True)
    t0 = time.time()
    with log_path.open("w") as logf:
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(tt_metal)).returncode
    dt = time.time() - t0
    try:
        runs = json.loads(out_path.read_text()).get("benchmarks") or []
    except (OSError, ValueError, AttributeError):
        runs = []
    errors = [r.get("error_message") or "skipped" for r in runs if r.get("error_occurred") or r.get("skipped")]
    gbps = {
        r["name"]: round(r["dev_bw"] / 1e9, 2) for r in runs if r.get("dev_bw")
    }  # the chip's kernels' rate, bytes/s
    status = PASS if rc == 0 and runs and not errors else FAIL
    data.update(rc=rc, duration_s=dt, result_file=str(out_path), gbps=gbps, errors=errors)
    rate = ", ".join(f"{v} GB/s" for v in gbps.values()) or "no result"
    emit_result(name, status, suffix=f"({dt:.1f}s)  chip 0: {rate}")
    details = (
        f"{'; '.join(errors) + ' ' if errors else ''}chip 0: {rate} rc={rc} dur={dt:.1f}s "
        f"passed={len(runs) - len(errors)} failed={len(errors)} log={log_path}"
    )
    phase.add(Check(name=name, status=status, details=details, data=data, ip="pcie"))
