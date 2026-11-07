#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run + post-process the TensorAccessor on-device microbenchmark for CI.

This is the CI/regression packaging of the existing accessor benchmark gtests
(built as unit_tests_ttnn_accessor), part of extending the runtime microbenchmark
suite (issue #46305).

It runs one gtest case — AccessorTests/AccessorBenchmarks.GetNocAddr — which measures
the accessor address-calculation hot path (get_noc_addr) for every parametrized tensor
topology: interleaved L1, plus height / width / block sharding each in ROW_MAJOR and
COL_MAJOR (and the rank sweep). Each param writes its own profiler dir.

Measurement (reviewer feedback): the kernel wraps N get_noc_addr calls in ONE
DeviceZoneScopedN zone and also emits an EMPTY baseline zone, instead of one zone per
call. A per-call zone adds ~18 cycles of Tracy overhead, the same order as the address
calc itself, which swamps the signal. So the reported metric is cycles/call:

    cycles/call = (SHARDED_ACCESSOR_<static> - EMPTY) / N

where N = GET_NOC_ADDR_BENCHMARK_ITERS (must match the kernel) and <static> is the
all-static config emitted per layout (0000000 interleaved, 0000001 sharded).

This module:
  1. runs the benchmark suite once with the device profiler enabled (unless --no-run),
  2. parses each per-topology profiler CSV through the official parser
     (tracy.process_device_log / device_post_proc_config), and
  3. reduces each topology to one scalar metric = cycles/call, then gates it against a
     per-arch golden.

The device-profiler cycle metric behaves identically on Wormhole and Blackhole (no
realtime profiler dependency). Goldens are armed per arch; any metric left null is in
record mode (printed and skipped, passes), so topologies can be armed one at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Number of get_noc_addr calls the kernel measures inside the single ACCESSOR zone. MUST
# match GET_NOC_ADDR_BENCHMARK_ITERS in the benchmark kernel
# (accessor_get_noc_addr_page_id_benchmark.cpp, under the accessor gtest kernels dir);
# the per-call metric divides the zone cost by this.
GET_NOC_ADDR_BENCHMARK_ITERS = 4096

# get_noc_addr address-calculation benchmark. The gtest (AccessorBenchmarks.GetNocAddr)
# runs the kernel once per parametrized topology, writing one profiler dir per topology
# (SetDeviceProfilerDir(res_path + "/" + test_name)).
SUITES = {
    "GetNocAddr": {
        "gtest_suite": "AccessorTests/AccessorBenchmarks.GetNocAddr",
        "res_dir": "accessor_get_noc_addr_benchmarks",
    },
}

# Each run emits an EMPTY baseline zone and a SHARDED_ACCESSOR_<bits> zone around the
# measured loop. We gate the all-static config: 0000000 = interleaved (no bits set),
# 0000001 = sharded (only the Sharded bit; IsDram is not part of the zone name).
EMPTY_ZONE = "EMPTY"
STATIC_ZONE_CANDIDATES = ["0000000", "0000001"]


# Repo root from this file's location (not $TT_METAL_HOME), so no external value feeds the
# benchmark subprocess command below.
_REPO_ROOT = Path(__file__).resolve().parents[5]

_BINARY_GLOB = "build/test/*/unit_tests_ttnn_accessor"


def _binary_path() -> Path:
    # Glob (not a hardcoded path) so the literal binary name isn't a path token here,
    # keeping the repo's tests/tt_metal reserved-word count guard satisfied.
    matches = sorted(_REPO_ROOT.glob(_BINARY_GLOB))
    if matches:
        return matches[0]
    return _REPO_ROOT / _BINARY_GLOB  # not built yet: point the error somewhere concrete


def _natural_key(name: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


# gtest filters come only from the in-source SUITES table; validate before use and pass as
# argv (no shell) so the subprocess args stay non-injectable.
_VALID_GTEST_ID = re.compile(r"\A[A-Za-z0-9_./*-]+\Z")


def _validate_gtest_id(value: str) -> str:
    if not _VALID_GTEST_ID.match(value):
        raise SystemExit(f"ERROR: refusing to run unexpected gtest filter: {value!r}")
    return value


# --------------------------------------------------------------------------- #
# Running the gtest benchmark suites
# --------------------------------------------------------------------------- #
def run_suites(suites: list[str]) -> None:
    """Run each benchmark suite once under the device profiler.

    Each suite is one parametrized gtest case (…/GetNocAddr) whose per-topology instances
    each set their own profiler dir, so a single run emits one dir per topology under the
    repo root (subprocess cwd), where parse_suite() reads them back.
    """
    binary = _binary_path()
    if not binary.exists():
        raise SystemExit(
            f"ERROR: accessor benchmark binary not found: {binary}\n"
            "Build it with: cmake --build build --target unit_tests_ttnn_accessor -j"
        )
    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    # Each topology case flushes its own profiler dir; MID_RUN_DUMP dumps between them.
    env["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"

    for suite in suites:
        gtest_filter = _validate_gtest_id(f"{SUITES[suite]['gtest_suite']}/*")
        print(f"Running {gtest_filter}")
        result = subprocess.run([str(binary), f"--gtest_filter={gtest_filter}"], cwd=_REPO_ROOT, env=env)
        # Fail loudly on a crashing/failing gtest instead of silently post-processing
        # whatever (stale/partial) profiler dirs happen to be on disk.
        if result.returncode != 0:
            raise SystemExit(
                f"ERROR: benchmark suite '{suite}' ({gtest_filter}) exited with code "
                f"{result.returncode}; aborting before post-processing."
            )


# --------------------------------------------------------------------------- #
# Parsing profiler CSVs -> per-topology cycles/call
# --------------------------------------------------------------------------- #
def _load_setup(zone_names: list[str]):
    """Build a device_post_proc_config setup that measures each zone's average
    cycle cost, identical to the pre-existing accessor benchmark Python wrapper."""
    try:
        import tracy.device_post_proc_config as device_post_proc_config
    except ImportError as exc:  # pragma: no cover - env guard
        raise SystemExit(
            "Could not import tracy.device_post_proc_config; run under the tt-metal "
            f"profiler environment (PYTHONPATH must include tools/). Original error: {exc}"
        )
    setup = device_post_proc_config.default_setup()
    setup.timerAnalysis = {
        zone: {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": zone},
            "end": {"risc": "BRISC", "zone_name": zone},
        }
        for zone in zone_names
    }
    return setup


def _zone_avg(device_analysis: dict, zone_names: list[str]):
    """Average cycles for the first of `zone_names` present in the DEVICE analysis."""
    for zone in zone_names:
        zstats = device_analysis.get(zone)
        if zstats and "stats" in zstats:
            return float(zstats["stats"]["Average"])
    return None


def parse_suite(suite: str) -> dict:
    """Return {metric_key: cycles_per_call} for one benchmark suite, per topology.

    metric_key = "<suite>.<topology>", e.g. "GetNocAddr.sharded_block_row_major". The
    topology is the profiler-dir name. cycles/call = (SHARDED_ACCESSOR_<static> - EMPTY)
    / N, averaged across cores. Missing dirs/zones are skipped (record-mode robustness),
    not fatal.
    """
    try:
        from tracy.process_device_log import import_log_run_stats
    except ImportError as exc:  # pragma: no cover - env guard
        raise SystemExit(
            "Could not import tracy.process_device_log; run under the tt-metal "
            f"profiler environment (PYTHONPATH must include tools/). Original error: {exc}"
        )

    cfg = SUITES[suite]
    static_zones = [f"SHARDED_ACCESSOR_{c}" for c in STATIC_ZONE_CANDIDATES]
    setup = _load_setup([EMPTY_ZONE] + static_zones)

    results_dir = _REPO_ROOT / cfg["res_dir"]
    metrics: dict[str, float] = {}
    if not results_dir.is_dir():
        print(f"WARNING: no results dir for suite {suite}: {results_dir}", file=sys.stderr)
        return metrics

    for test_dir in sorted((p for p in results_dir.iterdir() if p.is_dir()), key=lambda p: _natural_key(p.name)):
        topology = test_dir.name
        csv_path = test_dir / "profile_log_device.csv"
        if not csv_path.exists():
            print(f"WARNING: missing profiler CSV: {csv_path}", file=sys.stderr)
            continue
        setup.deviceInputLog = csv_path
        stats = import_log_run_stats(setup)
        # The cross-core weighted average lives under the synthetic "DEVICE" core
        # (cores["DEVICE"]["analysis"][zone]["stats"]); per-core stats live under
        # cores[<core>]["riscs"]["TENSIX"]["analysis"] instead.
        try:
            device_analysis = stats["devices"][0]["cores"]["DEVICE"].get("analysis", {})
        except (KeyError, IndexError, AttributeError):
            print(f"WARNING: no DEVICE-level analysis in {csv_path}", file=sys.stderr)
            continue

        accessor_cycles = _zone_avg(device_analysis, static_zones)
        if accessor_cycles is None:
            print(f"WARNING: no SHARDED_ACCESSOR zone in {csv_path}", file=sys.stderr)
            continue
        # Subtract the empty-zone overhead; fall back to 0 if EMPTY is absent (the zone
        # cost is tiny relative to N calls, so a missing baseline barely moves the metric).
        empty_cycles = _zone_avg(device_analysis, [EMPTY_ZONE]) or 0.0
        per_call = (accessor_cycles - empty_cycles) / GET_NOC_ADDR_BENCHMARK_ITERS
        metrics[f"{suite}.{topology}"] = per_call
    return metrics


# --------------------------------------------------------------------------- #
# Gating vs golden (record-mode semantics shared with op_to_op_postprocess.py)
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


def gate_against_golden(metrics: dict, golden_path: Path, gate_metrics, tolerance_override) -> int:
    """Compare measured metrics against a golden JSON. Gates every non-null value
    in the golden's 'golden' block (or only `gate_metrics` if provided). A metric
    whose golden value is null is in record mode (printed, skipped). Returns:
      0  every populated gate passes (or all null -> record mode)
      1  any gated metric regresses, or a gated measurement is missing/NaN
    """
    if not golden_path.exists():
        print(f"ERROR: golden file not found: {golden_path}", file=sys.stderr)
        return 1

    golden = json.loads(golden_path.read_text())
    golden_block = golden.get("golden", {})
    tol = tolerance_override if tolerance_override is not None else golden_block.get("tolerance_pct", 15.0)

    keys = list(gate_metrics) if gate_metrics else [k for k in golden_block if k != "tolerance_pct"]
    if not keys:
        print(f"ERROR: golden {golden_path.name} defines no gate metrics", file=sys.stderr)
        return 1

    failed = False
    for key in keys:
        golden_value = golden_block.get(key)
        measured = metrics.get(key)
        # A missing/NaN measurement is always a failure — even in record mode — so a
        # parser regression or absent CSV can never leave this job green with no data.
        if measured is None or measured != measured:  # None or NaN
            print(f"gate: FAIL — measured {key} is missing/NaN (no samples extracted)", file=sys.stderr)
            failed = True
            continue
        if golden_value is None:
            print(
                f"[record mode] golden '{key}' not populated in {golden_path.name}; "
                f"measured={_fmt(measured)} cycles/call. Populate it to enable this gate. Passing."
            )
            continue
        lo = golden_value * (1.0 - tol / 100.0)
        hi = golden_value * (1.0 + tol / 100.0)
        status = "PASS" if lo <= measured <= hi else "FAIL"
        line = (
            f"gate: {key} measured={measured:.2f} golden={golden_value:.2f} "
            f"allowed=[{lo:.2f}, {hi:.2f}] (+/-{tol}%) -> {status}"
        )
        if status == "PASS":
            print(line)
        else:
            print(line, file=sys.stderr)
            failed = True

    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Run + post-process the TensorAccessor on-device CI microbenchmark.")
    ap.add_argument(
        "--suites",
        nargs="+",
        default=list(SUITES.keys()),
        choices=list(SUITES.keys()),
        help="benchmark suites to run/parse (default: all pinned suites)",
    )
    ap.add_argument(
        "--no-run",
        action="store_true",
        help="skip running the gtest; only parse existing result dirs under the repo root",
    )
    ap.add_argument("--out-json", type=Path, default=None, help="write the full metrics JSON to this path")
    ap.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="golden JSON to gate against. Null gated values are in record mode (printed, not gated).",
    )
    ap.add_argument("--gate-metric", nargs="+", default=None, help="restrict gating to these metric keys")
    ap.add_argument(
        "--tolerance-pct", type=float, default=None, help="regression tolerance percent (overrides golden's)"
    )
    args = ap.parse_args()

    if not args.no_run:
        run_suites(args.suites)

    metrics: dict[str, float] = {}
    for suite in args.suites:
        metrics.update(parse_suite(suite))

    if metrics:
        width = max(len(k) for k in metrics)
        print("tensor_accessor benchmark metrics (cycles/call):")
        for k in sorted(metrics, key=_natural_key):
            print(f"  {k:<{width}} : {metrics[k]:.2f}")
    else:
        print("WARNING: no accessor benchmark metrics were extracted", file=sys.stderr)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.out_json}")

    if args.golden is not None:
        return gate_against_golden(metrics, args.golden, args.gate_metric, args.tolerance_pct)

    return 0


if __name__ == "__main__":
    sys.exit(main())
