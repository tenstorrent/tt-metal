#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run + post-process the TensorAccessor on-device microbenchmark for CI.

This is the CI/regression packaging of the existing accessor benchmark gtests
(tests/ttnn/unit_tests/gtests/accessor/test_accessor_benchmarks.cpp), part of
extending the runtime microbenchmark suite (issue #46305).

The gtest suite already sweeps the accessor cost across "different tensor
topologies": shape rank (2..7), sharded vs interleaved, and the 15 static/dynamic
argument configurations (bank coords / tensor shape / shard shape / num banks /
rank). Each benchmark kernel wraps the measured op (constructor, get_noc_addr, ...)
in a DeviceZoneScopedN("SHARDED_ACCESSOR_<bits>") loop, and the device profiler
records the per-iteration cycle cost.

This module:
  1. runs a pinned set of benchmark suites with the device profiler enabled
     (unless --no-run, in which case it just parses whatever result dirs exist),
  2. parses each per-(suite, rank) profiler CSV through the official parser
     (tracy.process_device_log / device_post_proc_config), reproducing the parse
     used by tests/ttnn/benchmark/python/test_accessor_benchmarks.py, and
  3. reduces the runs to one scalar metric per (suite, rank, config) = average
     cycles, then gates a small pinned subset against a per-arch golden.

The device-profiler cycle metric behaves identically on Wormhole and Blackhole
(no realtime profiler dependency). Goldens ship in record mode (null) until
populated on silicon, exactly like the op-to-op latency microbenchmark: a null
golden value is printed and skipped (passes), so the gate can be armed one
metric at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Full-chip address-calculation (get_noc_addr) benchmark. The gtest
# (AccessorFullChipBenchmarks.GetNocAddr) runs the kernel on the whole Tensix grid for
# one pinned all-static config per tensor topology, writing one profiler dir per
# topology (SetDeviceProfilerDir(res_path + "/" + test_name)). Each CSV holds a single
# DeviceZoneScopedN zone whose bitmask encodes the layout: 0000000 = interleaved,
# 0000001 = sharded (the Sharded bit; IsDram is not part of the zone name).
SUITES = {
    "GetNocAddr": {
        "gtest_suite": "AccessorTests/AccessorFullChipBenchmarks.GetNocAddr",
        "res_dir": "accessor_full_chip_get_noc_addr_benchmarks",
    },
}

# Topology profiler-dir name -> the SHARDED_ACCESSOR_<bits> zone the kernel emits for it.
# (Kept only for documentation; the parser probes both candidate zones so new topologies
# added to the gtest are picked up automatically.)
FULL_CHIP_TOPOLOGIES = {
    "interleaved_l1": "0000000",
    "interleaved_dram": "0000000",
    "sharded_l1_1d": "0000001",
    "sharded_l1_2d": "0000001",
    "sharded_l1_nd": "0000001",
    "sharded_dram_2d": "0000001",
    "sharded_dram_nd": "0000001",
}

# Candidate zone bitmasks probed per topology dir (interleaved vs sharded all-static).
ZONE_CANDIDATES = ["0000000", "0000001"]


def _binary_path(home: Path) -> Path:
    return home / "build" / "test" / "ttnn" / "unit_tests_ttnn_accessor"


def _natural_key(name: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


# --------------------------------------------------------------------------- #
# Running the gtest benchmark suites
# --------------------------------------------------------------------------- #
def _list_individual_tests(binary: Path, gtest_filter: str, env: dict) -> list[str]:
    """Expand a star filter (e.g. .../Constructor/*) into individual test ids."""
    result = subprocess.run(
        [str(binary), f"--gtest_filter={gtest_filter}", "--gtest_list_tests"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: --gtest_list_tests failed for {gtest_filter}: {result.stderr}", file=sys.stderr)
        return []
    tests, case = [], None
    for line in result.stdout.splitlines():
        if not line.strip() or "Running main() from" in line:
            continue
        if not line.startswith(" ") and line.endswith("."):
            case = line.rstrip(".")
        elif line.startswith(" ") and case:
            name = line.strip().split()[0]
            if name:
                tests.append(f"{case}.{name}")
    return tests


def run_suites(home: Path, suites: list[str]) -> None:
    binary = _binary_path(home)
    if not binary.exists():
        raise SystemExit(
            f"ERROR: accessor benchmark binary not found: {binary}\n"
            "Build it with: cmake --build build --target unit_tests_ttnn_accessor -j"
        )
    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    # Accessor benchmarks read one CSV per rank; MID_RUN_DUMP flushes the per-test
    # profiler dir the same way the existing ttnn benchmark wrapper does.
    env["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"

    for suite in suites:
        gtest_suite = SUITES[suite]["gtest_suite"]
        star = f"{gtest_suite}/*"
        individual = _list_individual_tests(binary, star, env)
        if not individual:
            # Fall back to running the whole star filter in one process.
            print(f"Running {star} (unexpanded)")
            subprocess.run([str(binary), f"--gtest_filter={star}"], env=env)
            continue
        print(f"Running {len(individual)} tests for {star}")
        for test_name in individual:
            subprocess.run([str(binary), f"--gtest_filter={test_name}"], env=env)


# --------------------------------------------------------------------------- #
# Parsing profiler CSVs -> per-(suite, rank, config) average cycles
# --------------------------------------------------------------------------- #
def _load_setup(zone_names: list[str]):
    """Build a device_post_proc_config setup that measures each zone's average
    cycle cost, identical to tests/ttnn/benchmark/python/test_accessor_benchmarks.py."""
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


def parse_suite(home: Path, suite: str) -> dict:
    """Return {metric_key: average_cycles} for one full-chip suite, per topology.

    metric_key = "<suite>.<topology>", e.g. "GetNocAddr.sharded_dram_nd". The topology
    is the profiler-dir name; the average is taken across all Tensix cores (the parser
    aggregates "across": "core"). Missing dirs/zones are skipped (record-mode
    robustness), not fatal.
    """
    try:
        from tracy.process_device_log import import_log_run_stats
    except ImportError as exc:  # pragma: no cover - env guard
        raise SystemExit(
            "Could not import tracy.process_device_log; run under the tt-metal "
            f"profiler environment (PYTHONPATH must include tools/). Original error: {exc}"
        )

    cfg = SUITES[suite]
    zone_names = [f"SHARDED_ACCESSOR_{c}" for c in ZONE_CANDIDATES]
    setup = _load_setup(zone_names)

    results_dir = home / cfg["res_dir"]
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
        # Read the cross-core weighted average that generate_device_level_summary stores
        # under the synthetic "DEVICE" core (cores["DEVICE"]["analysis"][zone]["stats"]).
        # This is the whole-chip number for a full-chip run; the per-core stats live under
        # cores[<core>]["riscs"]["TENSIX"]["analysis"] instead.
        try:
            device_analysis = stats["devices"][0]["cores"]["DEVICE"].get("analysis", {})
        except (KeyError, IndexError, AttributeError):
            print(f"WARNING: no DEVICE-level analysis in {csv_path}", file=sys.stderr)
            continue
        # One zone per topology dir; probe both candidate layouts and take whichever exists.
        for zone in zone_names:
            zstats = device_analysis.get(zone)
            if zstats and "stats" in zstats:
                metrics[f"{suite}.{topology}"] = float(zstats["stats"]["Average"])
                break
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
        if golden_value is None:
            print(
                f"[record mode] golden '{key}' not populated in {golden_path.name}; "
                f"measured={_fmt(measured)} cycles. Populate it to enable this gate. Passing."
            )
            continue
        if measured is None or measured != measured:  # None or NaN
            print(f"gate: FAIL — measured {key} is missing/NaN (no samples extracted)", file=sys.stderr)
            failed = True
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
        help="skip running the gtest; only parse existing result dirs under $TT_METAL_HOME",
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

    home = Path(os.environ.get("TT_METAL_HOME", "."))

    if not args.no_run:
        run_suites(home, args.suites)

    metrics: dict[str, float] = {}
    for suite in args.suites:
        metrics.update(parse_suite(home, suite))

    if metrics:
        width = max(len(k) for k in metrics)
        print("tensor_accessor benchmark metrics (average cycles):")
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
