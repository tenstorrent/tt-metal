#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CI driver + regression gate for the local JIT-build throughput microbenchmark.

Drives DISABLED_TensixCompileStress (tests/tt_metal/tt_metal/jit_build/
test_compile_stress.cpp) as a *local* host-compile benchmark against the real
attached device (TT_METAL_COMPILE_STRESS_MOCK=0), with no remote JIT compile
server, a fixed kernel count, and per-repetition unique {id, seed} tuples + an
isolated JIT cache so every repetition is a genuine cold compile of N kernels.
The device is only used for build config (arch/grid); kernels are compiled, not
dispatched, so device open/close stays outside the timed compile section.

Each repetition runs the gtest in its own process (fresh in-memory JitBuildCache)
and emits a result JSON via TT_METAL_COMPILE_STRESS_OUTPUT. We take the *fastest*
repetition (min wall-clock -> max throughput); a flake almost never makes the
compiler faster, so the min rejects upward CPU-contention noise on shared runners.

The fastest repetition is then compared against a per-arch golden:
  * A ``null`` golden metric stays in RECORD MODE - the measured value is printed
    but the job is not gated on it (lets this land before CI-SKU numbers exist).
  * A non-null golden metric is GATED: the job fails if the measured compile time
    regresses beyond ``tolerance_pct`` (i.e. gets slower).

Exit code is non-zero on a gated regression, a missing/failed repetition, or a
malformed golden.

Example (matches the runtime-perf pipeline invocation):
  ./tests/tt_metal/tt_metal/jit_build/compile_stress_ci.py \
      --arch "$ARCH_NAME" --num-kernels 1000 --repetitions 5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# We benchmark the real attached device (TT_METAL_COMPILE_STRESS_MOCK=0): the
# runtime-perf SKUs have hardware, and the test's mock real->mock transition
# throws once MeshDispatchFixture::SetUpTestSuite has opened the device. Strip
# simulator/emulator overrides so we measure the plain local-compile path.
SIM_FORCING_ENV_VARS = (
    "TT_METAL_SIMULATOR",
    "TT_METAL_EMULE_MODE",
    "TT_METAL_MOCK_CLUSTER_DESC_PATH",
)

# Fixed base seed so the compiled kernel set is reproducible across runs. Each
# repetition uses BASE_SEED + rep so its {id, seed} tuples are unique (cache
# miss) yet deterministic for a given repetition index.
BASE_SEED = 0x5EED

SCRIPT_DIR = Path(__file__).parent.resolve()

DEFAULT_GOLDEN_BY_ARCH = {
    "wormhole_b0": SCRIPT_DIR / "compile_stress_golden.json",
    "blackhole": SCRIPT_DIR / "compile_stress_blackhole_golden.json",
}


@dataclass
class RepResult:
    rep: int
    seed: int
    num_kernels: int
    num_programs: int
    total_elapsed_ms: float
    target_device_type: str

    @property
    def kernels_per_sec(self) -> float:
        return (self.num_kernels * 1000.0) / self.total_elapsed_ms if self.total_elapsed_ms > 0 else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--test-binary",
        type=Path,
        default=Path("./build/test/tt_metal/unit_tests_jit_build"),
        help="Path to the compiled unit_tests_jit_build gtest binary.",
    )
    p.add_argument(
        "--arch",
        default=os.environ.get("ARCH_NAME", "wormhole_b0"),
        help="Mock architecture to compile for (default: $ARCH_NAME or wormhole_b0).",
    )
    p.add_argument(
        "--num-kernels",
        type=int,
        default=1000,
        help="Kernels compiled per repetition (TT_METAL_COMPILE_STRESS_NUM_KERNELS).",
    )
    p.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of cold-compile repetitions; the fastest one is gated.",
    )
    p.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="Golden JSON. Defaults to the per-arch golden next to this script.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-rep result JSON, logs, and isolated caches (default: a temp dir).",
    )
    p.add_argument(
        "--keep-output",
        action="store_true",
        help="Do not delete a temp --output-dir on exit (for debugging).",
    )
    return p.parse_args(argv)


def resolve_golden(args: argparse.Namespace) -> Path:
    if args.golden is not None:
        return args.golden
    golden = DEFAULT_GOLDEN_BY_ARCH.get(args.arch)
    if golden is None:
        raise SystemExit(
            f"No default golden for arch {args.arch!r}; pass --golden explicitly. "
            f"Known arches: {sorted(DEFAULT_GOLDEN_BY_ARCH)}"
        )
    return golden


def run_repetition(args: argparse.Namespace, work_dir: Path, rep: int) -> RepResult:
    rep_dir = work_dir / f"rep_{rep:02d}"
    cache_dir = rep_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_file = rep_dir / "result.json"
    stdout_log = rep_dir / "stdout.log"
    seed = BASE_SEED + rep

    env = dict(os.environ)
    for var in SIM_FORCING_ENV_VARS:
        env.pop(var, None)
    # Force the local compile path: never route to a remote JIT compile server.
    env.pop("TT_METAL_JIT_SERVER_ENDPOINTS", None)
    env["TT_METAL_JIT_SERVER_ENABLE"] = "0"
    env.update(
        {
            "TT_METAL_COMPILE_STRESS_MOCK": "0",
            "TT_METAL_CACHE": str(cache_dir),
            "TT_METAL_COMPILE_STRESS_NUM_KERNELS": str(args.num_kernels),
            "TT_METAL_COMPILE_STRESS_ARCH": args.arch,
            "TT_METAL_COMPILE_STRESS_CLIENT_ID": str(rep),
            "TT_METAL_COMPILE_STRESS_SEED": str(seed),
            "TT_METAL_COMPILE_STRESS_SHARED_FRACTION": "0.000000",
            "TT_METAL_COMPILE_STRESS_OUTPUT": str(result_file),
        }
    )

    cmd = [
        str(args.test_binary),
        "--gtest_also_run_disabled_tests",
        "--gtest_filter=*TensixCompileStress*",
        "--gtest_color=no",
    ]
    print(f"[jit-build-perf] rep {rep}: seed={seed} num_kernels={args.num_kernels} arch={args.arch}")
    with open(stdout_log, "w") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f"[jit-build-perf] rep {rep} failed (exit {rc}). See {stdout_log}.")
    if not result_file.exists():
        raise SystemExit(f"[jit-build-perf] rep {rep} produced no JSON at {result_file}. See {stdout_log}.")

    with open(result_file) as f:
        data = json.load(f)
    result = RepResult(
        rep=rep,
        seed=seed,
        num_kernels=int(data["num_kernels"]),
        num_programs=int(data["num_programs"]),
        total_elapsed_ms=float(data["total_elapsed_ms"]),
        target_device_type=str(data["target_device_type"]),
    )
    print(
        f"[jit-build-perf] rep {rep}: {result.total_elapsed_ms:.1f} ms "
        f"({result.kernels_per_sec:.1f} kernels/sec, {result.num_programs} programs, "
        f"target={result.target_device_type})"
    )
    return result


def gate(reps: list[RepResult], golden_path: Path, args: argparse.Namespace) -> int:
    best = min(reps, key=lambda r: r.total_elapsed_ms)
    elapsed = [r.total_elapsed_ms for r in reps]
    compile_ms_min = best.total_elapsed_ms
    kernels_per_sec_max = best.kernels_per_sec

    print()
    print("=" * 72)
    print("JIT-build local compile throughput")
    print("=" * 72)
    print(f"arch:              {args.arch}")
    print(f"target device:     {best.target_device_type}")
    print(f"num_kernels:       {best.num_kernels}")
    print(f"num_programs:      {best.num_programs}")
    print(f"repetitions:       {len(reps)}")
    print(
        f"compile_ms  min:   {compile_ms_min:.1f}   median: {statistics.median(elapsed):.1f}   max: {max(elapsed):.1f}"
    )
    print(f"kernels/sec (best):{kernels_per_sec_max:.1f}")
    print()

    if not golden_path.exists():
        raise SystemExit(f"[jit-build-perf] golden not found: {golden_path}")
    with open(golden_path) as f:
        golden = json.load(f)

    tolerance_pct = float(golden.get("tolerance_pct", 15.0))
    metrics = golden.get("metrics", {})

    measured = {
        "compile_ms_min": compile_ms_min,
        "kernels_per_sec_max": kernels_per_sec_max,
    }
    # For each metric, "worse" is defined by direction: lower-is-better for a
    # time, higher-is-better for a throughput.
    lower_is_better = {"compile_ms_min": True, "kernels_per_sec_max": False}

    exit_code = 0
    for name, meas in measured.items():
        golden_val = metrics.get(name, None)
        if golden_val is None:
            print(f"[record] {name}: measured {meas:.3f} (golden null -> not gated)")
            continue
        golden_val = float(golden_val)
        if lower_is_better[name]:
            diff_pct = (meas / golden_val - 1.0) * 100.0 if golden_val > 0 else 0.0
            regressed = diff_pct > tolerance_pct
        else:
            diff_pct = (1.0 - meas / golden_val) * 100.0 if golden_val > 0 else 0.0
            regressed = diff_pct > tolerance_pct
        verdict = "FAIL" if regressed else "ok"
        print(
            f"[gated]  {name}: measured {meas:.3f} vs golden {golden_val:.3f} "
            f"({diff_pct:+.2f}% worse, tol {tolerance_pct:.1f}%) -> {verdict}"
        )
        if regressed:
            exit_code = 1

    print()
    if exit_code == 0:
        print("[jit-build-perf] PASS")
    else:
        print("[jit-build-perf] FAIL: compile throughput regressed beyond tolerance")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.test_binary.exists():
        raise SystemExit(f"--test-binary not found: {args.test_binary}")
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")
    if args.num_kernels < 1:
        raise SystemExit("--num-kernels must be >= 1")

    golden_path = resolve_golden(args)

    if args.output_dir is not None:
        work_dir = args.output_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="jit_build_perf_"))
        cleanup = not args.keep_output

    try:
        reps = [run_repetition(args, work_dir, rep) for rep in range(args.repetitions)]
        return gate(reps, golden_path, args)
    finally:
        if cleanup and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
