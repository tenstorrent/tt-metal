# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CI wrapper for the op-to-op latency microbenchmark.

Flow (the "run to completion, then post-process in Python" profiler path, and
the test_perf_op_report.py pattern of command -> capture -> post -> assert):

  1. Run test_op_to_op_latency at one pinned steady-state config with
     TT_METAL_DEVICE_PROFILER=1 so it dumps the device + realtime profiler CSVs.
  2. Post-process the CSVs (op_to_op_postprocess.compute_metrics).
  3. Compare the GATED metric (official KERNEL-zone op2op) against the per-arch
     golden within a tolerance; TRACK the RT gap (recorded, not gated).

The golden files ship with null values ("record mode"): until a golden is
populated from a real CI run the test only prints the measured metrics and does
not fail, so it can land before we have silicon numbers. Once the golden value
is filled in, the test gates on it.

Requires an ENABLE_TRACY=ON build (the single-card profiler pipeline builds this).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from op_to_op_postprocess import compute_metrics  # noqa: E402

# One pinned steady-state config. Trace + 2 warmup replays + 8 programs gives >=6
# clean steady-state transitions; --min-prog-id 3 (post-proc default) drops the
# cold ones. compute-nops keeps the program comfortably long vs dispatch noise.
# These values are intentionally fixed for CI comparability; do not sweep here.
PINNED_ARGS = [
    "--use-trace",
    "--trace-warmup-replays",
    "2",
    "--num-programs",
    "8",
    "--num-pages-per-core",
    "4",
    "--compute-nops",
    "2000",
    "--use-device-profiler",
    "--use-realtime-profiler",
]

MIN_PROG_ID = 3
GATED_METRIC = "official_op2op_us"

DEVICE_LOG = "generated/profiler/.logs/profile_log_device.csv"
RT_LOG = "generated/profiler/.logs/profile_log_device_rt.csv"


def _tt_metal_home() -> Path:
    home = os.environ.get("TT_METAL_HOME")
    assert home, "TT_METAL_HOME must be set"
    return Path(home)


def _binary_path() -> Path:
    return _tt_metal_home() / "build/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"


def _golden_path() -> Path:
    arch = os.environ.get("ARCH_NAME", "").lower()
    if "blackhole" in arch:
        return THIS_DIR / "op_to_op_blackhole_golden.json"
    return THIS_DIR / "op_to_op_golden.json"


def test_op_to_op_latency():
    home = _tt_metal_home()
    binary = _binary_path()
    assert binary.exists(), (
        f"benchmark binary not found: {binary}. Build it with a Tracy-enabled build: "
        "cmake --build build --target test_op_to_op_latency"
    )

    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    cmd = [str(binary)] + PINNED_ARGS
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, env=env, cwd=str(home))
    assert result.returncode == 0, f"benchmark exited with code {result.returncode}"

    device_csv = home / DEVICE_LOG
    rt_csv = home / RT_LOG
    assert device_csv.exists(), f"device profiler CSV not produced: {device_csv}"

    metrics = compute_metrics(device_csv, rt_csv, MIN_PROG_ID)
    print("measured metrics:")
    for k, v in metrics.items():
        print(f"  {k} = {v}")

    measured = metrics[GATED_METRIC]
    assert measured == measured, f"{GATED_METRIC} is NaN (no KERNEL-zone op2op samples extracted)"

    golden_path = _golden_path()
    golden = json.loads(golden_path.read_text())
    golden_value = golden.get("golden", {}).get(GATED_METRIC)
    tolerance_pct = golden.get("golden", {}).get("tolerance_pct", 15.0)

    if golden_value is None:
        pytest.skip(
            f"golden not populated in {golden_path.name}; measured {GATED_METRIC}={measured:.3f}. "
            f"Populate '{GATED_METRIC}' from this run to enable the gate."
        )

    lo = golden_value * (1.0 - tolerance_pct / 100.0)
    hi = golden_value * (1.0 + tolerance_pct / 100.0)
    print(
        f"gate: {GATED_METRIC} measured={measured:.3f} golden={golden_value:.3f} "
        f"allowed=[{lo:.3f}, {hi:.3f}] (+/-{tolerance_pct}%)"
    )
    assert lo <= measured <= hi, (
        f"{GATED_METRIC} regression: measured {measured:.3f} outside "
        f"[{lo:.3f}, {hi:.3f}] (golden {golden_value:.3f} +/-{tolerance_pct}%)"
    )
