# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Phase 0 of the Nsight-counters plan: trust the numbers.

Perf-counter utilization percentages are only worth reporting if they match
analytically-known work. This gate runs two closed-form workloads
(``counter_sanity_workload.py``) under the device profiler and asserts the
captured counters land within tolerance of the part's analytical peak:

  * compute-bound matmul -> achieved TFLOPs near the fidelity peak, hardware
    FPU utilization high and consistent with the analytical performance model.
  * bandwidth-bound eltwise -> achieved DRAM bandwidth near peak, FPU
    utilization low.

The cross-regime separation (matmul FPU util >> eltwise FPU util) is the core
claim: the counters correctly distinguish compute- from bandwidth-bound, which
is the whole point of capturing them. A failure here means the util math is
wrong and must be fixed before any downstream number is believed.

DRAM bandwidth here is computed as bytes-moved / device-time, not from NoC
counters (that path lands in Phase 3); this gate validates the compute-counter
math and the device time base.
"""

import json
import os
import tempfile

import pandas as pd
import pytest

from tracy.process_model_log import run_device_profiler, get_latest_ops_log_filename

# Analytical peaks per part. TFLOPs are bf16 at HiFi2; DRAM is GB/s.
# Source: tt-buddy profiler interpretation.md peak reference table.
PEAKS = {
    "blackhole": {"hifi2_tflops": 280.0, "dram_gbps": 512.0, "compute_cores": 140},
    "wormhole_b0": {"hifi2_tflops": 128.0, "dram_gbps": 288.0, "compute_cores": 64},
}

# Phase 0's real claim is counter trust, not kernel tuning: the grid-normalized
# FPU-utilization counter must equal the independently-computed achieved-FLOPs
# fraction, because FPU cycle-occupancy IS the fraction of the fidelity peak the
# op reaches. A tuned vs untuned matmul changes both numbers together; the gate
# is that they agree. Tolerance is in absolute percentage points.
COUNTER_VS_FLOPS_TOL_PCT = 8.0
MATMUL_FPU_ACTIVITY_FLOOR_PCT = 5.0  # the op must register as doing compute at all

ELTWISE_BW_MIN_FRAC = 0.35  # the eltwise op must actually be bandwidth-bound
PEAK_CEILING_FRAC = 1.05  # measured throughput above this is a math/time-base bug

ELTWISE_FPU_UTIL_MAX_PCT = 5.0  # bandwidth-bound op leaves the FPU idle
FPU_UTIL_SEPARATION_PCT = 10.0  # matmul FPU util must clear eltwise by this margin


def _device_fw_ns(rows):
    return pd.to_numeric(rows["DEVICE FW DURATION [ns]"], errors="coerce").dropna()


def _op_rows(df, op_code_substr):
    mask = df["OP CODE"].astype(str).str.contains(op_code_substr, case=False, na=False)
    rows = df[mask].copy()
    rows = rows[_device_fw_ns(rows) > 0]
    return rows


def _fpu_util_pct(rows):
    """Hardware-counter FPU utilization for the op, warm-window median.

    Prefers the full-grid normalized column; falls back to the per-core
    median. Returns None if neither carries data.
    """
    for col in ("Avg FPU util on full grid (%)", "FPU Util Median (%)", "FPU Util Avg (%)"):
        if col in rows.columns:
            vals = pd.to_numeric(rows[col], errors="coerce").dropna()
            vals = vals[vals > 0]
            if not vals.empty:
                return float(vals.median())
    return None


@pytest.fixture(scope="module")
def counter_sanity_run():
    manifest_fd, manifest_path = tempfile.mkstemp(prefix="counter_sanity_", suffix=".json")
    os.close(manifest_fd)

    name = "CounterUtilizationSanity"
    workload = "tests/ttnn/tracy/counter_sanity_workload.py"
    command = f"pytest {workload}"

    prev_env = {k: os.environ.get(k) for k in ("RUN_COUNTER_SANITY_WORKLOAD", "COUNTER_SANITY_MANIFEST")}
    os.environ["RUN_COUNTER_SANITY_WORKLOAD"] = "1"
    os.environ["COUNTER_SANITY_MANIFEST"] = manifest_path
    try:
        run_device_profiler(command, name, capture_perf_counters_groups=["fpu"])
    finally:
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    with open(manifest_path) as f:
        manifest = json.load(f)
    os.unlink(manifest_path)

    csv_path = get_latest_ops_log_filename(name)
    df = pd.read_csv(csv_path)

    arch = manifest["arch"]
    if arch not in PEAKS:
        pytest.skip(f"no analytical peak table for arch {arch!r}")

    return manifest, df, PEAKS[arch]


def test_matmul_fpu_counter_matches_achieved_flops(counter_sanity_run):
    """The headline counter-trust gate: the FPU utilization counter must equal
    the achieved fraction of the analytical FLOPs peak."""
    manifest, df, peak = counter_sanity_run
    rows = _op_rows(df, "matmul")
    assert not rows.empty, "no matmul rows with device time in the perf report"

    # Min duration = best/steady-state iteration, immune to cold-iter inflation.
    best_ns = _device_fw_ns(rows).min()
    achieved_tflops = manifest["matmul"]["flops"] / (best_ns * 1e-9) / 1e12
    peak_tflops = peak["hifi2_tflops"]
    achieved_frac_pct = achieved_tflops / peak_tflops * 100.0

    assert achieved_tflops <= peak_tflops * PEAK_CEILING_FRAC, (
        f"matmul achieved {achieved_tflops:.1f} TFLOPs exceeds {peak_tflops} peak x{PEAK_CEILING_FRAC} "
        f"-- FLOPs accounting or device time base is wrong"
    )

    hw_fpu = _fpu_util_pct(rows)  # grid-normalized, same 140-core basis as peak
    assert hw_fpu is not None, "no FPU utilization counter data captured for matmul"
    assert (
        hw_fpu >= MATMUL_FPU_ACTIVITY_FLOOR_PCT
    ), f"matmul FPU util {hw_fpu:.1f}% reads as idle while achieving {achieved_tflops:.1f} TFLOPs"
    assert abs(hw_fpu - achieved_frac_pct) <= COUNTER_VS_FLOPS_TOL_PCT, (
        f"FPU counter {hw_fpu:.1f}% disagrees with achieved-FLOPs fraction {achieved_frac_pct:.1f}% "
        f"by more than {COUNTER_VS_FLOPS_TOL_PCT} pts -- the FPU utilization math is not trustworthy"
    )


def test_eltwise_is_bandwidth_bound_near_peak(counter_sanity_run):
    manifest, df, peak = counter_sanity_run
    rows = _op_rows(df, "binary")
    assert not rows.empty, "no eltwise rows with device time in the perf report"

    best_ns = _device_fw_ns(rows).min()
    achieved_gbps = manifest["eltwise"]["bytes_moved"] / (best_ns * 1e-9) / 1e9
    peak_gbps = peak["dram_gbps"]

    assert achieved_gbps <= peak_gbps * PEAK_CEILING_FRAC, (
        f"eltwise achieved {achieved_gbps:.0f} GB/s exceeds {peak_gbps} peak x{PEAK_CEILING_FRAC} "
        f"-- bytes accounting or device time base is wrong"
    )
    assert achieved_gbps >= peak_gbps * ELTWISE_BW_MIN_FRAC, (
        f"eltwise achieved only {achieved_gbps:.0f} GB/s ({achieved_gbps / peak_gbps:.0%} of "
        f"{peak_gbps} peak); expected bandwidth-bound >= {ELTWISE_BW_MIN_FRAC:.0%}"
    )

    hw_fpu = _fpu_util_pct(rows)
    if hw_fpu is not None:
        assert hw_fpu <= ELTWISE_FPU_UTIL_MAX_PCT, (
            f"eltwise hardware FPU util {hw_fpu:.1f}% above {ELTWISE_FPU_UTIL_MAX_PCT}% -- "
            f"bandwidth-bound op should leave the FPU mostly idle"
        )


def test_counters_separate_compute_from_bandwidth(counter_sanity_run):
    """The headline claim: counters classify bound type correctly."""
    _manifest, df, _peak = counter_sanity_run
    mm_fpu = _fpu_util_pct(_op_rows(df, "matmul"))
    el_fpu = _fpu_util_pct(_op_rows(df, "binary"))

    assert mm_fpu is not None, "no FPU counter data for matmul"
    assert el_fpu is not None, "no FPU counter data for eltwise"
    assert mm_fpu - el_fpu >= FPU_UTIL_SEPARATION_PCT, (
        f"FPU util does not separate compute ({mm_fpu:.1f}%) from bandwidth ({el_fpu:.1f}%) "
        f"by the required {FPU_UTIL_SEPARATION_PCT}% -- the bound classifier is not trustworthy"
    )
