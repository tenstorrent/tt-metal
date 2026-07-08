# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Tests for the real-time-profiler "quick" perf report (rt_device_perf_report.csv, produced by
# TT_METAL_PROFILER_RT_QUICK=1). The headline test runs one workload with both the RT profiler and
# the classic device profiler on, joins the two per-program reports, and checks they agree:
#   * OP TO OP LATENCY matches tightly (both anchor on the same program boundaries).
#   * DEVICE KERNEL DURATION does not: the RT dispatch_s window wraps the worker-core kernel markers
#     the classic report measures, so RT exceeds classic by a roughly-constant dispatch overhead.
#     Hence we bound the offset, not the ratio (which is large for short kernels, ~1 for long ones).
# Enabling both profilers at once is a test technique (exact per-program join, no run-to-run noise),
# not production usage.

import os
import subprocess
import sys

import pandas as pd
import pytest
from loguru import logger

from tracy.common import (
    TT_METAL_HOME,
    PROFILER_LOGS_DIR,
    PROFILER_CPP_DEVICE_PERF_REPORT,
    RT_DEVICE_PERF_REPORT,
    clear_profiler_runtime_artifacts,
)

WORKLOAD = os.path.join(os.path.dirname(__file__), "rt_perf_report_workload.py")

KEYS = ["GLOBAL CALL COUNT", "DEVICE ID"]
DURATION_COL = "DEVICE KERNEL DURATION [ns]"
LATENCY_COL = "OP TO OP LATENCY [ns]"
START_COL = "DEVICE KERNEL START CYCLE"
END_COL = "DEVICE KERNEL END CYCLE"

LATENCY_REL_TOL = 0.10
LATENCY_FLOOR_NS = 2000
DURATION_MAX_OFFSET_NS = 8000
DURATION_UNDERSHOOT_NS = 500

# GLOBAL CALL COUNT packs the device id in its low bits; mirrors detail::EncodePerDeviceProgramID
# (and decode_run_host_id in test_device_profiler.py).
DEVICE_ID_NUM_BITS = 10

# Loose sanity cap: the workload's ops are tiny, so a plausible device kernel duration is well under
# 1s. A corrupted/half-swapped timestamp would blow past this.
MAX_PLAUSIBLE_DURATION_NS = 1_000_000_000


def _run_workload(env_extra):
    clear_profiler_runtime_artifacts()
    env = {**os.environ, **env_extra}
    subprocess.run([sys.executable, WORKLOAD], cwd=TT_METAL_HOME, env=env, check=True)


def _load(name):
    path = PROFILER_LOGS_DIR / name
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def _load_rt_or_skip():
    rt = _load(RT_DEVICE_PERF_REPORT)
    if rt is None or len(rt) == 0:
        pytest.skip(f"real-time profiler produced no records ({RT_DEVICE_PERF_REPORT} empty); not supported here")
    return rt


def test_rt_perf_report_matches_device_profiler():
    _run_workload(
        {
            "TT_METAL_DEVICE_PROFILER": "1",
            "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
            "TT_METAL_PROFILER_RT_QUICK": "1",
        }
    )

    rt = _load_rt_or_skip()
    cpp = _load(PROFILER_CPP_DEVICE_PERF_REPORT)
    assert cpp is not None and len(cpp) > 0, f"{PROFILER_CPP_DEVICE_PERF_REPORT} missing/empty; cannot compare"

    merged = rt.merge(cpp, on=KEYS, suffixes=("_rt", "_cpp"))
    logger.info(f"RT rows={len(rt)} classic rows={len(cpp)} joined={len(merged)}")
    assert len(rt) >= 10, f"workload produced too few RT rows ({len(rt)}) for a meaningful comparison"
    # RT is a subset of the classic report -- it may skip a benign startup-race record, so len(rt) can be
    # < len(cpp) -- but every RT row is a real program classic also captured, so all of them must join 1:1.
    assert len(merged) == len(
        rt
    ), f"expected all {len(rt)} RT rows to join the classic report (got {len(merged)}); classic has {len(cpp)}"

    lat_rt = _num(merged[f"{LATENCY_COL}_rt"])
    lat_cpp = _num(merged[f"{LATENCY_COL}_cpp"])
    lat_mask = lat_rt.notna() & lat_cpp.notna() & (lat_cpp > LATENCY_FLOOR_NS)
    assert lat_mask.sum() >= 3, f"workload exercised too few real op-to-op gaps (>{LATENCY_FLOOR_NS}ns) to compare"
    rel_err = (lat_rt[lat_mask] - lat_cpp[lat_mask]).abs() / lat_cpp[lat_mask]
    worst = merged.loc[lat_mask].assign(rel_err=rel_err).nlargest(5, "rel_err")
    assert rel_err.max() <= LATENCY_REL_TOL, (
        f"OP TO OP LATENCY disagreement exceeds {LATENCY_REL_TOL:.0%} (max {rel_err.max():.1%}). Worst:\n"
        f"{worst[KEYS + [f'{LATENCY_COL}_rt', f'{LATENCY_COL}_cpp', 'rel_err']].to_string(index=False)}"
    )

    dur_rt = _num(merged[f"{DURATION_COL}_rt"])
    dur_cpp = _num(merged[f"{DURATION_COL}_cpp"])
    dur_mask = dur_rt.notna() & dur_cpp.notna() & (dur_cpp > 0)
    assert dur_mask.sum() >= 3, "too few programs with a classic kernel duration to compare"
    offset = dur_rt[dur_mask] - dur_cpp[dur_mask]
    logger.info(
        f"duration offset (RT-classic) ns: min={offset.min():.0f} median={offset.median():.0f} max={offset.max():.0f}"
    )

    undershoot = merged.loc[dur_mask].assign(offset=offset).nsmallest(5, "offset")
    assert offset.min() >= -DURATION_UNDERSHOOT_NS, (
        f"RT kernel duration is materially smaller than classic (min offset {offset.min():.0f}ns); "
        f"the RT window should contain the kernel. Worst:\n"
        f"{undershoot[KEYS + [f'{DURATION_COL}_rt', f'{DURATION_COL}_cpp', 'offset']].to_string(index=False)}"
    )
    overshoot = merged.loc[dur_mask].assign(offset=offset).nlargest(5, "offset")
    assert offset.max() <= DURATION_MAX_OFFSET_NS, (
        f"RT kernel duration exceeds classic by more than the {DURATION_MAX_OFFSET_NS}ns dispatch-overhead bound "
        f"(max offset {offset.max():.0f}ns). Worst:\n"
        f"{overshoot[KEYS + [f'{DURATION_COL}_rt', f'{DURATION_COL}_cpp', 'offset']].to_string(index=False)}"
    )


def test_rt_perf_report_standalone_is_self_consistent():
    _run_workload({"TT_METAL_PROFILER_RT_QUICK": "1"})
    rt = _load_rt_or_skip()

    for col in [DURATION_COL, START_COL, END_COL, LATENCY_COL, *KEYS]:
        assert col in rt.columns, f"quick report missing column '{col}'"

    assert len(rt) >= 10, f"expected many program rows from the workload, got {len(rt)}"

    gcc = _num(rt["GLOBAL CALL COUNT"])
    dev = _num(rt["DEVICE ID"])
    start = _num(rt[START_COL])
    end = _num(rt[END_COL])
    dur = _num(rt[DURATION_COL])
    lat = _num(rt[LATENCY_COL])

    device_mask = (1 << DEVICE_ID_NUM_BITS) - 1
    assert ((gcc & device_mask) == dev).all(), "GLOBAL CALL COUNT low bits must equal DEVICE ID"
    assert (
        (gcc // (1 << DEVICE_ID_NUM_BITS)) > 0
    ).all(), "base program id (GLOBAL CALL COUNT high bits) must be non-zero"

    assert (end > start).all(), "device kernel end cycle must exceed start cycle"
    assert (dur > 0).all(), "device kernel duration must be positive"
    assert (
        dur < MAX_PLAUSIBLE_DURATION_NS
    ).all(), "device kernel duration is implausibly large (mis-decoded timestamp?)"

    # Latency is blank (NaN) for a negative gap and 0 for the first op on a chip; never negative.
    assert (lat.dropna() >= 0).all(), "op-to-op latency must be non-negative"
    # The earliest program on each device has no predecessor, so its latency is exactly 0.
    for device_id, group in rt.groupby("DEVICE ID"):
        first = group.loc[_num(group["GLOBAL CALL COUNT"]).idxmin()]
        assert (
            _num(pd.Series([first[LATENCY_COL]])).iloc[0] == 0
        ), f"first program on device {device_id} must report 0 latency"
