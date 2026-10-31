# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-side tests for tt-ember's analysis: synthetic samples, no device needed.

    python3 -m pytest --confcutdir=tools/tt-ember tools/tt-ember/tests

(--confcutdir keeps pytest from loading tt-metal's root conftest.py, which needs ttnn.)
"""

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402
import parser as ember_parser  # noqa: E402

T0 = datetime(2026, 1, 1, 12, 0, 0)


def ms(t: float) -> datetime:
    return T0 + timedelta(milliseconds=t)


def samples(start_ms: int, end_ms: int, amps: float):
    """One sample per millisecond in [start_ms, end_ms)."""
    return [ember_parser.Sample(ms(t), 0, 800.0, amps, amps * 0.8) for t in range(start_ms, end_ms)]


def interval(grid: str, start_ms: float, end_ms: float):
    return ember_parser.ProgramInterval(grid, 6, (end_ms - start_ms) / 1000, ms(start_ms), ms(end_ms))


def test_tdp_without_a_limit_parses():
    # Firmware without a TDP limit prints "N/A"; such lines used to be dropped.
    line = "2026-01-01 12:00:00.000 Device ID 0 TDP 14/N/A W, TDC 27 A, VCORE 800 mV"
    s = ember_parser.parse_line(line)
    assert s is not None and (s.tdp_w, s.tdc_a, s.vcore_mv) == (14.0, 27.0, 800.0)


@pytest.mark.parametrize("legacy, expect_clean", [(False, True), (True, False)])
def test_idle_baseline_keeps_out_run_samples(legacy, expect_clean):
    # Runs at 50 A around a 20 A pause. Updated mode shrinks the pause by the trim, so its
    # baseline is the idle current; legacy widens it into the runs, as the published analysis did.
    raw = samples(0, 100, 50.0) + samples(100, 200, 20.0) + samples(200, 300, 50.0)
    runs = [interval("3x2", 0, 100), interval("3x3", 200, 300)]
    base, count, _settle_ms = ember_parser.compute_pause_base_currents(raw, runs, trim_ms=5.0, legacy=legacy)
    assert count[0] > 0
    assert (base[0] == pytest.approx(20.0)) == expect_clean


@pytest.mark.parametrize("legacy, expected", [(False, 20.0), (True, None)])
def test_idle_baseline_leaves_out_the_ramp_down(legacy, expected):
    # After a run the current takes a while to settle: 40 A for the first 300 ms of a 1 s pause,
    # then 20 A idle. Updated mode leaves the ramp-down out; legacy averages all of it.
    raw = samples(0, 100, 50.0) + samples(100, 400, 40.0) + samples(400, 1100, 20.0) + samples(1100, 1200, 50.0)
    runs = [interval("3x2", 0, 100), interval("3x3", 1100, 1200)]
    base, _count, settle_ms = ember_parser.compute_pause_base_currents(raw, runs, trim_ms=1.0, legacy=legacy)
    if expected is None:
        assert base[0] > 25.0 and settle_ms[0] == 0.0
    else:
        assert base[0] == pytest.approx(expected) and settle_ms[0] == pytest.approx(300.0, abs=2.0)


@pytest.mark.parametrize("legacy, expected", [(False, -2.0), (True, 0.0)])
def test_dynamic_current_keeps_its_sign_in_updated_mode(legacy, expected):
    row = {"avg_current_a": 18.0, "peak_current_a": 18.0, "window_s": 1.0}
    (out,) = ember_parser.attach_dynamic_baseline([row], [20.0], legacy=legacy)
    assert out["dynamic_avg_current_a"] == pytest.approx(expected)


def test_grid_without_samples_gives_nan():
    raw = samples(0, 100, 30.0)
    (r,) = ember_parser.compute_interval_metrics(raw, [interval("3x2", 500, 600)], trim_ms=1.0)
    assert r["sample_count"] == 0 and math.isnan(r["avg_current_a"])


@pytest.mark.parametrize("legacy, expected", [(False, 2.0), (True, 1.0)])
def test_energy_is_scaled_to_the_run_the_flops_cover(legacy, expected):
    # 1 J over a 0.5 s trimmed window of a 1 s run: updated mode reports the whole run.
    row = {"energy_j_vi": "1.0", "window_s": "0.5", "algo_time_s_reported": "1.0"}
    assert common.whole_run_value(row, "energy_j_vi", legacy) == pytest.approx(expected)


def markers_for(cores: int, drop: str = ""):
    out = []
    for kernel, per_core in ember_parser._KERNEL_CORE_MULTIPLIER.items():
        active = "COMPUTE" if kernel == "COMPUTE" else "TRANSFER"
        for suffix in ("KERNEL_START", "KERNEL_END", "WAIT_CYCLES", f"{active}_CYCLES"):
            zone = f"{kernel}_{suffix}"
            n = cores * per_core - (1 if zone == drop else 0)
            out += [ember_parser.DeviceMarker(0, 0, "x", i, zone) for i in range(n)]
    return out


@pytest.mark.parametrize("drop", ["", "READER_KERNEL_END", "COMPUTE_WAIT_CYCLES", "WRITER_TRANSFER_CYCLES"])
def test_lost_markers_make_the_run_incomplete(drop):
    runs = [interval("3x2", 0, 100)]
    assert ember_parser.check_marker_counts(markers_for(6, drop), runs) == (drop == "")


def test_run_status_matches_settings(tmp_path):
    settings = common.run_settings(Path("/bin/true"), ["1", "2"], {"POWER_CASE": "1"}, 50, 1.0, 1, 0)
    assert common.run_status(tmp_path, settings) == "missing"
    common.write_run_complete(tmp_path, settings)
    assert common.run_status(tmp_path, settings) == "done"
    other = common.run_settings(Path("/bin/false"), ["1", "2"], {"POWER_CASE": "1"}, 50, 1.0, 1, 0)
    assert common.run_status(tmp_path, other) == "different"
