# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free test of PerfConfig report generation (#51244).

Drives ``PerfConfig.build_report_frame`` — the pure report-assembly seam of
``run()`` — with synthetic per-run-type results and parameters, so the report is
produced with **no chip**. Then it checks that the produced column set conforms
to the shared output schema (``perf_wide_schema.OUTPUT_SCHEMA``).

This is the exact validation the static gate only approximates: the columns come
from the real assembly code path, so it catches the optional-None and
dynamic-param cases that static reading cannot. Needs the LLK env (pandas,
ttexalens importable) but no hardware.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from helpers.llk_params import ApproximationMode, DestAccumulation, PerfRunType
from helpers.perf import PerfConfig, PerfReport
from helpers.perf_schema import MARKER, MEAN, STD, assert_unique_columns, stat_column
from helpers.perf_wide_schema import OUTPUT_SCHEMA
from helpers.profiler import Profiler, ProfilerData
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import APPROX_MODE, LOOP_FACTOR, TILE_COUNT

_SCHEMA_NAMES = {c.name for c in OUTPUT_SCHEMA}


def _fake_formats():
    """A stand-in for a formats_config entry: only the attributes the assembly reads."""
    fmt = SimpleNamespace(
        unpack_A_src="Float16_b",
        unpack_B_src="Float16_b",
        unpack_A_dst="Float16_b",
        unpack_B_dst="Float16_b",
        output_format="Float16_b",
    )
    return [fmt]


def _timing_result(run_type: PerfRunType) -> pd.DataFrame:
    """What get_stats would emit for one run type: marker + mean/std timing columns."""
    rt = run_type.name
    return pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            stat_column(rt, MEAN): [10.0, 20.0],
            stat_column(rt, STD): [1.0, 2.0],
        }
    )


def _build(formats_config):
    results = [
        _timing_result(PerfRunType.L1_TO_L1),
        _timing_result(PerfRunType.MATH_ISOLATE),
    ]
    code_sizes = {PerfRunType.L1_TO_L1: 4096, PerfRunType.MATH_ISOLATE: 2048}
    templates = [APPROX_MODE(ApproximationMode.No)]
    runtimes = [TILE_COUNT(tile_cnt=4), LOOP_FACTOR(loop_factor=1)]
    return PerfConfig.build_report_frame(
        results,
        code_sizes,
        formats_config,
        False,  # unpack_to_dest
        DestAccumulation.No,  # dest_acc
        templates,
        runtimes,
    )


def test_report_columns_conform_to_output_schema():
    combined = _build(_fake_formats())

    # Every produced column must be a known output-schema column.
    unknown = sorted(set(combined.columns) - _SCHEMA_NAMES)
    assert not unknown, (
        f"PerfConfig produced columns not in perf_wide_schema.OUTPUT_SCHEMA: "
        f"{unknown}. Either the report changed (add them to OUTPUT_SCHEMA as "
        f"nullable) or a column is malformed."
    )

    # No duplicate headers, and the expected columns are present.
    assert_unique_columns(combined.columns, context="hw-free report")
    for expected in [
        "marker",
        "tile_cnt",
        "loop_factor",
        "approx_mode",
        "formats.input_A",
        "unpack_to_dest",
        "dest_acc",
        "TEXT_SIZE(L1_TO_L1)",
        stat_column("L1_TO_L1", MEAN),
        stat_column("MATH_ISOLATE", STD),
    ]:
        assert expected in combined.columns, f"missing expected column {expected!r}"


def test_report_without_formats_still_conforms():
    # formats_config=None → no format columns; the rest must still be schema-clean.
    combined = _build(None)
    unknown = sorted(set(combined.columns) - _SCHEMA_NAMES)
    assert not unknown, f"columns not in OUTPUT_SCHEMA: {unknown}"
    assert not any(c.startswith("formats.") for c in combined.columns)


def test_single_row_per_config():
    # One test configuration -> exactly one sweep row cross-joined onto the markers.
    combined = _build(_fake_formats())
    assert len(combined) == 2  # INIT + TILE_LOOP markers


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: drive the REAL PerfConfig.run() with every hardware/build seam
# stubbed and Profiler.get_data returning synthetic events. Unlike the tests
# above (which hand-build stat frames and only exercise build_report_frame),
# this runs the real per-run-type loop AND the real stat aggregation
# (Profiler.STATS_FUNCTION) — the profiler-events -> mean/std math — with no chip.
# ─────────────────────────────────────────────────────────────────────────────

_MARKERS = (("INIT", 0), ("TILE_LOOP", 1))
_THREADS = ("unpack", "math", "pack")


def _one_run_events(seed: int) -> pd.DataFrame:
    """One run's raw profiler events: a ZONE_START/ZONE_END pair per (thread,
    marker), in the raw-event shape Profiler.get_data returns on hardware.
    Durations vary with ``seed`` so repeated runs give real (non-NaN) mean/std.
    """
    rows = []
    ts = 100
    for thread in _THREADS:
        for marker, mid in _MARKERS:
            dur = 10 + mid * 5 + seed  # distinct per marker, varies per run
            for etype, offset in (("ZONE_START", 0), ("ZONE_END", dur)):
                rows.append(
                    {
                        "thread": thread,
                        "type": etype,
                        MARKER: marker,
                        "timestamp": ts + offset,
                        "data": 0,
                        "marker_id": mid,
                        "file": "perf.cpp",
                        "line": 1,
                    }
                )
            ts += dur + 5
    return pd.DataFrame(rows)


def _run_hw_free(monkeypatch, run_types, run_count):
    """Run PerfConfig.run() end-to-end with no chip; return the produced frame."""
    # Class-level execution config: execute-only (skip build, don't pytest.skip),
    # counters off, artefacts dir unused (get_elf_text_size is stubbed).
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME)
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", False)
    monkeypatch.setattr(TestConfig, "ENABLE_PERF_COUNTERS", False)
    monkeypatch.setattr(TestConfig, "ARTEFACTS_DIR", Path("/tmp/hwfree"), raising=False)
    monkeypatch.setattr(TestConfig, "TENSIX_LOCATION", None, raising=False)
    monkeypatch.setattr(
        TestConfig, "get_elf_text_size", staticmethod(lambda path: 4096)
    )

    # Profiler read seam: synthetic events, a fresh seed per call (call = run).
    calls = {"n": 0}

    def fake_get_data(test_name, variant_id, location):
        df = _one_run_events(calls["n"])
        calls["n"] += 1
        return ProfilerData(df)

    monkeypatch.setattr(Profiler, "get_data", staticmethod(fake_get_data))

    cfg = PerfConfig(
        test_name="perf_hwfree",
        formats=None,  # no format inference; formats path covered by _build tests
        run_types=run_types,
        templates=[APPROX_MODE(ApproximationMode.No)],
        runtimes=[TILE_COUNT(tile_cnt=4), LOOP_FACTOR(loop_factor=1)],
    )
    # Device seams on the instance: no-ops.
    for seam in (
        "write_runtimes_to_L1",
        "run_elf_files",
        "wait_for_tensix_operations_finished",
    ):
        monkeypatch.setattr(cfg, seam, lambda *a, **k: None)

    report = PerfReport()
    cfg.run(report, run_count=run_count)
    return report._frames[-1]


def test_run_end_to_end_conforms_to_output_schema(monkeypatch):
    frame = _run_hw_free(
        monkeypatch,
        [PerfRunType.MATH_ISOLATE, PerfRunType.UNPACK_ISOLATE],
        run_count=2,
    )

    unknown = sorted(set(frame.columns) - _SCHEMA_NAMES)
    assert not unknown, f"run() produced columns not in OUTPUT_SCHEMA: {unknown}"

    assert_unique_columns(frame.columns, context="hw-free run()")
    for expected in [
        MARKER,
        "approx_mode",
        "tile_cnt",
        "loop_factor",
        stat_column("MATH_ISOLATE", MEAN),
        stat_column("UNPACK_ISOLATE", MEAN),
    ]:
        assert expected in frame.columns, f"missing expected column {expected!r}"


def test_run_multi_run_populates_std_columns(monkeypatch):
    # run_count>=2 -> std is defined per marker, so the std column is kept AND full.
    frame = _run_hw_free(monkeypatch, [PerfRunType.MATH_ISOLATE], run_count=2)
    std_col = stat_column("MATH_ISOLATE", STD)
    assert std_col in frame.columns, "multi-run should emit the std column"
    assert frame[std_col].notna().all(), "std must be populated with >=2 samples"


def test_run_single_run_drops_empty_std(monkeypatch):
    # run_count==1 -> std is NaN for every marker, so the std column is dropped.
    frame = _run_hw_free(monkeypatch, [PerfRunType.MATH_ISOLATE], run_count=1)
    assert stat_column("MATH_ISOLATE", MEAN) in frame.columns
    assert stat_column("MATH_ISOLATE", STD) not in frame.columns
