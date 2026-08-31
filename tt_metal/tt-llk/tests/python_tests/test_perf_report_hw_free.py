# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free test of PerfConfig report generation (#51244).

Runs the real report code with no chip and checks the produced columns against
helpers.perf.wide_schema.OUTPUT_SCHEMA. Two levels:
  - build_report_frame directly, with synthetic per-run-type stat frames;
  - the full PerfConfig.run(), with the device/build seams stubbed and
    Profiler.get_data returning synthetic events (so the real stat aggregation
    runs too).

Unlike the static gate, the columns come from the real code path, so this catches
the optional-None and dynamic-param cases static reading can't. Needs the LLK env
(pandas, ttexalens importable) but no hardware.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from helpers.llk_params import ApproximationMode, DestAccumulation, PerfRunType
from helpers.perf.core import (
    PerfConfig,
    PerfReport,
    combine_perf_reports,
    postprocess_tile_loop,
)
from helpers.perf.schema import (
    MARKER,
    MEAN,
    STD,
    PerfSchemaError,
    assert_unique_columns,
    stat_column,
)
from helpers.perf.wide_schema import DB_SCHEMA, DROPPED_COLUMNS, OUTPUT_SCHEMA
from helpers.profiler import Profiler, ProfilerData, _stats_l1_to_l1
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
        sfpu_src="Float16_b",
        sfpu_dst="Float16_b",
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
    unknown = sorted(set(combined.columns) - _SCHEMA_NAMES - DROPPED_COLUMNS)
    assert not unknown, (
        f"PerfConfig produced columns not in helpers.perf.wide_schema.OUTPUT_SCHEMA: "
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
    unknown = sorted(set(combined.columns) - _SCHEMA_NAMES - DROPPED_COLUMNS)
    assert not unknown, f"columns not in OUTPUT_SCHEMA: {unknown}"
    assert not any(c.startswith("formats.") for c in combined.columns)


def test_single_row_per_config():
    # One test configuration -> exactly one sweep row cross-joined onto the markers.
    combined = _build(_fake_formats())
    assert len(combined) == 2  # INIT + TILE_LOOP markers


# End-to-end: run the real PerfConfig.run() with the device/build seams stubbed
# and Profiler.get_data returning synthetic events. The tests above hand-build
# stat frames; these also run the real stat aggregation (Profiler.STATS_FUNCTION,
# i.e. the profiler-events -> mean/std math), still with no chip.

_MARKERS = (("INIT", 0), ("TILE_LOOP", 1))
_THREADS = ("unpack", "math", "pack")


def _parallel_events(zones) -> pd.DataFrame:
    rows = []
    for thread, start, end in zones:
        for event_type, timestamp in (("ZONE_START", start), ("ZONE_END", end)):
            rows.append(
                {
                    "thread": thread,
                    "type": event_type,
                    MARKER: "TILE_LOOP",
                    "timestamp": timestamp,
                    "data": 0,
                    "marker_id": 1,
                    "file": "perf.cpp",
                    "line": 1,
                    "run_index": 0,
                }
            )
    return pd.DataFrame(rows)


def test_l1_to_l1_four_trisc_aggregates_component_and_envelope_durations():
    data = ProfilerData(
        _parallel_events(
            [
                ("unpack", 100, 105),
                ("pack", 155, 160),
                ("sfpu", 110, 150),
            ]
        )
    )

    result = _stats_l1_to_l1(data)
    prefix = PerfRunType.L1_TO_L1.name
    fpu = result.loc[0, stat_column(f"{prefix}[FPU]", MEAN)]
    sfpu = result.loc[0, stat_column(f"{prefix}[SFPU]", MEAN)]
    overall = result.loc[0, stat_column(prefix, MEAN)]

    assert fpu == 60
    assert sfpu == 40
    assert overall == 60


def test_l1_to_l1_three_trisc_keeps_unpack_to_pack_duration():
    # A stub SFPU KERNEL zone must not switch L1_TO_L1 onto the 4-TRISC path.
    data = ProfilerData(
        pd.concat(
            [
                _parallel_events(
                    [
                        ("unpack", 100, 105),
                        ("pack", 155, 160),
                    ]
                ),
                _parallel_events(
                    [
                        ("unpack", 10, 15),
                        ("pack", 50, 55),
                        ("sfpu", 90, 200),
                    ]
                ).assign(**{MARKER: "KERNEL"}),
            ],
            ignore_index=True,
        )
    )

    result = _stats_l1_to_l1(data)
    prefix = PerfRunType.L1_TO_L1.name
    tile_loop = result[result[MARKER] == "TILE_LOOP"].iloc[0]
    assert tile_loop[stat_column(prefix, MEAN)] == 60
    assert stat_column(f"{prefix}[FPU]", MEAN) not in result.columns


def test_l1_to_l1_four_trisc_rejects_mismatched_zone_counts():
    data = ProfilerData(
        _parallel_events(
            [
                ("unpack", 100, 105),
                ("unpack", 200, 205),
                ("pack", 155, 160),
                ("sfpu", 110, 150),
            ]
        )
    )

    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="must be present and paired"
    ):
        _stats_l1_to_l1(data)


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
    # run() bumps the PerfConfig.TEST_COUNTER ClassVar, which the autouse
    # perf_report fixture reads to decide whether to write a CSV. This module
    # carries no perf marker, so it runs in the ordinary CI session; restore the
    # counter on teardown so we do not leave empty perf_data artefacts for every
    # later module on the worker (register it with monkeypatch to auto-restore).
    monkeypatch.setattr(PerfConfig, "TEST_COUNTER", PerfConfig.TEST_COUNTER)

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

    unknown = sorted(set(frame.columns) - _SCHEMA_NAMES - DROPPED_COLUMNS)
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


def test_run_rejects_empty_stats_for_requested_run_type(monkeypatch):
    monkeypatch.setitem(
        Profiler.STATS_FUNCTION,
        PerfRunType.MATH_ISOLATE,
        lambda data: pd.DataFrame(),
    )

    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError,
        match="no timing statistics for requested run type MATH_ISOLATE",
    ):
        _run_hw_free(monkeypatch, [PerfRunType.MATH_ISOLATE], run_count=1)


def test_run_rejects_negative_mean_timing(monkeypatch):
    mean_column = stat_column("MATH_ISOLATE", MEAN)
    monkeypatch.setitem(
        Profiler.STATS_FUNCTION,
        PerfRunType.MATH_ISOLATE,
        lambda data: pd.DataFrame(
            {
                MARKER: ["INIT", "TILE_LOOP"],
                mean_column: [-1.0, -2.0],
            }
        ),
    )

    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError,
        match="negative mean timing values.*MATH_ISOLATE",
    ):
        _run_hw_free(monkeypatch, [PerfRunType.MATH_ISOLATE], run_count=1)


def test_postprocess_tile_loop_derives_per_tile_from_raw():
    # Public per-tile derivation used downstream on the RAW (Parquet/CSV) table.
    raw = pd.DataFrame(
        {
            MARKER: ["INIT", "TILE_LOOP"],
            "loop_factor": [1, 2],
            "tile_cnt": [4, 4],
            stat_column("MATH_ISOLATE", MEAN): [100.0, 80.0],  # TILE_LOOP total = 80
            # a run-type-prefixed bounded %-metric — must NOT be divided
            "L1_TO_L1_mean(fpu_utilization_pct)": [50.0, 60.0],
        }
    )

    out = postprocess_tile_loop(raw.copy())

    tl = out[out[MARKER] == "TILE_LOOP"].iloc[0]
    init = out[out[MARKER] == "INIT"].iloc[0]
    # TILE_LOOP wall-clock divided by loop_factor(2) * tile_cnt(4): 80 / 8 = 10
    assert tl[stat_column("MATH_ISOLATE", MEAN)] == 10.0
    # non-TILE_LOOP row untouched
    assert init[stat_column("MATH_ISOLATE", MEAN)] == 100.0
    # the prefixed %-metric column is left alone even on the TILE_LOOP row
    assert tl["L1_TO_L1_mean(fpu_utilization_pct)"] == 60.0


def test_combine_perf_reports_emits_parquet_alongside_csv(tmp_path, monkeypatch):
    # A run publishes both CSV and a run-level Parquet batch (raw frames), with
    # provenance stamped from the CI environment.
    import pyarrow.parquet as pq

    workers = tmp_path / "workers"
    workers.mkdir()
    root = tmp_path / "root"
    monkeypatch.setattr(TestConfig, "PERF_DATA_DIR", workers)
    monkeypatch.setattr(TestConfig, "LLK_ROOT", root)
    monkeypatch.setenv("CHIP_ARCH", "wormhole")
    monkeypatch.setenv("GITHUB_SHA", "testsha")
    monkeypatch.setenv("GITHUB_RUN_ID", "testrun")
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)  # -> pipeline "nightly"

    # one raw per-worker CSV (the .gw* pattern combine globs for)
    pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            "tile_cnt": [4, 4],
            "loop_factor": [1, 1],
            stat_column("MATH_ISOLATE", MEAN): [10.0, 20.0],
        }
    ).to_csv(workers / "perf_x.gw0.csv", index=False)

    combine_perf_reports()

    # CSV still produced...
    assert (root / "perf_data" / "perf_x" / "perf_x.csv").exists()
    # ...and a run-level Parquet batch alongside it.
    parquet = root / "perf_data" / "testrun.parquet"
    assert parquet.exists()
    table = pq.read_table(parquet)
    assert table.schema.names == [c.name for c in DB_SCHEMA]
    df = table.to_pandas()
    assert set(df["test_name"]) == {"perf_x"}
    assert set(df["arch"]) == {"wormhole"}
    assert set(df["commit_sha"]) == {"testsha"}
    assert set(df["pipeline"]) == {"nightly"}


def test_combine_perf_reports_raises_on_unknown_parquet_columns(tmp_path, monkeypatch):
    # Schema drift must fail the session, not drop columns and continue. CSV is
    # already written by the time Parquet conversion runs.
    workers = tmp_path / "workers"
    workers.mkdir()
    root = tmp_path / "root"
    monkeypatch.setattr(TestConfig, "PERF_DATA_DIR", workers)
    monkeypatch.setattr(TestConfig, "LLK_ROOT", root)
    monkeypatch.setenv("CHIP_ARCH", "wormhole")
    monkeypatch.setenv("GITHUB_SHA", "testsha")
    monkeypatch.setenv("GITHUB_RUN_ID", "testrun")

    pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            "tile_cnt": [4, 4],
            "loop_factor": [1, 1],
            stat_column("MATH_ISOLATE", MEAN): [10.0, 20.0],
            "made_up_col": [1, 2],
        }
    ).to_csv(workers / "perf_x.gw0.csv", index=False)

    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        PerfSchemaError, match="made_up_col"
    ):
        combine_perf_reports()

    assert (root / "perf_data" / "perf_x" / "perf_x.csv").exists()
    assert not (root / "perf_data" / "testrun.parquet").exists()
