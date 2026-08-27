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

import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from helpers.llk_params import ApproximationMode, DestAccumulation, PerfRunType
from helpers.perf.core import (
    PerfConfig,
    PerfReport,
    _ci_provenance,
    _prune_runs,
    _refresh_latest,
    _reject_duplicate_keys,
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

    monkeypatch.setenv("PERF_RUN_TAG", "testrun-wormhole-0")

    combine_perf_reports()

    run_dir = root / "perf_data" / "runs" / "testrun-wormhole-0"
    # CSV still produced, inside this run's own directory...
    assert (run_dir / "perf_x" / "perf_x.csv").exists()
    # ...reachable through the stable `latest` path...
    assert (root / "perf_data" / "latest" / "perf_x" / "perf_x.csv").exists()
    # ...and a run-level Parquet batch alongside it.
    parquet = run_dir / "testrun.parquet"
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
    monkeypatch.setenv("PERF_RUN_TAG", "testrun-wormhole-0")

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

    run_dir = root / "perf_data" / "runs" / "testrun-wormhole-0"
    assert (run_dir / "perf_x" / "perf_x.csv").exists()
    assert not (run_dir / "testrun.parquet").exists()


def _seed_worker_csv(workers, base, mean):
    """One raw per-worker CSV, the `.gw*` pattern combine_perf_reports globs for."""
    pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            "tile_cnt": [4, 4],
            "loop_factor": [1, 1],
            stat_column("MATH_ISOLATE", MEAN): [mean, mean * 2],
        }
    ).to_csv(workers / f"{base}.gw0.csv", index=False)


def _perf_run(tmp_path, monkeypatch, tag, bases, mean=10.0):
    """Run combine_perf_reports once under `tag`, for the given test bases.

    A real run never names a tag: ``perf_run_tag()`` mints ``local-<UTC stamp>``
    on its own. These tests pin it because a directory named after the current
    second is not something an assertion can address.
    """
    workers = tmp_path / f"workers-{tag}"
    workers.mkdir()
    monkeypatch.setattr(TestConfig, "PERF_DATA_DIR", workers)
    monkeypatch.setenv("PERF_RUN_TAG", tag)
    for base in bases:
        _seed_worker_csv(workers, base, mean)
    combine_perf_reports()


def test_second_run_does_not_mix_into_the_first(tmp_path, monkeypatch):
    # The bug this layout exists to prevent: run 2 is narrower than run 1, so a
    # shared output directory kept run 1's perf_b/ alongside run 2's perf_a/ and
    # the tree read as one complete run.
    root = tmp_path / "root"
    monkeypatch.setattr(TestConfig, "LLK_ROOT", root)

    _perf_run(tmp_path, monkeypatch, "run-1", ["perf_a", "perf_b"])
    _perf_run(tmp_path, monkeypatch, "run-2", ["perf_a"])

    runs = root / "perf_data" / "runs"
    # Run 1 is intact -- history is preserved, not overwritten.
    assert (runs / "run-1" / "perf_a" / "perf_a.csv").exists()
    assert (runs / "run-1" / "perf_b" / "perf_b.csv").exists()
    # Run 2 holds only what run 2 measured. No perf_b leaking in from run 1.
    assert (runs / "run-2" / "perf_a" / "perf_a.csv").exists()
    assert not (runs / "run-2" / "perf_b").exists()


def test_latest_follows_the_most_recent_run(tmp_path, monkeypatch):
    root = tmp_path / "root"
    monkeypatch.setattr(TestConfig, "LLK_ROOT", root)

    _perf_run(tmp_path, monkeypatch, "run-1", ["perf_a"], mean=10.0)
    _perf_run(tmp_path, monkeypatch, "run-2", ["perf_a"], mean=99.0)

    latest = root / "perf_data" / "latest"
    assert latest.is_symlink()
    frame = pd.read_csv(latest / "perf_a" / "perf_a.csv")
    assert frame[stat_column("MATH_ISOLATE", MEAN)].min() == 99.0


def test_run_history_is_pruned_to_the_keep_limit(tmp_path, monkeypatch):
    root = tmp_path / "root"
    monkeypatch.setattr(TestConfig, "LLK_ROOT", root)
    monkeypatch.setenv("PERF_KEEP_RUNS", "2")

    for i in range(4):
        _perf_run(tmp_path, monkeypatch, f"run-{i}", ["perf_a"])

    kept = {d.name for d in (root / "perf_data" / "runs").iterdir() if d.is_dir()}
    assert len(kept) == 2
    assert "run-3" in kept  # the newest always survives


def test_local_run_id_is_unique_per_run(tmp_path, monkeypatch):
    # Off CI run_id used to be the constant "local", so two local runs of one
    # commit published colliding ROW_KEYs.
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("PERF_RUN_TAG", raising=False)
    first = TestConfig.perf_run_tag()

    monkeypatch.delenv("PERF_RUN_TAG", raising=False)
    monkeypatch.setenv("PERF_RUN_TAG", "second-tag")

    assert first.startswith("local-")
    assert TestConfig.perf_run_tag() == "second-tag"


def test_run_tag_fallback_timestamps_instead_of_naming_the_arch(monkeypatch):
    # CHIP_ARCH was once part of the fallback tag, but it cannot disambiguate what
    # actually collides: every shard of one architecture shares GITHUB_RUN_ID, and
    # only the workflow can see the shard index -- which is why CI sets
    # PERF_RUN_TAG itself. Here a timestamp is what keeps invocations apart.
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("CHIP_ARCH", "wormhole")
    monkeypatch.delenv("PERF_RUN_TAG", raising=False)

    tag = TestConfig.perf_run_tag()

    assert tag.startswith("999-")
    assert tag != "999"  # a bare run id is the collision this exists to avoid
    assert "wormhole" not in tag


def test_run_tag_is_stable_within_a_process(tmp_path, monkeypatch):
    # xdist workers inherit the environment; the tag must not be re-minted per
    # call, or one run would scatter across several directories.
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("PERF_RUN_TAG", raising=False)

    assert TestConfig.perf_run_tag() == TestConfig.perf_run_tag()


def test_ci_run_id_still_wins_for_provenance(monkeypatch):
    # All shards of one workflow must share run_id: it is a ROW_KEY column and
    # the data team's notion of a run spans shards.
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("PERF_RUN_TAG", "999-wormhole-3")

    assert _ci_provenance()["run_id"] == "999"


def test_rerun_of_a_workflow_publishes_under_its_own_run_id(monkeypatch):
    # "Re-run all/failed jobs" keeps GITHUB_RUN_ID and bumps GITHUB_RUN_ATTEMPT.
    # Attempt 2 is a second, different measurement: sharing attempt 1's ROW_KEY
    # (test_name, commit_sha, arch, run_id) would collide with rows already
    # published.
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("PERF_RUN_TAG", "999-wormhole-3")

    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    assert _ci_provenance()["run_id"] == "999-2"

    # Attempt 1 stays bare, so rows already archived keep the identity they were
    # published with.
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    assert _ci_provenance()["run_id"] == "999"


def test_prune_keeps_the_current_run_however_old_it_looks(tmp_path):
    # The current run survives by name, not by being the newest: an mtime that is
    # older than its neighbours (a clock step, a filesystem that lies) must not be
    # able to delete the report this invocation just wrote.
    runs = tmp_path / "runs"
    runs.mkdir()
    current = runs / "run-current"
    for i, d in enumerate([current, runs / "run-a", runs / "run-b"]):
        d.mkdir()
        os.utime(d, (0, i))  # current is the OLDEST

    _prune_runs(runs, keep=1, current=current)

    survivors = {d.name for d in runs.iterdir()}
    assert survivors == {"run-current", "run-b"}


def test_prune_survives_a_directory_it_cannot_stat(tmp_path, monkeypatch):
    # One unreadable entry costs that entry, not the whole prune -- otherwise a
    # single bad directory means history grows without bound forever after.
    runs = tmp_path / "runs"
    runs.mkdir()
    current = runs / "run-current"
    bad = runs / "run-bad"
    stale = runs / "run-stale"
    for i, d in enumerate([stale, bad, current]):
        d.mkdir()
        os.utime(d, (0, i))

    real_stat = Path.stat

    def stat_that_fails_on_bad(self, *args, **kwargs):
        if self == bad:
            raise OSError("stat refused")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_that_fails_on_bad)
    _prune_runs(runs, keep=1, current=current)
    monkeypatch.undo()

    survivors = {d.name for d in runs.iterdir()}
    assert "run-current" in survivors  # protected
    assert "run-bad" in survivors  # skipped, never pruned blindly
    assert "run-stale" not in survivors  # the prune still did its job


def test_latest_swap_leaves_no_debris_when_it_fails(tmp_path, monkeypatch):
    # The swap goes through a temporary name. If it fails, neither the old link
    # nor a stray .latest.tmp.<pid> may be left behind for the next run to trip on.
    perf_data = tmp_path / "perf_data"
    (perf_data / "runs" / "run-1").mkdir(parents=True)
    (perf_data / "runs" / "run-2").mkdir()
    (perf_data / "latest").symlink_to(Path("runs") / "run-1", target_is_directory=True)

    def replace_that_fails(self, *args, **kwargs):
        raise OSError("rename refused")

    monkeypatch.setattr(Path, "replace", replace_that_fails)
    _refresh_latest(perf_data / "runs" / "run-2")
    monkeypatch.undo()

    assert (perf_data / "latest").readlink() == Path("runs") / "run-1"
    assert not list(perf_data.glob(".latest.tmp.*"))


def test_duplicate_sweep_key_is_rejected_not_averaged():
    # Two rows with the same (sweep-params, marker) key make the measurement
    # ambiguous, so the run must fail instead of averaging them into one row.
    frame = pd.DataFrame(
        {
            "dest_acc": ["Yes", "Yes"],
            "tile_cnt": [8, 8],
            MARKER: ["TILE_LOOP", "TILE_LOOP"],
            stat_column("L1_TO_L1", MEAN): [100.0, 140.0],
        }
    )
    with pytest.raises(PerfSchemaError) as excinfo:
        _reject_duplicate_keys(frame, "perf_example.csv")
    message = str(excinfo.value)
    assert "perf_example.csv" in message
    assert "duplicate" in message


def test_distinct_sweep_keys_pass_through_unchanged():
    frame = pd.DataFrame(
        {
            "dest_acc": ["Yes", "No"],
            "tile_cnt": [8, 8],
            MARKER: ["TILE_LOOP", "TILE_LOOP"],
            stat_column("L1_TO_L1", MEAN): [100.0, 140.0],
        }
    )
    pd.testing.assert_frame_equal(
        _reject_duplicate_keys(frame, "perf_example.csv"), frame
    )
