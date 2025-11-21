# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import shutil
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import pytest
from helpers.device import (
    BootMode,
    reset_mailboxes,
    run_elf_files,
    wait_for_tensix_operations_finished,
)
from helpers.profiler import Profiler, ProfilerData
from helpers.test_config import ProfilerBuild, build_test


class PerfRunType(Enum):
    L1_TO_L1 = 1
    UNPACK_ISOLATE = 2
    MATH_ISOLATE = 3
    PACK_ISOLATE = 4
    L1_CONGESTION = 5


ALL_RUN_TYPES = [type for type in PerfRunType]


def _stats_timings(perf_data: pd.DataFrame) -> pd.DataFrame:

    # dont aggregate marker column
    timings = perf_data.columns.drop("marker")
    result = perf_data.groupby("marker", as_index=False)[timings].agg(["mean", "std"])

    columns = ["marker"]
    columns += [f"{stat}({col})" for col in timings for stat in ["mean", "std"]]

    result.columns = columns
    return result


def _stats_l1_to_l1(data: ProfilerData) -> pd.Series:
    groups = data.zones().raw().groupby(["marker"])

    timings = []
    for (marker,), group in groups:

        unpack_start = group[
            (group["thread"] == "UNPACK") & (group["type"] == "ZONE_START")
        ].reset_index(drop=True)

        pack_end = group[
            (group["thread"] == "PACK") & (group["type"] == "ZONE_END")
        ].reset_index(drop=True)

        if len(unpack_start) == 0 or len(pack_end) == 0:
            raise ValueError(
                "Zone must be captured on both unpack and pack for L1_TO_L1 to work properly"
            )

        if len(unpack_start) != len(pack_end):
            raise ValueError(
                f"Unpack and pack must be paired properly for L1_TO_L1 to work properly"
            )

        durations = pack_end["timestamp"] - unpack_start["timestamp"]

        marker_timings = pd.DataFrame(
            {
                "marker": marker,
                PerfRunType.L1_TO_L1.name: durations,
            }
        )
        timings.append(marker_timings)

    return _stats_timings(pd.concat(timings, ignore_index=True))


def _stats_thread(stat: str, raw_thread: pd.DataFrame) -> pd.DataFrame:
    start_entries = raw_thread[(raw_thread["type"] == "ZONE_START")].reset_index(
        drop=True
    )

    end_entries = raw_thread[(raw_thread["type"] == "ZONE_END")].reset_index(drop=True)

    if len(start_entries) != len(end_entries):
        raise ValueError(
            f"Mismatched start/end zones: {len(start_entries)} != {len(end_entries)}"
        )

    timings = pd.DataFrame(
        {
            "marker": start_entries["marker"],
            stat: end_entries["timestamp"] - start_entries["timestamp"],
        }
    )

    return _stats_timings(timings)


def _stats_unpack_isolate(data: ProfilerData) -> pd.DataFrame:
    return _stats_thread(PerfRunType.UNPACK_ISOLATE.name, data.unpack().raw())


def _stats_math_isolate(data: ProfilerData) -> pd.DataFrame:
    return _stats_thread(PerfRunType.MATH_ISOLATE.name, data.math().raw())


def _stats_pack_isolate(data: ProfilerData) -> pd.DataFrame:
    return _stats_thread(PerfRunType.PACK_ISOLATE.name, data.pack().raw())


def _stats_l1_congestion(data: ProfilerData) -> pd.DataFrame:
    stats = [
        _stats_thread(f"{PerfRunType.L1_CONGESTION.name}[UNPACK]", data.unpack().raw()),
        _stats_thread(f"{PerfRunType.L1_CONGESTION.name}[PACK]", data.pack().raw()),
    ]

    return pd.concat(stats, ignore_index=True)


def perf_benchmark(
    test_config, run_types: list[PerfRunType], run_count=2, boot_mode=BootMode.DEFAULT
):  # global override boot mode for perf tests here

    STATS_FUNCTION = {
        PerfRunType.L1_TO_L1: _stats_l1_to_l1,
        PerfRunType.UNPACK_ISOLATE: _stats_unpack_isolate,
        PerfRunType.MATH_ISOLATE: _stats_math_isolate,
        PerfRunType.PACK_ISOLATE: _stats_pack_isolate,
        PerfRunType.L1_CONGESTION: _stats_l1_congestion,
    }
    SUPPORTED_RUNS = STATS_FUNCTION.keys()

    results = []

    for run_type in run_types:
        assert run_type in SUPPORTED_RUNS, f"ERROR: run_type={run_type} not implemented"

        get_stats = STATS_FUNCTION[run_type]

        test_config["perf_run_type"] = run_type
        build_test(test_config, boot_mode, ProfilerBuild.Yes)

        runs = []
        for _ in range(run_count):
            reset_mailboxes()
            elfs = run_elf_files(test_config["testname"], boot_mode)
            wait_for_tensix_operations_finished(elfs)

            profiler_data = Profiler.get_data(test_config["testname"])

            runs.append(profiler_data)

        results.append(get_stats(ProfilerData.concat(runs)))

    results = pd.concat(results, ignore_index=True)

    # combine all run types into a single row in the dataframe
    report = results.groupby("marker").first().reset_index()

    return report


class PerfReport:
    """
    Lazy evaluation container for performance benchmark data.

    Allows for lazy evaluated query and append operation to the report.

    """

    def __init__(
        self,
        frames: list[pd.DataFrame] | None = None,
        masks: list[pd.Series] | None = None,
    ):
        self._frames = frames or [pd.DataFrame()]
        self._masks = masks or [pd.Series()]

    def append(self, frame: pd.DataFrame) -> "PerfReport":
        self._frames.append(frame)
        self._masks.append(pd.Series(True, index=frame.index))
        return self

    def frame(self) -> pd.DataFrame:
        # merge
        frame = pd.concat(self._frames, ignore_index=True)
        mask = pd.concat(self._masks, ignore_index=True)

        # apply masks
        frame = frame[mask]

        # save
        self._frames = [frame]
        self._masks = [pd.Series(True, index=frame.index)]

        return frame

    def filter(self, column: str, value: Any) -> "PerfReport":
        """Filter: Generic column filter"""
        mask_chain = [
            mask & (frame[column] == value)
            for frame, mask in zip(self._frames, self._masks)
        ]
        return PerfReport(frames=self._frames, masks=mask_chain)

    def marker(self, marker: str) -> "PerfReport":
        """Filter: Marker"""
        return self.filter("marker", marker)


# Generating the report


@pytest.fixture(scope="module")
def perf_report(request):
    report = PerfReport()

    test_module = request.path.stem

    delete_benchmark_dir(test_module)
    try:
        yield report
    except Exception as e:
        print("Perf: Unexpected error, Saving report anyway", e)

    dump_csv(test_module, f"{test_module}.csv", report)

    post = _postprocess_report(report)
    dump_csv(test_module, f"{test_module}.post.csv", post)
    # dump_scatter(test_module, post)


def _dataclass_names(parent, obj):
    """Provides the **names** of the columns for the report"""
    return [f"{parent}.{f.name}" for f in fields(obj)]


def _dataclass_values(obj):
    """Provides the **values** of the columns for the report"""
    return [getattr(obj, f.name) for f in fields(obj)]


def _get_sweep_names(params):
    names = []
    for param, value in params.items():
        if is_dataclass(value):
            names.extend(_dataclass_names(param, value))
        else:
            names.append(param)

    return names


def _get_sweep_values(params):
    return [
        value
        for param in params.values()
        for value in (_dataclass_values(param) if is_dataclass(param) else [param])
    ]


def _get_sweep(params):
    """Returns a DataFrame containing the sweep values for the given parameters"""

    names = _get_sweep_names(params)
    values = _get_sweep_values(params)

    return pd.DataFrame([values], columns=names)


def update_report(report: PerfReport, test_config, results):
    # TODO: make this more robust, handle nested dataclasses, etc.

    exclude = {
        "testname",
        "perf_run_type",
    }

    params = {
        param: value for param, value in test_config.items() if param not in exclude
    }

    sweep = _get_sweep(params)

    combined = sweep.merge(results, how="cross")

    report.append(combined)


def delete_benchmark_dir(testname: str):
    root = os.environ.get("LLK_HOME")
    if not root:
        raise AssertionError("Environment variable LLK_HOME is not set")

    path = Path(root) / "perf_data" / testname

    if path.exists() and path.is_dir():
        shutil.rmtree(path)


def get_benchmark_dir(testname: str) -> Path:
    root = os.environ.get("LLK_HOME")
    if not root:
        raise AssertionError("Environment variable LLK_HOME is not set")

    path = Path(root) / "perf_data" / testname

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return path


# Common postprocessing


def _postprocess_tile_loop(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame["marker"] == "TILE_LOOP"

    if not mask.any():
        return frame

    # Ensure columns exist and default missing values only for masked rows
    for col in ["loop_factor", "tile_cnt"]:
        if col not in frame.columns:
            frame[col] = 1
        frame[col] = frame[col].fillna(1)

    # Compute divisor as Series aligned with masked rows
    divisor = frame.loc[mask, "loop_factor"] * frame.loc[mask, "tile_cnt"]

    # Select only mean/std columns
    mean_columns = [c for c in frame.columns if c.startswith("mean(")]
    std_columns = [c for c in frame.columns if c.startswith("std(")]

    # Apply division
    for cols in (mean_columns, std_columns):
        if cols:
            frame.loc[mask, cols] = frame.loc[mask, cols].div(divisor, axis=0)

    return frame


def _postprocess_report(report: PerfReport) -> PerfReport:
    frame = report.frame().copy()
    frame = _postprocess_tile_loop(frame)

    return PerfReport().append(frame)


def dump_csv(testname: str, filename: str, post_report: PerfReport):
    benchmark_dir = get_benchmark_dir(testname)
    output_path = benchmark_dir / filename
    post_report.frame().to_csv(output_path, index=False)


def dump_scatter(testname: str, report: PerfReport):
    # FIXME: was broken by the new pandas implementation (https://github.com/tenstorrent/tt-llk/issues/857)

    # generate a scatter plot using plotly.graph_objects (no pandas required)

    if not report.sweep_names or not report.stat_names:
        # This is possible on CI when the whole split of the test is skipped
        return

    dir = create_benchmark_dir(testname)
    output_path = dir / f"{testname}.html"

    # x-axis: sweep values (left to right, zipped for each sweep)
    # y-axis: stat values (for each run type, for each stat)
    # stat_names: e.g. mean(L1_TO_L1), mean(UNPACK_ISOLATE), ...
    # sweep_names: e.g. tile_cnt, param2, ...

    fig = go.Figure()

    mean_columns = [
        (name, i) for i, name in enumerate(report.stat_names) if name.startswith("mean")
    ]

    hover = [
        ", ".join(f"{name}={val}" for name, val in zip(report.sweep_names, sweep))
        for sweep in report.sweep_values
    ]

    # For each stat column (run type), plot all points
    for stat_name, stat_idx in mean_columns:
        y_vals = [stat[stat_idx] for stat in report.stat_values]

        fig.add_trace(
            go.Scatter(
                x=list(range(len(report.sweep_values))),
                y=y_vals,
                mode="markers+lines",
                name=stat_name,
                text=hover,
                hoverinfo="text+y",
            )
        )

    # X-axis label
    xaxis_title = "Sweep index (see hover for values)"

    fig.update_layout(
        title=f"Performance Scatter Plot: {testname}",
        xaxis_title=xaxis_title,
        yaxis_title="Cycles / Tile",
        legend_title="Run Type / Stat",
    )

    fig.write_html(str(output_path))
