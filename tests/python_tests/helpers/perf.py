# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .test_config import TestConfig

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

    def append(self, frame: pd.DataFrame):
        self._frames.append(frame)
        self._masks.append(pd.Series(True, index=frame.index))

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

    def post_process(self):
        frame = pd.concat(self._frames, ignore_index=True)
        mask = pd.concat(self._masks, ignore_index=True)

        frame = _postprocess_tile_loop(frame[mask])

        self._frames = [pd.DataFrame(), frame]
        self._masks = [pd.Series(), pd.Series(True, index=frame.index)]

    def dump_csv(self, filename: str):
        benchmark_dir = TestConfig.LLK_ROOT / "perf_data"

        if not benchmark_dir.exists():
            benchmark_dir.mkdir(parents=True, exist_ok=True)

        frame = pd.concat(self._frames, ignore_index=True)
        mask = pd.concat(self._masks, ignore_index=True)

        # apply masks
        frame[mask].to_csv(benchmark_dir / filename, index=False)


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


def get_unique_base_names(input_dir: Path):
    """
    Extract unique base filenames from files matching *.gw*.csv pattern.
    For example: perf_unpack_untilize.gw0.csv -> perf_unpack_untilize
    """
    # Use Path.glob() for more Pythonic code
    csv_files = list(input_dir.glob("*.gw*.csv")) + list(
        input_dir.glob("*.master.*.csv")
    )

    # Extract base names with regex that handles both patterns
    unique_bases = {
        re.sub(r"\.(?:gw\d+|master\.\d+)(?:\.post)?\.csv$", "", f.name)
        for f in csv_files
    }

    return sorted(unique_bases)


def combine_perf_reports():
    """
    Combine performance report CSV files into two files per base name:
    - One for regular files (without .post.csv)
    - One for post files (with .post.csv)
    """

    output_dir = input_dir = TestConfig.LLK_ROOT / "perf_data"

    if not output_dir.exists():
        return

    for base_name in get_unique_base_names(input_dir):
        csv_files = glob.glob(os.path.join(input_dir, f"{base_name}.gw*.csv"))
        csv_files += glob.glob(os.path.join(input_dir, f"{base_name}.master.*.csv"))

        regular_files = [f for f in csv_files if not f.endswith(".post.csv")]
        post_files = [f for f in csv_files if f.endswith(".post.csv")]
        if regular_files:
            dfs_regular = []
            for file in sorted(regular_files):
                df = pd.read_csv(file)
                dfs_regular.append(df)

            combined_regular = pd.concat(dfs_regular, ignore_index=True)
            output_regular = os.path.join(output_dir, f"{base_name}.csv")
            combined_regular.to_csv(output_regular, index=False)

        if post_files:
            dfs_post = []
            for file in sorted(post_files):
                df = pd.read_csv(file)
                dfs_post.append(df)

            combined_post = pd.concat(dfs_post, ignore_index=True)
            output_post = os.path.join(output_dir, f"{base_name}.post.csv")
            combined_post.to_csv(output_post, index=False)

        for file in regular_files:
            Path(file).unlink()

        for file in post_files:
            Path(file).unlink()
