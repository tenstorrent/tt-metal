# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re
from dataclasses import fields
from functools import reduce
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest

from .device import BootMode, wait_for_tensix_operations_finished
from .format_config import FormatConfig
from .llk_params import DestAccumulation, L1Accumulation, PerfRunType
from .profiler import Profiler, ProfilerData
from .stimuli_config import StimuliConfig
from .test_config import ProfilerBuild, TestConfig, TestMode
from .test_variant_parameters import PERF_RUN_TYPE, RuntimeParameter, TemplateParameter

# Common postprocessing


def _postprocess_tile_loop(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame["marker"] == "TILE_LOOP"

    if not mask.any():
        return frame

    # Ensure columns exist and default missing values only for masked rows
    for col in ["loop_factor", "tile_cnt"]:
        if col not in frame.columns:
            col_idx = frame.columns.get_loc("marker")
            frame.insert(col_idx, col, 1)
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
        if not TestConfig.PERF_DATA_DIR.exists():
            TestConfig.PERF_DATA_DIR.mkdir(parents=True, exist_ok=True)

        frame = pd.concat(self._frames, ignore_index=True)
        mask = pd.concat(self._masks, ignore_index=True)

        # apply masks
        frame[mask].to_csv(TestConfig.PERF_DATA_DIR / filename, index=False)


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

    csv_files = list(input_dir.glob("*.gw*.csv")) + list(
        input_dir.glob("*.master*.csv")
    )

    # Extract base names with regex that handles both patterns
    unique_bases = {
        re.sub(r"\.(?:gw\d+|master)(?:\.post)?\.csv$", "", report_file.name)
        for report_file in csv_files
    }

    return sorted(unique_bases)


def combine_perf_reports():
    """
    Combine performance report CSV files into two files per base name:
    - One for regular files (without .post.csv)
    - One for post files (with .post.csv)
    """

    unique_module_names = get_unique_base_names(TestConfig.PERF_DATA_DIR)
    if not unique_module_names:
        return

    output_dir = TestConfig.LLK_ROOT / "perf_data"

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    for base_name in unique_module_names:
        csv_files = glob.glob(
            os.path.join(TestConfig.PERF_DATA_DIR, f"{base_name}.gw*.csv")
        )
        csv_files += glob.glob(
            os.path.join(TestConfig.PERF_DATA_DIR, f"{base_name}.master*.csv")
        )

        temp_output_dir = output_dir / base_name
        if not temp_output_dir.exists():
            temp_output_dir.mkdir(parents=True, exist_ok=True)

        regular_files, post_files = [], []
        regular_append = regular_files.append
        post_append = post_files.append

        for f in csv_files:
            if f.endswith(".post.csv"):
                post_append(f)
            else:
                regular_append(f)

        if regular_files:
            dfs_regular = []
            for file in sorted(regular_files):
                df = pd.read_csv(file)
                dfs_regular.append(df)

            combined_regular = pd.concat(dfs_regular, ignore_index=True)
            combined_regular = combined_regular.sort_values(
                by=combined_regular.columns.tolist()
            ).reset_index(drop=True)
            output_regular = os.path.join(temp_output_dir, f"{base_name}.csv")
            combined_regular.to_csv(output_regular, index=False)

        if post_files:
            dfs_post = []
            for file in sorted(post_files):
                df = pd.read_csv(file)
                dfs_post.append(df)

            combined_post = pd.concat(dfs_post, ignore_index=True)
            combined_post = combined_post.sort_values(
                by=combined_post.columns.tolist()
            ).reset_index(drop=True)
            output_post = os.path.join(temp_output_dir, f"{base_name}.post.csv")
            combined_post.to_csv(output_post, index=False)

        for file in regular_files:
            Path(file).unlink()

        for file in post_files:
            Path(file).unlink()


class PerfConfig(TestConfig):
    # === STATIC VARIABLES ===
    TEST_COUNTER: ClassVar[int] = 0

    def __init__(
        self,
        test_name: str,
        formats: FormatConfig = None,
        run_types: list[PerfRunType] = [],
        templates: list[TemplateParameter] = [],
        runtimes: list[RuntimeParameter] = [],
        variant_stimuli: StimuliConfig = None,
        unpack_to_dest=False,
        disable_format_inference=False,
        dest_acc=DestAccumulation.No,
        l1_acc=L1Accumulation.No,
    ):
        super().__init__(
            test_name,
            formats,
            templates,
            runtimes,
            variant_stimuli,
            BootMode.DEFAULT,
            ProfilerBuild.Yes,
            1,  # L1_2_L1s
            unpack_to_dest,
            disable_format_inference,
            dest_acc,
            l1_acc,
        )

        self.passed_templates = templates
        self.passed_runtimes = runtimes
        self.current_run_type = None

        # TODO Add check here for all selected runs, to see if the profiler/counter supports them

        self.run_configs = [
            (
                templates.copy() + [PERF_RUN_TYPE(run_type)],
                runtimes.copy(),
                run_type,
            )
            for run_type in run_types
        ]

    def generate_variant_hash(self):
        NON_COMPILATION_ARGUMENTS = [
            "variant_stimuli",
            "run_configs",
            "variant_id",
            "runtime_params_struct",
            "runtime_format",
            "runtimes",
            "passed_templates",
            "passed_runtimes",
            "current_run_type",
        ]
        temp_str = [
            str(value)
            for field_name, value in self.__dict__.items()
            if field_name not in NON_COMPILATION_ARGUMENTS
        ]

        self.variant_id = sha256(str(" | ".join(temp_str)).encode()).hexdigest()

    @staticmethod
    def _dataclass_name_and_values(obj):
        """Return (name, value) pairs for dataclass fields, used as columns for the report."""
        return [(f.name, getattr(obj, f.name)) for f in fields(obj)]

    def run(self, perf_report: PerfReport, run_count=2, location="0,0"):
        results = []

        if TestConfig.MODE in [TestMode.PRODUCE, TestMode.DEFAULT]:
            for templates, runtimes, run_type in self.run_configs:
                self.current_run_type = run_type
                self.templates = templates
                self.runtimes = runtimes
                self.generate_variant_hash()
                self.build_elfs()

        if TestConfig.MODE == TestMode.PRODUCE:
            pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

        PerfConfig.TEST_COUNTER += 1

        for templates, runtimes, run_type in self.run_configs:
            self.current_run_type = run_type
            self.templates = templates
            self.runtimes = runtimes
            self.generate_variant_hash()
            variant_raw_data = []
            for run_index in range(run_count):
                self.write_runtimes_to_L1(location)
                elfs = self.run_elf_files(location)
                wait_for_tensix_operations_finished(elfs, location)

                profiler_data = Profiler.get_data(
                    self.test_name, self.variant_id, location
                )
                # TODO You add additional data collections you want here

                # Tag profiler data with run index for proper L1-to-L1 pairing
                profiler_data.df["run_index"] = run_index
                variant_raw_data.append(profiler_data)

            get_stats = Profiler.STATS_FUNCTION[run_type]
            results.append(get_stats(ProfilerData.concat(variant_raw_data)))

        # Merge results with validation
        # how="outer" keeps all markers (some may not appear in all run types)
        # validate="1:1" catches duplicate markers within each run type
        run_results = reduce(
            lambda left, right: pd.merge(
                left, right, on="marker", how="outer", validate="1:1"
            ),
            results,
        )

        # Setting header fields that are always there
        names = ["formats.input", "formats.output"] if self.formats else []
        values = (
            [self.formats.input_format, self.formats.output_format]
            if self.formats
            else []
        )

        names += ["unpack_to_dest", "dest_acc"]
        values += [self.unpack_to_dest, self.dest_acc]

        for param in self.passed_templates + self.passed_runtimes:
            for name, value in PerfConfig._dataclass_name_and_values(param):
                if value is not None:
                    names.append(name)
                    values.append(value)

        sweep = pd.DataFrame([values], columns=names)
        combined = sweep.merge(run_results, how="cross")

        perf_report.append(combined)
