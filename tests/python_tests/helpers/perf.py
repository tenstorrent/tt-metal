# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import csv
import os
import shutil
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from statistics import mean, variance

from helpers.device import run_elf_files, wait_for_tensix_operations_finished
from helpers.profiler import Profiler, build_with_profiler


@dataclass
class PerfZone:
    start: int
    end: int
    duration: int


@dataclass
class PerfThreadData:
    init: PerfZone
    tile_loop: PerfZone
    kernel: PerfZone


@dataclass
class PerfData:
    unpack: PerfThreadData
    math: PerfThreadData
    pack: PerfThreadData


def _parse_thread(thread_data) -> PerfThreadData:
    zones = {}
    markers = {"kernel", "init", "tile_loop"}

    for entry in thread_data:
        marker = entry.full_marker.marker.lower()
        if marker in markers:
            zones[marker] = PerfZone(
                start=entry.start, end=entry.end, duration=entry.duration
            )

    if len(zones) < len(markers):
        missing = markers - zones.keys()
        raise AssertionError(
            f"Missing zones after perf run: {', '.join(sorted(missing))}"
        )

    return PerfThreadData(**zones)


def process_profiler_data(profiler_data) -> PerfData:
    return PerfData(
        unpack=_parse_thread(profiler_data.unpack),
        math=_parse_thread(profiler_data.math),
        pack=_parse_thread(profiler_data.pack),
    )


def timing_l1_to_l1(perf_data: PerfData) -> int:
    """Time to perform the whole operation (compute)"""
    return (perf_data.pack.tile_loop.end - perf_data.unpack.tile_loop.start,)


def timing_unpack(perf_data: PerfData) -> int:
    return (perf_data.unpack.tile_loop.duration,)


def timing_math(perf_data: PerfData) -> int:
    return (perf_data.math.tile_loop.duration,)


def timing_pack(perf_data: PerfData) -> int:
    return (perf_data.pack.tile_loop.duration,)


def timing_l1_congestion(perf_data: PerfData) -> int:
    return (
        perf_data.unpack.tile_loop.duration,
        perf_data.pack.tile_loop.duration,
    )


def process_runs(runs, test_config):
    tile_cnt = test_config.get("tile_cnt", 1)

    return tuple(
        {
            "mean": mean(column) / tile_cnt,
            "variance": variance(column) / (tile_cnt * tile_cnt),
        }
        for column in zip(*runs)
    )


class PerfRunType(Enum):
    L1_TO_L1 = 1
    UNPACK_ISOLATE = 2
    MATH_ISOLATE = 3
    PACK_ISOLATE = 4
    L1_CONGESTION = 5


ALL_RUN_TYPES = [type for type in PerfRunType]


def perf_benchmark(test_config, run_types: list[PerfRunType], run_count=8):

    RUN_CONFIGURATIONS = {
        PerfRunType.L1_TO_L1: timing_l1_to_l1,
        PerfRunType.UNPACK_ISOLATE: timing_unpack,
        PerfRunType.MATH_ISOLATE: timing_math,
        PerfRunType.PACK_ISOLATE: timing_pack,
        PerfRunType.L1_CONGESTION: timing_l1_congestion,
    }
    SUPPORTED_RUNS = RUN_CONFIGURATIONS.keys()

    results = {}

    for type in run_types:
        assert type in SUPPORTED_RUNS, f"ERROR: run_type={type} not implemented"
        get_timing = RUN_CONFIGURATIONS[type]

        test_config["perf_run_type"] = type
        build_with_profiler(test_config)

        runs = []
        for _ in range(run_count):
            run_elf_files(test_config["testname"])
            wait_for_tensix_operations_finished()

            profiler_data = Profiler.get_data(test_config["testname"])
            perf_data = process_profiler_data(profiler_data)

            runs.append(get_timing(perf_data))

        results[type] = process_runs(runs, test_config)

    return results


def delete_reports():
    root = os.environ.get("LLK_HOME")
    if not root:
        raise AssertionError("Environment variable LLK_HOME is not set")

    path = Path(root) / "perf"
    print(path)

    if path.exists() and path.is_dir():
        shutil.rmtree(path)


def _dataclass_names(parent, obj):
    """Provides the **names** of the columns for the report"""
    return [f"{parent}.{f.name}" for f in fields(obj)]


def _dataclass_values(obj):
    """Provides the **values** of the columns for the report"""
    return [getattr(obj, f.name) for f in fields(obj)]


def report_header(params, result):
    columns = []
    for param, value in params.items():
        if is_dataclass(value):
            columns.extend(_dataclass_names(param, value))
        else:
            columns.append(param)

    for run_type, stats in result.items():
        for i, _ in enumerate(stats):
            columns.extend(
                [
                    f"mean({run_type.name}[{i}])",
                    f"variance({run_type.name}[{i}])",
                ]
            )

    return columns


def write_to_report(test_config, result):
    root = os.environ.get("LLK_HOME")
    if not root:
        raise AssertionError("Environment variable LLK_HOME is not set")

    filename = f"{test_config['testname']}.csv"
    output_path = Path(root) / "perf" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exclude = {
        "testname",
        "perf_run_type",
        "tile_cnt",
    }

    params = {
        param: value for param, value in test_config.items() if param not in exclude
    }

    row = []
    for value in params.values():
        if is_dataclass(value):
            row.extend(_dataclass_values(value))
        else:
            row.append(value)

    for stats in result.values():
        for stat in stats:
            row.extend([stat["mean"], stat["variance"]])

    # Write to CSV
    first_entry = not os.path.exists(output_path)
    with open(output_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if first_entry:
            writer.writerow(report_header(params, result))
        writer.writerow(row)
