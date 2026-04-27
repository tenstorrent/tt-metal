# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Run DeepSeek CCL trace tests under tracy for per-CCL parameter sweeps, then
summarize top-level zone durations from profile_log_device.csv.

Op time definition:
- Parse only trace id 1 from profile_log_device.csv.
- For each top-level (RISC processor type, zone) bucket, average across all
  devices.
- Op time is the max of those per-bucket averages.
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[4]
if str(REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from models.demos.deepseek_v3_b1.micro_ops.sdpa_reduce_to_all.config import (
    SDPA_REDUCE_TUNING_BY_MAX_PAYLOAD_SIZE_BYTES,
    resolve_sdpa_reduce_config,
)

DEFAULT_MAX_PAYLOAD_SIZES = [2048, 4096, 8192, 15232]
DEFAULT_SDPA_NUM_L_CHUNKS = [1, 2, 4]
DEFAULT_SDPA_COMPUTE_BLOCK_SIZES = [4, 8]
SUPPORTED_SDPA_COMPUTE_BLOCK_SIZES = [1, 2, 4, 8]
TRACE_ID = 1
CHIP_FREQ_RE = re.compile(r"CHIP_FREQ\[MHz\]:\s*(\d+(?:\.\d+)?)")

SDPA_TRACE_NUM_CORES = 8
SDPA_TRACE_NUM_LINKS = 2
SDPA_TRACE_TILE_HEIGHT = 8
SDPA_TRACE_TILE_WIDTH = 32
SDPA_TRACE_BYTES_PER_ELEMENT = 2


def format_sdpa_tuning_matrix() -> str:
    return ", ".join(
        f"{payload} -> ({tuning.num_l_chunks}, {tuning.compute_block_size})"
        for payload, tuning in sorted(SDPA_REDUCE_TUNING_BY_MAX_PAYLOAD_SIZE_BYTES.items())
    )


@dataclass(frozen=True)
class CCLConfig:
    title: str
    test_target: str
    benchmark_shape: tuple[int, int]
    default_num_links: tuple[int, ...]
    supported_num_links: tuple[int, ...]
    transport_mode: str
    env_num_links: str | None
    env_max_payload_size: str
    top_level_zones: tuple[str, ...]
    zone_labels: dict[str, str]
    fixed_setup_description: str | None = None
    show_num_links_in_report: bool = True
    env_num_l_chunks: str | None = None
    default_num_l_chunks: tuple[int, ...] = ()
    supported_num_l_chunks: tuple[int, ...] = ()
    env_compute_block_size: str | None = None
    default_compute_block_sizes: tuple[int, ...] = ()
    supported_compute_block_sizes: tuple[int, ...] = ()


@dataclass(frozen=True)
class BucketStat:
    count: int
    avg_cycles: float
    min_cycles: float
    median_cycles: float


@dataclass(frozen=True)
class RunResult:
    ccl: str
    num_links: int
    max_payload_size: int
    report_path: str
    chip_freq_mhz: float
    device_stats: dict[tuple[str, str, str], BucketStat]
    bucket_stats: dict[tuple[str, str], BucketStat]
    op_bucket: tuple[str, str]
    op_cycles: float
    num_l_chunks: int | None = None
    tiles_per_l_chunk: int | None = None
    block_size: int | None = None


CCL_CONFIGS = {
    "all_gather": CCLConfig(
        title="CCL all-gather trace perf matrix",
        test_target=("models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::" "test_ccl_all_gather"),
        benchmark_shape=(1, 7168),
        default_num_links=(1,),
        supported_num_links=(1,),
        transport_mode="single-link only",
        env_num_links="CCL_ALL_GATHER_NUM_LINKS",
        env_max_payload_size="CCL_ALL_GATHER_MAX_PAYLOAD_SIZE_BYTES",
        top_level_zones=("ALLGATHER_GATHER", "ALLGATHER_TRANSPORT"),
        zone_labels={
            "ALLGATHER_GATHER": "gather",
            "ALLGATHER_TRANSPORT": "transport",
        },
    ),
    "all_reduce": CCLConfig(
        title="CCL all-reduce trace perf matrix",
        test_target=("models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::" "test_ccl_all_reduce"),
        benchmark_shape=(1, 2048),
        default_num_links=(1, 2),
        supported_num_links=(1, 2),
        transport_mode="configurable (1 or 2 links)",
        env_num_links="CCL_ALL_REDUCE_NUM_LINKS",
        env_max_payload_size="CCL_ALL_REDUCE_MAX_PAYLOAD_SIZE_BYTES",
        top_level_zones=("CCL_SENDER_WRITER", "CCL_RECEIVER", "CCL_COMPUTE"),
        zone_labels={
            "CCL_SENDER_WRITER": "sender_writer",
            "CCL_RECEIVER": "receiver",
            "CCL_COMPUTE": "compute",
        },
    ),
    "sdpa_reduce_to_all": CCLConfig(
        title="SDPA reduce-to-all trace perf matrix",
        test_target=(
            "models/demos/deepseek_v3_b1/tests/unit_tests/test_sdpa_reduce_to_all.py::" "test_sdpa_reduce_to_all_trace"
        ),
        benchmark_shape=(8, 4096),
        default_num_links=(2,),
        supported_num_links=(2,),
        transport_mode="fixed forwarder topology",
        env_num_links=None,
        env_max_payload_size="SDPA_REDUCE_TO_ALL_MAX_PAYLOAD_SIZE_BYTES",
        top_level_zones=(
            "SDPA_REDUCE_READER",
            "SDPA_REDUCE_WRITER",
            "SDPA_REDUCE_COMPUTE",
            "SDPA_REDUCE_FORWARDER",
        ),
        zone_labels={
            "SDPA_REDUCE_READER": "reader",
            "SDPA_REDUCE_WRITER": "writer",
            "SDPA_REDUCE_COMPUTE": "compute",
            "SDPA_REDUCE_FORWARDER": "forwarder",
        },
        fixed_setup_description=(
            "2 forwarder cores total; each forwarder core binds to one physical link, "
            "with BRISC/NCRISC handling the two traffic directions on that core."
        ),
        show_num_links_in_report=False,
        env_num_l_chunks="SDPA_REDUCE_TO_ALL_NUM_L_CHUNKS",
        default_num_l_chunks=tuple(DEFAULT_SDPA_NUM_L_CHUNKS),
        supported_num_l_chunks=tuple(DEFAULT_SDPA_NUM_L_CHUNKS),
        env_compute_block_size="SDPA_REDUCE_TO_ALL_COMPUTE_BLOCK_SIZE",
        default_compute_block_sizes=tuple(DEFAULT_SDPA_COMPUTE_BLOCK_SIZES),
        supported_compute_block_sizes=tuple(SUPPORTED_SDPA_COMPUTE_BLOCK_SIZES),
    ),
}

RISC_ORDER = {
    "BRISC": 0,
    "NCRISC": 1,
    "TRISC0": 2,
    "TRISC1": 3,
    "TRISC2": 4,
    "ERISC": 5,
}

PROFILE_LOG_REQUIRED_COLUMNS = {
    "PCIe slot",
    "core_x",
    "core_y",
    "RISC processor type",
    "time[cycles since reset]",
    "trace id",
    "trace id counter",
    "zone name",
    "type",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a DeepSeek CCL trace perf matrix and summarize top-level " "profile_log_device.csv zone durations."
        )
    )
    parser.add_argument(
        "--ccl",
        choices=sorted(CCL_CONFIGS),
        default=None,
        help="CCL to profile. If omitted, runs all configured CCLs.",
    )
    parser.add_argument(
        "--num-links",
        default=None,
        help="Comma or space separated num_links values for CCLs that expose num_links as a sweep knob.",
    )
    parser.add_argument(
        "--max-payload-sizes",
        default=",".join(str(v) for v in DEFAULT_MAX_PAYLOAD_SIZES),
        help="Comma or space separated max payload sizes in bytes (default: 2048,4096,8192,15232).",
    )
    parser.add_argument(
        "--sdpa-num-l-chunks",
        default=None,
        help=(
            "Comma or space separated SDPA num_l_chunks values. " "Used only for sdpa_reduce_to_all (default: 1,2,4)."
        ),
    )
    parser.add_argument(
        "--sdpa-compute-block-sizes",
        default=None,
        help=("Comma or space separated SDPA compute block sizes. " "Used only for sdpa_reduce_to_all (default: 4,8)."),
    )
    parser.add_argument(
        "--test-target",
        default=None,
        help="Optional pytest target override. Defaults to the selected CCL trace test.",
    )
    parser.add_argument(
        "--output-dir",
        default="generated/profiler",
        help="Directory to write markdown summaries (default: generated/profiler).",
    )
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    if raw is None:
        return []
    values = [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
    result = []
    for item in values:
        try:
            parsed = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid integer value: {item!r}") from exc
        if parsed <= 0:
            raise ValueError(f"Values must be > 0, got {parsed}")
        result.append(parsed)
    return result


def list_profile_logs(report_root: Path) -> list[Path]:
    if not report_root.exists():
        return []
    return list(report_root.rglob("profile_log_device.csv"))


def to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(row: list[str]) -> str:
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"

    lines = [fmt_row(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def format_shape(shape: tuple[int, int]) -> str:
    return f"[{shape[0]}, {shape[1]}]"


def format_float(value: float | None, decimals: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def is_sdpa_chunk_sweep(config: CCLConfig) -> bool:
    return config.env_num_l_chunks is not None


def derive_sdpa_total_l_tiles(config: CCLConfig) -> int:
    batch_size, per_device_l_width = config.benchmark_shape
    if per_device_l_width % SDPA_TRACE_NUM_CORES != 0:
        raise RuntimeError(
            "SDPA benchmark width must be divisible by the worker core count: "
            f"{per_device_l_width} % {SDPA_TRACE_NUM_CORES} != 0"
        )

    per_core_l_width = per_device_l_width // SDPA_TRACE_NUM_CORES
    if batch_size % SDPA_TRACE_TILE_HEIGHT != 0:
        raise RuntimeError(
            "SDPA benchmark batch size must be divisible by the tile height: "
            f"{batch_size} % {SDPA_TRACE_TILE_HEIGHT} != 0"
        )
    if per_core_l_width % SDPA_TRACE_TILE_WIDTH != 0:
        raise RuntimeError(
            "SDPA per-core width must be divisible by the tile width: "
            f"{per_core_l_width} % {SDPA_TRACE_TILE_WIDTH} != 0"
        )

    input_l_num_pages = (batch_size // SDPA_TRACE_TILE_HEIGHT) * (per_core_l_width // SDPA_TRACE_TILE_WIDTH)
    pnh = 8
    dh = input_l_num_pages * SDPA_TRACE_TILE_WIDTH
    dht = dh // SDPA_TRACE_TILE_WIDTH
    pnht = pnh // SDPA_TRACE_TILE_HEIGHT
    return pnht * dht


def derive_sdpa_sweep_metadata(
    config: CCLConfig,
    num_l_chunks: int,
    compute_block_size: int,
    max_payload_size: int | None = None,
) -> tuple[int, int, int, int]:
    if num_l_chunks <= 0:
        raise RuntimeError(f"num_l_chunks must be > 0, got {num_l_chunks}")

    resolved = resolve_sdpa_reduce_config(
        batch_size=config.benchmark_shape[0],
        l_width=config.benchmark_shape[1] // SDPA_TRACE_NUM_CORES,
        num_cores=SDPA_TRACE_NUM_CORES,
        tile_height=SDPA_TRACE_TILE_HEIGHT,
        tile_width=SDPA_TRACE_TILE_WIDTH,
        bytes_per_element=SDPA_TRACE_BYTES_PER_ELEMENT,
        num_links=SDPA_TRACE_NUM_LINKS,
        max_payload_size_bytes=max_payload_size,
        num_l_chunks_override=num_l_chunks,
        compute_block_size_override=compute_block_size,
    )
    return (
        resolved.out_tiles,
        resolved.tiles_per_l_chunk,
        resolved.l_chunk_size_bytes,
        resolved.compute_block_size,
    )


def is_valid_sdpa_sweep_point(
    config: CCLConfig, max_payload_size: int, num_l_chunks: int, compute_block_size: int
) -> bool:
    try:
        derive_sdpa_sweep_metadata(config, num_l_chunks, compute_block_size, max_payload_size)
    except ValueError:
        return False
    return True


def sort_results_for_display(config: CCLConfig, results: list[RunResult]) -> list[RunResult]:
    if is_sdpa_chunk_sweep(config):
        return sorted(
            results, key=lambda result: (result.num_l_chunks or 0, result.block_size or 0, result.max_payload_size)
        )
    if config.show_num_links_in_report:
        return sorted(results, key=lambda result: (result.num_links, result.max_payload_size))
    return sorted(results, key=lambda result: result.max_payload_size)


def get_selected_ccls(selected_ccl: str | None) -> list[str]:
    if selected_ccl is not None:
        return [selected_ccl]
    return list(CCL_CONFIGS)


def write_text_sync(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def get_repo_root() -> Path:
    tt_metal_home = os.environ.get("TT_METAL_HOME")
    if not tt_metal_home:
        raise RuntimeError("TT_METAL_HOME must be set")

    repo_root = Path(tt_metal_home).expanduser().resolve()
    if not repo_root.is_dir():
        raise RuntimeError(f"TT_METAL_HOME does not point to a directory: {repo_root}")

    return repo_root


def run_tracy_pytest(test_target: str, env: dict[str, str], repo_root: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "tracy",
        "-r",
        "-p",
        "-v",
        "-m",
        "pytest",
        "-svv",
        test_target,
    ]
    subprocess.run(cmd, env=env, cwd=repo_root, check=True)


def is_header_row(row: list[str]) -> bool:
    return PROFILE_LOG_REQUIRED_COLUMNS.issubset({cell.strip() for cell in row})


def load_header(reader: csv.reader) -> dict[str, int]:
    for row in reader:
        if not row:
            continue
        if is_header_row(row):
            header = [cell.strip() for cell in row]
            return {name: idx for idx, name in enumerate(header)}
    return {}


def parse_int(text: str) -> int | None:
    value = text.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def device_sort_key(device_id: str) -> tuple[int, int, str]:
    try:
        return (0, int(device_id), "")
    except ValueError:
        return (1, 0, device_id)


def risc_sort_key(risc_type: str) -> tuple[int, str]:
    return (RISC_ORDER.get(risc_type, 99), risc_type)


def bucket_sort_key(bucket: tuple[str, str], config: CCLConfig) -> tuple[int, tuple[int, str]]:
    risc_type, zone_name = bucket
    zone_index = (
        config.top_level_zones.index(zone_name) if zone_name in config.top_level_zones else len(config.top_level_zones)
    )
    return (zone_index, risc_sort_key(risc_type))


def bucket_display(bucket: tuple[str, str], config: CCLConfig) -> str:
    risc_type, zone_name = bucket
    zone_label = config.zone_labels.get(zone_name, zone_name)
    return f"{risc_type} {zone_label}"


def parse_chip_freq_mhz(profile_log: Path) -> float:
    with profile_log.open(encoding="utf-8") as handle:
        first_line = handle.readline()
    match = CHIP_FREQ_RE.search(first_line)
    if match is None:
        raise RuntimeError(f"Could not parse CHIP_FREQ[MHz] from {profile_log}")
    return float(match.group(1))


def cycles_to_ns(cycles: float, chip_freq_mhz: float) -> float:
    return (cycles * 1000.0) / chip_freq_mhz


def cycles_to_us(cycles: float, chip_freq_mhz: float) -> float:
    return cycles / chip_freq_mhz


def make_bucket_stat(values: list[float]) -> BucketStat:
    if not values:
        raise ValueError("Cannot build stats from an empty value list")
    return BucketStat(
        count=len(values),
        avg_cycles=sum(values) / len(values),
        min_cycles=min(values),
        median_cycles=statistics.median(values),
    )


def summarize_profile_log(
    profile_log: Path,
    config: CCLConfig,
) -> tuple[float, dict[tuple[str, str, str], BucketStat], dict[tuple[str, str], BucketStat], tuple[str, str], float]:
    chip_freq_mhz = parse_chip_freq_mhz(profile_log)
    zone_set = set(config.top_level_zones)

    with profile_log.open(newline="", encoding="utf-8") as handle:
        handle.readline()
        reader = csv.reader(handle)
        header_index = load_header(reader)
        if not header_index:
            raise RuntimeError(f"Header row not found in {profile_log}")

        missing = [column for column in PROFILE_LOG_REQUIRED_COLUMNS if column not in header_index]
        if missing:
            raise RuntimeError(f"Missing columns in {profile_log}: {', '.join(sorted(missing))}")

        run_host_idx = header_index.get("run host ID")
        stacks: dict[tuple[str, str, str, str, str, str, str], list[tuple[str, int]]] = defaultdict(list)
        durations_by_core_iter: dict[tuple[str, str, str, str, str, str], int] = defaultdict(int)
        unmatched_ends = 0
        unmatched_starts = 0
        mismatched_ends = 0
        negative_durations = 0

        for row in reader:
            if not row or is_header_row(row):
                continue

            trace_id = parse_int(row[header_index["trace id"]])
            if trace_id is None or trace_id != TRACE_ID:
                continue

            zone_name = row[header_index["zone name"]].strip()
            if zone_name not in zone_set:
                continue

            event_type = row[header_index["type"]].strip()
            if event_type not in ("ZONE_START", "ZONE_END"):
                continue

            time_cycles = parse_int(row[header_index["time[cycles since reset]"]])
            if time_cycles is None:
                continue

            device_id = row[header_index["PCIe slot"]].strip()
            core_x = row[header_index["core_x"]].strip()
            core_y = row[header_index["core_y"]].strip()
            risc_type = row[header_index["RISC processor type"]].strip()
            trace_counter = row[header_index["trace id counter"]].strip()
            run_host_id = row[run_host_idx].strip() if run_host_idx is not None and run_host_idx < len(row) else ""
            iter_key = run_host_id or trace_counter or row[header_index["time[cycles since reset]"]].strip()
            stream_key = (
                device_id,
                core_x,
                core_y,
                risc_type,
                str(trace_id),
                trace_counter,
                run_host_id,
            )

            if event_type == "ZONE_START":
                stacks[stream_key].append((zone_name, time_cycles))
                continue

            stack = stacks[stream_key]
            if not stack:
                unmatched_ends += 1
                continue

            active_zone, start_time = stack[-1]
            if active_zone != zone_name:
                mismatched_ends += 1
                continue

            stack.pop()
            duration = time_cycles - start_time
            if duration < 0:
                negative_durations += 1
                continue

            durations_by_core_iter[(device_id, risc_type, zone_name, iter_key, core_x, core_y)] += duration

        for stack in stacks.values():
            unmatched_starts += len(stack)

    if unmatched_ends or unmatched_starts or mismatched_ends or negative_durations:
        print(
            "WARNING: {} had unmatched starts={}, unmatched ends={}, mismatched ends={}, negative durations={}".format(
                profile_log,
                unmatched_starts,
                unmatched_ends,
                mismatched_ends,
                negative_durations,
            ),
            file=sys.stderr,
        )

    if not durations_by_core_iter:
        raise RuntimeError(f"No matching top-level zone durations found in {profile_log} for trace id {TRACE_ID}")

    per_iter_totals: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for (device_id, risc_type, zone_name, iter_key, _core_x, _core_y), total_cycles in durations_by_core_iter.items():
        per_iter_totals[(device_id, risc_type, zone_name, iter_key)].append(total_cycles)

    device_bucket_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (device_id, risc_type, zone_name, _iter_key), core_totals in per_iter_totals.items():
        device_bucket_values[(device_id, risc_type, zone_name)].append(float(max(core_totals)))

    device_stats = {key: make_bucket_stat(values) for key, values in device_bucket_values.items()}

    bucket_device_averages: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (_device_id, risc_type, zone_name), stat in device_stats.items():
        bucket_device_averages[(risc_type, zone_name)].append(stat.avg_cycles)

    bucket_stats = {key: make_bucket_stat(values) for key, values in bucket_device_averages.items()}
    if not bucket_stats:
        raise RuntimeError(f"No top-level bucket stats produced from {profile_log}")

    op_bucket = max(bucket_stats, key=lambda key: bucket_stats[key].avg_cycles)
    op_cycles = bucket_stats[op_bucket].avg_cycles
    return chip_freq_mhz, device_stats, bucket_stats, op_bucket, op_cycles


def collect_bucket_order(results: list[RunResult], config: CCLConfig) -> list[tuple[str, str]]:
    buckets = set()
    for result in results:
        buckets.update(result.bucket_stats.keys())
    return sorted(buckets, key=lambda bucket: bucket_sort_key(bucket, config))


def render_summary_section(
    config: CCLConfig,
    test_target: str,
    num_links_list: list[int],
    num_l_chunks_list: list[int],
    compute_block_size_list: list[int],
    max_payload_sizes: list[int],
    results: list[RunResult],
) -> str:
    lines = [
        f"Test target: {test_target}",
        f"Configured benchmark shape: {format_shape(config.benchmark_shape)}",
        f"Configured transport mode: {config.transport_mode}",
    ]
    if config.show_num_links_in_report:
        lines.append(f"Configured num_links: {', '.join(str(v) for v in num_links_list)}")
    elif config.fixed_setup_description is not None:
        lines.append(f"Setup: {config.fixed_setup_description}")
    if is_sdpa_chunk_sweep(config):
        lines.append(f"Configured num_l_chunks: {', '.join(str(v) for v in num_l_chunks_list)}")
        lines.append(f"Configured compute_block_sizes: {', '.join(str(v) for v in compute_block_size_list)}")
        total_l_tiles = derive_sdpa_total_l_tiles(config)
        lines.append("Sweep: num_l_chunks x compute_block_size x max_payload_size_bytes")
        lines.append(f"For this shape, total_l_tiles per worker = {total_l_tiles}.")
        lines.append(f"Tuned payload matrix: {format_sdpa_tuning_matrix()}.")
        lines.append(
            "Invalid cells are shown as n/a when a forced L chunk payload exceeds the configured fabric max payload "
            "or a compute block size does not divide the total L tiles."
        )
    lines.extend(
        [
            f"Trace id: {TRACE_ID}",
            "Source: profile_log_device.csv only (top-level zones only; no micro-profiling).",
            "Conversion: us = cycles / CHIP_FREQ[MHz], ns = cycles * 1000 / CHIP_FREQ[MHz].",
            "Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).",
            f"Top-level zones: {', '.join(config.top_level_zones)}",
            "",
        ]
    )

    if not results:
        lines.append("Results pending.")
        lines.append("")
        return "\n".join(lines)

    observed_freqs = sorted({result.chip_freq_mhz for result in results})
    lines.append("Observed CHIP_FREQ[MHz]: " + ", ".join(format_float(freq, 3) for freq in observed_freqs))
    lines.append("")

    summary_rows = []
    sorted_results = sort_results_for_display(config, results)
    if is_sdpa_chunk_sweep(config):
        summary_headers = [
            "num_l_chunks",
            "tiles_per_l_chunk",
            "block_size",
            "max_payload_size_bytes",
            "op_time_us",
            "op_cycles",
            "op_bucket",
        ]
        result_lookup = {}
        for result in sorted_results:
            row = [
                str(result.num_l_chunks),
                str(result.tiles_per_l_chunk),
                str(result.block_size),
                str(result.max_payload_size),
                format_float(cycles_to_us(result.op_cycles, result.chip_freq_mhz), 3),
                format_float(result.op_cycles, 1),
                bucket_display(result.op_bucket, config),
            ]
            summary_rows.append(row)
            result_lookup[(result.num_l_chunks, result.block_size, result.max_payload_size)] = result

        matrix_headers = ["num_l_chunks, block_size \\ max_payload"] + [str(payload) for payload in max_payload_sizes]
        matrix_rows_us = []
        matrix_rows_cycles = []
        for num_l_chunks in num_l_chunks_list:
            for compute_block_size in compute_block_size_list:
                try:
                    derive_sdpa_sweep_metadata(config, num_l_chunks, compute_block_size)
                    row_label = f"{num_l_chunks}, {compute_block_size}"
                except ValueError:
                    row_label = f"{num_l_chunks}, {compute_block_size}"
                row_us = [row_label]
                row_cycles = [row_label]
                for max_payload in max_payload_sizes:
                    result = result_lookup.get((num_l_chunks, compute_block_size, max_payload))
                    value_us = cycles_to_us(result.op_cycles, result.chip_freq_mhz) if result is not None else None
                    value_cycles = result.op_cycles if result is not None else None
                    row_us.append(format_float(value_us, 3))
                    row_cycles.append(format_float(value_cycles, 1))
                matrix_rows_us.append(row_us)
                matrix_rows_cycles.append(row_cycles)
    elif config.show_num_links_in_report:
        summary_headers = ["num_links", "max_payload_size_bytes", "op_time_us", "op_cycles", "op_bucket"]
        result_lookup = {}
        for result in sorted_results:
            row = [
                str(result.num_links),
                str(result.max_payload_size),
                format_float(cycles_to_us(result.op_cycles, result.chip_freq_mhz), 3),
                format_float(result.op_cycles, 1),
                bucket_display(result.op_bucket, config),
            ]
            summary_rows.append(row)
            result_lookup[(result.num_links, result.max_payload_size)] = result

        matrix_headers = ["num_links \\ max_payload"] + [str(payload) for payload in max_payload_sizes]
        matrix_rows_us = []
        matrix_rows_cycles = []
        for num_links in num_links_list:
            row_us = [str(num_links)]
            row_cycles = [str(num_links)]
            for max_payload in max_payload_sizes:
                result = result_lookup.get((num_links, max_payload))
                value_us = cycles_to_us(result.op_cycles, result.chip_freq_mhz) if result is not None else None
                value_cycles = result.op_cycles if result is not None else None
                row_us.append(format_float(value_us, 3))
                row_cycles.append(format_float(value_cycles, 1))
            matrix_rows_us.append(row_us)
            matrix_rows_cycles.append(row_cycles)
    else:
        summary_headers = ["max_payload_size_bytes", "op_time_us", "op_cycles", "op_bucket"]
        result_lookup = {}
        for result in sorted_results:
            row = [
                str(result.max_payload_size),
                format_float(cycles_to_us(result.op_cycles, result.chip_freq_mhz), 3),
                format_float(result.op_cycles, 1),
                bucket_display(result.op_bucket, config),
            ]
            summary_rows.append(row)
            result_lookup[result.max_payload_size] = result

        matrix_headers = ["metric"] + [str(payload) for payload in max_payload_sizes]
        row_us = ["op_time_us"]
        row_cycles = ["op_cycles"]
        for max_payload in max_payload_sizes:
            result = result_lookup.get(max_payload)
            value_us = cycles_to_us(result.op_cycles, result.chip_freq_mhz) if result is not None else None
            value_cycles = result.op_cycles if result is not None else None
            row_us.append(format_float(value_us, 3))
            row_cycles.append(format_float(value_cycles, 1))
        matrix_rows_us = [row_us]
        matrix_rows_cycles = [row_cycles]

    lines.append("### Per-run overview")
    lines.append("")
    lines.append(format_markdown_table(summary_headers, summary_rows))
    lines.append("")

    lines.append("### Perf matrix (op time, us)")
    lines.append("")
    lines.append(format_markdown_table(matrix_headers, matrix_rows_us))
    lines.append("")

    lines.append("### Perf matrix (op time, cycles)")
    lines.append("")
    lines.append(format_markdown_table(matrix_headers, matrix_rows_cycles))
    lines.append("")

    return "\n".join(lines)


def render_summary(
    selected_ccls: list[str],
    test_targets: dict[str, str],
    timestamp: str,
    details_rel: str,
    num_links_by_ccl: dict[str, list[int]],
    num_l_chunks_by_ccl: dict[str, list[int]],
    compute_block_sizes_by_ccl: dict[str, list[int]],
    max_payload_sizes: list[int],
    results_by_ccl: dict[str, list[RunResult]],
) -> str:
    lines = [
        "# DeepSeek CCL trace perf matrix",
        "",
        f"Timestamp: {timestamp}",
        f"CCLs: {', '.join(selected_ccls)}",
        f"Details file: {details_rel}",
        "",
    ]

    for ccl in selected_ccls:
        config = CCL_CONFIGS[ccl]
        lines.append(f"## {ccl}")
        lines.append("")
        lines.append(
            render_summary_section(
                config=config,
                test_target=test_targets[ccl],
                num_links_list=num_links_by_ccl[ccl],
                num_l_chunks_list=num_l_chunks_by_ccl[ccl],
                compute_block_size_list=compute_block_sizes_by_ccl[ccl],
                max_payload_sizes=max_payload_sizes,
                results=results_by_ccl[ccl],
            ).rstrip()
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def render_details_section(
    config: CCLConfig,
    test_target: str,
    num_links_list: list[int],
    num_l_chunks_list: list[int],
    compute_block_size_list: list[int],
    results: list[RunResult],
) -> str:
    lines = [
        f"Test target: {test_target}",
        f"Configured benchmark shape: {format_shape(config.benchmark_shape)}",
        f"Configured transport mode: {config.transport_mode}",
    ]
    if config.show_num_links_in_report:
        lines.append(f"Configured num_links: {', '.join(str(v) for v in num_links_list)}")
    elif config.fixed_setup_description is not None:
        lines.append(f"Setup: {config.fixed_setup_description}")
    if is_sdpa_chunk_sweep(config):
        lines.append(f"Configured num_l_chunks: {', '.join(str(v) for v in num_l_chunks_list)}")
        lines.append(f"Configured compute_block_sizes: {', '.join(str(v) for v in compute_block_size_list)}")
        total_l_tiles = derive_sdpa_total_l_tiles(config)
        lines.append("Sweep: num_l_chunks x compute_block_size x max_payload_size_bytes")
        lines.append(f"For this shape, total_l_tiles per worker = {total_l_tiles}.")
        lines.append(f"Tuned payload matrix: {format_sdpa_tuning_matrix()}.")
        lines.append(
            "Invalid cells are shown as n/a when a forced L chunk payload exceeds the configured fabric max payload "
            "or a compute block size does not divide the total L tiles."
        )
    lines.extend(
        [
            f"Trace id: {TRACE_ID}",
            "Source: profile_log_device.csv only (top-level zones only; no micro-profiling).",
            "",
        ]
    )

    if not results:
        lines.append("Results pending.")
        lines.append("")
        return "\n".join(lines)

    bucket_order = collect_bucket_order(results, config)
    for result in sort_results_for_display(config, results):
        if is_sdpa_chunk_sweep(config):
            lines.append(
                "### num_l_chunks={}, tiles_per_l_chunk={}, block_size={}, max_payload_size_bytes={}".format(
                    result.num_l_chunks,
                    result.tiles_per_l_chunk,
                    result.block_size,
                    result.max_payload_size,
                )
            )
        elif config.show_num_links_in_report:
            lines.append(f"### num_links={result.num_links}, max_payload_size_bytes={result.max_payload_size}")
        else:
            lines.append(f"### max_payload_size_bytes={result.max_payload_size}")
        lines.append("")
        lines.append(f"Report: {result.report_path}")
        lines.append(f"CHIP_FREQ[MHz]: {format_float(result.chip_freq_mhz, 3)}")
        lines.append("")

        device_headers = [
            "PCIe slot",
            "RISC processor type",
            "ZONE",
            "COUNT",
            "AVG_CYCLES",
            "AVG_NS",
            "AVG_US",
            "MIN_CYCLES",
            "MEDIAN_CYCLES",
        ]
        device_rows = []
        for (device_id, risc_type, zone_name), stat in sorted(
            result.device_stats.items(),
            key=lambda item: (
                device_sort_key(item[0][0]),
                bucket_sort_key((item[0][1], item[0][2]), config),
            ),
        ):
            device_rows.append(
                [
                    device_id,
                    risc_type,
                    zone_name,
                    str(stat.count),
                    format_float(stat.avg_cycles, 1),
                    format_float(cycles_to_ns(stat.avg_cycles, result.chip_freq_mhz), 3),
                    format_float(cycles_to_us(stat.avg_cycles, result.chip_freq_mhz), 3),
                    format_float(stat.min_cycles, 1),
                    format_float(stat.median_cycles, 1),
                ]
            )

        lines.append("#### Per-device averages")
        lines.append("")
        lines.append(format_markdown_table(device_headers, device_rows))
        lines.append("")

        bucket_headers = [
            "RISC processor type",
            "ZONE",
            "DEVICE_COUNT",
            "AVG_CYCLES",
            "AVG_NS",
            "AVG_US",
            "MIN_DEVICE_AVG_CYCLES",
            "MEDIAN_DEVICE_AVG_CYCLES",
        ]
        bucket_rows = []
        for bucket in bucket_order:
            stat = result.bucket_stats.get(bucket)
            if stat is None:
                continue
            risc_type, zone_name = bucket
            bucket_rows.append(
                [
                    risc_type,
                    zone_name,
                    str(stat.count),
                    format_float(stat.avg_cycles, 1),
                    format_float(cycles_to_ns(stat.avg_cycles, result.chip_freq_mhz), 3),
                    format_float(cycles_to_us(stat.avg_cycles, result.chip_freq_mhz), 3),
                    format_float(stat.min_cycles, 1),
                    format_float(stat.median_cycles, 1),
                ]
            )

        lines.append("#### Bucket averages across devices")
        lines.append("")
        lines.append(format_markdown_table(bucket_headers, bucket_rows))
        lines.append("")
        lines.append(
            "Final op time: {} us ({} cycles), bucket={}".format(
                format_float(cycles_to_us(result.op_cycles, result.chip_freq_mhz), 3),
                format_float(result.op_cycles, 1),
                bucket_display(result.op_bucket, config),
            )
        )
        lines.append("")

    return "\n".join(lines)


def render_details(
    selected_ccls: list[str],
    test_targets: dict[str, str],
    timestamp: str,
    num_links_by_ccl: dict[str, list[int]],
    num_l_chunks_by_ccl: dict[str, list[int]],
    compute_block_sizes_by_ccl: dict[str, list[int]],
    results_by_ccl: dict[str, list[RunResult]],
) -> str:
    lines = [
        "# DeepSeek CCL trace perf matrix details",
        "",
        f"Timestamp: {timestamp}",
        f"CCLs: {', '.join(selected_ccls)}",
        "",
    ]

    for ccl in selected_ccls:
        config = CCL_CONFIGS[ccl]
        lines.append(f"## {ccl}")
        lines.append("")
        lines.append(
            render_details_section(
                config=config,
                test_target=test_targets[ccl],
                num_links_list=num_links_by_ccl[ccl],
                num_l_chunks_list=num_l_chunks_by_ccl[ccl],
                compute_block_size_list=compute_block_sizes_by_ccl[ccl],
                results=results_by_ccl[ccl],
            ).rstrip()
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    selected_ccls = get_selected_ccls(args.ccl)
    if len(selected_ccls) > 1 and args.test_target is not None:
        raise RuntimeError("--test-target requires --ccl when profiling multiple CCLs")

    max_payload_sizes = parse_int_list(args.max_payload_sizes)
    if not max_payload_sizes:
        raise RuntimeError("max payload sizes list is empty")

    repo_root = get_repo_root()
    report_root = repo_root / "generated" / "profiler" / "reports"
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    summary_path = output_dir / f"ccl_trace_perf_summary_{timestamp}.md"
    details_path = output_dir / f"ccl_trace_perf_details_{timestamp}.md"
    details_rel = to_repo_relative(details_path, repo_root)

    test_targets = {}
    num_links_by_ccl = {}
    num_l_chunks_by_ccl = {}
    compute_block_sizes_by_ccl = {}
    for ccl in selected_ccls:
        config = CCL_CONFIGS[ccl]
        test_targets[ccl] = args.test_target or config.test_target
        if config.env_num_links is None:
            if args.num_links is not None:
                raise RuntimeError(f"{ccl} does not expose num_links as a sweep knob")
            num_links_list = list(config.default_num_links)
        else:
            num_links_list = (
                parse_int_list(args.num_links) if args.num_links is not None else list(config.default_num_links)
            )
        if not num_links_list:
            raise RuntimeError("num_links list is empty")
        invalid_num_links = [value for value in num_links_list if value not in config.supported_num_links]
        if invalid_num_links:
            supported = ", ".join(str(value) for value in config.supported_num_links)
            invalid = ", ".join(str(value) for value in invalid_num_links)
            raise RuntimeError(f"{ccl} supports num_links in {{{supported}}}, got {{{invalid}}}")
        num_links_by_ccl[ccl] = num_links_list

        if is_sdpa_chunk_sweep(config):
            num_l_chunks_list = (
                parse_int_list(args.sdpa_num_l_chunks)
                if args.sdpa_num_l_chunks is not None
                else list(config.default_num_l_chunks)
            )
            if not num_l_chunks_list:
                raise RuntimeError("num_l_chunks list is empty")
            invalid_num_l_chunks = [value for value in num_l_chunks_list if value not in config.supported_num_l_chunks]
            if invalid_num_l_chunks:
                supported = ", ".join(str(value) for value in config.supported_num_l_chunks)
                invalid = ", ".join(str(value) for value in invalid_num_l_chunks)
                raise RuntimeError(f"{ccl} supports num_l_chunks in {{{supported}}}, got {{{invalid}}}")
            compute_block_size_list = (
                parse_int_list(args.sdpa_compute_block_sizes)
                if args.sdpa_compute_block_sizes is not None
                else list(config.default_compute_block_sizes)
            )
            if not compute_block_size_list:
                raise RuntimeError("compute block size list is empty")
            invalid_compute_block_sizes = [
                value for value in compute_block_size_list if value not in config.supported_compute_block_sizes
            ]
            if invalid_compute_block_sizes:
                supported = ", ".join(str(value) for value in config.supported_compute_block_sizes)
                invalid = ", ".join(str(value) for value in invalid_compute_block_sizes)
                raise RuntimeError(f"{ccl} supports compute block sizes in {{{supported}}}, got {{{invalid}}}")
            for num_l_chunks in num_l_chunks_list:
                for compute_block_size in compute_block_size_list:
                    derive_sdpa_sweep_metadata(config, num_l_chunks, compute_block_size)
            num_l_chunks_by_ccl[ccl] = num_l_chunks_list
            compute_block_sizes_by_ccl[ccl] = compute_block_size_list
        else:
            num_l_chunks_by_ccl[ccl] = []
            compute_block_sizes_by_ccl[ccl] = []

    results_by_ccl: dict[str, list[RunResult]] = {ccl: [] for ccl in selected_ccls}
    for ccl in selected_ccls:
        config = CCL_CONFIGS[ccl]
        test_target = test_targets[ccl]
        num_links_list = num_links_by_ccl[ccl]
        num_l_chunks_list = num_l_chunks_by_ccl[ccl]
        compute_block_size_list = compute_block_sizes_by_ccl[ccl]
        run_num_l_chunks = num_l_chunks_list if is_sdpa_chunk_sweep(config) else [None]
        run_compute_block_sizes = compute_block_size_list if is_sdpa_chunk_sweep(config) else [None]

        for max_payload in max_payload_sizes:
            for num_links in num_links_list:
                for num_l_chunks in run_num_l_chunks:
                    for compute_block_size in run_compute_block_sizes:
                        if (
                            num_l_chunks is not None
                            and compute_block_size is not None
                            and not is_valid_sdpa_sweep_point(config, max_payload, num_l_chunks, compute_block_size)
                        ):
                            continue

                        env = os.environ.copy()
                        if config.env_num_links is not None:
                            env[config.env_num_links] = str(num_links)
                        env[config.env_max_payload_size] = str(max_payload)
                        if config.env_num_l_chunks is not None and num_l_chunks is not None:
                            env[config.env_num_l_chunks] = str(num_l_chunks)
                        if config.env_compute_block_size is not None and compute_block_size is not None:
                            env[config.env_compute_block_size] = str(compute_block_size)

                        before = set(list_profile_logs(report_root))
                        run_start = time.time()

                        run_tracy_pytest(test_target, env, repo_root)

                        after = list_profile_logs(report_root)
                        new_logs = [path for path in after if path not in before]
                        if not new_logs:
                            new_logs = [path for path in after if path.stat().st_mtime >= run_start - 1.0]
                        if not new_logs:
                            if not report_root.exists():
                                raise RuntimeError(
                                    "Profiler report directory was not created after the first run: "
                                    f"{report_root}. Check TT_METAL_HOME and profiler output setup."
                                )
                            raise RuntimeError(
                                "No new profile_log_device.csv found after the test run under " f"{report_root}"
                            )

                        profile_log = max(new_logs, key=lambda path: path.stat().st_mtime)
                        profile_log_rel = to_repo_relative(profile_log, repo_root)
                        chip_freq_mhz, device_stats, bucket_stats, op_bucket, op_cycles = summarize_profile_log(
                            profile_log,
                            config,
                        )
                        tiles_per_l_chunk = None
                        block_size = None
                        if num_l_chunks is not None and compute_block_size is not None:
                            (
                                _total_l_tiles,
                                tiles_per_l_chunk,
                                _l_chunk_size_bytes,
                                block_size,
                            ) = derive_sdpa_sweep_metadata(config, num_l_chunks, compute_block_size, max_payload)
                        results_by_ccl[ccl].append(
                            RunResult(
                                ccl=ccl,
                                num_links=num_links,
                                max_payload_size=max_payload,
                                report_path=profile_log_rel,
                                chip_freq_mhz=chip_freq_mhz,
                                device_stats=device_stats,
                                bucket_stats=bucket_stats,
                                op_bucket=op_bucket,
                                op_cycles=op_cycles,
                                num_l_chunks=num_l_chunks,
                                tiles_per_l_chunk=tiles_per_l_chunk,
                                block_size=block_size,
                            )
                        )

                        summary_text = render_summary(
                            selected_ccls=selected_ccls,
                            test_targets=test_targets,
                            timestamp=timestamp,
                            details_rel=details_rel,
                            num_links_by_ccl=num_links_by_ccl,
                            num_l_chunks_by_ccl=num_l_chunks_by_ccl,
                            compute_block_sizes_by_ccl=compute_block_sizes_by_ccl,
                            max_payload_sizes=max_payload_sizes,
                            results_by_ccl=results_by_ccl,
                        )
                        details_text = render_details(
                            selected_ccls=selected_ccls,
                            test_targets=test_targets,
                            timestamp=timestamp,
                            num_links_by_ccl=num_links_by_ccl,
                            num_l_chunks_by_ccl=num_l_chunks_by_ccl,
                            compute_block_sizes_by_ccl=compute_block_sizes_by_ccl,
                            results_by_ccl=results_by_ccl,
                        )
                        write_text_sync(summary_path, summary_text)
                        write_text_sync(details_path, details_text)

                        if is_sdpa_chunk_sweep(config):
                            print(
                                "Updated reports after ccl={}, num_l_chunks={}, compute_block_size={}, "
                                "max_payload_size_bytes={}: op_time={} us ({:.1f} cycles)".format(
                                    ccl,
                                    num_l_chunks,
                                    compute_block_size,
                                    max_payload,
                                    format_float(cycles_to_us(op_cycles, chip_freq_mhz), 3),
                                    op_cycles,
                                )
                            )
                        elif config.show_num_links_in_report:
                            print(
                                "Updated reports after ccl={}, num_links={}, max_payload_size_bytes={}: op_time={} us ({:.1f} cycles)".format(
                                    ccl,
                                    num_links,
                                    max_payload,
                                    format_float(cycles_to_us(op_cycles, chip_freq_mhz), 3),
                                    op_cycles,
                                )
                            )
                        else:
                            print(
                                "Updated reports after ccl={}, max_payload_size_bytes={}: op_time={} us ({:.1f} cycles)".format(
                                    ccl,
                                    max_payload,
                                    format_float(cycles_to_us(op_cycles, chip_freq_mhz), 3),
                                    op_cycles,
                                )
                            )

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote details: {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
