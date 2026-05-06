# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Run DeepSeek CCL trace tests under tracy for per-CCL parameter sweeps, then
summarize top-level zone durations from profile_log_device.csv.

Op time definition:
- Most CCLs parse only trace id 1 from profile_log_device.csv and group
  per-launch samples by `run host ID`.
- Most CCLs: for each top-level (RISC processor type, zone) bucket, average
  across all devices; op time is the max of those per-bucket averages.
- reduce_to_one uses a fixed rotated fresh-trace benchmark: capture one
  single-iteration trace per root per sample, replay once with blocking=True,
  interleave roots across samples, then report the trimmed mean of the
  interleaved cycle means after dropping the single highest and lowest means.
  Its per-sample grouping uses `trace id counter`.
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
from dataclasses import dataclass, field, replace
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
REDUCE_TO_ONE_TRACE_NUM_WARMUP_SAMPLES_ENV = "CCL_REDUCE_TO_ONE_TRACE_NUM_WARMUP_SAMPLES"
REDUCE_TO_ONE_TRACE_NUM_PERF_SAMPLES_ENV = "CCL_REDUCE_TO_ONE_TRACE_NUM_PERF_SAMPLES"
REDUCE_TO_ONE_ROTATED_ROOTS = ((1, 0), (1, 1), (2, 0), (2, 1))
DEFAULT_REDUCE_TO_ONE_NUM_WARMUP_SAMPLES = 15
DEFAULT_REDUCE_TO_ONE_NUM_PERF_SAMPLES = 30

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
    op_time_aggregation: str = "cross_device_bucket_avg"
    render_cross_device_bucket_averages: bool = True
    iter_key_source: str = "run_host_id"


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
    op_bucket: tuple[str, str] | None
    op_cycles: float
    op_device: str | None = None
    num_l_chunks: int | None = None
    tiles_per_l_chunk: int | None = None
    block_size: int | None = None
    setup_details: tuple[str, ...] = ()
    analysis_description: str | None = None
    sample_count: int | None = None
    root_spread_cycles: float | None = None
    notes: tuple[str, ...] = ()
    sample_op_stats: tuple["SampleOpStat", ...] = ()
    root_device_stats: dict[str, dict[tuple[str, str, str], BucketStat]] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleOpStat:
    trace_id: int
    iter_key: str
    sample_index: int | None
    root_label: str | None
    op_cycles: float
    op_bucket: tuple[str, str]
    op_device: str


@dataclass(frozen=True)
class ReduceToOneTraceConfig:
    num_warmup_samples: int
    num_perf_samples: int


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
    "broadcast": CCLConfig(
        title="CCL broadcast trace perf matrix",
        test_target=("models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_broadcast.py::" "test_ccl_broadcast"),
        benchmark_shape=(1, 7168),
        default_num_links=(1, 2),
        supported_num_links=(1, 2),
        transport_mode="configurable (1 or 2 links)",
        env_num_links="CCL_BROADCAST_NUM_LINKS",
        env_max_payload_size="CCL_BROADCAST_MAX_PAYLOAD_SIZE_BYTES",
        top_level_zones=(
            "CCL_BROADCAST_READER",
            "CCL_BROADCAST_TRANSPORT",
        ),
        zone_labels={
            "CCL_BROADCAST_READER": "reader",
            "CCL_BROADCAST_TRANSPORT": "transport",
        },
        fixed_setup_description=(
            "2D neighbor-exchange tree broadcast on a 4x2 submesh from sender=(1,0), "
            "using one worker core per device."
        ),
        op_time_aggregation="max_device_bucket",
        render_cross_device_bucket_averages=False,
    ),
    "reduce_to_one": CCLConfig(
        title="Reduce-to-one trace perf matrix",
        test_target=(
            "models/demos/deepseek_v3_b1/tests/unit_tests/test_reduce_to_one_b1.py::" "test_reduce_to_one_trace"
        ),
        benchmark_shape=(1, 8192),
        default_num_links=(1,),
        supported_num_links=(1,),
        transport_mode="fixed 2-column forwarder topology",
        env_num_links=None,
        env_max_payload_size="CCL_REDUCE_TO_ONE_MAX_PAYLOAD_SIZE_BYTES",
        top_level_zones=(
            "REDUCE_TO_ONE_READER",
            "REDUCE_TO_ONE_WRITER",
            "REDUCE_TO_ONE_FORWARDER",
            "REDUCE_TO_ONE_COMPUTE",
        ),
        zone_labels={
            "REDUCE_TO_ONE_READER": "reader",
            "REDUCE_TO_ONE_WRITER": "writer",
            "REDUCE_TO_ONE_FORWARDER": "forwarder",
            "REDUCE_TO_ONE_COMPUTE": "compute",
        },
        fixed_setup_description=(
            "2D only; standalone padded 8-shard geometry with pinned workers "
            "(1024 elems per shard) and 2 local reduce forwarders per device."
        ),
        show_num_links_in_report=False,
        op_time_aggregation="max_device_bucket",
        render_cross_device_bucket_averages=False,
        iter_key_source="trace_counter",
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
    parser.add_argument(
        "--reduce-to-one-num-warmup-samples",
        type=int,
        default=DEFAULT_REDUCE_TO_ONE_NUM_WARMUP_SAMPLES,
        help="Warmup samples per root for reduce_to_one's fixed rotated-fresh benchmark (default: 15).",
    )
    parser.add_argument(
        "--reduce-to-one-num-perf-samples",
        type=int,
        default=DEFAULT_REDUCE_TO_ONE_NUM_PERF_SAMPLES,
        help="Perf samples per root for reduce_to_one's fixed rotated-fresh benchmark (default: 30).",
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


def format_root_coord(root_coord: tuple[int, int]) -> str:
    return f"({root_coord[0]},{root_coord[1]})"


def build_reduce_to_one_trace_config(args: argparse.Namespace) -> ReduceToOneTraceConfig:
    config = ReduceToOneTraceConfig(
        num_warmup_samples=args.reduce_to_one_num_warmup_samples,
        num_perf_samples=args.reduce_to_one_num_perf_samples,
    )
    if config.num_warmup_samples <= 0:
        raise RuntimeError(f"--reduce-to-one-num-warmup-samples must be > 0, got {config.num_warmup_samples}")
    if config.num_perf_samples <= 0:
        raise RuntimeError(f"--reduce-to-one-num-perf-samples must be > 0, got {config.num_perf_samples}")
    return config


def reduce_to_one_setup_details(trace_config: ReduceToOneTraceConfig) -> tuple[str, ...]:
    return (
        "Reduce-to-one trace mode: single_trace_fresh_replay",
        "Reduce-to-one root mode: rotate_interleaved",
        "Reduce-to-one blocking replay: true",
        f"Reduce-to-one warmup samples per root: {trace_config.num_warmup_samples}",
        f"Reduce-to-one perf samples per root: {trace_config.num_perf_samples}",
    )


def resolve_reduce_to_one_perf_trace_map(trace_config: ReduceToOneTraceConfig) -> dict[int, str]:
    warmup_trace_count = len(REDUCE_TO_ONE_ROTATED_ROOTS) * trace_config.num_warmup_samples
    perf_start = warmup_trace_count
    trace_map = {}
    trace_id = perf_start
    # The benchmark captures traces sample-major: for each sample, capture one
    # fresh trace per root in root order.
    for _sample_idx in range(trace_config.num_perf_samples):
        for root_coord in REDUCE_TO_ONE_ROTATED_ROOTS:
            trace_map[trace_id] = format_root_coord(root_coord)
            trace_id += 1
    return trace_map


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


def uses_max_device_bucket_op_time(config: CCLConfig) -> bool:
    return config.op_time_aggregation == "max_device_bucket"


def op_time_description(config: CCLConfig) -> str:
    if config.op_time_aggregation == "cross_device_bucket_avg":
        return "Op time = max(avg per top-level (RISC processor type, zone) bucket across devices)."
    if config.op_time_aggregation == "max_device_bucket":
        return (
            "Op time = slowest device's slowest top-level bucket, " "using per-device averages across iterations only."
        )
    raise ValueError(f"Unsupported op_time_aggregation: {config.op_time_aggregation}")


def resolve_op_time_description(config: CCLConfig, results: list[RunResult]) -> str:
    for result in results:
        if result.analysis_description:
            return result.analysis_description
    return op_time_description(config)


def is_reduce_to_one_config(config: CCLConfig) -> bool:
    return "test_reduce_to_one_b1.py" in config.test_target


def resolve_trace_reference_line(config: CCLConfig) -> str | None:
    if is_reduce_to_one_config(config):
        return None
    return f"Trace id: {TRACE_ID}"


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


def drop_extrema(values: list[float], *, count_per_side: int) -> list[float]:
    if count_per_side < 0:
        raise ValueError(f"count_per_side must be >= 0, got {count_per_side}")
    sorted_values = sorted(values)
    if count_per_side == 0 or len(sorted_values) <= 2 * count_per_side:
        return sorted_values
    return sorted_values[count_per_side:-count_per_side]


def resolve_profile_iter_key(
    config: CCLConfig, trace_counter: str, run_host_id: str, fallback: str
) -> tuple[str, int | None]:
    if config.iter_key_source == "trace_counter":
        if trace_counter:
            return trace_counter, parse_int(trace_counter)
        if run_host_id:
            return run_host_id, parse_int(run_host_id)
        return fallback, None
    if config.iter_key_source == "run_host_id":
        if run_host_id:
            return run_host_id, parse_int(run_host_id)
        if trace_counter:
            return trace_counter, parse_int(trace_counter)
        return fallback, None
    raise ValueError(f"Unsupported iter_key_source: {config.iter_key_source}")


def summarize_profile_log(
    profile_log: Path,
    config: CCLConfig,
    *,
    trace_ids: set[int] | None = None,
    trace_id_to_root_label: dict[int, str] | None = None,
) -> tuple[
    float,
    dict[tuple[str, str, str], BucketStat],
    dict[tuple[str, str], BucketStat],
    str | None,
    tuple[str, str] | None,
    float,
    list[SampleOpStat],
    dict[str, dict[tuple[str, str, str], BucketStat]],
]:
    chip_freq_mhz = parse_chip_freq_mhz(profile_log)
    zone_set = set(config.top_level_zones)
    selected_trace_ids = {TRACE_ID} if trace_ids is None else set(trace_ids)

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
        durations_by_core_iter: dict[tuple[int, str, str, str, str, str, str], int] = defaultdict(int)
        unmatched_ends = 0
        unmatched_starts = 0
        mismatched_ends = 0
        negative_durations = 0

        for row in reader:
            if not row or is_header_row(row):
                continue

            trace_id = parse_int(row[header_index["trace id"]])
            if trace_id is None or trace_id not in selected_trace_ids:
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
            iter_key, _sample_index = resolve_profile_iter_key(
                config,
                trace_counter,
                run_host_id,
                row[header_index["time[cycles since reset]"]].strip(),
            )
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

            durations_by_core_iter[(trace_id, device_id, risc_type, zone_name, iter_key, core_x, core_y)] += duration

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
        trace_list = ", ".join(str(trace_id) for trace_id in sorted(selected_trace_ids))
        raise RuntimeError(
            f"No matching top-level zone durations found in {profile_log} for trace ids {{{trace_list}}}"
        )

    per_iter_totals: dict[tuple[int, str, str, str, str], list[int]] = defaultdict(list)
    for (
        trace_id,
        device_id,
        risc_type,
        zone_name,
        iter_key,
        _core_x,
        _core_y,
    ), total_cycles in durations_by_core_iter.items():
        per_iter_totals[(trace_id, device_id, risc_type, zone_name, iter_key)].append(total_cycles)

    device_bucket_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    root_device_bucket_values: dict[str, dict[tuple[str, str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_sample_bucket_values: dict[tuple[int, str], dict[tuple[str, str, str], float]] = defaultdict(dict)
    for (trace_id, device_id, risc_type, zone_name, iter_key), core_totals in per_iter_totals.items():
        bucket_cycles = float(max(core_totals))
        device_bucket_values[(device_id, risc_type, zone_name)].append(bucket_cycles)
        if trace_id_to_root_label is not None and trace_id in trace_id_to_root_label:
            root_label = trace_id_to_root_label[trace_id]
            root_device_bucket_values[root_label][(device_id, risc_type, zone_name)].append(bucket_cycles)
        per_sample_bucket_values[(trace_id, iter_key)][(device_id, risc_type, zone_name)] = bucket_cycles

    device_stats = {key: make_bucket_stat(values) for key, values in device_bucket_values.items()}
    if not device_stats:
        raise RuntimeError(f"No per-device stats produced from {profile_log}")
    root_device_stats = {
        root_label: {key: make_bucket_stat(values) for key, values in bucket_values.items()}
        for root_label, bucket_values in root_device_bucket_values.items()
    }

    sample_op_stats = []
    for (trace_id, iter_key), sample_bucket_values in per_sample_bucket_values.items():
        (device_id, risc_type, zone_name), op_cycles = max(sample_bucket_values.items(), key=lambda item: item[1])
        sample_index = parse_int(iter_key)
        sample_op_stats.append(
            SampleOpStat(
                trace_id=trace_id,
                iter_key=iter_key,
                sample_index=sample_index,
                root_label=trace_id_to_root_label.get(trace_id) if trace_id_to_root_label is not None else None,
                op_cycles=op_cycles,
                op_bucket=(risc_type, zone_name),
                op_device=device_id,
            )
        )

    sample_op_stats.sort(
        key=lambda stat: (
            stat.sample_index is None,
            stat.sample_index if stat.sample_index is not None else 0,
            stat.trace_id,
            stat.iter_key,
        )
    )

    if config.op_time_aggregation == "max_device_bucket":
        (op_device, op_risc_type, op_zone_name), op_stat = max(
            device_stats.items(), key=lambda item: item[1].avg_cycles
        )
        bucket_stats: dict[tuple[str, str], BucketStat] = {}
        op_bucket = (op_risc_type, op_zone_name)
        op_cycles = op_stat.avg_cycles
        return (
            chip_freq_mhz,
            device_stats,
            bucket_stats,
            op_device,
            op_bucket,
            op_cycles,
            sample_op_stats,
            root_device_stats,
        )

    if config.op_time_aggregation == "cross_device_bucket_avg":
        bucket_device_averages: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (_device_id, risc_type, zone_name), stat in device_stats.items():
            bucket_device_averages[(risc_type, zone_name)].append(stat.avg_cycles)

        bucket_stats = {key: make_bucket_stat(values) for key, values in bucket_device_averages.items()}
        if not bucket_stats:
            raise RuntimeError(f"No top-level bucket stats produced from {profile_log}")

        op_bucket = max(bucket_stats, key=lambda key: bucket_stats[key].avg_cycles)
        op_cycles = bucket_stats[op_bucket].avg_cycles
        return chip_freq_mhz, device_stats, bucket_stats, None, op_bucket, op_cycles, sample_op_stats, root_device_stats

    raise ValueError(f"Unsupported op_time_aggregation: {config.op_time_aggregation}")


def build_reduce_to_one_run_result(
    base_result: RunResult,
    trace_config: ReduceToOneTraceConfig,
    sample_op_stats: list[SampleOpStat],
) -> RunResult:
    if not sample_op_stats:
        raise RuntimeError("reduce_to_one sample analysis requires at least one per-replay sample")

    setup_details = reduce_to_one_setup_details(trace_config)
    notes = [
        f"Perf replay samples analyzed: {len(sample_op_stats)}",
    ]

    per_root_values: dict[str, list[float]] = defaultdict(list)
    for sample in sample_op_stats:
        if sample.root_label is None:
            raise RuntimeError("reduce_to_one rotated root analysis needs trace_id -> root mapping")
        per_root_values[sample.root_label].append(sample.op_cycles)

    expected_root_labels = [format_root_coord(root_coord) for root_coord in REDUCE_TO_ONE_ROTATED_ROOTS]
    missing_root_labels = [label for label in expected_root_labels if label not in per_root_values]
    if missing_root_labels:
        raise RuntimeError("Missing reduce_to_one perf samples for rotated roots: " + ", ".join(missing_root_labels))

    root_averages = {label: statistics.mean(per_root_values[label]) for label in expected_root_labels}
    root_spread_cycles = max(root_averages.values()) - min(root_averages.values())
    notes.append(
        "Per-root averages [cycles]: "
        + ", ".join(f"{label}={format_float(root_averages[label], 1)}" for label in expected_root_labels)
    )
    notes.append(
        "Root spread: {} cycles ({} us)".format(
            format_float(root_spread_cycles, 1),
            format_float(cycles_to_us(root_spread_cycles, base_result.chip_freq_mhz), 3),
        )
    )

    ordered_root_sequences = [per_root_values[label] for label in expected_root_labels]
    cycle_count = min(len(sequence) for sequence in ordered_root_sequences)
    if cycle_count <= 0:
        raise RuntimeError("No interleaved reduce_to_one root cycles were captured")
    cycle_means = [
        sum(sequence[idx] for sequence in ordered_root_sequences) / len(ordered_root_sequences)
        for idx in range(cycle_count)
    ]
    cycle_stat = make_bucket_stat(cycle_means)
    trimmed_cycle_means = drop_extrema(cycle_means, count_per_side=1)
    trimmed_cycle_stat = make_bucket_stat(trimmed_cycle_means)
    analysis_description = (
        "Op time = trimmed mean of per-cycle means across interleaved roots, dropping the single highest and "
        "single lowest cycle mean and averaging the remainder, using per-replay slowest-device buckets."
    )
    notes.append(
        "Interleaved cycle means [cycles]: count={}, trimmed_avg={}, avg={}, median={}, min={}, max={}".format(
            cycle_stat.count,
            format_float(trimmed_cycle_stat.avg_cycles, 1),
            format_float(cycle_stat.avg_cycles, 1),
            format_float(cycle_stat.median_cycles, 1),
            format_float(cycle_stat.min_cycles, 1),
            format_float(max(cycle_means), 1),
        )
    )

    return replace(
        base_result,
        op_bucket=None,
        op_cycles=trimmed_cycle_stat.avg_cycles,
        op_device=None,
        setup_details=setup_details,
        analysis_description=analysis_description,
        sample_count=len(sample_op_stats),
        root_spread_cycles=root_spread_cycles,
        notes=tuple(notes),
        sample_op_stats=tuple(sample_op_stats),
    )


def collect_reduce_to_one_root_labels(result: RunResult) -> list[str]:
    root_labels = {sample.root_label for sample in result.sample_op_stats if sample.root_label is not None}
    ordered = [format_root_coord(root_coord) for root_coord in REDUCE_TO_ONE_ROTATED_ROOTS]
    return [label for label in ordered if label in root_labels] + sorted(
        label for label in root_labels if label not in ordered
    )


def group_reduce_to_one_samples_by_root(result: RunResult) -> dict[str, list[SampleOpStat]]:
    grouped: dict[str, list[SampleOpStat]] = defaultdict(list)
    for sample in result.sample_op_stats:
        if sample.root_label is None:
            continue
        grouped[sample.root_label].append(sample)
    for root_label in grouped:
        grouped[root_label].sort(
            key=lambda sample: (
                sample.sample_index is None,
                sample.sample_index if sample.sample_index is not None else 0,
                sample.trace_id,
                sample.iter_key,
            )
        )
    return dict(grouped)


def render_reduce_to_one_rotated_sections(result: RunResult) -> list[str]:
    root_labels = collect_reduce_to_one_root_labels(result)
    if len(root_labels) <= 1:
        return []

    grouped = group_reduce_to_one_samples_by_root(result)
    lines = []

    root_summary_headers = [
        "ROOT",
        "SAMPLE_COUNT",
        "AVG_CYCLES",
        "STDDEV_CYCLES",
        "AVG_US",
        "MIN_CYCLES",
        "MEDIAN_CYCLES",
        "MAX_CYCLES",
    ]
    root_summary_rows = []
    for root_label in root_labels:
        samples = grouped.get(root_label, [])
        if not samples:
            continue
        cycles = [sample.op_cycles for sample in samples]
        stat = make_bucket_stat(cycles)
        stddev_cycles = statistics.stdev(cycles) if len(cycles) > 1 else 0.0
        root_summary_rows.append(
            [
                root_label,
                str(stat.count),
                format_float(stat.avg_cycles, 1),
                format_float(stddev_cycles, 1),
                format_float(cycles_to_us(stat.avg_cycles, result.chip_freq_mhz), 3),
                format_float(stat.min_cycles, 1),
                format_float(stat.median_cycles, 1),
                format_float(max(cycles), 1),
            ]
        )

    lines.append("#### Per-root replay summaries")
    lines.append("")
    lines.append(format_markdown_table(root_summary_headers, root_summary_rows))
    lines.append("")

    cycle_count = min(len(grouped.get(root_label, [])) for root_label in root_labels)
    cycle_headers = ["CYCLE"] + root_labels + ["MEAN_CYCLES", "MEAN_US"]
    cycle_rows = []
    for cycle_idx in range(cycle_count):
        root_cycles = [grouped[root_label][cycle_idx].op_cycles for root_label in root_labels]
        mean_cycles = sum(root_cycles) / len(root_cycles)
        cycle_rows.append(
            [str(cycle_idx + 1)]
            + [format_float(cycles, 1) for cycles in root_cycles]
            + [format_float(mean_cycles, 1), format_float(cycles_to_us(mean_cycles, result.chip_freq_mhz), 3)]
        )
    lines.append("#### Interleaved cycle means")
    lines.append("")
    lines.append(format_markdown_table(cycle_headers, cycle_rows))
    lines.append("")

    return lines


def build_device_rows(
    device_stats: dict[tuple[str, str, str], BucketStat],
    *,
    chip_freq_mhz: float,
    config: CCLConfig,
) -> list[list[str]]:
    rows = []
    for (device_id, risc_type, zone_name), stat in sorted(
        device_stats.items(),
        key=lambda item: (
            device_sort_key(item[0][0]),
            bucket_sort_key((item[0][1], item[0][2]), config),
        ),
    ):
        rows.append(
            [
                device_id,
                risc_type,
                zone_name,
                str(stat.count),
                format_float(stat.avg_cycles, 1),
                format_float(cycles_to_ns(stat.avg_cycles, chip_freq_mhz), 3),
                format_float(cycles_to_us(stat.avg_cycles, chip_freq_mhz), 3),
                format_float(stat.min_cycles, 1),
                format_float(stat.median_cycles, 1),
            ]
        )
    return rows


def render_reduce_to_one_rotated_device_tables(result: RunResult, config: CCLConfig) -> list[str]:
    if not result.root_device_stats:
        return []

    root_labels = collect_reduce_to_one_root_labels(result)
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
    lines = []
    for root_label in root_labels:
        root_stats = result.root_device_stats.get(root_label)
        if not root_stats:
            continue
        lines.append(f"#### Per-device averages for root {root_label}")
        lines.append("")
        lines.append(
            format_markdown_table(
                device_headers,
                build_device_rows(root_stats, chip_freq_mhz=result.chip_freq_mhz, config=config),
            )
        )
        lines.append("")
    return lines


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
    trace_reference_line = resolve_trace_reference_line(config)
    if trace_reference_line is not None:
        lines.append(trace_reference_line)
    lines.extend(
        [
            "Source: profile_log_device.csv only (top-level zones only; no micro-profiling).",
            "Conversion: us = cycles / CHIP_FREQ[MHz], ns = cycles * 1000 / CHIP_FREQ[MHz].",
            resolve_op_time_description(config, results),
            f"Top-level zones: {', '.join(config.top_level_zones)}",
            "",
        ]
    )
    if results and results[0].setup_details:
        lines.extend(results[0].setup_details)
        lines.append("")
    if not config.render_cross_device_bucket_averages:
        lines.append("Cross-device bucket averages are omitted for this CCL.")
        lines.append("")

    if not results:
        lines.append("Results pending.")
        lines.append("")
        return "\n".join(lines)

    observed_freqs = sorted({result.chip_freq_mhz for result in results})
    lines.append("Observed CHIP_FREQ[MHz]: " + ", ".join(format_float(freq, 3) for freq in observed_freqs))
    lines.append("")

    summary_rows = []
    sorted_results = sort_results_for_display(config, results)
    include_sample_count = any(result.sample_count is not None for result in sorted_results)
    include_root_spread = any(result.root_spread_cycles is not None for result in sorted_results)
    include_op_bucket = all(result.op_bucket is not None for result in sorted_results)
    include_op_device = (
        uses_max_device_bucket_op_time(config)
        and include_op_bucket
        and any(result.op_device is not None for result in sorted_results)
    )
    if is_sdpa_chunk_sweep(config):
        summary_headers = [
            "num_l_chunks",
            "tiles_per_l_chunk",
            "block_size",
            "max_payload_size_bytes",
            "op_time_us",
            "op_cycles",
        ]
        if include_sample_count:
            summary_headers.append("sample_count")
        if include_root_spread:
            summary_headers.append("root_spread_us")
        if include_op_device:
            summary_headers.append("op_device")
        if include_op_bucket:
            summary_headers.append("op_bucket")
        result_lookup = {}
        for result in sorted_results:
            row = [
                str(result.num_l_chunks),
                str(result.tiles_per_l_chunk),
                str(result.block_size),
                str(result.max_payload_size),
                format_float(cycles_to_us(result.op_cycles, result.chip_freq_mhz), 3),
                format_float(result.op_cycles, 1),
            ]
            if include_sample_count:
                row.append(str(result.sample_count) if result.sample_count is not None else "n/a")
            if include_root_spread:
                spread_us = (
                    cycles_to_us(result.root_spread_cycles, result.chip_freq_mhz)
                    if result.root_spread_cycles is not None
                    else None
                )
                row.append(format_float(spread_us, 3))
            if include_op_device:
                row.append(result.op_device or "-")
            if include_op_bucket:
                row.append(bucket_display(result.op_bucket, config) if result.op_bucket is not None else "-")
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
        summary_headers = ["num_links", "max_payload_size_bytes", "op_time_us", "op_cycles"]
        if include_sample_count:
            summary_headers.append("sample_count")
        if include_root_spread:
            summary_headers.append("root_spread_us")
        if include_op_device:
            summary_headers.append("op_device")
        if include_op_bucket:
            summary_headers.append("op_bucket")
        result_lookup = {}
        for result in sorted_results:
            row = [
                str(result.num_links),
                str(result.max_payload_size),
                format_float(cycles_to_us(result.op_cycles, result.chip_freq_mhz), 3),
                format_float(result.op_cycles, 1),
            ]
            if include_sample_count:
                row.append(str(result.sample_count) if result.sample_count is not None else "n/a")
            if include_root_spread:
                spread_us = (
                    cycles_to_us(result.root_spread_cycles, result.chip_freq_mhz)
                    if result.root_spread_cycles is not None
                    else None
                )
                row.append(format_float(spread_us, 3))
            if include_op_device:
                row.append(result.op_device or "-")
            if include_op_bucket:
                row.append(bucket_display(result.op_bucket, config) if result.op_bucket is not None else "-")
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
        summary_headers = ["max_payload_size_bytes", "op_time_us", "op_cycles"]
        if include_sample_count:
            summary_headers.append("sample_count")
        if include_root_spread:
            summary_headers.append("root_spread_us")
        if include_op_device:
            summary_headers.append("op_device")
        if include_op_bucket:
            summary_headers.append("op_bucket")
        result_lookup = {}
        for result in sorted_results:
            row = [
                str(result.max_payload_size),
                format_float(cycles_to_us(result.op_cycles, result.chip_freq_mhz), 3),
                format_float(result.op_cycles, 1),
            ]
            if include_sample_count:
                row.append(str(result.sample_count) if result.sample_count is not None else "n/a")
            if include_root_spread:
                spread_us = (
                    cycles_to_us(result.root_spread_cycles, result.chip_freq_mhz)
                    if result.root_spread_cycles is not None
                    else None
                )
                row.append(format_float(spread_us, 3))
            if include_op_device:
                row.append(result.op_device or "-")
            if include_op_bucket:
                row.append(bucket_display(result.op_bucket, config) if result.op_bucket is not None else "-")
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
    trace_reference_line = resolve_trace_reference_line(config)
    if trace_reference_line is not None:
        lines.append(trace_reference_line)
    lines.extend(
        [
            "Source: profile_log_device.csv only (top-level zones only; no micro-profiling).",
            resolve_op_time_description(config, results),
            "",
        ]
    )
    if results and results[0].setup_details:
        lines.extend(results[0].setup_details)
        lines.append("")
    if not config.render_cross_device_bucket_averages:
        lines.append("Cross-device bucket averages are omitted for this CCL.")
        lines.append("")

    if not results:
        lines.append("Results pending.")
        lines.append("")
        return "\n".join(lines)

    bucket_order = collect_bucket_order(results, config) if config.render_cross_device_bucket_averages else []
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

        if result.notes:
            lines.append("#### Trace Analysis")
            lines.append("")
            for note in result.notes:
                lines.append(f"- {note}")
            lines.append("")

        rotated_lines = render_reduce_to_one_rotated_sections(result)
        if rotated_lines:
            lines.extend(rotated_lines)

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
        rotated_device_lines = render_reduce_to_one_rotated_device_tables(result, config)
        if rotated_device_lines:
            lines.extend(rotated_device_lines)
        else:
            lines.append("#### Per-device averages")
            lines.append("")
            lines.append(
                format_markdown_table(
                    device_headers,
                    build_device_rows(result.device_stats, chip_freq_mhz=result.chip_freq_mhz, config=config),
                )
            )
            lines.append("")

        if config.render_cross_device_bucket_averages:
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
        if result.op_bucket is None:
            lines.append(
                "Final op time: {} us ({} cycles)".format(
                    format_float(cycles_to_us(result.op_cycles, result.chip_freq_mhz), 3),
                    format_float(result.op_cycles, 1),
                )
            )
        elif uses_max_device_bucket_op_time(config):
            lines.append(
                "Final op time: {} us ({} cycles), device={}, bucket={}".format(
                    format_float(cycles_to_us(result.op_cycles, result.chip_freq_mhz), 3),
                    format_float(result.op_cycles, 1),
                    result.op_device or "-",
                    bucket_display(result.op_bucket, config),
                )
            )
        else:
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
    reduce_to_one_trace_config = build_reduce_to_one_trace_config(args)
    reduce_to_one_perf_trace_map = resolve_reduce_to_one_perf_trace_map(reduce_to_one_trace_config)
    reduce_to_one_perf_trace_ids = set(reduce_to_one_perf_trace_map)

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
                        if ccl == "reduce_to_one":
                            env[REDUCE_TO_ONE_TRACE_NUM_WARMUP_SAMPLES_ENV] = str(
                                reduce_to_one_trace_config.num_warmup_samples
                            )
                            env[REDUCE_TO_ONE_TRACE_NUM_PERF_SAMPLES_ENV] = str(
                                reduce_to_one_trace_config.num_perf_samples
                            )

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
                        (
                            chip_freq_mhz,
                            device_stats,
                            bucket_stats,
                            op_device,
                            op_bucket,
                            op_cycles,
                            sample_op_stats,
                            root_device_stats,
                        ) = summarize_profile_log(
                            profile_log,
                            config,
                            trace_ids=(reduce_to_one_perf_trace_ids if ccl == "reduce_to_one" else None),
                            trace_id_to_root_label=(reduce_to_one_perf_trace_map if ccl == "reduce_to_one" else None),
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
                        run_result = RunResult(
                            ccl=ccl,
                            num_links=num_links,
                            max_payload_size=max_payload,
                            report_path=profile_log_rel,
                            chip_freq_mhz=chip_freq_mhz,
                            device_stats=device_stats,
                            bucket_stats=bucket_stats,
                            op_bucket=op_bucket,
                            op_cycles=op_cycles,
                            op_device=op_device,
                            num_l_chunks=num_l_chunks,
                            tiles_per_l_chunk=tiles_per_l_chunk,
                            block_size=block_size,
                            root_device_stats=root_device_stats,
                        )
                        if ccl == "reduce_to_one":
                            run_result = build_reduce_to_one_run_result(
                                run_result,
                                reduce_to_one_trace_config,
                                sample_op_stats,
                            )
                        results_by_ccl[ccl].append(run_result)

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
                                    format_float(cycles_to_us(run_result.op_cycles, chip_freq_mhz), 3),
                                    run_result.op_cycles,
                                )
                            )
                        elif config.show_num_links_in_report:
                            print(
                                "Updated reports after ccl={}, num_links={}, max_payload_size_bytes={}: op_time={} us ({:.1f} cycles)".format(
                                    ccl,
                                    num_links,
                                    max_payload,
                                    format_float(cycles_to_us(run_result.op_cycles, chip_freq_mhz), 3),
                                    run_result.op_cycles,
                                )
                            )
                        else:
                            print(
                                "Updated reports after ccl={}, max_payload_size_bytes={}: op_time={} us ({:.1f} cycles)".format(
                                    ccl,
                                    max_payload,
                                    format_float(cycles_to_us(run_result.op_cycles, chip_freq_mhz), 3),
                                    run_result.op_cycles,
                                )
                            )

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote details: {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
