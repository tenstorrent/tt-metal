# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import math
import os
import subprocess
from pathlib import Path


BENCHMARK_REL_TOLERANCE = 0.05
BENCHMARK_FILTER = "mux_v2/standalone_mux_throughput/.*"
NOC_SORT_ORDER = {"RISCV_0_default": 0, "RISCV_1_default": 1}
FAMILY_SORT_ORDER = {
    "buffer_sweep": 0,
    "payload_sweep": 1,
    "sender_sweep": 2,
    "high_sender_sweep": 3,
}

SUMMARY_HEADERS = [
    "Case name",
    "Senders",
    "Payload bytes",
    "Num packets",
    "Buffers per channel",
    "Forwarder NOC",
    "Aggregate bytes",
    "Max sender cycles",
    "Bytes per cycle",
    "Cycles per packet",
    "Throughput GB/s",
    "Speedup vs golden",
    "Status",
]

GOLDEN_HEADERS = [
    "Case name",
    "Senders",
    "Payload bytes",
    "Num packets",
    "Buffers per channel",
    "Forwarder NOC",
    "Bytes per cycle",
    "Throughput GB/s",
]

GOLDEN_KEY_FIELDS = GOLDEN_HEADERS[:-2]


def get_tt_metal_home() -> Path:
    return Path(os.environ.get("TT_METAL_HOME", Path.cwd())).resolve()


def sanitize_name(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_") or "unknown"


def get_benchmark_binary(tt_metal_home: Path) -> Path:
    override = os.environ.get("FABRIC_MUX_V2_THROUGHPUT_BIN")
    if override:
        return Path(override).resolve()
    return tt_metal_home / "build/test/tt_metal/perf_microbenchmark/routing/benchmark_fabric_mux_v2_throughput"


def get_output_dir(tt_metal_home: Path) -> Path:
    override = os.environ.get("FABRIC_MUX_V2_THROUGHPUT_OUTPUT_DIR")
    if override:
        return Path(override).resolve()
    return tt_metal_home / "generated/fabric_mux_v2_throughput"


def get_golden_path(tt_metal_home: Path, arch_name: str) -> Path:
    override = os.environ.get("FABRIC_MUX_V2_THROUGHPUT_GOLDEN")
    if override:
        return Path(override).resolve()
    return (
        tt_metal_home / "tests/tt_metal/microbenchmarks/ethernet" / f"fabric_mux_v2_throughput_golden_{arch_name}.csv"
    )


def should_update_golden() -> bool:
    return os.environ.get("FABRIC_MUX_V2_THROUGHPUT_UPDATE_GOLDEN", "").lower() in {"1", "true", "yes"}


def extract_case_name(benchmark_name: str) -> str:
    prefix = "mux_v2/standalone_mux_throughput/"
    if benchmark_name.startswith(prefix):
        return benchmark_name[len(prefix) :].split("/", maxsplit=1)[0]
    return benchmark_name.split("/", maxsplit=1)[0]


def infer_forwarder_noc(case_name: str) -> str:
    return "RISCV_1_default" if "_r1" in case_name else "RISCV_0_default"


def as_int(value) -> int:
    return int(round(float(value)))


def as_float(value) -> float:
    return float(value)


def normalize_benchmark_row(benchmark: dict) -> dict:
    case_name = extract_case_name(benchmark["name"])
    sender_count = as_int(benchmark["senders"])
    num_packets = as_int(benchmark["num_packets"])
    max_sender_cycles = as_int(benchmark["max_sender_cycles"])
    aggregate_packet_count = sender_count * num_packets
    bytes_per_cycle = as_float(benchmark["bytes_per_cycle"])
    cycles_per_packet = max_sender_cycles / aggregate_packet_count
    throughput_gb_per_s = as_float(benchmark["throughput_bytes_per_s"]) / 1.0e9
    return {
        "Case name": case_name,
        "Senders": sender_count,
        "Payload bytes": as_int(benchmark["payload_bytes"]),
        "Num packets": num_packets,
        "Buffers per channel": as_int(benchmark["buffers_per_channel"]),
        "Forwarder NOC": infer_forwarder_noc(case_name),
        "Aggregate bytes": as_int(benchmark["aggregate_case_bytes"]),
        "Max sender cycles": max_sender_cycles,
        "Bytes per cycle": bytes_per_cycle,
        "Cycles per packet": cycles_per_packet,
        "Throughput GB/s": throughput_gb_per_s,
    }


def get_arch_name_from_results(results: dict, results_path: Path) -> str:
    arch_name = results.get("context", {}).get("arch")
    if not arch_name:
        raise AssertionError(f"Benchmark JSON did not include context.arch: {results_path}")
    return sanitize_name(arch_name)


def get_family_sort_order(case_name: str) -> int:
    for family_prefix, sort_order in FAMILY_SORT_ORDER.items():
        if case_name.startswith(family_prefix):
            return sort_order
    return len(FAMILY_SORT_ORDER)


def get_family_axis_sort_value(row: dict) -> int:
    case_name = row["Case name"]
    if case_name.startswith("buffer_sweep"):
        return row["Buffers per channel"]
    if case_name.startswith("payload_sweep"):
        return row["Payload bytes"]
    if case_name.startswith("sender_sweep") or case_name.startswith("high_sender_sweep"):
        return row["Senders"]
    return 0


def sort_key(row: dict) -> tuple[int, int, int, str]:
    return (
        NOC_SORT_ORDER.get(row["Forwarder NOC"], len(NOC_SORT_ORDER)),
        get_family_sort_order(row["Case name"]),
        get_family_axis_sort_value(row),
        row["Case name"],
    )


def read_benchmark_results(results_path: Path) -> dict:
    with results_path.open() as results_file:
        return json.load(results_file)


def parse_benchmark_results(results: dict, results_path: Path) -> list[dict]:
    rows = []
    for benchmark in results.get("benchmarks", []):
        if benchmark.get("run_type") != "iteration":
            continue
        if "bytes_per_cycle" not in benchmark:
            continue
        rows.append(normalize_benchmark_row(benchmark))

    if not rows:
        raise AssertionError(f"No mux-v2 throughput benchmark rows found in {results_path}")
    return sorted(rows, key=sort_key)


def read_golden_rows(golden_path: Path) -> dict[str, dict]:
    if not golden_path.exists():
        raise AssertionError(f"Missing golden file: {golden_path}")

    with golden_path.open(newline="") as golden_file:
        reader = csv.DictReader(golden_file)
        missing_columns = [field for field in GOLDEN_HEADERS if field not in (reader.fieldnames or [])]
        if missing_columns:
            raise AssertionError(f"Golden file {golden_path} is missing columns: {missing_columns}")
        return {row["Case name"]: row for row in reader}


def validate_against_golden(rows: list[dict], golden_rows: dict[str, dict]) -> tuple[list[float], list[str]]:
    speedups = []
    errors = []

    for row in rows:
        case_name = row["Case name"]
        golden_row = golden_rows.get(case_name)
        if golden_row is None:
            errors.append(f"No golden row for case {case_name}")
            row["Speedup vs golden"] = ""
            row["Status"] = "missing-golden"
            continue

        stale_fields = [field for field in GOLDEN_KEY_FIELDS if str(row[field]) != str(golden_row[field])]
        if stale_fields:
            errors.append(f"Golden row for {case_name} has stale fields: {stale_fields}")
            row["Speedup vs golden"] = ""
            row["Status"] = "stale-golden"
            continue

        golden_bytes_per_cycle = float(golden_row["Bytes per cycle"])
        current_bytes_per_cycle = float(row["Bytes per cycle"])
        speedup = current_bytes_per_cycle / golden_bytes_per_cycle
        speedups.append(speedup)
        row["Speedup vs golden"] = speedup

        lower_bound = 1.0 - BENCHMARK_REL_TOLERANCE
        upper_bound = 1.0 + BENCHMARK_REL_TOLERANCE
        if lower_bound <= speedup <= upper_bound:
            row["Status"] = "pass"
        else:
            row["Status"] = "fail"
            errors.append(
                f"{case_name}: {current_bytes_per_cycle:.6f} B/c vs golden "
                f"{golden_bytes_per_cycle:.6f} B/c ({speedup:.4f}x)"
            )

    return speedups, errors


def write_csv_summary(summary_path: Path, rows: list[dict]) -> None:
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_summary_row(row))


def write_golden_rows(golden_path: Path, rows: list[dict]) -> None:
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with golden_path.open("w", newline="") as golden_file:
        writer = csv.DictWriter(golden_file, fieldnames=GOLDEN_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_golden_field(row, field) for field in GOLDEN_HEADERS})


def format_golden_field(row: dict, field: str):
    if field == "Bytes per cycle":
        return f"{row[field]:.12f}"
    if field == "Throughput GB/s":
        return f"{row[field]:.12f}"
    return row[field]


def format_summary_row(row: dict) -> dict:
    formatted = row.copy()
    formatted["Bytes per cycle"] = f"{row['Bytes per cycle']:.6f}"
    formatted["Cycles per packet"] = f"{row['Cycles per packet']:.6f}"
    formatted["Throughput GB/s"] = f"{row['Throughput GB/s']:.6f}"
    if row.get("Speedup vs golden") != "":
        formatted["Speedup vs golden"] = f"{row['Speedup vs golden']:.6f}"
    return formatted


def calculate_geomean(values: list[float]) -> float:
    if not values:
        return 1.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def format_text_table(rows: list[dict], geomean_speedup: float) -> str:
    display_headers = [
        "Case",
        "Senders",
        "Payload",
        "Bufs",
        "NOC",
        "B/c",
        "Cyc/pkt",
        "GB/s",
        "vs golden",
        "Status",
    ]
    display_rows = []
    for row in rows:
        display_rows.append(
            [
                row["Case name"],
                str(row["Senders"]),
                str(row["Payload bytes"]),
                str(row["Buffers per channel"]),
                "r1" if row["Forwarder NOC"] == "RISCV_1_default" else "r0",
                f"{row['Bytes per cycle']:.3f}",
                f"{row['Cycles per packet']:.2f}",
                f"{row['Throughput GB/s']:.3f}",
                f"{row['Speedup vs golden']:.3f}x" if row.get("Speedup vs golden") != "" else "",
                row.get("Status", ""),
            ]
        )

    widths = [
        max(len(display_headers[col_idx]), *(len(row[col_idx]) for row in display_rows))
        for col_idx in range(len(display_headers))
    ]
    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(display_headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in display_rows:
        lines.append(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    lines.append("")
    lines.append(f"Geomean speedup vs golden: {geomean_speedup:.6f}x")
    lines.append(f"Tolerance: +/- {BENCHMARK_REL_TOLERANCE * 100:.1f}% per case")
    return "\n".join(lines)


def run_benchmark(benchmark_binary: Path, output_json_path: Path, tt_metal_home: Path) -> None:
    if not benchmark_binary.exists():
        raise AssertionError(f"Benchmark binary does not exist: {benchmark_binary}")

    benchmark_filter = os.environ.get("FABRIC_MUX_V2_THROUGHPUT_FILTER", BENCHMARK_FILTER)
    command = [
        str(benchmark_binary),
        f"--benchmark_filter={benchmark_filter}",
        f"--benchmark_out={output_json_path}",
        "--benchmark_out_format=json",
    ]
    completed = subprocess.run(command, cwd=tt_metal_home, check=False)
    if completed.returncode != 0:
        raise AssertionError(f"Mux-v2 throughput benchmark failed with exit code {completed.returncode}")


def test_fabric_mux_v2_throughput():
    tt_metal_home = get_tt_metal_home()
    output_dir = get_output_dir(tt_metal_home)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_json_path = output_dir / "results.json"

    run_benchmark(get_benchmark_binary(tt_metal_home), raw_json_path, tt_metal_home)

    benchmark_results = read_benchmark_results(raw_json_path)
    arch_name = get_arch_name_from_results(benchmark_results, raw_json_path)
    summary_csv_path = output_dir / f"summary_{arch_name}.csv"
    summary_txt_path = output_dir / f"summary_{arch_name}.txt"

    rows = parse_benchmark_results(benchmark_results, raw_json_path)
    golden_path = get_golden_path(tt_metal_home, arch_name)
    if should_update_golden():
        for row in rows:
            row["Speedup vs golden"] = 1.0
            row["Status"] = "updated-golden"
        write_golden_rows(golden_path, rows)
        speedups = [1.0 for _ in rows]
        validation_errors = []
    else:
        speedups, validation_errors = validate_against_golden(rows, read_golden_rows(golden_path))
    geomean_speedup = calculate_geomean(speedups)

    write_csv_summary(summary_csv_path, rows)
    summary_text = format_text_table(rows, geomean_speedup)
    summary_txt_path.write_text(summary_text)
    print(summary_text)
    print(f"\nRaw JSON: {raw_json_path}")
    print(f"CSV summary: {summary_csv_path}")
    print(f"Text summary: {summary_txt_path}")
    print(f"Golden CSV: {golden_path}")

    if validation_errors:
        raise AssertionError("\n".join(validation_errors))
