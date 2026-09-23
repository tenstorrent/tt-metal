# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side fabric builder timings for cold and hot kernel caches.

Runs fabric_builder_benchmark under tracy-capture, extracts the fabric builder zones inside the
cold and hot phase markers, and checks their durations against the golden CSV for this machine.
Set FABRIC_BUILDER_PERF_UPDATE_GOLDEN=1 to write the measured durations to the golden instead.
"""

import csv
import json
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

ZONES = [
    "DeviceManager::initialize_fabric_and_dispatch_fw",
    "FabricFirmwareInitializer::init",
    "ControlPlane::write_routing_tables_to_all_chips",
    "FabricFirmwareInitializer::compile_and_configure_fabric",
    "FabricFirmwareInitializer::configure",
]
CACHES = ["cold", "hot"]
PHASE_MARKER = "FabricBuilderBenchmark::{cache}"

DEFAULT_TOLERANCE_PERCENT = 10.0
BENCHMARK_TIMEOUT_S = 180
CAPTURE_EXIT_TIMEOUT_S = 60
TRACY_PORTS = range(8086, 8500)

ZONE_CSV_COLUMNS = ["name", "ns_since_start", "exec_time_ns"]
GOLDEN_HEADERS = ["fabric_config", "cache", "zone", "golden_ms", "tolerance_percent"]
SUMMARY_HEADERS = [
    "fabric_config",
    "cache",
    "zone",
    "measured_ms",
    "golden_ms",
    "delta_percent",
    "tolerance_percent",
    "status",
]


@dataclass(frozen=True)
class Zone:
    name: str
    start_ns: int
    duration_ns: int

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.duration_ns

    def contains(self, other: "Zone") -> bool:
        return self.start_ns <= other.start_ns and other.end_ns <= self.end_ns


# Paths and environment.
def get_tt_metal_home() -> Path:
    return Path(os.environ.get("TT_METAL_HOME", Path.cwd())).resolve()


def get_benchmark_binary(tt_metal_home: Path) -> Path:
    return tt_metal_home / "build/test/tt_metal/tt_fabric/fabric_builder_benchmark"


def get_tracy_tool(tt_metal_home: Path, tool_name: str) -> Path:
    return tt_metal_home / "build/tools/profiler/bin" / tool_name


def get_output_dir(tt_metal_home: Path) -> Path:
    override = os.environ.get("FABRIC_BUILDER_PERF_OUTPUT_DIR")
    if override:
        return Path(override).resolve()
    return tt_metal_home / "generated/fabric_builder_perf"


def get_golden_path(tt_metal_home: Path, arch: str, cluster_type: str) -> Path:
    return (
        tt_metal_home
        / "tests/tt_metal/tt_fabric/fabric_builder_perf/golden_data"
        / f"fabric_builder_perf_golden_{arch}_{cluster_type}.csv"
    )


def should_update_golden() -> bool:
    return os.environ.get("FABRIC_BUILDER_PERF_UPDATE_GOLDEN", "").lower() in {"1", "true", "yes"}


def get_benchmark_env(cache_dir: Path, tracy_port: int) -> dict:
    env = dict(os.environ)
    env.update(
        TT_METAL_CACHE=str(cache_dir),  # populated with artifacts from the cold phase which are used in hot phase
        TT_METAL_DEVICE_PROFILER="0",  # disable device profiler to avoid profiler overhead
        TRACY_PORT=str(tracy_port),  # set the tracy port
    )
    return env


def find_free_tracy_port() -> int:
    for port in TRACY_PORTS:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise AssertionError(f"No free Tracy port in {TRACY_PORTS.start}-{TRACY_PORTS.stop - 1}")


# Connects to tracy and runs the benchmark binary
def run_benchmark_under_tracy(tt_metal_home: Path, fabric_config: str, case_dir: Path) -> None:
    capture_tool = get_tracy_tool(tt_metal_home, "tracy-capture")
    benchmark_binary = get_benchmark_binary(tt_metal_home)
    assert capture_tool.exists(), f"Tracy capture tool not found: {capture_tool}"
    assert benchmark_binary.exists(), f"Benchmark binary not found: {benchmark_binary}"

    port = find_free_tracy_port()
    benchmark_log = case_dir / "benchmark.log"
    capture_log = case_dir / "capture.log"
    capture_command = [str(capture_tool), "-o", str(case_dir / "capture.tracy"), "-f", "-p", str(port)]
    benchmark_command = [
        str(benchmark_binary),
        "--output",
        str(case_dir / "results.json"),
        "--fabric-config",
        fabric_config,
    ]

    with capture_log.open("w") as capture_out:
        capture = subprocess.Popen(capture_command, stdout=capture_out, stderr=subprocess.STDOUT)
        # run the actual benchmark
        try:
            with benchmark_log.open("w") as benchmark_out:
                benchmark = subprocess.run(
                    benchmark_command,
                    env=get_benchmark_env(case_dir / "cache", port),
                    cwd=tt_metal_home,
                    stdout=benchmark_out,
                    stderr=subprocess.STDOUT,
                    timeout=BENCHMARK_TIMEOUT_S,
                    check=False,
                )
            assert (
                benchmark.returncode == 0
            ), f"Benchmark failed with exit code {benchmark.returncode}; see {benchmark_log}"
            capture.wait(timeout=CAPTURE_EXIT_TIMEOUT_S)
        finally:
            if capture.poll() is None:
                capture.terminate()
                capture.wait()
    assert capture.returncode == 0, f"tracy-capture failed with exit code {capture.returncode}; see {capture_log}"


def export_zones(tt_metal_home: Path, case_dir: Path) -> Path:
    csvexport_tool = get_tracy_tool(tt_metal_home, "tracy-csvexport")
    zones_csv = case_dir / "zones.csv"
    export_log = case_dir / "export.log"
    with zones_csv.open("w") as zones_out, export_log.open("w") as log_out:
        completed = subprocess.run(
            [str(csvexport_tool), "-u", str(case_dir / "capture.tracy")],
            stdout=zones_out,
            stderr=log_out,
            check=False,
        )
    assert completed.returncode == 0, f"tracy-csvexport failed with exit code {completed.returncode}; see {export_log}"
    return zones_csv


# Zone extraction.
def read_zones(zones_csv: Path) -> list[Zone]:
    with zones_csv.open(newline="") as zones_file:
        reader = csv.DictReader(zones_file)
        missing_columns = [column for column in ZONE_CSV_COLUMNS if column not in (reader.fieldnames or [])]
        assert not missing_columns, f"{zones_csv} is missing columns {missing_columns}"
        return [Zone(row["name"], int(row["ns_since_start"]), int(row["exec_time_ns"])) for row in reader]


def find_single_zone(zones: list[Zone], name: str, within: Zone | None = None) -> Zone:
    matches = [zone for zone in zones if zone.name == name and (within is None or within.contains(zone))]
    location = f" inside {within.name}" if within else ""
    assert len(matches) == 1, f"Expected one '{name}' zone{location}, found {len(matches)}"
    return matches[0]


def extract_durations_ms(zones: list[Zone]) -> dict[tuple[str, str], float]:
    durations = {}
    for cache in CACHES:
        marker = find_single_zone(zones, PHASE_MARKER.format(cache=cache))
        for zone_name in ZONES:
            durations[(cache, zone_name)] = find_single_zone(zones, zone_name, within=marker).duration_ns / 1e6
    return durations


# Cache state validation.
def validate_cache_state(phases: dict) -> None:
    cold_before = phases["cold"]["artifacts_before"]
    hot_before = phases["hot"]["artifacts_before"]
    hot_after = phases["hot"]["artifacts_after"]
    assert cold_before == 0, f"Cold phase started with {cold_before} cached artifacts"
    assert hot_before > 0, "Hot phase started with an empty kernel cache"
    assert hot_after == hot_before, f"Hot phase compiled {hot_after - hot_before} new artifacts; the cache was not hot"


# Golden CSV.
def read_golden_rows(golden_path: Path) -> list[dict]:
    assert (
        golden_path.exists()
    ), f"Missing golden file: {golden_path}. Run with FABRIC_BUILDER_PERF_UPDATE_GOLDEN=1 on this machine to create it."
    with golden_path.open(newline="") as golden_file:
        reader = csv.DictReader(golden_file)
        missing_columns = [column for column in GOLDEN_HEADERS if column not in (reader.fieldnames or [])]
        assert not missing_columns, f"Golden file {golden_path} is missing columns {missing_columns}"
        return list(reader)


# Replaces this fabric config's rows, keeping each row's tolerance and other fabric configs' rows.
def write_golden_rows(golden_path: Path, fabric_config: str, durations: dict[tuple[str, str], float]) -> None:
    existing_rows = read_golden_rows(golden_path) if golden_path.exists() else []
    tolerances = {
        (row["cache"], row["zone"]): row["tolerance_percent"]
        for row in existing_rows
        if row["fabric_config"] == fabric_config
    }
    other_rows = [row for row in existing_rows if row["fabric_config"] != fabric_config]
    new_rows = [
        {
            "fabric_config": fabric_config,
            "cache": cache,
            "zone": zone,
            "golden_ms": f"{duration_ms:.3f}",
            "tolerance_percent": tolerances.get((cache, zone), f"{DEFAULT_TOLERANCE_PERCENT:.1f}"),
        }
        for (cache, zone), duration_ms in durations.items()
    ]
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with golden_path.open("w", newline="") as golden_file:
        writer = csv.DictWriter(golden_file, fieldnames=GOLDEN_HEADERS)
        writer.writeheader()
        writer.writerows(other_rows + new_rows)


# Golden comparison. Tolerance is two-sided, so a large speedup also fails and the golden gets refreshed.
def compare_to_golden(
    fabric_config: str, durations: dict[tuple[str, str], float], golden_rows: list[dict]
) -> tuple[list[dict], list[str]]:
    goldens = {(row["cache"], row["zone"]): row for row in golden_rows if row["fabric_config"] == fabric_config}
    rows = []
    errors = []

    for (cache, zone), measured_ms in durations.items():
        golden = goldens.pop((cache, zone), None)
        if golden is None:
            rows.append(make_summary_row(fabric_config, cache, zone, measured_ms, status="missing-golden"))
            errors.append(f"{cache} {zone}: no golden row")
            continue

        golden_ms = float(golden["golden_ms"])
        tolerance_percent = float(golden["tolerance_percent"])
        delta_percent = (measured_ms / golden_ms - 1.0) * 100.0
        status = "pass" if abs(delta_percent) <= tolerance_percent else "fail"
        rows.append(
            make_summary_row(
                fabric_config,
                cache,
                zone,
                measured_ms,
                golden_ms=golden["golden_ms"],
                delta_percent=f"{delta_percent:+.1f}",
                tolerance_percent=golden["tolerance_percent"],
                status=status,
            )
        )
        if status == "fail":
            errors.append(
                f"{cache} {zone}: {measured_ms:.3f} ms vs golden {golden_ms:.3f} ms "
                f"({delta_percent:+.1f}%, tolerance {tolerance_percent}%)"
            )

    # Whatever is left in the golden was not measured.
    for cache, zone in goldens:
        rows.append(make_summary_row(fabric_config, cache, zone, status="stale-golden"))
        errors.append(f"{cache} {zone}: golden row was not measured")

    return rows, errors


# Summary output.
def make_summary_row(fabric_config: str, cache: str, zone: str, measured_ms: float | None = None, **fields) -> dict:
    row = dict.fromkeys(SUMMARY_HEADERS, "")
    row.update(fabric_config=fabric_config, cache=cache, zone=zone, **fields)
    if measured_ms is not None:
        row["measured_ms"] = f"{measured_ms:.3f}"
    return row


def format_text_table(rows: list[dict]) -> str:
    display_rows = [[str(row[header]) for header in SUMMARY_HEADERS] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in display_rows)) for index, header in enumerate(SUMMARY_HEADERS)
    ]
    lines = [
        " | ".join(header.ljust(widths[index]) for index, header in enumerate(SUMMARY_HEADERS)),
        "-+-".join("-" * width for width in widths),
    ]
    lines += [" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in display_rows]
    return "\n".join(lines)


def write_summary(case_dir: Path, summary_name: str, rows: list[dict]) -> str:
    with (case_dir / f"{summary_name}.csv").open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    summary_text = format_text_table(rows)
    (case_dir / f"{summary_name}.txt").write_text(summary_text + "\n")
    return summary_text


@pytest.mark.parametrize("fabric_config", ["FABRIC_2D"])
def test_fabric_builder_perf(fabric_config):
    # Setup
    tt_metal_home = get_tt_metal_home()
    case_dir = get_output_dir(tt_metal_home) / fabric_config
    # Clean up the output directory, including wiping any existing cache
    if case_dir.exists():
        shutil.rmtree(case_dir)
    (case_dir / "cache").mkdir(parents=True)

    # Measure
    run_benchmark_under_tracy(tt_metal_home, fabric_config, case_dir)
    results = json.loads((case_dir / "results.json").read_text())
    validate_cache_state(results["phases"])
    durations = extract_durations_ms(read_zones(export_zones(tt_metal_home, case_dir)))

    # Report
    context = results["context"]
    golden_path = get_golden_path(tt_metal_home, context["arch"], context["cluster_type"])

    # Refresh the golden values when requested, otherwise check against them
    if should_update_golden():
        write_golden_rows(golden_path, fabric_config, durations)
        rows = [
            make_summary_row(fabric_config, cache, zone, measured_ms, status="updated-golden")
            for (cache, zone), measured_ms in durations.items()
        ]
        errors = []
    else:
        rows, errors = compare_to_golden(fabric_config, durations, read_golden_rows(golden_path))

    # Write the summary before failing so it is available for every run
    summary_text = write_summary(case_dir, f"summary_{context['arch']}_{context['cluster_type']}", rows)
    print(summary_text)
    print(f"\nResults: {case_dir}")
    print(f"Golden CSV: {golden_path}")

    assert not errors, "Fabric builder perf does not match the golden:\n" + "\n".join(errors)
