#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import logging
import os
import sys
import glob
from typing import List, Dict, Tuple, Optional, Union, Callable


# ── Data types: Parsing layer ────────────────────────────────────────────────


@dataclass
class TraceMetadata:
    """Metadata from the profile_log_device.csv header line."""

    arch: str
    chip_freq_mhz: int
    max_compute_cores: int


@dataclass
class TraceEvent:
    """A single row from profile_log_device.csv — no interpretation."""

    device_id: int
    core_x: int
    core_y: int
    risc_type: str
    zone_name: str
    zone_type: str  # "ZONE_START" or "ZONE_END"
    time_cycles: int


@dataclass
class TraceSpan:
    """A matched ZONE_START/ZONE_END pair = one duration measurement."""

    device_id: int
    core_x: int
    core_y: int
    risc_type: str
    zone_name: str
    start_cycles: int
    end_cycles: int

    @property
    def duration_cycles(self) -> int:
        return self.end_cycles - self.start_cycles

    def duration_us(self, freq_mhz: int) -> float:
        return self.duration_cycles / freq_mhz

    def duration_ns(self, freq_mhz: int) -> float:
        return (self.duration_cycles / freq_mhz) * 1000

    def duration_ms(self, freq_mhz: int) -> float:
        return self.duration_cycles / freq_mhz / 1000


# ── Data types: Ops perf (separate CSV) ──────────────────────────────────────


@dataclass
class OpsPerfData:
    """POD for ops perf results data."""

    fpu_util: float
    device_kernel_duration_ns: float


# ── Data types: Statistics & reporting ───────────────────────────────────────


@dataclass
class DeviceStats:
    """POD for device statistics."""

    cores: int
    min: float
    max: float
    avg: float
    p50: float
    p75: float
    p90: float
    p99: float
    workload_balance: float
    fpu_util: float
    device_kernel_duration_ns: float


@dataclass
class DeviceStatsConfig:
    """Configuration for a device metric."""

    label: str
    accessor: Callable[[DeviceStats], Union[int, float]]
    format_spec: str
    units: str = ""

    def format_value(self, stats: DeviceStats) -> str:
        """Format the metric value for display."""
        value = self.accessor(stats)
        return self.format_spec.format(value)

    def get_table_label(self) -> str:
        """Get label for table header (with units if present)."""
        return f"{self.label} ({self.units})" if self.units else self.label


# Shared metrics configuration used by both summary table and device reports
DEVICE_STATS = [
    DeviceStatsConfig("Total Cores", lambda s: s.cores, "{}", ""),
    DeviceStatsConfig("Min Duration", lambda s: s.min, "{:.2f}", "ms"),
    DeviceStatsConfig("Max Duration", lambda s: s.max, "{:.2f}", "ms"),
    DeviceStatsConfig("Avg Duration", lambda s: s.avg, "{:.2f}", "ms"),
    DeviceStatsConfig("P50 (Median)", lambda s: s.p50, "{:.2f}", "ms"),
    DeviceStatsConfig("P75", lambda s: s.p75, "{:.2f}", "ms"),
    DeviceStatsConfig("P90", lambda s: s.p90, "{:.2f}", "ms"),
    DeviceStatsConfig("P99", lambda s: s.p99, "{:.2f}", "ms"),
    DeviceStatsConfig("Workload Balance", lambda s: s.workload_balance, "{:.2f}", "%"),
    DeviceStatsConfig("FPU Utilization", lambda s: s.fpu_util, "{:.2f}", "%"),
    DeviceStatsConfig("Device Kernel Duration", lambda s: s.device_kernel_duration_ns / 1_000_000, "{:.2f}", "ms"),
]


# ── Utilities ────────────────────────────────────────────────────────────────


def percentile(data: List[float], p: float) -> float:
    """Calculate the p-th percentile of data."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        return 0
    k = (n - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_data[-1]
    if f == k:
        return sorted_data[f]
    # Linear interpolation
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


# ── Parsing layer ────────────────────────────────────────────────────────────


def parse_profile_log(path: str) -> Tuple[TraceMetadata, List[TraceEvent]]:
    """Parse profile_log_device.csv into metadata and typed events.

    Single pass. No filtering, no pairing, no unit conversion.

    Raises:
        FileNotFoundError: if path does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"profile_log_device.csv not found: {path}")

    events: List[TraceEvent] = []

    with open(path, "r") as f:
        lines = f.readlines()

        # Parse metadata from first line
        # Format: "ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 120"
        metadata_parts: Dict[str, str] = {}
        for part in lines[0].strip().split(","):
            part = part.strip()
            if ":" in part:
                key, value = part.split(":", 1)
                metadata_parts[key.strip()] = value.strip()

        metadata = TraceMetadata(
            arch=metadata_parts.get("ARCH", ""),
            chip_freq_mhz=int(metadata_parts.get("CHIP_FREQ[MHz]", "0")),
            max_compute_cores=int(metadata_parts.get("Max Compute Cores", "0")),
        )

        # Parse CSV data (header is line 2, data starts line 3)
        reader = csv.DictReader(lines[1:])
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            events.append(
                TraceEvent(
                    device_id=int(row["PCIe slot"]),
                    core_x=int(row["core_x"]),
                    core_y=int(row["core_y"]),
                    risc_type=row["RISC processor type"],
                    zone_name=row["zone name"],
                    zone_type=row["type"],
                    time_cycles=int(row["time[cycles since reset]"]),
                )
            )

    return metadata, events


def pair_events(events: List[TraceEvent]) -> List[TraceSpan]:
    """Pair ZONE_START/ZONE_END events into TraceSpans using FIFO matching.

    Groups by (device_id, core_x, core_y, risc_type, zone_name).
    Drops pairs with non-positive duration.
    """
    pending_starts: Dict[Tuple[int, int, int, str, str], List[int]] = defaultdict(list)
    spans: List[TraceSpan] = []

    for event in events:
        key = (event.device_id, event.core_x, event.core_y, event.risc_type, event.zone_name)

        if event.zone_type == "ZONE_START":
            pending_starts[key].append(event.time_cycles)
        elif event.zone_type == "ZONE_END":
            if pending_starts[key]:
                start_cycles = pending_starts[key].pop(0)
                if event.time_cycles > start_cycles:
                    spans.append(
                        TraceSpan(
                            device_id=event.device_id,
                            core_x=event.core_x,
                            core_y=event.core_y,
                            risc_type=event.risc_type,
                            zone_name=event.zone_name,
                            start_cycles=start_cycles,
                            end_cycles=event.time_cycles,
                        )
                    )

    return spans


def parse_ops_perf_data(ops_perf_file: str) -> Dict[int, OpsPerfData]:
    """Parse ops perf results data from ops_perf_results CSV.

    Raises:
        FileNotFoundError: if ops_perf_file does not exist
    """
    if not os.path.exists(ops_perf_file):
        raise FileNotFoundError(f"ops_perf_results file not found: {ops_perf_file}")

    ops_perf_data: Dict[int, OpsPerfData] = {}
    with open(ops_perf_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            device_id = int(row["DEVICE ID"].strip())

            # Parse FPU utilization
            fpu_util_str = row.get("PM FPU UTIL (%)", "").strip()
            if not fpu_util_str:
                fpu_util_str = row.get("Avg FPU util on full grid (%)", "").strip()
            fpu_util = float(fpu_util_str) if fpu_util_str else 0.0

            # Parse device kernel duration
            device_kernel_duration_str = row.get("DEVICE KERNEL DURATION [ns]", "").strip()
            device_kernel_duration_ns = float(device_kernel_duration_str) if device_kernel_duration_str else 0.0

            ops_perf_data[device_id] = OpsPerfData(
                fpu_util=fpu_util, device_kernel_duration_ns=device_kernel_duration_ns
            )

    return ops_perf_data


# ── File discovery helpers ───────────────────────────────────────────────────


def find_ops_perf_results(tracy_dir: str) -> str:
    """Find and return the ops_perf_results CSV file path.

    Raises:
        FileNotFoundError: if no matching file found
    """
    ops_perf_files = glob.glob(os.path.join(tracy_dir, "ops_perf_results_*.csv"))
    if not ops_perf_files:
        raise FileNotFoundError(f"No ops_perf_results CSV file found in {tracy_dir}")
    return ops_perf_files[0]


def extract_report_date(ops_perf_file: str) -> str:
    """Extract report date from ops_perf_results filename."""
    basename = os.path.basename(ops_perf_file)
    report_timestamp = basename.replace("ops_perf_results_", "").replace(".csv", "")
    # Convert to readable format: YYYY-MM-DD HH:MM:SS
    return f"{report_timestamp[:4]}-{report_timestamp[5:7]}-{report_timestamp[8:10]} {report_timestamp[11:13]}:{report_timestamp[14:16]}:{report_timestamp[17:19]}"


# ── Statistics ───────────────────────────────────────────────────────────────


def calculate_device_stats(spans: List[TraceSpan], chip_freq_mhz: int, ops_perf: OpsPerfData) -> Optional[DeviceStats]:
    """Calculate statistics for a single device from its kernel spans."""
    if not spans:
        return None

    durations_ms: List[float] = [s.duration_ms(chip_freq_mhz) for s in spans]

    num_cores: int = len(durations_ms)
    max_duration: float = max(durations_ms)
    sum_duration: float = sum(durations_ms)
    workload_balance: float = (sum_duration / (num_cores * max_duration)) * 100

    return DeviceStats(
        cores=num_cores,
        min=min(durations_ms),
        max=max_duration,
        avg=sum_duration / num_cores,
        p50=percentile(durations_ms, 50),
        p75=percentile(durations_ms, 75),
        p90=percentile(durations_ms, 90),
        p99=percentile(durations_ms, 99),
        workload_balance=workload_balance,
        fpu_util=ops_perf.fpu_util,
        device_kernel_duration_ns=ops_perf.device_kernel_duration_ns,
    )


# ── Report generation ────────────────────────────────────────────────────────


def generate_device_report(
    device: int, spans: List[TraceSpan], chip_freq_mhz: int, stats: Optional[DeviceStats]
) -> List[str]:
    """Generate analysis report for a single device."""
    device_report: List[str] = []
    device_report.append(f"## Device {device}\n\n")

    if not spans or not stats:
        device_report.append("No kernel data found.\n\n")
        return device_report

    x_coords = sorted(set(s.core_x for s in spans))
    y_coords = sorted(set(s.core_y for s in spans))

    grid: Dict[Tuple[int, int], float] = {}
    for s in spans:
        grid[(s.core_x, s.core_y)] = s.duration_ms(chip_freq_mhz)

    # Generate table
    device_report.append("### Kernel Duration Per Core (milliseconds)\n\n")

    header = "| Y\\X |"
    for x in x_coords:
        header += f" {x:2d} |"
    device_report.append(header + "\n")

    separator = "|-----|"
    for _ in x_coords:
        separator += "--------|"
    device_report.append(separator + "\n")

    for y in y_coords:
        row = f"| {y:2d}  |"
        for x in x_coords:
            if (x, y) in grid:
                row += f" {grid[(x, y)]:6.2f} |"
            else:
                row += "      - |"
        device_report.append(row + "\n")

    # Append statistics from shared metrics configuration
    device_report.append(f"\n**Statistics:**\n")
    for metric in DEVICE_STATS:
        formatted_value = metric.format_value(stats)
        unit_str = f" {metric.units}" if metric.units else ""
        device_report.append(f"- {metric.label}: {formatted_value}{unit_str}\n")
    device_report.append("\n")

    return device_report


def generate_summary_table(device_stats: Dict[int, DeviceStats]) -> List[str]:
    """Generate cross-device comparison summary table."""
    summary: List[str] = []
    summary.append("## Summary: Cross-Device Comparison\n\n")

    devices_sorted = sorted(device_stats.keys())

    header = "| Metric |"
    for dev in devices_sorted:
        header += f" Device {dev} |"
    summary.append(header + "\n")

    separator = "|--------|"
    for _ in devices_sorted:
        separator += "----------|"
    summary.append(separator + "\n")

    for metric in DEVICE_STATS:
        row = f"| {metric.get_table_label()} |"
        for dev in devices_sorted:
            row += f" {metric.format_value(device_stats[dev])} |"
        summary.append(row + "\n")

    summary.append("\n")
    return summary


def generate_zone_percentile_report(zone_name: str, spans: List[TraceSpan], chip_freq_mhz: int) -> List[str]:
    """Generate percentile report for a specific zone."""
    report: List[str] = []
    report.append(f"## Zone: {zone_name}\n\n")

    if not spans:
        report.append("No measurements found.\n\n")
        return report

    durations_us = [s.duration_us(chip_freq_mhz) for s in spans]
    report.append(f"**Total measurements:** {len(spans)}\n\n")

    # Overall stats table
    report.append("### Overall (all devices)\n\n")
    report.append("| Percentile | Duration (us) |\n")
    report.append("|------------|---------------|\n")
    report.append(f"| Min        | {min(durations_us):>13.2f} |\n")
    report.append(f"| P50        | {percentile(durations_us, 50):>13.2f} |\n")
    report.append(f"| P75        | {percentile(durations_us, 75):>13.2f} |\n")
    report.append(f"| P90        | {percentile(durations_us, 90):>13.2f} |\n")
    report.append(f"| P95        | {percentile(durations_us, 95):>13.2f} |\n")
    report.append(f"| Max        | {max(durations_us):>13.2f} |\n")
    report.append(f"| Mean       | {sum(durations_us)/len(durations_us):>13.2f} |\n")
    report.append("\n")

    # Group by device and by (device, core)
    by_device: Dict[int, List[float]] = defaultdict(list)
    by_device_core: Dict[int, Dict[Tuple[int, int], List[float]]] = defaultdict(lambda: defaultdict(list))
    for s in spans:
        dur = s.duration_us(chip_freq_mhz)
        by_device[s.device_id].append(dur)
        by_device_core[s.device_id][(s.core_x, s.core_y)].append(dur)

    if len(by_device) > 1:
        report.append("### Per Device Summary\n\n")
        report.append("| Device | Count | Min (us) | P50 (us) | P75 (us) | P90 (us) | P95 (us) | Max (us) |\n")
        report.append("|--------|-------|----------|----------|----------|----------|----------|----------|\n")
        for dev in sorted(by_device.keys()):
            d = by_device[dev]
            report.append(
                f"| {dev:>6} | {len(d):>5} | {min(d):>8.2f} | {percentile(d, 50):>8.2f} | "
                f"{percentile(d, 75):>8.2f} | {percentile(d, 90):>8.2f} | {percentile(d, 95):>8.2f} | {max(d):>8.2f} |\n"
            )
        report.append("\n")

    # Per-core min/max tables for each device
    for device in sorted(by_device_core.keys()):
        core_data = by_device_core[device]
        x_coords = sorted(set(k[0] for k in core_data.keys()))
        y_coords = sorted(set(k[1] for k in core_data.keys()))

        grid_max: Dict[Tuple[int, int], float] = {}
        grid_min: Dict[Tuple[int, int], float] = {}
        for (cx, cy), durations in core_data.items():
            grid_max[(cx, cy)] = max(durations)
            grid_min[(cx, cy)] = min(durations)

        def generate_core_table(grid: Dict[Tuple[int, int], float], title: str) -> None:
            report.append(f"### Device {device} - {title} (us)\n\n")
            header = "| Y\\X |"
            for x in x_coords:
                header += f" {x:>5} |"
            report.append(header + "\n")

            separator = "|-----|"
            for _ in x_coords:
                separator += "-------|"
            report.append(separator + "\n")

            for y in y_coords:
                row = f"| {y:>3} |"
                for x in x_coords:
                    if (x, y) in grid:
                        row += f" {grid[(x, y)]:>5.1f} |"
                    else:
                        row += "     - |"
                report.append(row + "\n")
            report.append("\n")

        generate_core_table(grid_min, "Min Duration Per Core")
        generate_core_table(grid_max, "Max Duration Per Core")

    # Per-iteration analysis across all devices
    max_iters = 0
    for device in by_device_core:
        for core, durations in by_device_core[device].items():
            max_iters = max(max_iters, len(durations))

    if max_iters > 0:
        report.append("### Per-Iteration Statistics (all devices)\n\n")
        report.append("| Iter | Min (us) | P50 (us) | P75 (us) | P90 (us) | Max (us) | Cores |\n")
        report.append("|------|----------|----------|----------|----------|----------|-------|\n")

        for iter_idx in range(min(max_iters, 50)):
            iter_durations: List[float] = []
            for device in by_device_core:
                for core, durations in by_device_core[device].items():
                    if iter_idx < len(durations):
                        iter_durations.append(durations[iter_idx])

            if iter_durations:
                report.append(
                    f"| {iter_idx:>4} | {min(iter_durations):>8.2f} | {percentile(iter_durations, 50):>8.2f} | "
                    f"{percentile(iter_durations, 75):>8.2f} | {percentile(iter_durations, 90):>8.2f} | "
                    f"{max(iter_durations):>8.2f} | {len(iter_durations):>5} |\n"
                )

        if max_iters > 50:
            report.append(f"\n*Showing first 50 of {max_iters} iterations*\n")
        report.append("\n")

    # Per-iteration core tables for device 0
    if 0 in by_device_core:
        core_data = by_device_core[0]
        x_coords = sorted(set(k[0] for k in core_data.keys()))
        y_coords = sorted(set(k[1] for k in core_data.keys()))

        dev0_max_iters = max(len(durations) for durations in core_data.values())

        report.append("### Device 0 - Duration Per Core By Iteration (us)\n\n")

        for iter_idx in range(min(dev0_max_iters, 20)):
            report.append(f"**Iteration {iter_idx}**\n\n")

            header = "| Y\\X |"
            for x in x_coords:
                header += f" {x:>5} |"
            report.append(header + "\n")

            separator = "|-----|"
            for _ in x_coords:
                separator += "-------|"
            report.append(separator + "\n")

            for y in y_coords:
                row = f"| {y:>3} |"
                for x in x_coords:
                    if (x, y) in core_data and iter_idx < len(core_data[(x, y)]):
                        row += f" {core_data[(x, y)][iter_idx]:>5.1f} |"
                    else:
                        row += "     - |"
                report.append(row + "\n")
            report.append("\n")

        if dev0_max_iters > 20:
            report.append(f"*Showing first 20 of {dev0_max_iters} iterations*\n\n")

    return report


def write_report(
    output_path: str,
    report_date: str,
    metadata: TraceMetadata,
    device_stats: Dict[int, DeviceStats],
    device_reports: List[List[str]],
    zone_reports: Optional[List[List[str]]] = None,
) -> None:
    """Write the complete analysis report to file."""
    report: List[str] = []
    report.append("# Tracy Profiling Analysis\n")
    report.append(f"**Report Date:** {report_date}\n")
    report.append(f"**Architecture:** {metadata.arch.title()}\n")
    report.append(f"**Chip Frequency:** {metadata.chip_freq_mhz} MHz\n")
    report.append(f"**Operation:** RingJointSDPADeviceOperation\n\n")

    # Add summary table
    report.extend(generate_summary_table(device_stats))

    # Add zone percentile reports if provided
    if zone_reports:
        report.append("# Zone Percentile Analysis\n\n")
        for zone_report in zone_reports:
            report.extend(zone_report)

    # Add device detail reports
    for device_report in device_reports:
        report.extend(device_report)

    # Write to file
    with open(output_path, "w") as f:
        f.writelines(report)

    logging.info(f"Analysis complete! Report written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Tracy profiling logs and generate analysis report")
    parser.add_argument(
        "tracy_dir", help="Directory containing Tracy profiling logs (profile_log_device.csv, ops_perf_results_*.csv)"
    )
    parser.add_argument("-o", "--output", help="Output path for analysis report (default: <tracy_dir>/analysis.md)")
    parser.add_argument(
        "--zones", nargs="+", default=[], help="Zone names for percentile analysis (e.g., 'K fwd' 'K receive')"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tracy_dir = os.path.abspath(args.tracy_dir)
    if not os.path.isdir(tracy_dir):
        logging.error(f"Tracy directory not found: {tracy_dir}")
        sys.exit(1)
    output_path = os.path.abspath(args.output) if args.output else os.path.join(tracy_dir, "analysis.md")

    # Parse — single pass through profile_log_device.csv
    ops_perf_file = find_ops_perf_results(tracy_dir)
    report_date = extract_report_date(ops_perf_file)
    ops_perf_data = parse_ops_perf_data(ops_perf_file)
    metadata, events = parse_profile_log(os.path.join(tracy_dir, "profile_log_device.csv"))
    spans = pair_events(events)

    # Filter kernel spans for device reports (replaces calculate_core_durations)
    kernel_spans = [s for s in spans if s.zone_name == "BRISC-KERNEL" and s.risc_type == "BRISC"]

    # Group by device
    spans_by_device: Dict[int, List[TraceSpan]] = defaultdict(list)
    for s in kernel_spans:
        spans_by_device[s.device_id].append(s)

    # Calculate statistics and generate device reports
    device_stats: Dict[int, DeviceStats] = {}
    device_reports: List[List[str]] = []
    for device in sorted(spans_by_device.keys()):
        ops_perf = ops_perf_data.get(device, OpsPerfData(fpu_util=0.0, device_kernel_duration_ns=0.0))
        stats = calculate_device_stats(spans_by_device[device], metadata.chip_freq_mhz, ops_perf)
        report = generate_device_report(device, spans_by_device[device], metadata.chip_freq_mhz, stats)
        if stats:
            device_stats[device] = stats
        device_reports.append(report)

    # Zone percentile reports — filter from the same span list, no re-parse
    zone_reports: Optional[List[List[str]]] = None
    if args.zones:
        zone_names_set = set(args.zones)
        zone_spans: Dict[str, List[TraceSpan]] = {name: [] for name in args.zones}
        for s in spans:
            if s.zone_name in zone_names_set:
                zone_spans[s.zone_name].append(s)
        zone_reports = [
            generate_zone_percentile_report(name, zone_spans[name], metadata.chip_freq_mhz) for name in args.zones
        ]

    write_report(output_path, report_date, metadata, device_stats, device_reports, zone_reports)
