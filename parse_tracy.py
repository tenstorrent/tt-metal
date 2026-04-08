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


@dataclass
class KernelZoneEvent:
    """POD for kernel zone trace event."""

    device_id: int
    core_x: int
    core_y: int
    risc_type: str
    zone_name: str
    start_cycles: int
    end_cycles: int


@dataclass
class TensixCore:
    """POD for Tensix core profiling data."""

    device_id: int
    core_x: int
    core_y: int
    duration_cycles: int
    duration_ns: float


@dataclass
class ZoneDuration:
    """POD for individual zone duration measurement."""

    device_id: int
    core_x: int
    core_y: int
    zone_name: str
    duration_us: float


@dataclass
class OpsPerfData:
    """POD for ops perf results data."""

    fpu_util: float
    device_kernel_duration_ns: float


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


def validate_tracy_directory(tracy_dir: str) -> None:
    """Validate that the Tracy directory exists."""
    if not os.path.isdir(tracy_dir):
        logging.error(f"Tracy directory not found: {tracy_dir}")
        sys.exit(1)


def find_ops_perf_results(tracy_dir: str) -> str:
    """Find and return the ops_perf_results CSV file path."""
    ops_perf_files = glob.glob(os.path.join(tracy_dir, "ops_perf_results_*.csv"))
    if not ops_perf_files:
        logging.error(f"No ops_perf_results CSV file found in {tracy_dir}")
        sys.exit(1)
    return ops_perf_files[0]


def extract_report_date(ops_perf_file: str) -> str:
    """Extract report date from ops_perf_results filename."""
    basename = os.path.basename(ops_perf_file)
    report_timestamp = basename.replace("ops_perf_results_", "").replace(".csv", "")
    # Convert to readable format: YYYY-MM-DD HH:MM:SS
    return f"{report_timestamp[:4]}-{report_timestamp[5:7]}-{report_timestamp[8:10]} {report_timestamp[11:13]}:{report_timestamp[14:16]}:{report_timestamp[17:19]}"


def parse_ops_perf_data(ops_perf_file: str) -> Dict[int, OpsPerfData]:
    """Parse ops perf results data from ops_perf_results CSV."""
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


def parse_tracy_events(tracy_dir: str) -> Tuple[List[KernelZoneEvent], int]:
    """Parse all trace events from profile_log_device.csv and return list of KernelZoneEvent objects and chip frequency."""
    ZONE_START_TAG = "ZONE_START"
    ZONE_END_TAG = "ZONE_END"

    profile_log_path = os.path.join(tracy_dir, "profile_log_device.csv")
    if not os.path.exists(profile_log_path):
        logging.error(f"profile_log_device.csv not found in {tracy_dir}")
        sys.exit(1)

    # Temporary storage for pairing start/end events
    event_pairs: Dict[Tuple[int, int, int, str, str], Dict[str, int]] = {}
    chip_freq_mhz: int = 0

    with open(profile_log_path, "r") as f:
        lines = f.readlines()

        # Parse metadata from first line
        metadata = lines[0].strip()
        for part in metadata.split(","):
            if "CHIP_FREQ[MHz]" in part:
                chip_freq_mhz = int(part.split(":")[1].strip())

        # Parse CSV data starting from line 2
        reader = csv.DictReader(lines[1:])

        for row in reader:
            # Strip whitespace from column names
            row = {k.strip(): v.strip() for k, v in row.items()}

            device = int(row["PCIe slot"])
            core_x = int(row["core_x"])
            core_y = int(row["core_y"])
            risc_type = row["RISC processor type"]
            zone_name = row["zone name"]
            zone_type = row["type"]
            time_cycles = int(row["time[cycles since reset]"])

            # Pair start/end events for all zones
            event_key = (device, core_x, core_y, risc_type, zone_name)
            if event_key not in event_pairs:
                event_pairs[event_key] = {}

            if zone_type == ZONE_START_TAG:
                event_pairs[event_key]["start"] = time_cycles
            elif zone_type == ZONE_END_TAG:
                event_pairs[event_key]["end"] = time_cycles

    # Convert paired events to KernelZoneEvent objects
    kernel_events: List[KernelZoneEvent] = []
    for (device_id, core_x, core_y, risc_type, zone_name), event_data in event_pairs.items():
        if "start" in event_data and "end" in event_data:
            kernel_events.append(
                KernelZoneEvent(
                    device_id=device_id,
                    core_x=core_x,
                    core_y=core_y,
                    risc_type=risc_type,
                    zone_name=zone_name,
                    start_cycles=event_data["start"],
                    end_cycles=event_data["end"],
                )
            )

    return kernel_events, chip_freq_mhz


def parse_zone_durations(tracy_dir: str, zone_names: List[str]) -> Tuple[Dict[str, List[ZoneDuration]], int]:
    """Parse all zone durations for specified zone names.

    Collects ALL start/end pairs (not just one per key) using FIFO matching.

    Returns:
        Tuple of (dict mapping zone_name -> list of ZoneDuration, chip_freq_mhz)
    """
    profile_log_path = os.path.join(tracy_dir, "profile_log_device.csv")
    if not os.path.exists(profile_log_path):
        logging.error(f"profile_log_device.csv not found in {tracy_dir}")
        sys.exit(1)

    pending_starts: Dict[Tuple[int, int, int, str], List[int]] = defaultdict(list)
    durations: Dict[str, List[ZoneDuration]] = {name: [] for name in zone_names}
    chip_freq_mhz = 0

    with open(profile_log_path, "r") as f:
        lines = f.readlines()

        # Parse metadata from first line
        for part in lines[0].strip().split(","):
            if "CHIP_FREQ[MHz]" in part:
                chip_freq_mhz = int(part.split(":")[1].strip())

        reader = csv.DictReader(lines[1:])
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            zone_name = row["zone name"]

            if zone_name not in zone_names:
                continue

            device = int(row["PCIe slot"])
            core_x = int(row["core_x"])
            core_y = int(row["core_y"])
            zone_type = row["type"]
            time_cycles = int(row["time[cycles since reset]"])

            key = (device, core_x, core_y, zone_name)

            if zone_type == "ZONE_START":
                pending_starts[key].append(time_cycles)
            elif zone_type == "ZONE_END":
                if pending_starts[key]:
                    start_cycles = pending_starts[key].pop(0)  # FIFO matching
                    duration_cycles = time_cycles - start_cycles
                    if duration_cycles > 0:
                        duration_us = duration_cycles / chip_freq_mhz  # cycles / MHz = us
                        durations[zone_name].append(
                            ZoneDuration(
                                device_id=device,
                                core_x=core_x,
                                core_y=core_y,
                                zone_name=zone_name,
                                duration_us=duration_us,
                            )
                        )

    return durations, chip_freq_mhz


def calculate_core_durations(kernel_events: List[KernelZoneEvent], chip_freq_mhz: int) -> List[TensixCore]:
    """Calculate kernel durations per core from KernelZoneEvent objects.

    Filters for BRISC-KERNEL events only and returns list of TensixCore objects.
    """
    KERNEL_ZONE_NAME = "BRISC-KERNEL"
    RISC_TYPE = "BRISC"

    def cycles_to_ns(cycles: int) -> float:
        return (cycles / chip_freq_mhz) * 1000

    tensix_cores: List[TensixCore] = []

    for event in kernel_events:
        # Filter for BRISC-KERNEL zones only
        if event.zone_name == KERNEL_ZONE_NAME and event.risc_type == RISC_TYPE:
            duration_cycles = event.end_cycles - event.start_cycles
            duration_ns = cycles_to_ns(duration_cycles)

            if duration_cycles > 0:
                tensix_cores.append(
                    TensixCore(
                        device_id=event.device_id,
                        core_x=event.core_x,
                        core_y=event.core_y,
                        duration_cycles=duration_cycles,
                        duration_ns=duration_ns,
                    )
                )

    return tensix_cores


def calculate_device_stats(cores: List[TensixCore], ops_perf: OpsPerfData) -> Optional[DeviceStats]:
    """Calculate statistics for a single device.

    Args:
        cores: List of TensixCore objects for this device
        ops_perf: OpsPerfData object containing FPU utilization and kernel duration

    Returns:
        DeviceStats object or None if no cores
    """
    if not cores:
        return None

    durations_ms: List[float] = [c.duration_ns / 1_000_000 for c in cores]  # Convert ns to ms

    # Calculate workload balance: sum(duration) / (#cores * max(duration))
    num_cores: int = len(durations_ms)
    max_duration: float = max(durations_ms)
    sum_duration: float = sum(durations_ms)
    workload_balance: float = (sum_duration / (num_cores * max_duration)) * 100  # as percentage

    # Create statistics object
    return DeviceStats(
        cores=num_cores,
        min=min(durations_ms),
        max=max_duration,
        avg=sum(durations_ms) / len(durations_ms),
        p50=percentile(durations_ms, 50),
        p75=percentile(durations_ms, 75),
        p90=percentile(durations_ms, 90),
        p99=percentile(durations_ms, 99),
        workload_balance=workload_balance,
        fpu_util=ops_perf.fpu_util,
        device_kernel_duration_ns=ops_perf.device_kernel_duration_ns,
    )


def generate_device_report(device: int, cores: List[TensixCore], stats: Optional[DeviceStats]) -> List[str]:
    """Generate analysis report for a single device.

    Args:
        device: Device ID (int)
        cores: List of TensixCore objects for this device
        stats: DeviceStats object or None if no data

    Returns:
        List of report lines
    """
    device_report: List[str] = []
    device_report.append(f"## Device {device}\n\n")

    if not cores or not stats:
        device_report.append("No kernel data found.\n\n")
        return device_report

    # Determine grid dimensions
    x_coords = sorted(set(c.core_x for c in cores))
    y_coords = sorted(set(c.core_y for c in cores))

    # Create grid lookup
    grid: Dict[Tuple[int, int], float] = {}
    for core in cores:
        grid[(core.core_x, core.core_y)] = core.duration_ns

    # Generate table
    device_report.append("### Kernel Duration Per Core (milliseconds)\n\n")

    # Header row
    header = "| Y\\X |"
    for x in x_coords:
        header += f" {x:2d} |"
    device_report.append(header + "\n")

    # Separator
    separator = "|-----|"
    for _ in x_coords:
        separator += "--------|"
    device_report.append(separator + "\n")

    # Data rows
    for y in y_coords:
        row = f"| {y:2d}  |"
        for x in x_coords:
            if (x, y) in grid:
                duration_ms = grid[(x, y)] / 1_000_000  # Convert ns to ms
                row += f" {duration_ms:6.2f} |"
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

    # Generate header dynamically
    header = "| Metric |"
    for dev in devices_sorted:
        header += f" Device {dev} |"
    summary.append(header + "\n")

    # Generate separator dynamically
    separator = "|--------|"
    for _ in devices_sorted:
        separator += "----------|"
    summary.append(separator + "\n")

    # Generate rows from shared metrics configuration
    for metric in DEVICE_STATS:
        row = f"| {metric.get_table_label()} |"
        for dev in devices_sorted:
            row += f" {metric.format_value(device_stats[dev])} |"
        summary.append(row + "\n")

    summary.append("\n")
    return summary


def generate_zone_percentile_report(zone_name: str, measurements: List[ZoneDuration]) -> List[str]:
    """Generate percentile report for a specific zone.

    Args:
        zone_name: Name of the zone
        measurements: List of ZoneDuration objects

    Returns:
        List of report lines in markdown format
    """
    report: List[str] = []
    report.append(f"## Zone: {zone_name}\n\n")

    if not measurements:
        report.append("No measurements found.\n\n")
        return report

    durations_us = [m.duration_us for m in measurements]
    report.append(f"**Total measurements:** {len(measurements)}\n\n")

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

    # Per-device breakdown
    by_device: Dict[int, List[float]] = defaultdict(list)
    by_device_core: Dict[int, Dict[Tuple[int, int], List[float]]] = defaultdict(lambda: defaultdict(list))
    # Track iteration order per core: by_device_core_iter[device][(x,y)] = [dur0, dur1, dur2, ...]
    by_device_core_iter: Dict[int, Dict[Tuple[int, int], List[float]]] = defaultdict(lambda: defaultdict(list))
    for m in measurements:
        by_device[m.device_id].append(m.duration_us)
        by_device_core[m.device_id][(m.core_x, m.core_y)].append(m.duration_us)
        by_device_core_iter[m.device_id][(m.core_x, m.core_y)].append(m.duration_us)

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

    # Per-core tables for each device
    for device in sorted(by_device_core.keys()):
        core_data = by_device_core[device]
        x_coords = sorted(set(k[0] for k in core_data.keys()))
        y_coords = sorted(set(k[1] for k in core_data.keys()))

        # Build grids of min and max durations
        grid_max: Dict[Tuple[int, int], float] = {}
        grid_min: Dict[Tuple[int, int], float] = {}
        for (cx, cy), durations in core_data.items():
            grid_max[(cx, cy)] = max(durations)
            grid_min[(cx, cy)] = min(durations)

        # Helper to generate a per-core table
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
    # Find max iterations across all cores
    max_iters = 0
    for device in by_device_core_iter:
        for core, durations in by_device_core_iter[device].items():
            max_iters = max(max_iters, len(durations))

    if max_iters > 0:
        report.append("### Per-Iteration Statistics (all devices)\n\n")
        report.append("| Iter | Min (us) | P50 (us) | P75 (us) | P90 (us) | Max (us) | Cores |\n")
        report.append("|------|----------|----------|----------|----------|----------|-------|\n")

        # Collect durations for each iteration across all devices/cores
        for iter_idx in range(min(max_iters, 50)):  # Limit to first 50 iterations
            iter_durations: List[float] = []
            for device in by_device_core_iter:
                for core, durations in by_device_core_iter[device].items():
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
    if 0 in by_device_core_iter:
        core_data = by_device_core_iter[0]
        x_coords = sorted(set(k[0] for k in core_data.keys()))
        y_coords = sorted(set(k[1] for k in core_data.keys()))

        # Find max iterations for device 0
        dev0_max_iters = max(len(durations) for durations in core_data.values())

        report.append("### Device 0 - Duration Per Core By Iteration (us)\n\n")

        for iter_idx in range(min(dev0_max_iters, 20)):  # First 20 iterations
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
    chip_freq_mhz: int,
    device_stats: Dict[int, DeviceStats],
    device_reports: List[List[str]],
    zone_reports: Optional[List[List[str]]] = None,
) -> None:
    """Write the complete analysis report to file."""
    report: List[str] = []
    report.append("# Tracy Profiling Analysis\n")
    report.append(f"**Report Date:** {report_date}\n")
    report.append(f"**Architecture:** Blackhole\n")
    report.append(f"**Chip Frequency:** {chip_freq_mhz} MHz\n")
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parse Tracy profiling logs and generate analysis report")
    parser.add_argument(
        "tracy_dir", help="Directory containing Tracy profiling logs (profile_log_device.csv, ops_perf_results_*.csv)"
    )
    parser.add_argument("-o", "--output", help="Output path for analysis report (default: <tracy_dir>/analysis.md)")
    parser.add_argument(
        "--zones", nargs="+", default=[], help="Zone names for percentile analysis (e.g., 'K fwd' 'K receive')"
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Validate and setup paths
    tracy_dir = os.path.abspath(args.tracy_dir)
    validate_tracy_directory(tracy_dir)
    output_path = os.path.abspath(args.output) if args.output else os.path.join(tracy_dir, "analysis.md")

    # Parse data
    ops_perf_file = find_ops_perf_results(tracy_dir)
    report_date = extract_report_date(ops_perf_file)
    ops_perf_data = parse_ops_perf_data(ops_perf_file)
    kernel_events, chip_freq_mhz = parse_tracy_events(tracy_dir)
    tensix_cores = calculate_core_durations(kernel_events, chip_freq_mhz)

    # Group cores by device
    cores_by_device = defaultdict(list)
    for core in tensix_cores:
        cores_by_device[core.device_id].append(core)

    # Calculate statistics and generate reports
    device_stats: Dict[int, DeviceStats] = {}
    device_reports: List[List[str]] = []
    for device in sorted(cores_by_device.keys()):
        ops_perf = ops_perf_data.get(device, OpsPerfData(fpu_util=0.0, device_kernel_duration_ns=0.0))
        stats = calculate_device_stats(cores_by_device[device], ops_perf)
        report = generate_device_report(device, cores_by_device[device], stats)
        if stats:
            device_stats[device] = stats
        device_reports.append(report)

    # Parse zone durations if zones specified
    zone_reports: Optional[List[List[str]]] = None
    if args.zones:
        zone_durations, _ = parse_zone_durations(tracy_dir, args.zones)
        zone_reports = []
        for zone_name in args.zones:
            zone_reports.append(generate_zone_percentile_report(zone_name, zone_durations[zone_name]))

    # Write output
    write_report(output_path, report_date, chip_freq_mhz, device_stats, device_reports, zone_reports)
