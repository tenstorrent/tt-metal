#!/usr/bin/env python3
"""
Device Profiler Analysis Script for reduce_to_all optimization.

Usage:
    python analyze_device_profile.py <csv_path> [--all-iters] [--iter-range START END]

By default, only analyzes the main perf trace iterations (trace_id=1).
"""

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import statistics


@dataclass
class ZoneDuration:
    """A single zone duration measurement."""

    iteration: int
    device_id: int
    core: Tuple[int, int]
    risc: str
    zone_name: str
    duration_cycles: int
    trace_id: Optional[str]
    start_cycle: int = 0  # Absolute start timestamp
    end_cycle: int = 0  # Absolute end timestamp


@dataclass
class ZoneEvent:
    """A single zone start/end event with absolute timestamp."""

    pcie_slot: int
    core: Tuple[int, int]
    risc: str
    cycle: int  # Absolute timestamp
    run_host_id: int
    trace_id: Optional[str]
    zone_name: str
    zone_type: str  # ZONE_START or ZONE_END

    @property
    def device_id(self) -> int:
        return self.run_host_id & 0x3FF

    @property
    def iteration(self) -> int:
        return self.run_host_id >> 10


@dataclass
class ProfileData:
    """Container for all parsed profile data."""

    clock_mhz: float = 1350.0
    arch: str = ""
    durations: List[ZoneDuration] = field(default_factory=list)
    events: List[ZoneEvent] = field(default_factory=list)  # Raw events for timestamp analysis

    def cycles_to_ns(self, cycles: int) -> float:
        return cycles * 1000 / self.clock_mhz


def parse_csv(csv_path: str) -> ProfileData:
    """Parse the profile_log_device.csv file."""
    data = ProfileData()

    # Track zone starts for matching with ends
    zone_starts: Dict[Tuple, int] = {}

    # Zones to skip (firmware/kernel wrapper zones)
    skip_zones = {
        "BRISC-FW",
        "BRISC-KERNEL",
        "NCRISC-FW",
        "NCRISC-KERNEL",
        "TRISC-FW",
        "TRISC-KERNEL",
        "TRISC_0-FW",
        "TRISC_0-KERNEL",
        "TRISC_1-FW",
        "TRISC_1-KERNEL",
        "TRISC_2-FW",
        "TRISC_2-KERNEL",
    }

    with open(csv_path, "r") as f:
        reader = csv.reader(f)

        # First line: ARCH info
        arch_line = next(reader)
        if arch_line:
            # Parse "ARCH: blackhole, CHIP_FREQ[MHz]: 1350, ..."
            for part in arch_line:
                part = part.strip()
                if part.startswith("ARCH:"):
                    data.arch = part.split(":")[1].strip()
                elif "CHIP_FREQ" in part:
                    try:
                        data.clock_mhz = float(part.split(":")[1].strip())
                    except:
                        pass

        # Second line: Headers
        headers = next(reader)

        # Parse data rows
        for row in reader:
            if len(row) < 12:
                continue

            try:
                core_x = int(row[1])
                core_y = int(row[2])
                risc = row[3].strip()
                cycle = int(row[5])
                run_host_id = int(row[7])
                trace_id = row[8].strip() if len(row) > 8 else ""
                zone_name = row[10].strip() if len(row) > 10 else ""
                zone_type = row[11].strip() if len(row) > 11 else ""

                # Skip wrapper zones
                if zone_name in skip_zones or not zone_name:
                    continue

                # Decode run_host_id
                device_id = run_host_id & 0x3FF
                iteration = run_host_id >> 10

                # Key for matching start/end
                key = (core_x, core_y, risc, zone_name, run_host_id)

                if zone_type == "ZONE_START":
                    zone_starts[key] = cycle
                    # Also store the raw event for timestamp analysis
                    pcie_slot = int(row[0]) if row[0] else 0
                    data.events.append(
                        ZoneEvent(
                            pcie_slot=pcie_slot,
                            core=(core_x, core_y),
                            risc=risc,
                            cycle=cycle,
                            run_host_id=run_host_id,
                            trace_id=trace_id if trace_id else None,
                            zone_name=zone_name,
                            zone_type=zone_type,
                        )
                    )
                elif zone_type == "ZONE_END" and key in zone_starts:
                    start_cycle = zone_starts[key]
                    duration = cycle - start_cycle

                    data.durations.append(
                        ZoneDuration(
                            iteration=iteration,
                            device_id=device_id,
                            core=(core_x, core_y),
                            risc=risc,
                            zone_name=zone_name,
                            duration_cycles=duration,
                            trace_id=trace_id if trace_id else None,
                            start_cycle=start_cycle,
                            end_cycle=cycle,
                        )
                    )
                    del zone_starts[key]

                    # Also store the end event
                    pcie_slot = int(row[0]) if row[0] else 0
                    data.events.append(
                        ZoneEvent(
                            pcie_slot=pcie_slot,
                            core=(core_x, core_y),
                            risc=risc,
                            cycle=cycle,
                            run_host_id=run_host_id,
                            trace_id=trace_id if trace_id else None,
                            zone_name=zone_name,
                            zone_type=zone_type,
                        )
                    )

            except (ValueError, IndexError):
                continue

    return data


def filter_data(
    data: ProfileData, iter_range: Optional[Tuple[int, int]] = None, trace_id_filter: Optional[str] = None
) -> ProfileData:
    """Filter data by iteration range and/or trace_id."""
    filtered = ProfileData(clock_mhz=data.clock_mhz, arch=data.arch)

    for d in data.durations:
        # Filter by iteration range
        if iter_range and not (iter_range[0] <= d.iteration <= iter_range[1]):
            continue

        # Filter by trace_id
        if trace_id_filter is not None and d.trace_id != trace_id_filter:
            continue

        filtered.durations.append(d)

    return filtered


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values."""
    if not values:
        return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0,
    }


def print_summary_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def analyze_per_zone(data: ProfileData):
    """Analyze performance per zone (across all iterations/devices/cores)."""
    print_summary_header("PER-ZONE SUMMARY (all iterations, all devices, all cores)")

    # Group by zone name
    zone_durations: Dict[str, List[float]] = defaultdict(list)

    for d in data.durations:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        zone_durations[d.zone_name].append(dur_ns)

    # Print stats for each zone
    print(
        f"\n{'Zone Name':<25} {'Count':>8} {'Mean(ns)':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'Std':>10} {'Max/Mean':>10}"
    )
    print("-" * 105)

    for zone in sorted(zone_durations.keys()):
        stats = compute_stats(zone_durations[zone])
        ratio = stats["max"] / stats["mean"] if stats["mean"] > 0 else 0
        flag = "⚠️" if ratio > 2.0 else ""
        print(
            f"{zone:<25} {stats['count']:>8} {stats['mean']:>10.0f} {stats['median']:>10.0f} "
            f"{stats['min']:>10.0f} {stats['max']:>10.0f} {stats['std']:>10.0f} {ratio:>9.2f}x {flag}"
        )


def analyze_per_trisc(data: ProfileData):
    """Analyze compute kernel zones separated by TRISC (Unpack/Math/Pack).

    Each compute core has 3 RISC processors:
    - TRISC_0 (Unpack): Reads CBs → SRC registers
    - TRISC_1 (Math):   FPU operations
    - TRISC_2 (Pack):   DST registers → CBs

    The same zone marker appears on all 3 TRISCs, but each sees different code
    (UNPACK/MATH/PACK macros compile out non-matching code). This analysis
    separates them to identify true bottlenecks vs waiting.
    """
    print_summary_header("PER-TRISC ANALYSIS (Compute zones by Unpack/Math/Pack)")

    # Filter to only TRISC cores
    trisc_data = [d for d in data.durations if d.risc.startswith("TRISC")]

    if not trisc_data:
        print("No TRISC data found.")
        return

    # Map TRISC to role
    trisc_roles = {
        "TRISC_0": "Unpack",
        "TRISC_1": "Math",
        "TRISC_2": "Pack",
        "TRISC": "Combined",  # Fallback for old format
    }

    # Group by (zone, trisc)
    zone_trisc_durations: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for d in trisc_data:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        zone_trisc_durations[(d.zone_name, d.risc)].append(dur_ns)

    # Get all unique zones that appear on TRISCs
    zones = sorted(set(z for z, _ in zone_trisc_durations.keys()))
    triscs = ["TRISC_0", "TRISC_1", "TRISC_2"]

    print(f"\nFound {len(zones)} zones on TRISC cores")
    print(f"\nNote: Same zone on different TRISCs measures different code paths:")
    print("  - TRISC_0 (Unpack): Data movement from CB to SRC registers")
    print("  - TRISC_1 (Math):   Actual compute operations (FPU)")
    print("  - TRISC_2 (Pack):   Data movement from DST registers to CB")
    print("  Stalls (tile_regs_wait/commit) show as idle time on waiting TRISC.\n")

    # Print header
    print(
        f"{'Zone':<22} {'TRISC':<8} {'Role':<7} {'Count':>6} {'Mean(ns)':>10} {'Med':>10} {'Min':>10} {'Max':>10} {'Bottleneck?':<12}"
    )
    print("-" * 115)

    for zone in zones:
        # Get stats for each TRISC on this zone
        trisc_stats = {}
        for trisc in triscs:
            vals = zone_trisc_durations.get((zone, trisc), [])
            if vals:
                trisc_stats[trisc] = compute_stats(vals)

        if not trisc_stats:
            continue

        # Find which TRISC is slowest (potential bottleneck)
        slowest_trisc = max(trisc_stats.keys(), key=lambda t: trisc_stats[t]["mean"])
        slowest_mean = trisc_stats[slowest_trisc]["mean"]

        # Print each TRISC's stats for this zone
        first = True
        for trisc in triscs:
            if trisc not in trisc_stats:
                continue
            stats = trisc_stats[trisc]
            role = trisc_roles.get(trisc, "?")

            # Flag if this is the bottleneck (slowest and significantly slower)
            is_bottleneck = (
                trisc == slowest_trisc
                and len(trisc_stats) > 1
                and stats["mean"] > 1.2 * min(s["mean"] for s in trisc_stats.values())
            )
            bottleneck_flag = "← BOTTLENECK" if is_bottleneck else ""

            zone_display = zone if first else ""
            first = False

            print(
                f"{zone_display:<22} {trisc:<8} {role:<7} {stats['count']:>6} "
                f"{stats['mean']:>10.0f} {stats['median']:>10.0f} "
                f"{stats['min']:>10.0f} {stats['max']:>10.0f} {bottleneck_flag:<12}"
            )

        # Separator between zones
        if len(trisc_stats) > 1:
            print()

    # Summary: which TRISC is generally slowest?
    print("\n--- TRISC Bottleneck Summary ---")
    trisc_totals: Dict[str, List[float]] = defaultdict(list)
    for (zone, trisc), vals in zone_trisc_durations.items():
        trisc_totals[trisc].extend(vals)

    for trisc in triscs:
        if trisc in trisc_totals:
            stats = compute_stats(trisc_totals[trisc])
            role = trisc_roles.get(trisc, "?")
            print(f"  {trisc} ({role}): Mean={stats['mean']:.0f} ns, Total samples={stats['count']}")

    # Identify overall bottleneck
    if trisc_totals:
        slowest = max(trisc_totals.keys(), key=lambda t: statistics.mean(trisc_totals[t]))
        fastest = min(trisc_totals.keys(), key=lambda t: statistics.mean(trisc_totals[t]))
        slowest_mean = statistics.mean(trisc_totals[slowest])
        fastest_mean = statistics.mean(trisc_totals[fastest])

        if slowest_mean > 1.2 * fastest_mean:
            role = trisc_roles.get(slowest, "?")
            print(f"\n  → Overall bottleneck: {slowest} ({role})")
            print(f"    {role} is {slowest_mean/fastest_mean:.1f}x slower than fastest TRISC")
            if slowest == "TRISC_1":
                print(f"    Consider: Optimize math ops, reduce tile count, or pipeline better")
            elif slowest == "TRISC_0":
                print(f"    Consider: CB starvation? Check data dependencies, pack delays")
            elif slowest == "TRISC_2":
                print(f"    Consider: Dst register contention? Check tile_regs_commit timing")
        else:
            print(f"\n  → TRISCs are balanced (no single bottleneck)")


def analyze_per_device(data: ProfileData):
    """Analyze performance per device (across all iterations/zones)."""
    print_summary_header("PER-DEVICE SUMMARY (all iterations, all zones)")

    # Group by device
    device_durations: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for d in data.durations:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        device_durations[d.device_id][d.zone_name].append(dur_ns)

    # Get all zones
    all_zones = sorted(set(d.zone_name for d in data.durations))

    # Print header
    print(f"\n{'Device':<8}", end="")
    for zone in all_zones[:6]:  # Limit to first 6 zones for readability
        print(f" {zone[:12]:>14}", end="")
    print()
    print("-" * (8 + 15 * min(6, len(all_zones))))

    for device in sorted(device_durations.keys()):
        print(f"D{device:<7}", end="")
        for zone in all_zones[:6]:
            if zone in device_durations[device]:
                mean = statistics.mean(device_durations[device][zone])
                print(f" {mean:>14.0f}", end="")
            else:
                print(f" {'--':>14}", end="")
        print()


def analyze_per_iteration(data: ProfileData):
    """Analyze performance per iteration (across all devices/zones)."""
    print_summary_header("PER-ITERATION SUMMARY (all devices, key zones)")

    # Group by iteration
    iter_durations: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for d in data.durations:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        iter_durations[d.iteration][d.zone_name].append(dur_ns)

    # Key zones to show (adjust based on what zones exist)
    all_zones = sorted(set(d.zone_name for d in data.durations))
    key_zones = [z for z in all_zones if "WAIT" in z or "COMPUTE" in z or "SEND" in z][:4]
    if not key_zones:
        key_zones = all_zones[:4]

    # Print header
    print(f"\n{'Iter':<6}", end="")
    for zone in key_zones:
        print(f" {zone[:15]:>17}", end="")
    print("  Total(ns)")
    print("-" * (6 + 18 * len(key_zones) + 12))

    iterations = sorted(iter_durations.keys())

    for iteration in iterations[:30]:  # Limit to first 30
        print(f"{iteration:<6}", end="")
        total = 0
        for zone in key_zones:
            if zone in iter_durations[iteration]:
                mean = statistics.mean(iter_durations[iteration][zone])
                total += mean
                print(f" {mean:>17.0f}", end="")
            else:
                print(f" {'--':>17}", end="")
        print(f"  {total:>10.0f}")


def analyze_per_iteration_zone_spread(data: ProfileData):
    """Show min/max/spread per zone for each iteration across all devices."""
    print_summary_header("PER-ITERATION ZONE SPREAD (min/max across devices per iteration)")

    # Group by (iteration, zone, device) -> list of durations
    iter_zone_dev: Dict[Tuple[int, str, int], List[float]] = defaultdict(list)

    for d in data.durations:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        iter_zone_dev[(d.iteration, d.zone_name, d.device_id)].append(dur_ns)

    # Get all zones and iterations
    all_zones = sorted(set(d.zone_name for d in data.durations))
    all_iters = sorted(set(d.iteration for d in data.durations))

    # Focus on key zones for readability
    key_zones = [z for z in all_zones if any(kw in z for kw in ["WAIT", "SEND", "COMPUTE", "FWD", "LOCAL"])]
    if not key_zones:
        key_zones = all_zones[:8]

    print(f"\nShowing {len(key_zones)} key zones across {len(all_iters)} iterations")
    print("For each zone: Min(device) | Max(device) | Spread(ns) | Which device is slowest")

    for zone in key_zones:
        print(f"\n--- {zone} ---")
        print(f"{'Iter':<6} {'Min(ns)':>10} {'MinDev':>7} {'Max(ns)':>10} {'MaxDev':>7} {'Spread':>10} {'Spread%':>8}")
        print("-" * 65)

        for iteration in all_iters[:30]:  # Limit to 30 iterations
            # Get mean duration per device for this (iter, zone)
            dev_means = {}
            for dev in range(4):
                vals = iter_zone_dev.get((iteration, zone, dev), [])
                if vals:
                    dev_means[dev] = statistics.mean(vals)

            if dev_means:
                min_dev = min(dev_means, key=dev_means.get)
                max_dev = max(dev_means, key=dev_means.get)
                min_val = dev_means[min_dev]
                max_val = dev_means[max_dev]
                spread = max_val - min_val
                spread_pct = (spread / min_val * 100) if min_val > 0 else 0

                print(
                    f"{iteration:<6} {min_val:>10.0f} {'D'+str(min_dev):>7} {max_val:>10.0f} {'D'+str(max_dev):>7} "
                    f"{spread:>10.0f} {spread_pct:>7.1f}%"
                )

    # Summary: which device is slowest most often per zone
    print(f"\n{'='*65}")
    print("SLOWEST DEVICE FREQUENCY (which device has max time most often)")
    print(f"{'='*65}")
    print(f"{'Zone':<30} {'D0':>6} {'D1':>6} {'D2':>6} {'D3':>6} {'Dominant':>10}")
    print("-" * 70)

    for zone in key_zones:
        slowest_count = {0: 0, 1: 0, 2: 0, 3: 0}

        for iteration in all_iters:
            dev_means = {}
            for dev in range(4):
                vals = iter_zone_dev.get((iteration, zone, dev), [])
                if vals:
                    dev_means[dev] = statistics.mean(vals)

            if dev_means:
                max_dev = max(dev_means, key=dev_means.get)
                slowest_count[max_dev] += 1

        total = sum(slowest_count.values())
        if total > 0:
            dominant = max(slowest_count, key=slowest_count.get)
            dominant_pct = slowest_count[dominant] / total * 100
            print(
                f"{zone:<30} {slowest_count[0]:>6} {slowest_count[1]:>6} {slowest_count[2]:>6} {slowest_count[3]:>6} "
                f"D{dominant}({dominant_pct:.0f}%)"
            )


def analyze_timestamp_skew(data: ProfileData):
    """
    Analyze timestamps to understand device timing relationships.

    IMPORTANT: Each PCIe device has its own independent clock that started at a different
    time. We CANNOT compare absolute timestamps across devices - they're in different
    clock domains. The massive offsets (e.g., 196ms) are just differences in when each
    chip was reset.

    What we CAN analyze:
    1. Per-device iteration-to-iteration timing patterns
    2. Duration variations within each device
    3. Whether high-duration iterations correlate with iteration parity
    """
    print_summary_header("TIMESTAMP ANALYSIS (Per-Device Iteration Timing)")

    print("NOTE: Each device has an independent clock. Cross-device absolute")
    print("      timestamp comparison is NOT meaningful.")

    # Target zones for analysis
    target_zones = ["R1-WAIT-NEIGHBOR", "R2-WAIT-NEIGHBOR"]

    # Group durations by (zone, device, iteration) -> duration
    zone_dev_iter: Dict[Tuple[str, int, int], int] = {}

    for d in data.durations:
        if d.zone_name not in target_zones:
            continue
        key = (d.zone_name, d.device_id, d.iteration)
        # Take first occurrence (representative core)
        if key not in zone_dev_iter:
            zone_dev_iter[key] = d.duration_cycles

    if not zone_dev_iter:
        print("No wait zones found.")
        return

    # Get all iterations
    all_iters = sorted(set(k[2] for k in zone_dev_iter.keys()))

    for zone in target_zones:
        print(f"\n{'='*90}")
        print(f"ZONE: {zone} - Per-Device Iteration Analysis")
        print(f"{'='*90}")

        # Collect per-device iteration sequences
        dev_sequences = {d: [] for d in range(4)}

        for iteration in all_iters:
            for dev in range(4):
                key = (zone, dev, iteration)
                if key in zone_dev_iter:
                    dur_ns = data.cycles_to_ns(zone_dev_iter[key])
                    dev_sequences[dev].append((iteration, dur_ns))

        # Analyze each device's pattern
        print(f"\n--- Per-Device Statistics ---")
        print(f"{'Device':<8} {'Mean(ns)':>12} {'StdDev':>12} {'Min':>12} {'Max':>12} {'CoV%':>10}")
        print("-" * 70)

        for dev in range(4):
            if dev_sequences[dev]:
                durs = [d[1] for d in dev_sequences[dev]]
                mean = statistics.mean(durs)
                std = statistics.stdev(durs) if len(durs) > 1 else 0
                cov = (std / mean * 100) if mean > 0 else 0
                print(f"D{dev:<7} {mean:>12.0f} {std:>12.0f} {min(durs):>12.0f} {max(durs):>12.0f} {cov:>9.1f}%")

        # Analyze even/odd iteration pattern per device
        print(f"\n--- Even vs Odd Iteration Pattern ---")
        print(f"{'Device':<8} {'Even Mean':>12} {'Odd Mean':>12} {'Diff':>12} {'Pattern':>20}")
        print("-" * 70)

        for dev in range(4):
            if dev_sequences[dev]:
                even_durs = [d[1] for d in dev_sequences[dev] if d[0] % 2 == 0]
                odd_durs = [d[1] for d in dev_sequences[dev] if d[0] % 2 == 1]

                if even_durs and odd_durs:
                    even_mean = statistics.mean(even_durs)
                    odd_mean = statistics.mean(odd_durs)
                    diff = odd_mean - even_mean

                    if abs(diff) > min(even_mean, odd_mean) * 0.5:
                        pattern = "ODD HIGH" if diff > 0 else "EVEN HIGH"
                    else:
                        pattern = "~SIMILAR"

                    print(f"D{dev:<7} {even_mean:>12.0f} {odd_mean:>12.0f} {diff:>+12.0f} {pattern:>20}")

        # Analyze iteration-to-iteration autocorrelation per device
        print(f"\n--- Lag-1 Autocorrelation (iteration N vs N+1) ---")

        for dev in range(4):
            if len(dev_sequences[dev]) > 2:
                durs = [d[1] for d in sorted(dev_sequences[dev], key=lambda x: x[0])]

                # Compute lag-1 autocorrelation
                n = len(durs)
                mean = statistics.mean(durs)
                var = sum((x - mean) ** 2 for x in durs) / n

                if var > 0:
                    cov = sum((durs[i] - mean) * (durs[i + 1] - mean) for i in range(n - 1)) / (n - 1)
                    autocorr = cov / var

                    pattern = ""
                    if autocorr < -0.5:
                        pattern = "STRONG ALTERNATING"
                    elif autocorr < -0.2:
                        pattern = "WEAK ALTERNATING"
                    elif autocorr > 0.5:
                        pattern = "STRONG PERSISTENT"
                    elif autocorr > 0.2:
                        pattern = "WEAK PERSISTENT"
                    else:
                        pattern = "RANDOM"

                    print(f"  D{dev}: {autocorr:>+.3f} ({pattern})")

        # Show per-device iteration breakdown for first 15 iterations
        print(f"\n--- First 15 Iterations: Duration per Device (ns) ---")
        print(f"{'Iter':<6}", end="")
        for dev in range(4):
            print(f" {'D'+str(dev):>10}", end="")
        print(f" {'Spread':>10} {'SlowDev':>8}")
        print("-" * 60)

        for iteration in all_iters[:15]:
            print(f"{iteration:<6}", end="")
            durs = {}
            for dev in range(4):
                key = (zone, dev, iteration)
                if key in zone_dev_iter:
                    dur_ns = data.cycles_to_ns(zone_dev_iter[key])
                    durs[dev] = dur_ns
                    print(f" {dur_ns:>10.0f}", end="")
                else:
                    print(f" {'--':>10}", end="")

            if durs:
                spread = max(durs.values()) - min(durs.values())
                slow_dev = max(durs, key=durs.get)
                print(f" {spread:>10.0f} D{slow_dev:>7}")
            else:
                print()

        # Pattern analysis: group devices by behavior
        print(f"\n--- Device Grouping (by even/odd preference) ---")
        even_preferred = []
        odd_preferred = []

        for dev in range(4):
            if dev_sequences[dev]:
                even_durs = [d[1] for d in dev_sequences[dev] if d[0] % 2 == 0]
                odd_durs = [d[1] for d in dev_sequences[dev] if d[0] % 2 == 1]

                if even_durs and odd_durs:
                    even_mean = statistics.mean(even_durs)
                    odd_mean = statistics.mean(odd_durs)

                    if even_mean < odd_mean * 0.8:  # Even is significantly faster
                        even_preferred.append(dev)
                    elif odd_mean < even_mean * 0.8:  # Odd is significantly faster
                        odd_preferred.append(dev)

        print(f"  Fast on EVEN iterations: {even_preferred}")
        print(f"  Fast on ODD iterations:  {odd_preferred}")

        if set(even_preferred) == {0, 2} and set(odd_preferred) == {1, 3}:
            print(f"\n  ✓ Pattern matches R1 pairs! (D0,D2) vs (D1,D3)")
        elif set(even_preferred) == {1, 3} and set(odd_preferred) == {0, 2}:
            print(f"\n  ✓ Pattern matches R1 pairs! (D1,D3) vs (D0,D2)")
        elif set(even_preferred) == {0, 3} and set(odd_preferred) == {1, 2}:
            print(f"\n  ✓ Pattern matches R2 pairs! (D0,D3) vs (D1,D2)")
        elif set(even_preferred) == {1, 2} and set(odd_preferred) == {0, 3}:
            print(f"\n  ✓ Pattern matches R2 pairs! (D1,D2) vs (D0,D3)")
        else:
            print(f"\n  ? No clear pairing pattern detected")


def analyze_device_imbalance(data: ProfileData):
    """Analyze device imbalance for each zone."""
    print_summary_header("DEVICE IMBALANCE ANALYSIS")

    # Group by zone -> device -> durations
    zone_device: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    for d in data.durations:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        zone_device[d.zone_name][d.device_id].append(dur_ns)

    print(f"\n{'Zone Name':<25} {'D0 Mean':>10} {'D1 Mean':>10} {'D2 Mean':>10} {'D3 Mean':>10} {'Imbalance':>12}")
    print("-" * 87)

    for zone in sorted(zone_device.keys()):
        devices = zone_device[zone]
        means = {}
        for dev_id in sorted(devices.keys()):
            means[dev_id] = statistics.mean(devices[dev_id])

        if means:
            min_mean = min(means.values())
            max_mean = max(means.values())
            imbalance = (max_mean / min_mean - 1) * 100 if min_mean > 0 else 0

            print(f"{zone:<25}", end="")
            for dev_id in range(4):
                if dev_id in means:
                    print(f" {means[dev_id]:>10.0f}", end="")
                else:
                    print(f" {'--':>10}", end="")
            flag = "⚠️" if imbalance > 20 else ""
            print(f" {imbalance:>10.1f}% {flag}")


def analyze_outliers(data: ProfileData, threshold: float = 2.0):
    """Identify iterations with outlier performance."""
    print_summary_header(f"OUTLIER DETECTION (>{threshold}x mean)")

    # Compute global mean per zone
    zone_means: Dict[str, float] = {}
    zone_all: Dict[str, List[float]] = defaultdict(list)

    for d in data.durations:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        zone_all[d.zone_name].append(dur_ns)

    for zone, vals in zone_all.items():
        zone_means[zone] = statistics.mean(vals)

    # Find outliers per iteration
    iter_outliers: Dict[int, List[Tuple[str, int, float, float]]] = defaultdict(list)

    for d in data.durations:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        mean = zone_means.get(d.zone_name, 0)
        if mean > 0 and dur_ns > mean * threshold:
            iter_outliers[d.iteration].append((d.zone_name, d.device_id, dur_ns, dur_ns / mean))

    # Print iterations with most outliers
    sorted_iters = sorted(iter_outliers.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"\nIterations with outliers (showing top 10):\n")
    for iteration, outliers in sorted_iters[:10]:
        print(f"Iteration {iteration}: {len(outliers)} outlier(s)")
        for zone, dev, dur, ratio in sorted(outliers, key=lambda x: x[3], reverse=True)[:3]:
            print(f"    {zone:<25} D{dev}: {dur:>8.0f} ns ({ratio:.2f}x mean)")
        print()


def analyze_device_spread_per_iteration(data: ProfileData):
    """
    For each zone, show per-device timing for each iteration.
    This helps identify if device imbalance is systematic (e.g., alternating pattern).
    """
    print_summary_header("PER-ITERATION DEVICE SPREAD ANALYSIS")

    # Group by (iteration, device, zone) -> list of durations (across cores)
    iter_dev_zone: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)

    for d in data.durations:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        key = (d.iteration, d.device_id, d.zone_name)
        iter_dev_zone[key].append(dur_ns)

    # Get all zones and iterations
    all_zones = sorted(set(d.zone_name for d in data.durations))
    all_iterations = sorted(set(d.iteration for d in data.durations))
    all_devices = sorted(set(d.device_id for d in data.durations))

    # For key zones, print detailed per-iteration per-device breakdown
    key_zones = [z for z in all_zones if "COMPUTE" in z or "WAIT" in z][:3]
    if not key_zones:
        key_zones = all_zones[:2]

    for zone in key_zones:
        print(f"\n--- Zone: {zone} ---")
        print(f"{'Iter':<6}", end="")
        for dev in all_devices:
            print(f"{'D'+str(dev):>10}", end="")
        print(f"{'Spread':>10} {'Max Dev':>8} {'Pattern':>10}")
        print("-" * (6 + 10 * len(all_devices) + 30))

        # Collect data for pattern analysis
        max_devices = []

        for iteration in all_iterations:
            print(f"{iteration:<6}", end="")

            dev_means = {}
            for dev in all_devices:
                key = (iteration, dev, zone)
                if key in iter_dev_zone:
                    mean = statistics.mean(iter_dev_zone[key])
                    dev_means[dev] = mean
                    print(f"{mean:>10.0f}", end="")
                else:
                    print(f"{'--':>10}", end="")

            if dev_means:
                spread = max(dev_means.values()) - min(dev_means.values())
                max_dev = max(dev_means, key=dev_means.get)
                max_devices.append(max_dev)
                print(f"{spread:>10.0f} {'D'+str(max_dev):>8}", end="")

                # Check for pattern
                if len(max_devices) >= 2:
                    if max_devices[-1] != max_devices[-2]:
                        print(f"{'alt':>10}", end="")
                    else:
                        print(f"{'same':>10}", end="")
                else:
                    print(f"{'--':>10}", end="")
            print()

        # Summarize pattern
        print()
        print(f"  Pattern summary for {zone}:")
        dev_counts = defaultdict(int)
        for d in max_devices:
            dev_counts[d] += 1
        for dev in sorted(dev_counts.keys()):
            print(f"    D{dev} was slowest in {dev_counts[dev]}/{len(max_devices)} iterations")

        # Check for alternating pattern
        alternations = sum(1 for i in range(1, len(max_devices)) if max_devices[i] != max_devices[i - 1])
        print(f"    Alternations: {alternations}/{len(max_devices)-1} ({100*alternations/(len(max_devices)-1):.0f}%)")

        # Check odd/even pattern
        if len(max_devices) >= 4:
            odd_iters = [max_devices[i] for i in range(len(max_devices)) if all_iterations[i] % 2 == 1]
            even_iters = [max_devices[i] for i in range(len(max_devices)) if all_iterations[i] % 2 == 0]

            if odd_iters:
                odd_mode = max(set(odd_iters), key=odd_iters.count)
                odd_consistency = odd_iters.count(odd_mode) / len(odd_iters) * 100
                print(f"    Odd iterations: D{odd_mode} slowest {odd_consistency:.0f}% of time")

            if even_iters:
                even_mode = max(set(even_iters), key=even_iters.count)
                even_consistency = even_iters.count(even_mode) / len(even_iters) * 100
                print(f"    Even iterations: D{even_mode} slowest {even_consistency:.0f}% of time")

        # Compute overall spread statistics
        spreads = []
        for iteration in all_iterations:
            dev_means = {}
            for dev in all_devices:
                key = (iteration, dev, zone)
                if key in iter_dev_zone:
                    dev_means[dev] = statistics.mean(iter_dev_zone[key])
            if dev_means:
                spreads.append(max(dev_means.values()) - min(dev_means.values()))

        if spreads:
            print(f"\n  Spread statistics:")
            print(f"    Mean spread: {statistics.mean(spreads):.0f} ns")
            print(f"    Min spread:  {min(spreads):.0f} ns")
            print(f"    Max spread:  {max(spreads):.0f} ns")


def analyze_iter_to_iter_variation(data: ProfileData):
    """Analyze iteration-to-iteration variation for key zones."""
    print_summary_header("ITERATION-TO-ITERATION VARIATION")

    # Group by zone -> iteration -> mean duration (across all devices/cores)
    zone_iter: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    for d in data.durations:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        zone_iter[d.zone_name][d.iteration].append(dur_ns)


def analyze_ring_communication(data: ProfileData):
    """
    Analyze communication patterns in the 4-device ring topology.

    PHYSICAL TO LOGICAL MAPPING (from physical_chip_mesh_coordinate_mapping):
      Chip 0 → Coord [0, 0] (device_index=0, is_even=true)
      Chip 2 → Coord [1, 0] (device_index=1, is_even=false)
      Chip 1 → Coord [2, 0] (device_index=2, is_even=true)
      Chip 3 → Coord [3, 0] (device_index=3, is_even=false)

    Ring topology (by logical coord): [0,0] → [1,0] → [2,0] → [3,0] → [0,0]
    Ring topology (by physical chip): Chip0 → Chip2 → Chip1 → Chip3 → Chip0

    Round 1 pairs (adjacent in ring by logical coord 0↔1 and 2↔3):
      - Coord 0↔1 = Chip0↔Chip2 (physical D0↔D2)
      - Coord 2↔3 = Chip1↔Chip3 (physical D1↔D3)

    Round 2 pairs (cross pairs by logical coord 0↔3 and 1↔2):
      - Coord 0↔3 = Chip0↔Chip3 (physical D0↔D3)
      - Coord 1↔2 = Chip2↔Chip1 (physical D2↔D1)

    FWD/BWD aggregator assignment (is_even = coord[0] % 2 == 0):
      - Even coords (0,2 = Chip0,Chip1): R1 uses FWD aggregator
      - Odd coords (1,3 = Chip2,Chip3):  R1 uses BWD aggregator
    """
    print_summary_header("RING COMMUNICATION PATTERN ANALYSIS")

    print("\n=== Physical ↔ Logical Mapping ===")
    print("  Chip 0 (D0) → Coord [0,0], is_even=true,  R1 uses FWD agg")
    print("  Chip 2 (D2) → Coord [1,0], is_even=false, R1 uses BWD agg")
    print("  Chip 1 (D1) → Coord [2,0], is_even=true,  R1 uses FWD agg")
    print("  Chip 3 (D3) → Coord [3,0], is_even=false, R1 uses BWD agg")
    print("\n  Ring order: Chip0(D0) → Chip2(D2) → Chip1(D1) → Chip3(D3) → Chip0")

    # Group by (iteration, device, zone) -> list of durations
    iter_dev_zone: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)

    for d in data.durations:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        key = (d.iteration, d.device_id, d.zone_name)
        iter_dev_zone[key].append(dur_ns)

    all_iterations = sorted(set(d.iteration for d in data.durations))

    print("\n=== Device Pair Analysis ===")
    print("  R1 pairs (by physical): D0↔D2 and D1↔D3")
    print("  R2 pairs (by physical): D0↔D3 and D2↔D1")

    # For WAIT zones, analyze the pattern of which device waits longer
    for zone in ["R1-WAIT-NEIGHBOR", "R2-WAIT-NEIGHBOR"]:
        zone_data = [
            (it, dev, dur)
            for (it, dev, z), durs in iter_dev_zone.items()
            if z == zone
            for dur in [statistics.mean(durs)]
            for dev in [dev]
        ]
        if not zone_data:
            continue

        print(f"\n--- {zone} ---")

        if "R1" in zone:
            # R1: D0↔D2 pair and D1↔D3 pair (CORRECTED!)
            print("  R1 pairs: Chip0↔Chip2 (D0↔D2) and Chip1↔Chip3 (D1↔D3)")
            pairs = [((0, 2), "D0↔D2"), ((1, 3), "D1↔D3")]
        else:
            # R2: D0↔D3 pair and D2↔D1 pair (CORRECTED!)
            print("  R2 pairs: Chip0↔Chip3 (D0↔D3) and Chip2↔Chip1 (D2↔D1)")
            pairs = [((0, 3), "D0↔D3"), ((2, 1), "D2↔D1")]

        # Analyze per-iteration which device in each pair waits longer
        for (dev_a, dev_b), pair_name in pairs:
            pair_winner = []

            for iteration in all_iterations:
                da = iter_dev_zone.get((iteration, dev_a, zone), [0])
                db = iter_dev_zone.get((iteration, dev_b, zone), [0])
                da_mean = statistics.mean(da) if da else 0
                db_mean = statistics.mean(db) if db else 0
                pair_winner.append(dev_a if da_mean > db_mean else dev_b)

            # Count winners
            from collections import Counter

            counts = Counter(pair_winner)

            print(
                f"\n  Pair {pair_name}: D{dev_a} waits longer {counts.get(dev_a,0)}/{len(pair_winner)} times, "
                f"D{dev_b} waits longer {counts.get(dev_b,0)}/{len(pair_winner)} times"
            )

            # Check if winners alternate
            alternations = sum(1 for i in range(1, len(pair_winner)) if pair_winner[i] != pair_winner[i - 1])
            print(
                f"    Alternation: {alternations}/{len(pair_winner)-1} "
                f"({100*alternations/(len(pair_winner)-1):.0f}%)"
            )

    # Analyze overall timeline correlation
    print("\n=== Cross-Round Correlation ===\n")
    print("Checking if R1 performance affects R2 performance...")

    r1_zone = "R1-COMPUTE"
    r2_zone = "R2-COMPUTE"

    for dev in range(4):
        r1_times = []
        r2_times = []
        for iteration in all_iterations:
            r1_data = iter_dev_zone.get((iteration, dev, r1_zone), [])
            r2_data = iter_dev_zone.get((iteration, dev, r2_zone), [])
            if r1_data and r2_data:
                r1_times.append(statistics.mean(r1_data))
                r2_times.append(statistics.mean(r2_data))

        if len(r1_times) > 2:
            # Compute correlation
            n = len(r1_times)
            mean_r1 = sum(r1_times) / n
            mean_r2 = sum(r2_times) / n

            cov = sum((r1_times[i] - mean_r1) * (r2_times[i] - mean_r2) for i in range(n)) / n
            std_r1 = (sum((x - mean_r1) ** 2 for x in r1_times) / n) ** 0.5
            std_r2 = (sum((x - mean_r2) ** 2 for x in r2_times) / n) ** 0.5

            if std_r1 > 0 and std_r2 > 0:
                corr = cov / (std_r1 * std_r2)
                # Map physical to logical for clarity
                logical_coord = {0: 0, 2: 1, 1: 2, 3: 3}[dev]
                print(
                    f"  D{dev} (Coord {logical_coord}): R1↔R2 correlation = {corr:.3f} "
                    f"({'positive' if corr > 0.3 else 'negative' if corr < -0.3 else 'weak'})"
                )


def analyze_compute_vs_wait_bottleneck(data: ProfileData):
    """
    Analyze whether compute or fabric wait is the bottleneck.

    IMPORTANT INSIGHT:
    The R1-COMPUTE zone INCLUDES the time waiting for neighbor CB data!
    Inside sdpa_reduce(), cb_wait_front(cb_l2, ...) blocks until Reader pushes neighbor CB.
    So R1-COMPUTE time = CB wait time + actual compute time.

    Reader's R1-WAIT-NEIGHBOR and Compute's cb_wait_front are waiting for the SAME event:
    the fabric packet to arrive and be pushed to CB.

    To separate compute from wait:
      actual_compute_time ≈ R1-COMPUTE - R1-WAIT-NEIGHBOR

    This is an approximation because:
      - Reader and Compute run on different RISCs (NCRISC vs TRISC)
      - Some overlap is possible
      - But CB wait inside compute should roughly match Reader's wait
    """
    print_summary_header("COMPUTE vs WAIT BOTTLENECK ANALYSIS")

    # Group by (iteration, device, zone) -> mean duration
    iter_dev_zone: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)

    for d in data.durations:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        key = (d.iteration, d.device_id, d.zone_name)
        iter_dev_zone[key].append(dur_ns)

    # Average across cores for each (iter, device, zone)
    iter_dev_zone_mean = {}
    for key, vals in iter_dev_zone.items():
        iter_dev_zone_mean[key] = statistics.mean(vals)

    all_iterations = sorted(set(d.iteration for d in data.durations))

    # === Part 1: Overall time breakdown ===
    print("\n=== Time Breakdown (R1 phase) ===")

    r1_compute_all = []
    r1_wait_all = []
    r1_send_all = []

    for (it, dev, zone), dur in iter_dev_zone_mean.items():
        if zone == "R1-COMPUTE":
            r1_compute_all.append(dur)
        elif zone == "R1-WAIT-NEIGHBOR":
            r1_wait_all.append(dur)
        elif zone == "R1-SEND":
            r1_send_all.append(dur)

    total_compute = statistics.mean(r1_compute_all) if r1_compute_all else 0
    total_wait = statistics.mean(r1_wait_all) if r1_wait_all else 0
    total_send = statistics.mean(r1_send_all) if r1_send_all else 0

    print(f"\nMean time per device per iteration (R1):")
    print(f"  R1-COMPUTE (includes CB wait): {total_compute:>8.0f} ns")
    print(f"  R1-WAIT-NEIGHBOR (Reader's wait):  {total_wait:>8.0f} ns")
    print(f"  R1-SEND (aggregator forward):      {total_send:>8.0f} ns")

    # Estimate actual compute time
    actual_compute = total_compute - total_wait
    print(f"\n  Estimated actual compute: {actual_compute:>8.0f} ns")
    print(f"    (= R1-COMPUTE - WAIT-NEIGHBOR)")
    if total_compute > 0:
        print(f"\n  Breakdown of R1-COMPUTE time:")
        print(f"    CB wait for neighbor data: {total_wait:>6.0f} ns ({100*total_wait/total_compute:.1f}%)")
        print(f"    Actual compute work:       {actual_compute:>6.0f} ns ({100*actual_compute/total_compute:.1f}%)")
    else:
        print(f"\n  No R1-COMPUTE data available")

    # === Part 2: Per-device breakdown ===
    print("\n=== Per-Device Time Breakdown ===")
    print(f"{'Device':<8} {'R1-COMPUTE':>12} {'WAIT-NEIGHBOR':>14} {'Est.Compute':>12} {'Wait%':>8}")
    print("-" * 60)

    for dev in range(4):
        compute_times = [
            iter_dev_zone_mean.get((it, dev, "R1-COMPUTE"), 0)
            for it in all_iterations
            if (it, dev, "R1-COMPUTE") in iter_dev_zone_mean
        ]
        wait_times = [
            iter_dev_zone_mean.get((it, dev, "R1-WAIT-NEIGHBOR"), 0)
            for it in all_iterations
            if (it, dev, "R1-WAIT-NEIGHBOR") in iter_dev_zone_mean
        ]

        if compute_times and wait_times:
            comp_mean = statistics.mean(compute_times)
            wait_mean = statistics.mean(wait_times)
            actual = comp_mean - wait_mean
            wait_pct = 100 * wait_mean / comp_mean if comp_mean > 0 else 0
            print(f"D{dev:<7} {comp_mean:>12.0f} {wait_mean:>14.0f} {actual:>12.0f} {wait_pct:>7.1f}%")

    # === Part 3: Per-iteration estimated compute ===
    print("\n=== Estimated Actual Compute per Iteration (first 10) ===")
    print(f"{'Iter':<6} {'D0':>8} {'D1':>8} {'D2':>8} {'D3':>8} {'Spread':>8}")
    print("-" * 50)

    est_compute_spreads = []
    count = 0
    for it in all_iterations:
        if count >= 10:
            break

        dev_est = {}
        for dev in range(4):
            comp = iter_dev_zone_mean.get((it, dev, "R1-COMPUTE"))
            wait = iter_dev_zone_mean.get((it, dev, "R1-WAIT-NEIGHBOR"))
            if comp is not None and wait is not None:
                dev_est[dev] = comp - wait

        if len(dev_est) == 4:
            spread = max(dev_est.values()) - min(dev_est.values())
            est_compute_spreads.append(spread)
            print(f"{it:<6} {dev_est[0]:>8.0f} {dev_est[1]:>8.0f} {dev_est[2]:>8.0f} {dev_est[3]:>8.0f} {spread:>8.0f}")
            count += 1

    # === Part 4: Analyze the estimated compute spread ===
    print("\n=== Estimated Compute Variability ===")

    # Collect all estimated compute times per device
    all_est_compute = defaultdict(list)
    for it in all_iterations:
        for dev in range(4):
            comp = iter_dev_zone_mean.get((it, dev, "R1-COMPUTE"))
            wait = iter_dev_zone_mean.get((it, dev, "R1-WAIT-NEIGHBOR"))
            if comp is not None and wait is not None:
                all_est_compute[dev].append(comp - wait)

    print(f"{'Device':<8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'CoV':>10}")
    print("-" * 60)
    for dev in range(4):
        if all_est_compute[dev]:
            vals = all_est_compute[dev]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0
            cov = 100 * std / mean if mean > 0 else 0
            print(f"D{dev:<7} {mean:>10.0f} {std:>10.0f} {min(vals):>10.0f} {max(vals):>10.0f} {cov:>9.1f}%")

    # === Summary ===
    print("\n=== BOTTLENECK DETERMINATION ===")
    print()

    if est_compute_spreads:
        mean_est_spread = statistics.mean(est_compute_spreads)
    else:
        mean_est_spread = 0

    # Compare wait variability vs compute variability
    wait_stds = [statistics.stdev(all_est_compute[d]) for d in range(4) if len(all_est_compute[d]) > 1]
    avg_wait_std = statistics.mean(wait_stds) if wait_stds else 0

    if total_compute > 0:
        print(f"  Wait portion of R1-COMPUTE: {100*total_wait/total_compute:.0f}%")
        print(f"  Compute portion:            {100*actual_compute/total_compute:.0f}%")
    else:
        print(f"  No R1-COMPUTE data to compute wait/compute ratio")
    print()

    if total_wait > actual_compute:
        print("  CONCLUSION: FABRIC WAIT is the BOTTLENECK")
        print("    - More than 50% of R1-COMPUTE time is waiting for neighbor data")
    else:
        print("  CONCLUSION: ACTUAL COMPUTE is the BOTTLENECK")
        print("    - More than 50% of R1-COMPUTE time is actual computation")
    print()

    if mean_est_spread < 1000:
        print(f"  Estimated compute spread ({mean_est_spread:.0f} ns) is SMALL")
        print("    → Compute work is well-balanced across devices")
        print("    → Wait time variability likely comes from fabric timing")
    else:
        print(f"  Estimated compute spread ({mean_est_spread:.0f} ns) is MODERATE/LARGE")
        print("    → Some compute imbalance exists")


def analyze_r1_vs_r2_wait(data: ProfileData):
    """
    Compare R1-WAIT-NEIGHBOR vs R2-WAIT-NEIGHBOR statistics.

    If R2 wait times are stable, they represent the baseline fabric latency.
    This helps identify how much R1 can potentially be optimized.

    Key insight:
      - R2 "winner" devices (D0, D1) consistently arrive first → their wait = fabric latency
      - R2 "loser" devices (D2, D3) consistently arrive later → their wait = fabric + delay
      - R1 alternates, so both slow and fast times occur on all devices
    """
    print_summary_header("R1 vs R2 WAIT TIME COMPARISON")

    # Group by (device, zone) -> list of durations
    dev_zone: Dict[Tuple[int, str], List[float]] = defaultdict(list)

    for d in data.durations:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        key = (d.device_id, d.zone_name)
        dev_zone[key].append(dur_ns)

    # Get R1 and R2 wait times
    r1_all = []
    r2_all = []
    for (dev, zone), vals in dev_zone.items():
        if zone == "R1-WAIT-NEIGHBOR":
            r1_all.extend(vals)
        elif zone == "R2-WAIT-NEIGHBOR":
            r2_all.extend(vals)

    if not r1_all or not r2_all:
        print("  No R1-WAIT-NEIGHBOR or R2-WAIT-NEIGHBOR data found.")
        return

    # Overall comparison
    print("\n=== Overall Comparison ===")
    print(f"{'':20} {'R1-WAIT':>12} {'R2-WAIT':>12} {'Ratio(R1/R2)':>14}")
    print("-" * 60)

    r1_stats = compute_stats(r1_all)
    r2_stats = compute_stats(r2_all)

    def safe_ratio(a, b):
        return a / b if b > 0 else 0

    print(f"{'Count:':<20} {r1_stats['count']:>12} {r2_stats['count']:>12}")
    print(
        f"{'Min:':<20} {r1_stats['min']:>10.0f}ns {r2_stats['min']:>10.0f}ns {safe_ratio(r1_stats['min'], r2_stats['min']):>12.2f}x"
    )
    print(
        f"{'Max:':<20} {r1_stats['max']:>10.0f}ns {r2_stats['max']:>10.0f}ns {safe_ratio(r1_stats['max'], r2_stats['max']):>12.2f}x"
    )
    print(
        f"{'Mean:':<20} {r1_stats['mean']:>10.0f}ns {r2_stats['mean']:>10.0f}ns {safe_ratio(r1_stats['mean'], r2_stats['mean']):>12.2f}x"
    )
    print(
        f"{'Median:':<20} {r1_stats['median']:>10.0f}ns {r2_stats['median']:>10.0f}ns {safe_ratio(r1_stats['median'], r2_stats['median']):>12.2f}x"
    )
    print(
        f"{'Std Dev:':<20} {r1_stats['std']:>10.0f}ns {r2_stats['std']:>10.0f}ns {safe_ratio(r1_stats['std'], r2_stats['std']):>12.2f}x"
    )

    cov_r1 = r1_stats["std"] / r1_stats["mean"] * 100 if r1_stats["mean"] > 0 else 0
    cov_r2 = r2_stats["std"] / r2_stats["mean"] * 100 if r2_stats["mean"] > 0 else 0
    print(f"{'CoV (std/mean):':<20} {cov_r1:>10.1f}% {cov_r2:>10.1f}%")

    # Per-device breakdown
    print("\n=== Per-Device Breakdown ===")
    print(
        f"{'Device':<8} {'R1 Min':>10} {'R1 Max':>10} {'R1 Mean':>10} {'R1 Std':>10} | {'R2 Min':>10} {'R2 Max':>10} {'R2 Mean':>10} {'R2 Std':>10}"
    )
    print("-" * 100)

    for dev in range(4):
        r1_dev = dev_zone.get((dev, "R1-WAIT-NEIGHBOR"), [])
        r2_dev = dev_zone.get((dev, "R2-WAIT-NEIGHBOR"), [])

        r1_s = compute_stats(r1_dev) if r1_dev else {"min": 0, "max": 0, "mean": 0, "std": 0}
        r2_s = compute_stats(r2_dev) if r2_dev else {"min": 0, "max": 0, "mean": 0, "std": 0}

        print(
            f"D{dev:<7} {r1_s['min']:>10.0f} {r1_s['max']:>10.0f} {r1_s['mean']:>10.0f} {r1_s['std']:>10.0f} | "
            f"{r2_s['min']:>10.0f} {r2_s['max']:>10.0f} {r2_s['mean']:>10.0f} {r2_s['std']:>10.0f}"
        )


def analyze_phase_drift_pattern(data: ProfileData):
    """
    Analyze phase drift between devices in the reduction ring.

    Key insight: In a reduction, if device A finishes compute slightly early,
    it sends first and waits. Device B receives late and has to catch up.
    This creates an alternating winner/loser pattern that can persist.

    NOTE: Cycle counters are per-device and not synchronized across devices.
    We can only compare RELATIVE timing within a device, not absolute timing
    across devices.
    """
    print("\n" + "=" * 100)
    print("PHASE DRIFT ANALYSIS (REDUCTION RING TIMING)")
    print("=" * 100)

    print("\n  NOTE: Device cycle counters are not synchronized across devices.")
    print("  We analyze RELATIVE patterns within each device instead.\n")

    # ==========================================================================
    # 1. Per-device: Time between send completion and receiving neighbor data
    # ==========================================================================
    # This tells us how long after sending this device waited for its pair

    send_to_receive: Dict[Tuple[int, int], float] = {}  # (device, iteration) -> ns

    for d in data.durations:
        key = (d.device_id, d.iteration)
        if d.zone_name == "R1-WAIT-NEIGHBOR":
            ns = data.cycles_to_ns(d.duration_cycles)
            send_to_receive[key] = max(send_to_receive.get(key, 0), ns)

    all_iters = sorted(set(k[1] for k in send_to_receive.keys()))
    all_devices = sorted(set(k[0] for k in send_to_receive.keys()))

    if len(all_devices) < 2:
        print("  Need at least 2 devices for phase analysis")
        return

    # ==========================================================================
    # 2. R1 Pair Analysis: Phys0↔Phys2 and Phys1↔Phys3
    # ==========================================================================
    # Based on mesh coordinate mapping and even/odd logic:
    # - MeshCoord [0,0] = Phys0, [1,0] = Phys2, [2,0] = Phys1, [3,0] = Phys3
    # - Even mesh coords use FWD, odd use BWD for R1
    # - R1 pairs: (Phys0↔Phys2) and (Phys1↔Phys3) - direct 1-hop neighbors!

    print("=== R1 PAIR TIMING ANALYSIS ===")
    print("(R1 pairs: Phys0↔Phys2, Phys1↔Phys3 - comparing wait times within pairs)\n")

    r1_pairs = [(0, 2), (1, 3)]  # Correct physical device pairing!

    for a, b in r1_pairs:
        print(f"--- Pair D{a}↔D{b} ---")

        a_waits = []
        b_waits = []
        a_minus_b = []

        for it in all_iters:
            a_val = send_to_receive.get((a, it), None)
            b_val = send_to_receive.get((b, it), None)
            if a_val is not None and b_val is not None:
                a_waits.append(a_val)
                b_waits.append(b_val)
                a_minus_b.append(a_val - b_val)

        if a_waits:
            a_stats = compute_stats(a_waits)
            b_stats = compute_stats(b_waits)
            diff_stats = compute_stats(a_minus_b)

            print(f"  D{a} wait: mean={a_stats['mean']:.0f}ns, std={a_stats['std']:.0f}ns")
            print(f"  D{b} wait: mean={b_stats['mean']:.0f}ns, std={b_stats['std']:.0f}ns")
            print(f"  D{a}-D{b} diff: mean={diff_stats['mean']:.0f}ns (positive = D{a} waits more)")

            # Check if they alternate: when A waits more, next iter B waits more?
            sign_flips = 0
            for i in range(len(a_minus_b) - 1):
                if (a_minus_b[i] > 0) != (a_minus_b[i + 1] > 0):
                    sign_flips += 1

            flip_rate = sign_flips / (len(a_minus_b) - 1) * 100 if len(a_minus_b) > 1 else 0
            print(f"  Sign flips (D{a}>D{b} → D{b}>D{a}): {flip_rate:.0f}% of iterations")

            if flip_rate > 60:
                print(f"  → ALTERNATING pattern: pairs trade off who waits more")
            elif flip_rate < 30:
                print(f"  → PERSISTENT pattern: same device consistently waits more")
            else:
                print(f"  → MIXED pattern")
            print()

    # ==========================================================================
    # 2b. R2 Pair Analysis: Phys0↔Phys3 and Phys2↔Phys1
    # ==========================================================================
    # R2 pairs based on mesh coord BWD direction for even, FWD for odd:
    # - [0,0](Phys0) BWD → [3,0](Phys3), [3,0](Phys3) FWD → [0,0](Phys0)
    # - [1,0](Phys2) BWD → [0,0](Phys0)? No - need to check the actual mapping
    # Actually: R2 uses opposite direction from R1
    # Even devices use BWD for R2: Phys0→Phys3, Phys2→Phys1(? based on bwd_coord)
    # Let me derive from the log: Phys0 BWD=[3,0]=Phys3, Phys2 BWD=[0,0]=Phys0
    # So R2 pairs: (Phys0↔Phys3), (Phys2↔Phys1)

    # Get R2 wait times
    r2_send_to_receive: Dict[Tuple[int, int], float] = {}
    for d in data.durations:
        key = (d.device_id, d.iteration)
        if d.zone_name == "R2-WAIT-NEIGHBOR":
            ns = data.cycles_to_ns(d.duration_cycles)
            r2_send_to_receive[key] = max(r2_send_to_receive.get(key, 0), ns)

    if r2_send_to_receive:
        print("=== R2 PAIR TIMING ANALYSIS ===")
        print("(R2 pairs: Phys0↔Phys3, Phys1↔Phys2 - comparing wait times within pairs)\n")

        r2_pairs = [(0, 3), (1, 2)]  # R2 physical device pairing

        for a, b in r2_pairs:
            print(f"--- R2 Pair D{a}↔D{b} ---")

            a_waits = []
            b_waits = []
            a_minus_b = []

            for it in all_iters:
                a_val = r2_send_to_receive.get((a, it), None)
                b_val = r2_send_to_receive.get((b, it), None)
                if a_val is not None and b_val is not None:
                    a_waits.append(a_val)
                    b_waits.append(b_val)
                    a_minus_b.append(a_val - b_val)

            if a_waits:
                a_stats = compute_stats(a_waits)
                b_stats = compute_stats(b_waits)
                diff_stats = compute_stats(a_minus_b)

                print(f"  D{a} wait: mean={a_stats['mean']:.0f}ns, std={a_stats['std']:.0f}ns")
                print(f"  D{b} wait: mean={b_stats['mean']:.0f}ns, std={b_stats['std']:.0f}ns")
                print(f"  D{a}-D{b} diff: mean={diff_stats['mean']:.0f}ns (positive = D{a} waits more)")

                sign_flips = 0
                for i in range(len(a_minus_b) - 1):
                    if (a_minus_b[i] > 0) != (a_minus_b[i + 1] > 0):
                        sign_flips += 1

                flip_rate = sign_flips / (len(a_minus_b) - 1) * 100 if len(a_minus_b) > 1 else 0
                print(f"  Sign flips (D{a}>D{b} → D{b}>D{a}): {flip_rate:.0f}% of iterations")

                if flip_rate > 60:
                    print(f"  → ALTERNATING pattern: pairs trade off who waits more")
                elif flip_rate < 30:
                    print(f"  → PERSISTENT pattern: same device consistently waits more")
                else:
                    print(f"  → MIXED pattern")
                print()

    # ==========================================================================
    # 3. Cross-pair analysis: Does Phys0/Phys2 pair spike when Phys1/Phys3 pair doesn't?
    # ==========================================================================
    print("=== CROSS-PAIR CORRELATION ===")
    print("(Do (Phys0,Phys2) and (Phys1,Phys3) spike independently?)\n")

    pair_02_waits = []  # Correct R1 pair: Phys0↔Phys2
    pair_13_waits = []  # Correct R1 pair: Phys1↔Phys3

    for it in all_iters:
        d0 = send_to_receive.get((0, it), 0)
        d1 = send_to_receive.get((1, it), 0)
        d2 = send_to_receive.get((2, it), 0)
        d3 = send_to_receive.get((3, it), 0)

        if d0 and d1 and d2 and d3:
            pair_02_waits.append(max(d0, d2))  # Worst wait in pair (Phys0↔Phys2)
            pair_13_waits.append(max(d1, d3))  # Worst wait in pair (Phys1↔Phys3)

    if len(pair_02_waits) >= 10:
        # Compute correlation
        m1 = statistics.mean(pair_02_waits)
        m2 = statistics.mean(pair_13_waits)

        num = sum((a - m1) * (b - m2) for a, b in zip(pair_02_waits, pair_13_waits))
        d1 = sum((a - m1) ** 2 for a in pair_02_waits) ** 0.5
        d2 = sum((b - m2) ** 2 for b in pair_13_waits) ** 0.5

        if d1 > 0 and d2 > 0:
            corr = num / (d1 * d2)
            print(f"  Pair (Phys0,Phys2) vs (Phys1,Phys3) correlation: {corr:.3f}")

            if corr > 0.5:
                print(f"  → SYNCHRONIZED: Both pairs spike together")
                print(f"    Suggests GLOBAL bottleneck (shared router/link)")
            elif corr < -0.3:
                print(f"  → ANTI-CORRELATED: When (0,2) spikes, (1,3) doesn't")
                print(f"    Suggests CONTENTION between pairs for shared resource")
            else:
                print(f"  → INDEPENDENT: Pairs spike independently")
                print(f"    Suggests LOCAL bottleneck within each pair")

    # ==========================================================================
    # 4. Per-device wait time trend over iterations
    # ==========================================================================
    print("\n=== WAIT TIME TREND (per device) ===")
    print("(Is wait time increasing, decreasing, or stable over iterations?)\n")

    for dev in all_devices:
        waits = [(it, send_to_receive.get((dev, it), 0)) for it in all_iters if (dev, it) in send_to_receive]

        if len(waits) >= 10:
            # Simple linear regression: is there a trend?
            x_vals = [i for i, _ in enumerate(waits)]
            y_vals = [w for _, w in waits]

            x_mean = statistics.mean(x_vals)
            y_mean = statistics.mean(y_vals)

            num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
            denom = sum((x - x_mean) ** 2 for x in x_vals)

            if denom > 0:
                slope = num / denom  # ns per iteration

                trend = "INCREASING" if slope > 10 else "DECREASING" if slope < -10 else "STABLE"
                print(f"  D{dev}: slope = {slope:+.1f} ns/iter → {trend}")

    # ==========================================================================
    # 5. Check for systematic "winner" device patterns
    # ==========================================================================
    print("\n=== SYSTEMATIC WINNER PATTERNS ===")
    print("(Which device consistently waits LEAST in each phase?)\n")

    # For R1
    r1_winner_counts = defaultdict(int)
    for it in all_iters:
        waits = [
            (dev, send_to_receive.get((dev, it), float("inf"))) for dev in all_devices if (dev, it) in send_to_receive
        ]
        if waits:
            winner = min(waits, key=lambda x: x[1])[0]
            r1_winner_counts[winner] += 1

    print(f"  R1 winner (shortest wait) frequency:")
    for dev in all_devices:
        pct = r1_winner_counts[dev] / len(all_iters) * 100
        print(f"    D{dev}: {pct:.0f}%")

    # Check if there's a dominant winner
    max_winner = max(r1_winner_counts.values())
    if max_winner / len(all_iters) > 0.5:
        dominant = max(r1_winner_counts.items(), key=lambda x: x[1])[0]
        print(f"\n  ⚠️ D{dominant} is the dominant R1 winner ({max_winner}/{len(all_iters)} iters)")
        print(f"    This suggests D{dominant}'s compute is consistently slower OR")
        print(f"    D{dominant}'s partner consistently sends early")


def analyze_alternating_spike_pattern(data: ProfileData):
    """
    Analyze the alternating spike pattern in fabric latency.

    Key questions:
    1. Is the spike truly alternating (every other iteration)?
    2. Which devices/rounds show spikes?
    3. Is there R1↔R2 correlation (when R1 spikes, does R2 also spike)?
    4. Does the pattern correlate with ring position or direction?
    """
    print("\n" + "=" * 100)
    print("ALTERNATING SPIKE PATTERN ANALYSIS")
    print("=" * 100)

    # Collect R1 and R2 wait times per (device, iteration)
    r1_waits: Dict[Tuple[int, int], float] = {}  # (device, iteration) -> ns
    r2_waits: Dict[Tuple[int, int], float] = {}

    for d in data.durations:
        key = (d.device_id, d.iteration)
        ns = data.cycles_to_ns(d.duration_cycles)
        if d.zone_name == "R1-WAIT-NEIGHBOR":
            # Take max if multiple cores (though should be 1 per device for this zone)
            r1_waits[key] = max(r1_waits.get(key, 0), ns)
        elif d.zone_name == "R2-WAIT-NEIGHBOR":
            r2_waits[key] = max(r2_waits.get(key, 0), ns)

    if not r1_waits:
        print("  No R1 wait data found.")
        return

    # Get all iterations and devices
    all_iters = sorted(set(k[1] for k in r1_waits.keys()))
    all_devices = sorted(set(k[0] for k in r1_waits.keys()))

    if len(all_iters) < 10:
        print(f"  Only {len(all_iters)} iterations - need more data for pattern analysis")
        return

    print(f"\n  Analyzing {len(all_iters)} iterations across {len(all_devices)} devices")

    # ==========================================================================
    # 1. Check for alternation pattern per device
    # ==========================================================================
    print("\n=== PER-DEVICE ALTERNATION PATTERN ===")
    print("(Checking if consecutive iterations alternate between high/low latency)\n")

    # For each device, check if odd iterations have higher/lower latency than even
    for dev in all_devices:
        dev_r1 = [(it, r1_waits.get((dev, it), 0)) for it in all_iters]
        even_vals = [v for (it, v) in dev_r1 if it % 2 == 0 and v > 0]
        odd_vals = [v for (it, v) in dev_r1 if it % 2 == 1 and v > 0]

        if even_vals and odd_vals:
            even_mean = statistics.mean(even_vals)
            odd_mean = statistics.mean(odd_vals)
            diff = abs(even_mean - odd_mean)
            diff_pct = diff / max(even_mean, odd_mean) * 100

            higher = "EVEN" if even_mean > odd_mean else "ODD"
            pattern = "STRONG" if diff_pct > 20 else "WEAK" if diff_pct > 5 else "NONE"

            print(
                f"  D{dev}: Even mean={even_mean:5.0f}ns, Odd mean={odd_mean:5.0f}ns, "
                f"Δ={diff:5.0f}ns ({diff_pct:4.1f}%) → {pattern} alternation, {higher} higher"
            )

    # ==========================================================================
    # 2. Cross-device correlation: Do all devices spike on the same iteration?
    # ==========================================================================
    print("\n=== CROSS-DEVICE SPIKE CORRELATION ===")
    print("(Do all devices experience high latency on the same iteration?)\n")

    # For each iteration, compute mean and spread across devices
    iter_means = []
    iter_spreads = []
    high_spread_iters = []

    for it in all_iters:
        vals = [r1_waits.get((dev, it), 0) for dev in all_devices if (dev, it) in r1_waits]
        if len(vals) >= 2:
            mean_val = statistics.mean(vals)
            spread = max(vals) - min(vals)
            spread_pct = spread / mean_val * 100 if mean_val > 0 else 0
            iter_means.append(mean_val)
            iter_spreads.append(spread_pct)

            if spread_pct > 50:  # High spread = devices diverge
                high_spread_iters.append((it, mean_val, spread, spread_pct))

    if iter_spreads:
        mean_spread = statistics.mean(iter_spreads)
        print(f"  Mean cross-device spread: {mean_spread:.1f}%")

        if mean_spread < 20:
            print(f"  ✓ Devices are well-synchronized (all spike/dip together)")
            print(f"    → Suggests a GLOBAL bottleneck (router, link capacity)")
        else:
            print(f"  ⚠ Devices are NOT synchronized (some spike while others don't)")
            print(f"    → Suggests LOCAL bottleneck (aggregator, compute)")

        if high_spread_iters:
            print(f"\n  High-spread iterations (>50% device divergence):")
            for it, m, s, sp in high_spread_iters[:10]:
                print(f"    Iter {it}: mean={m:.0f}ns, spread={s:.0f}ns ({sp:.0f}%)")

    # ==========================================================================
    # 3. R1↔R2 correlation: When R1 spikes, does R2 also spike?
    # ==========================================================================
    print("\n=== R1 ↔ R2 CORRELATION ===")
    print("(When R1 is slow, is R2 also slow on the same iteration?)\n")

    # For each (device, iteration), check R1 vs R2
    r1_r2_pairs = []
    for key in r1_waits:
        if key in r2_waits:
            r1_r2_pairs.append((r1_waits[key], r2_waits[key]))

    if len(r1_r2_pairs) >= 10:
        r1_vals = [p[0] for p in r1_r2_pairs]
        r2_vals = [p[1] for p in r1_r2_pairs]

        # Compute Pearson correlation (simplified)
        r1_mean = statistics.mean(r1_vals)
        r2_mean = statistics.mean(r2_vals)

        numerator = sum((r1 - r1_mean) * (r2 - r2_mean) for r1, r2 in r1_r2_pairs)
        r1_var = sum((r1 - r1_mean) ** 2 for r1 in r1_vals)
        r2_var = sum((r2 - r2_mean) ** 2 for r2 in r2_vals)

        if r1_var > 0 and r2_var > 0:
            correlation = numerator / ((r1_var**0.5) * (r2_var**0.5))
            print(f"  R1-R2 Pearson correlation: {correlation:.3f}")

            if correlation > 0.7:
                print(f"  ✓ STRONG positive correlation - R1 and R2 spike together")
                print(f"    → Both use same router/link (unlikely) OR shared upstream bottleneck")
            elif correlation < -0.3:
                print(f"  ✓ NEGATIVE correlation - R1 and R2 alternate")
                print(f"    → Resource contention between R1 and R2 phases")
            else:
                print(f"  ~ WEAK correlation - R1 and R2 are independent")
                print(f"    → Different links/routers with independent behavior")

    # ==========================================================================
    # 4. Autocorrelation: Does latency at iter N predict latency at iter N+1?
    # ==========================================================================
    print("\n=== TEMPORAL AUTOCORRELATION (iter N vs N+1) ===")
    print("(Does a slow iteration predict the next will be fast?)\n")

    for dev in all_devices:
        # Get time series for this device
        ts = [(it, r1_waits.get((dev, it), 0)) for it in all_iters if (dev, it) in r1_waits]
        if len(ts) < 20:
            continue

        # Compute lag-1 autocorrelation
        vals = [v for (_, v) in ts]
        mean_val = statistics.mean(vals)

        numerator = sum((vals[i] - mean_val) * (vals[i + 1] - mean_val) for i in range(len(vals) - 1))
        denominator = sum((v - mean_val) ** 2 for v in vals)

        if denominator > 0:
            autocorr = numerator / denominator

            interpretation = ""
            if autocorr < -0.3:
                interpretation = "ALTERNATING (high→low→high)"
            elif autocorr > 0.3:
                interpretation = "PERSISTENT (high→high or low→low)"
            else:
                interpretation = "NO PATTERN"

            print(f"  D{dev}: lag-1 autocorr = {autocorr:+.3f} → {interpretation}")

    # ==========================================================================
    # 5. Identify the specific pattern for top iterations
    # ==========================================================================
    print("\n=== TOP 10 HIGHEST LATENCY ITERATIONS ===")

    # Get top iterations by max R1 wait across devices
    iter_max_wait = {}
    for it in all_iters:
        vals = [r1_waits.get((dev, it), 0) for dev in all_devices]
        if vals:
            iter_max_wait[it] = max(vals)

    top_iters = sorted(iter_max_wait.items(), key=lambda x: -x[1])[:10]

    print(f"\n  {'Iter':<6} {'Max R1':<10} {'Which Device':<15} {'Prev Iter':<12} {'Next Iter':<12} {'Pattern'}")
    print(f"  {'-'*6} {'-'*10} {'-'*15} {'-'*12} {'-'*12} {'-'*20}")

    for it, max_wait in top_iters:
        # Which device had the max?
        max_dev = None
        for dev in all_devices:
            if r1_waits.get((dev, it), 0) == max_wait:
                max_dev = dev
                break

        # Check prev/next
        prev_wait = iter_max_wait.get(it - 1, 0)
        next_wait = iter_max_wait.get(it + 1, 0)

        pattern = ""
        if prev_wait > 0 and next_wait > 0:
            if max_wait > prev_wait * 1.5 and max_wait > next_wait * 1.5:
                pattern = "ISOLATED SPIKE"
            elif prev_wait > max_wait * 0.8 and next_wait < max_wait * 0.5:
                pattern = "DOWNWARD TREND"
            elif prev_wait < max_wait * 0.5 and next_wait > max_wait * 0.8:
                pattern = "UPWARD TREND"
            else:
                pattern = "CLUSTER"

        print(f"  {it:<6} {max_wait:<10.0f} D{max_dev:<14} {prev_wait:<12.0f} {next_wait:<12.0f} {pattern}")

    # ==========================================================================
    # 6. Physical link analysis based on ring topology
    # ==========================================================================
    print("\n=== PHYSICAL LINK / DIRECTION ANALYSIS ===")
    print("(Mapping R1/R2 to physical FWD/BWD links)\n")

    # Ring topology (physical): Phys0 → Phys2 → Phys1 → Phys3 → Phys0 (FWD)
    # Mesh coordinate mapping: [0,0]=Phys0, [1,0]=Phys2, [2,0]=Phys1, [3,0]=Phys3
    # R1 pairs (based on even/odd mesh coord): (Phys0↔Phys2), (Phys1↔Phys3)
    # R2 pairs: (Phys0↔Phys3), (Phys2↔Phys1)

    print("  Physical ring: Phys0 → Phys2 → Phys1 → Phys3 → Phys0 (FWD direction)")
    print("  R1 pairing: Phys0↔Phys2, Phys1↔Phys3 (direct 1-hop neighbors)")
    print("  R2 pairing: Phys0↔Phys3, Phys2↔Phys1 (direct 1-hop neighbors)")
    print()

    # Check if certain devices consistently have higher latency
    device_mean_r1 = {}
    device_mean_r2 = {}
    for dev in all_devices:
        r1_vals = [r1_waits.get((dev, it), 0) for it in all_iters if (dev, it) in r1_waits]
        r2_vals = [r2_waits.get((dev, it), 0) for it in all_iters if (dev, it) in r2_waits]
        if r1_vals:
            device_mean_r1[dev] = statistics.mean(r1_vals)
        if r2_vals:
            device_mean_r2[dev] = statistics.mean(r2_vals)

    if device_mean_r1:
        print(f"  Mean R1 latency per device:")
        for dev in sorted(device_mean_r1.keys()):
            direction = "FWD agg" if dev % 2 == 0 else "BWD agg"
            print(f"    D{dev} ({direction}): {device_mean_r1[dev]:.0f} ns")

        # Check even vs odd device pattern
        even_devs = [v for d, v in device_mean_r1.items() if d % 2 == 0]
        odd_devs = [v for d, v in device_mean_r1.items() if d % 2 == 1]
        if even_devs and odd_devs:
            even_mean = statistics.mean(even_devs)
            odd_mean = statistics.mean(odd_devs)
            print(f"\n  Even devices (D0,D2, use FWD agg for R1): mean = {even_mean:.0f} ns")
            print(f"  Odd devices (D1,D3, use BWD agg for R1):  mean = {odd_mean:.0f} ns")

            if abs(even_mean - odd_mean) / max(even_mean, odd_mean) > 0.2:
                faster = "EVEN (FWD)" if even_mean < odd_mean else "ODD (BWD)"
                print(f"  ⚠ Significant direction asymmetry - {faster} is faster")
            else:
                print(f"  ✓ Both directions have similar latency")


def analyze_writer_to_aggregator_latency(data: ProfileData):
    """
    Analyze writer send zones and forwarder forward zones.

    The forwarder-based design has workers write directly to forwarder L1 slots
    via NoC, then forwarder forwards via fabric.

    Key zones:
    - Writer: R1-SEND, R2-SEND-STREAMING
    - Forwarder: FWD-FORWARD-LOOP
    """
    print("\n" + "=" * 100)
    print("WRITER → FORWARDER TIMING ANALYSIS")
    print("=" * 100)

    # Group zones by device_id and iteration
    writer_zones: Dict[Tuple[int, int], List[ZoneDuration]] = defaultdict(list)
    forwarder_zones: Dict[Tuple[int, int], List[ZoneDuration]] = defaultdict(list)

    for d in data.durations:
        key = (d.device_id, d.iteration)
        if d.zone_name in ("R1-SEND", "R2-SEND-STREAMING"):
            writer_zones[key].append(d)
        elif d.zone_name.startswith("FWD-"):
            forwarder_zones[key].append(d)

    if not writer_zones:
        print("  No writer send zones found (R1-SEND, R2-SEND-STREAMING)")
        return

    # Compute statistics for writer send zones
    print("\n=== Writer Send Zone Durations ===")

    r1_send_durations = []
    r2_send_durations = []

    for key, zones in writer_zones.items():
        for z in zones:
            dur_ns = data.cycles_to_ns(z.duration_cycles)
            if z.zone_name == "R1-SEND":
                r1_send_durations.append(dur_ns)
            elif z.zone_name == "R2-SEND-STREAMING":
                r2_send_durations.append(dur_ns)

    print(f"{'Zone':<25} {'Count':>8} {'Mean(ns)':>12} {'Median':>12} {'Min':>10} {'Max':>10}")
    print("-" * 80)

    if r1_send_durations:
        stats = compute_stats(r1_send_durations)
        print(
            f"{'R1-SEND':<25} {stats['count']:>8.0f} {stats['mean']:>12.1f} {stats['median']:>12.1f} "
            f"{stats['min']:>10.1f} {stats['max']:>10.1f}"
        )

    if r2_send_durations:
        stats = compute_stats(r2_send_durations)
        print(
            f"{'R2-SEND-STREAMING':<25} {stats['count']:>8.0f} {stats['mean']:>12.1f} {stats['median']:>12.1f} "
            f"{stats['min']:>10.1f} {stats['max']:>10.1f}"
        )

    # Forwarder zones (if profiled)
    if forwarder_zones:
        print("\n=== Forwarder Zone Durations ===")

        fwd_zone_durations: Dict[str, List[float]] = defaultdict(list)
        for key, zones in forwarder_zones.items():
            for z in zones:
                fwd_zone_durations[z.zone_name].append(data.cycles_to_ns(z.duration_cycles))

        print(f"{'Zone':<25} {'Count':>8} {'Mean(ns)':>12} {'Median':>12} {'Min':>10} {'Max':>10}")
        print("-" * 80)

        for zone_name in sorted(fwd_zone_durations.keys()):
            vals = fwd_zone_durations[zone_name]
            stats = compute_stats(vals)
            print(
                f"{zone_name:<25} {stats['count']:>8.0f} {stats['mean']:>12.1f} {stats['median']:>12.1f} "
                f"{stats['min']:>10.1f} {stats['max']:>10.1f}"
            )
    else:
        print("\n  No forwarder zones found (FWD-*)")

    # Interpretation
    print("\n=== Interpretation ===")
    if r1_send_durations and r2_send_durations:
        r1_mean = statistics.mean(r1_send_durations)
        r2_mean = statistics.mean(r2_send_durations)

        print(f"  R1-SEND mean:           {r1_mean:.0f} ns (pack header + NoC write + sem signal)")
        print(f"  R2-SEND-STREAMING mean: {r2_mean:.0f} ns (streaming: cb_wait + pack + NoC + sem)")

        print(f"  R2 is ~{r2_mean/r1_mean:.1f}x longer than R1 (expected: R2 streams from compute)")
        print(f"  R2 includes wait time for compute to produce each chunk")

    # Fabric back pressure analysis
    if forwarder_zones:
        fwd_loop_durations = []
        fwd_wait_durations = []
        for key, zones in forwarder_zones.items():
            for z in zones:
                dur_ns = data.cycles_to_ns(z.duration_cycles)
                if z.zone_name == "FWD-FORWARD-LOOP":
                    fwd_loop_durations.append(dur_ns)
                elif z.zone_name == "FWD-FABRIC-WAIT":
                    fwd_wait_durations.append(dur_ns)

        if fwd_wait_durations:
            print("\n=== Fabric Back Pressure Analysis ===")
            wait_stats = compute_stats(fwd_wait_durations)
            print(f"  FWD-FABRIC-WAIT instances:  {wait_stats['count']:.0f}")
            print(f"  Per-wait mean:              {wait_stats['mean']:.0f} ns")
            print(f"  Per-wait median:            {wait_stats['median']:.0f} ns")
            print(f"  Per-wait min/max:           {wait_stats['min']:.0f} / {wait_stats['max']:.0f} ns")

            # Aggregate: total fabric wait vs total forward loop per (device, iteration)
            wait_by_key: Dict[Tuple[int, int], float] = defaultdict(float)
            loop_by_key: Dict[Tuple[int, int], float] = defaultdict(float)
            for key, zones in forwarder_zones.items():
                for z in zones:
                    dur_ns = data.cycles_to_ns(z.duration_cycles)
                    if z.zone_name == "FWD-FABRIC-WAIT":
                        wait_by_key[key] += dur_ns
                    elif z.zone_name == "FWD-FORWARD-LOOP":
                        loop_by_key[key] += dur_ns

            pct_list = []
            for key in loop_by_key:
                loop_ns = loop_by_key[key]
                wait_ns = wait_by_key.get(key, 0.0)
                if loop_ns > 0:
                    pct_list.append(wait_ns / loop_ns * 100.0)

            if pct_list:
                pct_mean = statistics.mean(pct_list)
                pct_median = statistics.median(pct_list)
                pct_max = max(pct_list)
                print(f"\n  Total fabric wait as % of forward loop:")
                print(f"    Mean:   {pct_mean:.1f}%")
                print(f"    Median: {pct_median:.1f}%")
                print(f"    Max:    {pct_max:.1f}%")
                if pct_mean > 50:
                    print(
                        f"  ⚠ SIGNIFICANT BACK PRESSURE: forwarder spends >{pct_mean:.0f}% of loop time "
                        f"waiting for empty fabric slots"
                    )
                    print(f"    → Larger packet sizes may saturate fabric bandwidth")
                elif pct_mean > 20:
                    print(f"  ⚡ MODERATE BACK PRESSURE: {pct_mean:.0f}% of forward loop is fabric wait")
                else:
                    print(f"  ✓ LOW BACK PRESSURE: only {pct_mean:.0f}% of forward loop is fabric wait")


def analyze_waterfall_timeline(data: ProfileData):
    """
    Build a waterfall/Gantt-style timeline showing all 3 kernels in parallel.

    This visualization shows Reader, Writer, and Compute kernels as parallel lanes,
    with zones stacked sequentially within each lane based on their execution order.

    Key insights this provides:
    - Which kernel is the bottleneck at each phase
    - Where blocking/waiting occurs across kernel boundaries
    - Opportunities for better pipelining or overlap

    Assumptions:
    - All 3 kernels start at t=0 (approximately true - they're dispatched together)
    - Zones within a kernel execute sequentially
    - For compute zones, we use TRISC_1 (Math) as the reference since it closely matches
      actual kernel execution time. TRISC_0 (Unpack) and TRISC_2 (Pack) run in parallel
      and synchronize via semaphores, so they don't add to wall-clock time.
    """
    print_summary_header("WATERFALL TIMELINE (3-kernel parallel view)")

    # Define zone ordering per kernel (from source code analysis)
    READER_ZONES = [
        "R1-LOCAL-INPUT",
        "R1-WAIT-NEIGHBOR",
        "R2-WAIT-NEIGHBOR",
    ]

    # Writer zones - packet-based forwarding through forwarder cores
    WRITER_ZONES = [
        "R1-SEND",  # Send local input to R1 neighbor (all packets at once)
        "R2-SEND-STREAMING",  # Stream R1 compute results to R2 neighbor (overlapped with compute)
    ]

    # Forwarder kernel zones (runs on forwarder cores, BRISC=FWD, NCRISC=BWD)
    # Non-blocking design: polls bit-packed semaphore and forwards any ready slot immediately
    FORWARDER_ZONES = [
        "FWD-FORWARD-LOOP",  # Main loop: poll semaphore, forward ready slots via fabric
        "FWD-FABRIC-WAIT",  # Time spent waiting for empty fabric write slot (back pressure)
    ]

    # Compute zones - SDPA tail streaming reduction
    COMPUTE_ZONES_TOP = [
        "R1-COMPUTE",  # Round 1: reduce(local, R1 neighbor) → R1 result
        "R2-COMPUTE",  # Round 2: reduce(R1 result, R2 neighbor) → final output
    ]

    SDPA_SUBZONES = []

    # Collect durations - separate by RISC type for compute zones
    zone_avg_ns: Dict[str, float] = {}
    zone_by_trisc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    zone_durations: Dict[str, List[float]] = defaultdict(list)

    for d in data.durations:
        dur_ns = data.cycles_to_ns(d.duration_cycles)
        zone_durations[d.zone_name].append(dur_ns)

        # Track per-TRISC for compute zones
        if d.risc.startswith("TRISC"):
            zone_by_trisc[d.zone_name][d.risc].append(dur_ns)

    for zone, durations in zone_durations.items():
        zone_avg_ns[zone] = statistics.mean(durations) if durations else 0

    def get_trisc_duration(zone: str, trisc: str = "TRISC_1") -> float:
        """Get the duration for a compute zone using specified TRISC (default: Math/TRISC_1)."""
        if zone in zone_by_trisc and trisc in zone_by_trisc[zone]:
            return statistics.mean(zone_by_trisc[zone][trisc])
        # Fallback to overall average if specified TRISC not found
        return zone_avg_ns.get(zone, 0)

    def get_all_trisc_durations(zone: str) -> Dict[str, float]:
        """Get duration per TRISC for a zone."""
        result = {}
        if zone in zone_by_trisc:
            for risc in ["TRISC_0", "TRISC_1", "TRISC_2"]:
                if risc in zone_by_trisc[zone]:
                    result[risc] = statistics.mean(zone_by_trisc[zone][risc])
        return result

    # Check zone coverage
    print(f"\nZone coverage check:")
    all_expected = set(READER_ZONES + WRITER_ZONES + FORWARDER_ZONES + COMPUTE_ZONES_TOP + SDPA_SUBZONES)
    found_zones = set(zone_avg_ns.keys())
    missing = all_expected - found_zones
    extra = (
        found_zones
        - all_expected
        - {"BRISC-FW", "NCRISC-FW", "TRISC-FW", "BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL"}
    )

    if missing:
        print(f"  Missing zones (not in data): {sorted(missing)}")
    if extra:
        extra_filtered = [z for z in extra if not z.endswith("-FW") and not z.endswith("-KERNEL")]
        if extra_filtered:
            print(f"  Extra zones (in data but not mapped): {sorted(extra_filtered)}")

    # Show per-TRISC breakdown for compute zones
    print(f"\n  Per-TRISC durations for compute zones:")
    print(f"  {'Zone':<30} {'TRISC_0':>10} {'TRISC_1':>10} {'TRISC_2':>10} {'Used':>10}")
    print(f"  " + "-" * 75)
    for zone in COMPUTE_ZONES_TOP + SDPA_SUBZONES:
        triscs = get_all_trisc_durations(zone)
        if triscs:
            t0 = triscs.get("TRISC_0", 0)
            t1 = triscs.get("TRISC_1", 0)
            t2 = triscs.get("TRISC_2", 0)
            used = get_trisc_duration(zone, "TRISC_1")
            print(f"  {zone:<30} {t0:>10.0f} {t1:>10.0f} {t2:>10.0f} {used:>10.0f}")

    # Build timeline per kernel
    def build_timeline(zones: List[str], kernel_name: str) -> List[Tuple[str, float, float]]:
        """Build [(zone_name, start_ns, end_ns), ...] for a kernel."""
        timeline = []
        current_time = 0.0

        for zone in zones:
            if zone in zone_avg_ns:
                duration = zone_avg_ns[zone]
                timeline.append((zone, current_time, current_time + duration))
                current_time += duration

        return timeline

    reader_timeline = build_timeline(READER_ZONES, "Reader")
    writer_timeline = build_timeline(WRITER_ZONES, "Writer")

    # Build compute timeline using TRISC_1 (Math) durations
    # TRISC_1 is used because it represents the actual execution path through the kernel
    # and matches the measured kernel duration closely
    compute_timeline = []
    current_time = 0.0

    # For sub-zones, we track them separately to show nesting
    sdpa_subzone_info = {}  # zone -> [(subzone, duration), ...]

    for zone in COMPUTE_ZONES_TOP:
        duration = get_trisc_duration(zone, "TRISC_1")
        if duration > 0:
            compute_timeline.append((zone, current_time, current_time + duration))
            current_time += duration

    # Calculate totals
    reader_total = reader_timeline[-1][2] if reader_timeline else 0
    writer_total = writer_timeline[-1][2] if writer_timeline else 0
    compute_total = compute_timeline[-1][2] if compute_timeline else 0
    max_total = max(reader_total, writer_total, compute_total)

    print(f"\n" + "=" * 90)
    print(f"KERNEL TIMELINE SUMMARY (based on average zone durations)")
    print(f"=" * 90)
    print(f"\n  Reader total:  {reader_total:>10.0f} ns (NCRISC)")
    print(f"  Writer total:  {writer_total:>10.0f} ns (BRISC)")
    print(f"  Compute total: {compute_total:>10.0f} ns (TRISC_1/Math path)")
    print(f"  ---------------------")
    print(f"  Critical path: {max_total:>10.0f} ns (longest kernel)")

    # Identify the bottleneck
    if compute_total >= max(reader_total, writer_total):
        print(f"\n  ⏱️  COMPUTE is the critical path")
    elif writer_total >= max(reader_total, compute_total):
        print(f"\n  ⏱️  WRITER is the critical path")
    else:
        print(f"\n  ⏱️  READER is the critical path")

    # Print detailed timeline per kernel
    def print_kernel_timeline(
        name: str, timeline: List[Tuple[str, float, float]], total: float, subzone_info: Dict = None
    ):
        print(f"\n{name} Timeline:")
        print(f"  {'Zone':<35} {'Start':>10} {'End':>10} {'Duration':>10} {'%':>6}")
        print(f"  " + "-" * 75)
        for zone, start, end in timeline:
            duration = end - start
            pct = (duration / total * 100) if total > 0 else 0
            print(f"  {zone:<35} {start:>10.0f} {end:>10.0f} {duration:>10.0f} {pct:>5.1f}%")
            # Show sub-zones if available
            if subzone_info and zone in subzone_info:
                for subzone, sub_dur in subzone_info[zone]:
                    sub_pct = (sub_dur / total * 100) if total > 0 else 0
                    print(f"    └─ {subzone:<31} {'':>10} {'':>10} {sub_dur:>10.0f} {sub_pct:>5.1f}%")
        print(f"  " + "-" * 75)
        print(f"  {'TOTAL':<35} {'':>10} {total:>10.0f}")

    print_kernel_timeline("READER (NCRISC)", reader_timeline, reader_total)
    print_kernel_timeline("WRITER (BRISC)", writer_timeline, writer_total)
    print_kernel_timeline("COMPUTE (TRISC_1/Math)", compute_timeline, compute_total, sdpa_subzone_info)

    # ASCII waterfall visualization
    print(f"\n" + "=" * 90)
    print(f"ASCII WATERFALL (time flows left→right, scale: 1 char ≈ {max_total/60:.0f} ns)")
    print(f"=" * 90)

    def render_bar(timeline: List[Tuple[str, float, float]], width: int = 60) -> str:
        """Render a timeline as an ASCII bar."""
        if not timeline or max_total == 0:
            return " " * width

        bar = [" "] * width
        for zone, start, end in timeline:
            start_pos = int(start / max_total * width)
            end_pos = int(end / max_total * width)
            # Use different chars for different zone types
            if "WAIT" in zone:
                char = "░"  # Waiting/blocking
            elif "SEND" in zone or "FWD" in zone:
                char = "▓"  # Data movement
            elif "COMPUTE" in zone:
                char = "█"  # Compute
            else:
                char = "▒"  # Setup/misc
            for i in range(start_pos, min(end_pos + 1, width)):
                bar[i] = char
        return "".join(bar)

    scale_bar = "".join(["." if i % 10 else "|" for i in range(60)])
    print(f"\n  Time:   |{scale_bar}|")
    print(f"          0{' '*28}{max_total/2:.0f}{' '*27}{max_total:.0f} ns")
    print(f"  Reader: |{render_bar(reader_timeline)}|")
    print(f"  Writer: |{render_bar(writer_timeline)}|")
    print(f"  Compute:|{render_bar(compute_timeline)}|")
    print(f"\n  Legend: █ compute  ▓ data move  ░ wait/connect  ▒ setup")

    # ==================================================================================
    # CROSS-KERNEL WATERFALL (Option C: kernels as columns, time flows down)
    # ==================================================================================
    print(f"\n" + "=" * 110)
    print(f"CROSS-KERNEL WATERFALL (time ↓ down, kernels → across)")
    print(f"=" * 110)
    print(f"All times in ns. Shows what each kernel is doing at key boundary times.")

    # Collect all zone boundaries across all kernels
    all_boundaries = set([0.0])  # Start with t=0

    for zone, start, end in reader_timeline:
        all_boundaries.add(start)
        all_boundaries.add(end)
    for zone, start, end in writer_timeline:
        all_boundaries.add(start)
        all_boundaries.add(end)
    for zone, start, end in compute_timeline:
        all_boundaries.add(start)
        all_boundaries.add(end)

    # Sort boundaries
    sorted_times = sorted(all_boundaries)

    # Helper to find active zone at a given time
    def get_active_zone(timeline: List[Tuple[str, float, float]], t: float) -> Tuple[str, float, float]:
        """Return (zone_name, start, end) active at time t, or None if done."""
        for zone, start, end in timeline:
            if start <= t < end:
                return (zone, start, end)
        # Check if we're past the end
        if timeline and t >= timeline[-1][2]:
            return None
        return None

    # Helper to format zone cell
    COL_WIDTH = 22  # Column width for zone names

    def format_zone_cell(
        active: Tuple[str, float, float],
        prev_active: Tuple[str, float, float],
        kernel_done_time: float,
        t: float,
        width: int = COL_WIDTH,
    ) -> str:
        """Format a cell showing the active zone."""
        if active is None:
            if t >= kernel_done_time:
                return f"{'(done)':^{width}}"
            return f"{'':^{width}}"

        zone, start, end = active
        duration = end - start

        # Determine zone type for styling
        if "WAIT" in zone:
            prefix = "░"
        elif "SEND" in zone or "FWD" in zone:
            prefix = "▓"
        elif "COMPUTE" in zone:
            prefix = "█"
        else:
            prefix = "▒"

        # Show zone name (truncated), or continuation arrow if same as previous
        if prev_active and prev_active[0] == zone:
            # Same zone continuing - show arrow
            return f"{'   ↓':^{width}}"
        else:
            # New zone - show name (truncate to fit)
            short_name = zone[: width - 3] if len(zone) > width - 3 else zone
            return f"{prefix} {short_name:<{width-2}}"

    # Track sync events for annotation
    sync_events = {}

    # Pre-compute sync points
    # Writer R2-SEND-STREAMING blocks on cb_wait_front until Compute produces data
    for zone, start, end in writer_timeline:
        if zone == "R2-SEND-STREAMING":
            sync_events[start] = ("W←C", "Writer streams R1 result as compute produces it")

    # Compute finishes R1 → enables R2-SEND-STREAMING and R2-WAIT-NEIGHBOR
    for zone, start, end in compute_timeline:
        if zone == "R1-COMPUTE":
            sync_events[end] = ("R1✓", "R1 compute done")

    # Print header
    w = COL_WIDTH
    print(f"\n{'Time':>8} │ {'Reader (NCRISC)':<{w}} │ {'Writer (BRISC)':<{w}} │ {'Compute (TRISC_1)':<{w}} │ Sync")
    print(f"{'─'*8}─┼─{'─'*w}─┼─{'─'*w}─┼─{'─'*w}─┼─{'─'*6}")

    reader_done = reader_timeline[-1][2] if reader_timeline else 0
    writer_done = writer_timeline[-1][2] if writer_timeline else 0
    compute_done = compute_timeline[-1][2] if compute_timeline else 0

    prev_reader = None
    prev_writer = None
    prev_compute = None

    # Only show key time points (zone transitions)
    shown_count = 0
    max_rows = 40  # Limit output

    for i, t in enumerate(sorted_times):
        if shown_count >= max_rows:
            print(f"{'...':>10} │ {'':^24} │ {'':^24} │ {'':^24} │")
            remaining = len(sorted_times) - i
            print(f"           │ ({remaining} more time points, truncated)")
            break

        reader_active = get_active_zone(reader_timeline, t)
        writer_active = get_active_zone(writer_timeline, t)
        compute_active = get_active_zone(compute_timeline, t)

        # Only show rows where something changes
        reader_changed = reader_active != prev_reader
        writer_changed = writer_active != prev_writer
        compute_changed = compute_active != prev_compute

        if not (reader_changed or writer_changed or compute_changed):
            continue

        # Get sync annotation
        sync = sync_events.get(t, ("", ""))[0]

        reader_cell = format_zone_cell(reader_active, prev_reader, reader_done, t)
        writer_cell = format_zone_cell(writer_active, prev_writer, writer_done, t)
        compute_cell = format_zone_cell(compute_active, prev_compute, compute_done, t)

        print(f"{t:>8.0f} │ {reader_cell} │ {writer_cell} │ {compute_cell} │ {sync}")

        # For important zones, show duration info on next line
        if reader_changed and reader_active:
            zone, start, end = reader_active
            # Don't show extra detail, keep it compact

        prev_reader = reader_active
        prev_writer = writer_active
        prev_compute = compute_active
        shown_count += 1

    # Final row
    print(f"{'─'*8}─┼─{'─'*w}─┼─{'─'*w}─┼─{'─'*w}─┼─{'─'*6}")
    print(f"{'TOTAL':>8} │ {reader_done:>{w-3}.0f} ns │ {writer_done:>{w-3}.0f} ns │ {compute_done:>{w-3}.0f} ns │")

    # Print zone duration legend
    print(f"\n  Zone Durations (for reference):")
    print(f"  ├─ Reader zones:")
    for zone, start, end in reader_timeline:
        print(f"  │    {zone:<25} [{start:>6.0f} → {end:>6.0f}] {end-start:>6.0f} ns")
    print(f"  ├─ Writer zones:")
    for zone, start, end in writer_timeline:
        print(f"  │    {zone:<25} [{start:>6.0f} → {end:>6.0f}] {end-start:>6.0f} ns")
    print(f"  └─ Compute zones:")
    for zone, start, end in compute_timeline:
        print(f"       {zone:<25} [{start:>6.0f} → {end:>6.0f}] {end-start:>6.0f} ns")

    print(f"\n  Legend: ░ wait/blocking  ▓ data move  █ compute  ▒ setup  ↓ continuing")

    # ==================================================================================
    # Cross-kernel dependency analysis (simplified)
    # ==================================================================================
    print(f"\n" + "=" * 110)
    print(f"CROSS-KERNEL SYNC ANALYSIS")
    print(f"=" * 110)
    print(f"\n  Key synchronization points (based on avg zone durations):")
    print(f"  NOTE: All 3 kernels start at t=0. Wait durations reflect actual blocking time.")

    # Find key synchronization points
    # 1. Reader pushes R1 neighbor data → Compute can process R1
    reader_r1_wait_end = 0
    for zone, start, end in reader_timeline:
        if zone == "R1-WAIT-NEIGHBOR":
            reader_r1_wait_end = end
            break

    r1_compute_time = 0
    for zone, start, end in compute_timeline:
        if zone == "R1-COMPUTE":
            r1_compute_time = end - start
            break

    print(f"\n  1. R1 Neighbor Data Flow (Reader → Compute):")
    print(f"     Reader finishes R1-WAIT-NEIGHBOR at: {reader_r1_wait_end:>8.0f} ns")
    print(f"     R1-COMPUTE duration (TRISC_1):       {r1_compute_time:>8.0f} ns")
    print(f"     ├─ Compute's cb_wait_front blocks until Reader pushes R1 data")
    print(f"     └─ R1-COMPUTE includes both CB wait + actual compute")

    # 2. Writer streams R1 results as compute produces them
    writer_r2_stream_start = 0
    writer_r2_stream_duration = 0
    for zone, start, end in writer_timeline:
        if zone == "R2-SEND-STREAMING":
            writer_r2_stream_start = start
            writer_r2_stream_duration = end - start
            break

    print(f"\n  2. R1 Result Streaming (Compute → Writer for R2 send):")
    print(f"     Writer starts R2-SEND-STREAMING at:  {writer_r2_stream_start:>8.0f} ns")
    print(f"     R2-SEND-STREAMING duration:          {writer_r2_stream_duration:>8.0f} ns")
    print(f"     ├─ Writer uses cb_wait_front to stream chunks as compute produces them")
    print(f"     └─ Overlaps fabric transfer with R1 compute")

    # 3. Reader pushes R2 neighbor data → Compute can process R2
    reader_r2_done = reader_timeline[-1][2] if reader_timeline else 0

    r2_compute_time = 0
    for zone, start, end in compute_timeline:
        if zone == "R2-COMPUTE":
            r2_compute_time = end - start
            break

    print(f"\n  3. R2 Neighbor Data Flow (Reader → Compute):")
    print(f"     Reader finishes all at:              {reader_r2_done:>8.0f} ns")
    print(f"     R2-COMPUTE duration (TRISC_1):       {r2_compute_time:>8.0f} ns")
    print(f"     ├─ R2-COMPUTE blocks on cb_wait_front until Reader pushes R2 data")
    print(f"     └─ R2-COMPUTE includes both CB wait + actual compute")

    # Summary insights
    print(f"\n  OPTIMIZATION INSIGHTS:")

    total_compute_time = r1_compute_time + r2_compute_time
    reader_total_wait = 0
    for zone, start, end in reader_timeline:
        if "WAIT" in zone:
            reader_total_wait += end - start

    print(f"    Compute kernel breakdown:")
    print(f"      - R1-COMPUTE:          {r1_compute_time:>8.0f} ns")
    print(f"      - R2-COMPUTE:          {r2_compute_time:>8.0f} ns")
    print(f"      - Total compute:       {total_compute_time:>8.0f} ns")
    print(f"    Reader wait breakdown:")
    print(f"      - R1-WAIT-NEIGHBOR:    {zone_avg_ns.get('R1-WAIT-NEIGHBOR', 0):>8.0f} ns")
    print(f"      - R2-WAIT-NEIGHBOR:    {zone_avg_ns.get('R2-WAIT-NEIGHBOR', 0):>8.0f} ns")
    print(f"      - Total reader wait:   {reader_total_wait:>8.0f} ns")

    if reader_total > compute_total:
        gap = reader_total - compute_total
        print(f"\n    ⚠️  READER is the critical path (by {gap:.0f} ns)")
        print(f"       Reader waits dominate - fabric latency is the bottleneck")
    elif compute_total > reader_total:
        gap = compute_total - reader_total
        print(f"\n    ⚠️  COMPUTE is the critical path (by {gap:.0f} ns)")
        print(f"       Focus on math/pack optimization")
    else:
        print(f"\n    ✓  Reader and Compute are balanced")


def main():
    parser = argparse.ArgumentParser(description="Analyze device profiler CSV data")
    parser.add_argument("csv_path", help="Path to profile_log_device.csv")
    parser.add_argument(
        "--all-iters", action="store_true", help="Include all iterations (default: only main perf trace)"
    )
    parser.add_argument(
        "--iter-range", nargs=2, type=int, metavar=("START", "END"), help="Filter to specific iteration range"
    )
    parser.add_argument("--trace-id", type=str, default=None, help="Filter to specific trace_id (0=warmup, 1=main)")

    args = parser.parse_args()

    print(f"Loading data from: {args.csv_path}")
    data = parse_csv(args.csv_path)
    print(f"Parsed {len(data.durations)} zone durations")
    print(f"Architecture: {data.arch}, Clock: {data.clock_mhz} MHz")

    # Get iteration range
    all_iters = sorted(set(d.iteration for d in data.durations))
    print(f"Iteration range in data: {min(all_iters)} to {max(all_iters)}")

    # Get trace_id distribution
    trace_ids = defaultdict(int)
    for d in data.durations:
        trace_ids[d.trace_id] += 1
    print(f"Trace ID distribution: {dict(trace_ids)}")

    # Apply filters
    if args.iter_range:
        data = filter_data(data, iter_range=tuple(args.iter_range))
        print(f"Filtered to iterations {args.iter_range[0]}-{args.iter_range[1]}")
    elif args.trace_id:
        data = filter_data(data, trace_id_filter=args.trace_id)
        print(f"Filtered to trace_id={args.trace_id}")
    elif not args.all_iters:
        # Default: main perf trace only (trace_id=1)
        data = filter_data(data, trace_id_filter="1")
        print(f"Filtered to main perf trace (trace_id=1)")

    remaining_iters = sorted(set(d.iteration for d in data.durations))
    print(f"Analyzing {len(data.durations)} measurements across {len(remaining_iters)} iterations")

    # Run analyses
    analyze_per_zone(data)
    analyze_per_trisc(data)
    analyze_waterfall_timeline(data)  # New 3-kernel parallel view
    analyze_per_device(data)
    analyze_per_iteration(data)
    analyze_per_iteration_zone_spread(data)
    analyze_timestamp_skew(data)
    analyze_device_imbalance(data)
    analyze_iter_to_iter_variation(data)
    analyze_ring_communication(data)
    analyze_compute_vs_wait_bottleneck(data)
    analyze_r1_vs_r2_wait(data)
    analyze_device_spread_per_iteration(data)
    analyze_outliers(data)
    analyze_phase_drift_pattern(data)
    analyze_alternating_spike_pattern(data)
    analyze_writer_to_aggregator_latency(data)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
