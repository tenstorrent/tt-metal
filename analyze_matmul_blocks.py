#!/usr/bin/env python3
"""
Analyze MATMUL_SINGLE_TILE blocks from profiler CSV and calculate their durations.
"""

import csv
import argparse
from collections import defaultdict


def analyze_matmul_blocks(csv_file):
    """Parse CSV and calculate MATMUL_SINGLE_TILE block durations."""

    # Store start times for each unique block
    # Key: (pcie_slot, core_x, core_y, risc_type, run_host_id)
    start_times = {}

    # Store calculated durations
    durations = []

    with open(csv_file, "r") as f:
        lines = f.readlines()

        # Skip the first line (metadata) and use the second line as header
        header_line = lines[1].strip()
        data_lines = lines[2:]

        # Parse CSV with proper header
        from io import StringIO

        csv_content = header_line + "\n" + "".join(data_lines)
        reader = csv.DictReader(StringIO(csv_content))

        for row in reader:
            zone_name = row.get(" zone name", "").strip()

            if zone_name == "MATMUL_SINGLE_TILE":
                # Create unique key for this block
                pcie_slot = row["PCIe slot"]
                core_x = row[" core_x"]
                core_y = row[" core_y"]
                risc_type = row[" RISC processor type"]
                run_host_id = row[" run host ID"]
                zone_type = row[" type"]

                block_key = (pcie_slot, core_x, core_y, risc_type, run_host_id)

                if zone_type == "ZONE_START":
                    # Store start time
                    start_times[block_key] = int(row[" time[cycles since reset]"])

                elif zone_type == "ZONE_END":
                    # Calculate duration if we have a start time
                    if block_key in start_times:
                        end_time = int(row[" time[cycles since reset]"])
                        start_time = start_times[block_key]
                        duration = end_time - start_time

                        durations.append(
                            {
                                "pcie_slot": pcie_slot,
                                "core_x": core_x.strip(),
                                "core_y": core_y.strip(),
                                "risc_type": risc_type.strip(),
                                "run_host_id": run_host_id.strip(),
                                "start_time": start_time,
                                "end_time": end_time,
                                "duration_cycles": duration,
                            }
                        )

                        # Remove the start time entry
                        del start_times[block_key]

    return durations


def print_results(durations, verbose=False):
    """Print formatted results."""

    if not durations:
        print("No MATMUL_SINGLE_TILE blocks found.")
        return

    # Group by host ID
    by_host_id = defaultdict(list)
    for d in durations:
        by_host_id[d["run_host_id"]].append(d)

    if verbose:
        print("\n" + "=" * 120)
        print("MATMUL_SINGLE_TILE Block Durations - Grouped by Host ID")
        print("=" * 120)

        for host_id in sorted(by_host_id.keys()):
            host_durations = by_host_id[host_id]

            print(f"\n{'='*120}")
            print(f"HOST ID: {host_id}")
            print(f"{'='*120}")

            # Print detailed table for this host
            print(
                f"{'#':<4} {'PCIe':<5} {'Core':<8} {'RISC Type':<10} {'Start Time':<20} {'End Time':<20} {'Duration (cycles)':<20}"
            )
            print("-" * 120)

            for i, d in enumerate(host_durations, 1):
                core_coord = f"({d['core_x']},{d['core_y']})"
                print(
                    f"{i:<4} {d['pcie_slot']:<5} {core_coord:<8} {d['risc_type']:<10} "
                    f"{d['start_time']:<20} {d['end_time']:<20} {d['duration_cycles']:<20}"
                )

            print("-" * 120)

            # Statistics for this host
            host_durations_values = [d["duration_cycles"] for d in host_durations]
            avg_duration = sum(host_durations_values) / len(host_durations_values)
            min_duration = min(host_durations_values)
            max_duration = max(host_durations_values)

            print(f"Total blocks: {len(host_durations)}")
            print(f"Average duration: {avg_duration:,.2f} cycles")
            print(f"Min duration:     {min_duration:,} cycles")
            print(f"Max duration:     {max_duration:,} cycles")

            # Overview by TRISC core for this host
            print(f"\n{'─'*120}")
            print(f"TRISC Core Overview for Host ID {host_id}:")
            print(f"{'─'*120}")

            # Group by TRISC core
            by_trisc = defaultdict(list)
            for d in host_durations:
                if d["risc_type"].startswith("TRISC"):
                    by_trisc[d["risc_type"]].append(d["duration_cycles"])

            if by_trisc:
                print(
                    f"{'TRISC Core':<12} {'Count':<8} {'Avg Duration':<18} {'Min Duration':<18} {'Max Duration':<18} {'Total Duration':<18}"
                )
                print("-" * 120)

                for trisc_type in sorted(by_trisc.keys()):
                    values = by_trisc[trisc_type]
                    count = len(values)
                    avg = sum(values) / count
                    min_val = min(values)
                    max_val = max(values)
                    total = sum(values)

                    print(f"{trisc_type:<12} {count:<8} {avg:>15,.2f}   {min_val:>15,}   {max_val:>15,}   {total:>15,}")
            else:
                print("No TRISC blocks found for this host.")

    # TRISC_0 Summary (default view)
    print(f"\n{'='*120}")
    print("TRISC_0 MATMUL_SINGLE_TILE Duration Summary")
    print(f"{'='*120}")

    # Filter only TRISC_0 blocks
    trisc0_by_host = defaultdict(list)
    for d in durations:
        if d["risc_type"] == "TRISC_0":
            trisc0_by_host[d["run_host_id"]].append(d["duration_cycles"])

    if trisc0_by_host:
        print(f"{'Host ID':<10} {'Duration (cycles)':<20}")
        print("-" * 30)

        for host_id in sorted(trisc0_by_host.keys()):
            values = trisc0_by_host[host_id]
            # Get the full info for display
            trisc0_blocks = [d for d in by_host_id[host_id] if d["risc_type"] == "TRISC_0"]
            for block in trisc0_blocks:
                print(f"{host_id:<10} {block['duration_cycles']:<20,}")

        print("-" * 30)
    else:
        print("No TRISC_0 blocks found.")

    print(f"\n{'='*120}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MATMUL_SINGLE_TILE blocks from profiler CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default="/localdev/skrsmanovic/gitrepos/tt-metal/generated/profiler/.logs/profile_log_device.csv",
        help="Path to the profile CSV file",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed breakdown by host ID and all TRISC cores"
    )

    args = parser.parse_args()

    print(f"Analyzing: {args.csv}")
    durations = analyze_matmul_blocks(args.csv)
    print_results(durations, verbose=args.verbose)
