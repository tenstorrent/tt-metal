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

    # Track TRISC_0 start and TRISC_2 end for total duration calculation
    # Key: (pcie_slot, core_x, core_y, run_host_id)
    trisc0_start_times = {}
    trisc2_end_times = {}

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
                host_block_key = (pcie_slot, core_x, core_y, run_host_id)

                if zone_type == "ZONE_START":
                    # Store start time
                    start_times[block_key] = int(row[" time[cycles since reset]"])

                    # Track TRISC_0 start
                    if risc_type.strip() == "TRISC_0":
                        trisc0_start_times[host_block_key] = int(row[" time[cycles since reset]"])

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

                    # Track TRISC_2 end
                    if risc_type.strip() == "TRISC_2":
                        trisc2_end_times[host_block_key] = int(row[" time[cycles since reset]"])

    # Calculate total duration from TRISC_0 start to TRISC_2 end
    total_durations = {}
    for host_block_key in trisc0_start_times:
        if host_block_key in trisc2_end_times:
            total_duration = trisc2_end_times[host_block_key] - trisc0_start_times[host_block_key]
            # Extract host_id from the key
            run_host_id = host_block_key[3]  # (pcie_slot, core_x, core_y, run_host_id)
            total_durations[run_host_id] = total_duration

    return durations, total_durations


def print_results(durations, total_durations, verbose=False):
    """Print formatted results."""

    if not durations:
        print("No MATMUL_SINGLE_TILE blocks found.")
        return

    # Host ID to in0_tile_r_dim mapping
    host_id_map = {
        "1024": 1,
        "2048": 2,
        "3072": 4,
        "4096": 8,
        "5120": 16,
        "6144": 32,
    }

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

    # All TRISC Summary (default view)
    print(f"\n{'='*120}")
    print("MATMUL_SINGLE_TILE Duration Summary - All TRISC Cores")
    print(f"{'='*120}")

    # Group by host ID and TRISC type
    trisc_by_host = defaultdict(lambda: defaultdict(list))
    for d in durations:
        if d["risc_type"].startswith("TRISC"):
            trisc_by_host[d["run_host_id"]][d["risc_type"]].append(d["duration_cycles"])

    if trisc_by_host:
        print(f"{'in0_tile_r_dim':<15} {'TRISC_0':<20} {'TRISC_1':<20} {'TRISC_2':<20} {'Total (T0→T2)':<20}")
        print("-" * 95)

        for host_id in sorted(trisc_by_host.keys()):
            trisc_data = trisc_by_host[host_id]
            in0_tile_r_dim = host_id_map.get(host_id, "?")

            # Get duration for each TRISC (assuming one block per TRISC per host)
            trisc0_dur = trisc_data.get("TRISC_0", [0])[0] if trisc_data.get("TRISC_0") else 0
            trisc1_dur = trisc_data.get("TRISC_1", [0])[0] if trisc_data.get("TRISC_1") else 0
            trisc2_dur = trisc_data.get("TRISC_2", [0])[0] if trisc_data.get("TRISC_2") else 0
            total_dur = total_durations.get(host_id, 0)

            print(f"{in0_tile_r_dim:<15} {trisc0_dur:<20,} {trisc1_dur:<20,} {trisc2_dur:<20,} {total_dur:<20,}")

        print("-" * 95)
    else:
        print("No TRISC blocks found.")

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
    durations, total_durations = analyze_matmul_blocks(args.csv)
    print_results(durations, total_durations, verbose=args.verbose)
