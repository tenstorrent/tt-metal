import csv
from pathlib import Path
from collections import defaultdict
import statistics
import os


def process_profiler_log(csv_file_path, zone_names=None):
    """
    Process profiler log to extract zone durations for specified zone names.

    Args:
        csv_file_path: Path to the profiler CSV file
        zone_names: List of zone names to filter (default: ['TRISC-KERNEL', 'UNTILIZE-BLOCK', 'UNTILIZE-OP'])

    Returns a dictionary with zone names as keys, each containing a dict of RISC processor types
    and their durations.
    """
    if zone_names is None:
        zone_names = ["TRISC-KERNEL", "UNTILIZE-BLOCK", "UNTILIZE-OP"]

    # Dictionary to store zone data: {zone_name: {risc_type: [durations]}}
    zone_data = {zone_name: defaultdict(list) for zone_name in zone_names}

    # Read the CSV file
    with open(csv_file_path, "r") as f:
        # Skip the first line (architecture info)
        next(f)

        reader = csv.DictReader(f, skipinitialspace=True)

        # Group rows by zone name and RISC processor type
        grouped_data = {zone_name: defaultdict(list) for zone_name in zone_names}

        for row in reader:
            # Strip all keys and values to handle spacing issues
            row = {k.strip(): v.strip() for k, v in row.items()}

            zone_name = row.get("zone name", "")
            risc_type = row.get("RISC processor type", "")

            # Filter only specified zones
            if zone_name in zone_names:
                grouped_data[zone_name][risc_type].append(row)

        # Process each zone name and RISC processor type
        for zone_name in zone_names:
            for risc_type, rows in grouped_data[zone_name].items():
                i = 0
                while i < len(rows) - 1:
                    current_row = rows[i]
                    next_row = rows[i + 1]

                    current_type = current_row.get("type", "")
                    next_type = next_row.get("type", "")

                    # Check for ZONE_START followed by ZONE_END
                    if current_type == "ZONE_START" and next_type == "ZONE_END":
                        start_time = int(current_row["time[cycles since reset]"])
                        end_time = int(next_row["time[cycles since reset]"])
                        duration = end_time - start_time

                        zone_data[zone_name][risc_type].append(duration)
                        i += 2  # Skip to next pair
                    else:
                        i += 1  # Move to next row

    return zone_data


def print_zone_statistics(zone_data):
    """Print statistics for each zone name and RISC processor type."""
    for zone_name in sorted(zone_data.keys()):
        print(f"\n{'='*60}")
        print(f"Zone: {zone_name}")
        print(f"{'='*60}\n")

        risc_data = zone_data[zone_name]

        for risc_type in sorted(risc_data.keys()):
            durations = risc_data[risc_type]

            print(f"{risc_type}:")
            print(f"  Number of zones: {len(durations)}")

            if durations:
                print(f"  Min: {min(durations):,} cycles")
                print(f"  Max: {max(durations):,} cycles")
                print(f"  Average: {statistics.mean(durations):,.2f} cycles")
                if len(durations) > 1:
                    print(f"  Std Dev: {statistics.stdev(durations):,.2f} cycles")
                print(f"  Total: {sum(durations):,} cycles")

                # Show first few durations as sample
                if len(durations) <= 5:
                    print(f"  Durations: {durations}")
                else:
                    print(f"  First 5 durations: {durations[:5]}")
            print()

    return zone_data


def print_summary_comparison(zone_data):
    """Print a summary comparison across all zones and RISC types."""
    print(f"\n{'='*60}")
    print("Summary Comparison")
    print(f"{'='*60}\n")

    for zone_name in sorted(zone_data.keys()):
        print(f"{zone_name}:")
        risc_data = zone_data[zone_name]

        for risc_type in sorted(risc_data.keys()):
            durations = risc_data[risc_type]
            if durations:
                avg = statistics.mean(durations)
                print(f"  {risc_type}: avg={avg:,.2f} cycles, count={len(durations)}")
        print()


# Example usage
if __name__ == "__main__":
    ENVS = dict(os.environ)
    TT_METAL_HOME = Path(ENVS.get("TT_METAL_HOME", "."))

    profiler_log_path = TT_METAL_HOME / "generated" / "profiler" / ".logs" / "profile_log_device.csv"

    # Process the log file for TRISC-KERNEL, UNTILIZE-BLOCK, and UNTILIZE-OP zones
    zone_data = process_profiler_log(profiler_log_path, zone_names=["TRISC-KERNEL", "UNTILIZE-BLOCK", "UNTILIZE-OP"])

    # Print detailed statistics
    print_zone_statistics(zone_data)

    # Print summary comparison
    print_summary_comparison(zone_data)

    # Access specific data
    print("\n" + "=" * 60)
    print("Specific Data Access Examples")
    print("=" * 60 + "\n")

    if "TRISC-KERNEL" in zone_data:
        print("TRISC-KERNEL zones:")
        for risc_type, durations in zone_data["TRISC-KERNEL"].items():
            print(f"  {risc_type}: {len(durations)} zones")

    if "UNTILIZE-BLOCK" in zone_data:
        print("\nUNTILIZE-BLOCK zones:")
        for risc_type, durations in zone_data["UNTILIZE-BLOCK"].items():
            print(f"  {risc_type}: {len(durations)} zones")

    if "UNTILIZE-OP" in zone_data:
        print("\nUNTILIZE-OP zones:")
        for risc_type, durations in zone_data["UNTILIZE-OP"].items():
            print(f"  {risc_type}: {len(durations)} zones")
