#!/usr/bin/env python3
"""
Script to scan Galaxy cluster connections and extract device mapping information.
Runs test_system_health for each visible device and parses the output.
"""

import subprocess
import re
import sys
from typing import List, Dict, Tuple


def run_test_for_device(device_id: int, test_binary: str) -> str:
    """
    Run the test_system_health binary for a specific device.

    Args:
        device_id: The device ID to test
        test_binary: Path to the test binary

    Returns:
        The filtered output containing only "Chip:" lines
    """
    env = {"TT_VISIBLE_DEVICES": str(device_id)}
    cmd = [test_binary, "--gtest_filter=Cluster.ReportSystemHealth"]

    try:
        result = subprocess.run(
            cmd, env={**subprocess.os.environ.copy(), **env}, capture_output=True, text=True, timeout=30
        )

        # Filter lines containing "Chip:"
        chip_lines = [line for line in result.stdout.split("\n") if "Chip:" in line]
        return "\n".join(chip_lines)

    except subprocess.TimeoutExpired:
        print(f"Warning: Timeout for device {device_id}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"Error running test for device {device_id}: {e}", file=sys.stderr)
        return ""


def parse_chip_info(line: str) -> Tuple[str, str, str]:
    """
    Parse a chip info line to extract tray and N location.

    Args:
        line: A line containing chip information

    Returns:
        Tuple of (unique_id, tray_num, n_loc) or empty strings if parsing fails
    """
    # Pattern: Chip: X Unique ID: <id> Tray: <tray_num> N<n_loc>
    unique_id_match = re.search(r"Unique ID:\s+(\w+)", line)
    tray_match = re.search(r"Tray:\s+(\d+)", line)
    n_match = re.search(r"N(\d+)", line)

    unique_id = unique_id_match.group(1) if unique_id_match else ""
    tray_num = tray_match.group(1) if tray_match else ""
    n_loc = n_match.group(1) if n_match else ""

    return unique_id, tray_num, n_loc


def scan_galaxy_connections(
    num_devices: int = 32, test_binary: str = "./build/test/tt_metal/tt_fabric/test_system_health"
) -> List[Dict[str, str]]:
    """
    Scan all devices in the Galaxy cluster and collect connection information.

    Args:
        num_devices: Number of devices to scan (default: 32 for Galaxy)
        test_binary: Path to the test binary

    Returns:
        List of dictionaries containing device mapping information
    """
    results = []

    for device_id in range(num_devices):
        print(f"Scanning device {device_id}...", file=sys.stderr)
        output = run_test_for_device(device_id, test_binary)

        if output:
            for line in output.split("\n"):
                if line.strip():
                    unique_id, tray_num, n_loc = parse_chip_info(line)
                    if unique_id and tray_num and n_loc:
                        results.append(
                            {"device_id": str(device_id), "unique_id": unique_id, "tray": tray_num, "n_loc": n_loc}
                        )
                        print(f"{device_id=} {unique_id=} {tray_num=} {n_loc=}")

    return results


def get_device_pairs(results: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """Extract device ID pairs from results.

    Args:
        results: List of device mapping information

    Returns:
        List of (device_id1, device_id2) tuples
    """
    # Group results by tray
    tray_map = {}
    for result in results:
        tray = result["tray"]
        if tray not in tray_map:
            tray_map[tray] = {}
        n_loc = result["n_loc"]
        tray_map[tray][n_loc] = result["device_id"]

    # Extract pairs
    pairs_list = []
    for tray in sorted(tray_map.keys(), key=int):
        n_devices = tray_map[tray]

        # Get pairs: N1-N2, N3-N4, N5-N6, N7-N8
        pairs = [("1", "2"), ("3", "4"), ("5", "6"), ("7", "8")]
        for n1, n2 in pairs:
            dev1 = n_devices.get(n1)
            dev2 = n_devices.get(n2)
            if dev1 and dev2:
                pairs_list.append((dev1, dev2))

    return pairs_list


def print_results(results: List[Dict[str, str]]):
    """Print results in a formatted table grouped by tray with N pairs."""
    if not results:
        print("No results found")
        return

    # Group results by tray
    tray_map = {}
    for result in results:
        tray = result["tray"]
        if tray not in tray_map:
            tray_map[tray] = {}
        n_loc = result["n_loc"]
        tray_map[tray][n_loc] = result["device_id"]

    print("\nDevice Mapping by Tray:")
    print("=" * 80)

    for tray in sorted(tray_map.keys(), key=int):
        print(f"\nTray {tray}:")
        print("-" * 80)

        n_devices = tray_map[tray]

        # Print pairs: N1-N2, N3-N4, N5-N6, N7-N8
        pairs = [("1", "2"), ("3", "4"), ("5", "6"), ("7", "8")]
        for n1, n2 in pairs:
            dev1 = n_devices.get(n1, "N/A")
            dev2 = n_devices.get(n2, "N/A")
            print(f"  N{n1}-N{n2}: ({dev1},{dev2})")

    print("\n" + "=" * 80)
    print(f"Total entries: {len(results)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scan Galaxy cluster connections and extract device mapping")
    parser.add_argument("--num-devices", type=int, default=32, help="Number of devices to scan (default: 32)")
    parser.add_argument(
        "--test-binary",
        type=str,
        default="./build/test/tt_metal/tt_fabric/test_system_health",
        help="Path to test_system_health binary",
    )
    parser.add_argument("--output", type=str, help="Optional output file to save results (CSV format)")

    args = parser.parse_args()

    results = scan_galaxy_connections(args.num_devices, args.test_binary)
    print_results(results)

    if args.output:
        import csv

        with open(args.output, "w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"\nResults saved to {args.output}")

    # Always write pairs.csv
    import csv

    pairs = get_device_pairs(results)
    with open("pairs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["device_id1", "device_id2"])
        writer.writerows(pairs)
    print(f"Pairs saved to pairs.csv")


if __name__ == "__main__":
    main()
