#!/usr/bin/env python3
"""
Script to map PCIe devices by tray and N location.
Runs test_system_health and parses the output to create a device mapping.
"""

import subprocess
import re
import sys
from typing import Dict, List, Tuple


def run_test_system_health(test_binary: str) -> str:
    """
    Run the test_system_health binary and capture output.

    Args:
        test_binary: Path to the test binary

    Returns:
        The output containing chip information
    """
    cmd = [test_binary, "--gtest_filter=Cluster.ReportSystemHealth"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.stdout
    except subprocess.TimeoutExpired:
        print("Error: Timeout while running test", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"Error running test: {e}", file=sys.stderr)
        return ""


def parse_chip_line(line: str) -> Dict[str, str]:
    """
    Parse a chip info line to extract relevant information.

    Args:
        line: A line containing chip information
        Example: "Chip: 31 PCIe: 15 Unique ID: 835323530303144 Tray: 2 N8"

    Returns:
        Dictionary with parsed information or empty dict if parsing fails
    """
    # Pattern to match the chip line
    pattern = r"Chip:\s+(\d+)\s+PCIe:\s+(\d+)\s+Unique ID:\s+(\w+)\s+Tray:\s+(\d+)\s+N(\d+)"
    match = re.search(pattern, line)

    if match:
        return {
            "chip": match.group(1),
            "pcie_id": match.group(2),
            "unique_id": match.group(3),
            "tray": match.group(4),
            "n_loc": match.group(5),
        }
    return {}


def create_device_mapping(output: str) -> List[Dict[str, str]]:
    """
    Create device mapping from test output.

    Args:
        output: The output from test_system_health

    Returns:
        List of dictionaries containing device mapping information
    """
    # Filter lines containing "PCIe:"
    chip_lines = [line for line in output.split("\n") if "PCIe:" in line]

    # Create list of device info
    results = []
    for line in chip_lines:
        info = parse_chip_line(line)
        if info:
            results.append(
                {
                    "pcie_id": info["pcie_id"],
                    "unique_id": info["unique_id"],
                    "tray": info["tray"],
                    "n_loc": info["n_loc"],
                }
            )

    return results


def get_device_pairs(results: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """Extract device ID pairs from results.

    Args:
        results: List of device mapping information

    Returns:
        List of (pcie_id1, pcie_id2) tuples
    """
    # Group results by tray
    tray_map = {}
    for result in results:
        tray = result["tray"]
        if tray not in tray_map:
            tray_map[tray] = {}
        n_loc = result["n_loc"]
        tray_map[tray][n_loc] = result["pcie_id"]

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


def print_device_mapping(results: List[Dict[str, str]]):
    """
    Print device mapping by tray in the requested format.

    Args:
        results: List of device mapping information
    """
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
        tray_map[tray][n_loc] = result["pcie_id"]

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
    import csv

    parser = argparse.ArgumentParser(description="Map PCIe devices by tray and N location")
    parser.add_argument(
        "--test-binary",
        type=str,
        default="./build/test/tt_metal/tt_fabric/test_system_health",
        help="Path to test_system_health binary",
    )
    parser.add_argument("--output", type=str, help="Optional output file to save results (CSV format)")

    args = parser.parse_args()

    # Run the test
    print(f"Running {args.test_binary}...", file=sys.stderr)
    output = run_test_system_health(args.test_binary)

    if not output:
        print("Error: No output from test", file=sys.stderr)
        sys.exit(1)

    # Create and print the mapping
    results = create_device_mapping(output)

    if not results:
        print("Error: No chip information found in output", file=sys.stderr)
        sys.exit(1)

    print_device_mapping(results)

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")

    # Always write pairs.csv
    pairs = get_device_pairs(results)
    with open("pairs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["device_id1", "device_id2"])
        writer.writerows(pairs)
    print(f"Pairs saved to pairs.csv")


if __name__ == "__main__":
    main()
