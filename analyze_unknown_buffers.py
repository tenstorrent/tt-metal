#!/usr/bin/env python3
"""Analyze unknown buffer deallocations from allocation server log"""
import re
from collections import defaultdict
import sys


def analyze_log(log_file):
    unknown_buffers = defaultdict(lambda: {"count": 0, "devices": set()})

    print(f"Reading {log_file}...")
    with open(log_file, "r") as f:
        for line in f:
            if "Deallocation for unknown buffer" in line:
                match = re.search(r"buffer (\d+) on device (\d+)", line)
                if match:
                    buffer_id = int(match.group(1))
                    device_id = int(match.group(2))
                    unknown_buffers[buffer_id]["count"] += 1
                    unknown_buffers[buffer_id]["devices"].add(device_id)

    print("\n" + "=" * 70)
    print("UNKNOWN BUFFER DEALLOCATION ANALYSIS")
    print("=" * 70)
    print(f"\nTotal unique unknown buffer IDs: {len(unknown_buffers)}")
    print(f"Total deallocation events: {sum(b['count'] for b in unknown_buffers.values())}")
    print()

    if not unknown_buffers:
        print("✓ No unknown buffer deallocations found!")
        return

    # Sort by count
    sorted_buffers = sorted(unknown_buffers.items(), key=lambda x: x[1]["count"], reverse=True)

    print("Top 20 most frequently freed unknown buffers:")
    print("-" * 70)
    print(f"{'Buffer ID (hex)':>16} | {'Count':>6} | {'Devices':>20}")
    print("-" * 70)

    for buffer_id, info in sorted_buffers[:20]:
        devices_str = ",".join(map(str, sorted(info["devices"])))
        print(f"  0x{buffer_id:08x}     | {info['count']:6d} | {devices_str:>20}")

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Analyze patterns
    all_device_buffers = [bid for bid, info in unknown_buffers.items() if len(info["devices"]) == 8]
    single_device_buffers = [bid for bid, info in unknown_buffers.items() if len(info["devices"]) == 1]

    print(f"\nBuffer Distribution:")
    print(f"  - Buffers on ALL 8 devices: {len(all_device_buffers)}")
    print(f"    (System-wide resources like dispatch queues)")
    print(f"  - Buffers on single device: {len(single_device_buffers)}")
    print(f"    (Device-specific initialization)")
    print(
        f"  - Buffers on multiple devices: {len(unknown_buffers) - len(all_device_buffers) - len(single_device_buffers)}"
    )

    if len(unknown_buffers) < 50:
        print("\n✓ Small number of unknown buffers - typical device initialization")
    else:
        print("\n⚠ Large number of unknown buffers - tracking may have started late")

    # Most common count
    most_common_count = sorted_buffers[0][1]["count"] if sorted_buffers else 0
    if most_common_count > 10:
        print(f"\n⚠ Some buffers freed {most_common_count} times")
        print("  This might indicate repeated allocation/deallocation cycles")

    print("\n" + "=" * 70)
    print("WHAT ARE THESE?")
    print("=" * 70)
    print(
        """
These buffers were allocated BEFORE tracking started:
  • Device initialization buffers (command queues, dispatch cores)
  • System memory allocations (hugepages, kernel launch)
  • NOC routing tables and configuration

They're being properly freed - just not tracked!

To track them, start the allocation server BEFORE creating MeshDevice.
"""
    )


if __name__ == "__main__":
    log_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/debug-llama.log"
    )

    analyze_log(log_file)
