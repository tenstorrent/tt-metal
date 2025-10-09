#!/usr/bin/env python3
"""
Correlate leaked buffers from server debug.log with allocation patterns
Shows which buffers leaked and their allocation/deallocation history
"""

import re
import sys
from collections import defaultdict


def parse_server_log(filename):
    """Parse allocation server debug.log"""

    allocations = defaultdict(list)  # {(device, buffer_id): [events]}
    deallocations = defaultdict(list)

    alloc_pattern = re.compile(
        r"✓ \[PID (\d+)\] Allocated (\d+) bytes of (\w+) on device (\d+) \(buffer_id=(\d+)(?:, ref_count=(\d+))?\)"
    )
    free_pattern = re.compile(
        r"✗ \[PID (\d+)\] Freed buffer (\d+) on device (\d+) \((\d+) bytes(?:, (?:ref_count=(\d+)|FINAL))?\)"
    )
    unknown_free_pattern = re.compile(r"⚠ \[PID (\d+)\] Deallocation for unknown buffer (\d+) on device (\d+)")

    print(f"Parsing {filename}...")
    line_num = 0

    with open(filename, "r") as f:
        for line in f:
            line_num += 1

            # Match allocation
            match = alloc_pattern.search(line)
            if match:
                pid = int(match.group(1))
                size = int(match.group(2))
                buf_type = match.group(3)
                device = int(match.group(4))
                buffer_id = int(match.group(5))
                ref_count = int(match.group(6)) if match.group(6) else 1

                key = (device, buffer_id)
                allocations[key].append(
                    {"line": line_num, "pid": pid, "size": size, "type": buf_type, "ref_count": ref_count}
                )
                continue

            # Match free
            match = free_pattern.search(line)
            if match:
                pid = int(match.group(1))
                buffer_id = int(match.group(2))
                device = int(match.group(3))
                size = int(match.group(4))
                is_final = "FINAL" in line

                key = (device, buffer_id)
                deallocations[key].append({"line": line_num, "pid": pid, "size": size, "is_final": is_final})
                continue

            # Match unknown free
            match = unknown_free_pattern.search(line)
            if match:
                pid = int(match.group(1))
                buffer_id = int(match.group(2))
                device = int(match.group(3))

                key = (device, buffer_id)
                deallocations[key].append({"line": line_num, "pid": pid, "size": 0, "unknown": True})

    return allocations, deallocations


def analyze_buffers(allocations, deallocations):
    """Analyze buffer lifecycle"""

    print("\n" + "=" * 80)
    print("BUFFER LIFECYCLE ANALYSIS")
    print("=" * 80)

    # Group by device
    by_device = defaultdict(lambda: {"alloc": [], "dealloc": []})

    for key in set(list(allocations.keys()) + list(deallocations.keys())):
        device, buffer_id = key
        by_device[device]["alloc"].extend(allocations[key])
        by_device[device]["dealloc"].extend(deallocations[key])

    # Analyze each device
    for device in sorted(by_device.keys()):
        device_data = by_device[device]

        print(f"\n{'='*80}")
        print(f"DEVICE {device}")
        print(f"{'='*80}")

        # Find buffers with mismatched alloc/dealloc counts
        buffer_balance = defaultdict(lambda: {"alloc": 0, "dealloc": 0, "events": []})

        for key, alloc_events in allocations.items():
            dev, buf_id = key
            if dev != device:
                continue

            dealloc_events = deallocations.get(key, [])

            buffer_balance[buf_id]["alloc"] = len(alloc_events)
            buffer_balance[buf_id]["dealloc"] = len(dealloc_events)
            buffer_balance[buf_id]["events"].extend([("alloc", e) for e in alloc_events])
            buffer_balance[buf_id]["events"].extend([("dealloc", e) for e in dealloc_events])
            buffer_balance[buf_id]["events"].sort(key=lambda x: x[1]["line"])

        # Find leaked buffers (more allocs than deallocs)
        leaked = {bid: data for bid, data in buffer_balance.items() if data["alloc"] > data["dealloc"]}

        # Find double-freed buffers (more deallocs than allocs)
        double_freed = {bid: data for bid, data in buffer_balance.items() if data["dealloc"] > data["alloc"]}

        # Perfectly balanced
        balanced = {
            bid: data for bid, data in buffer_balance.items() if data["alloc"] == data["dealloc"] and data["alloc"] > 0
        }

        print(f"\nSummary:")
        print(f"  Leaked buffers: {len(leaked)} (alloc > dealloc)")
        print(f"  Double-freed buffers: {len(double_freed)} (dealloc > alloc)")
        print(f"  Balanced buffers: {len(balanced)} (alloc == dealloc)")

        # Show leaked buffers
        if leaked:
            print(f"\nLEAKED BUFFERS (Top 20):")
            print(f"-" * 80)

            for buffer_id, data in sorted(leaked.items(), key=lambda x: x[1]["alloc"] - x[1]["dealloc"], reverse=True)[
                :20
            ]:
                print(f"\nBuffer 0x{buffer_id:x} ({buffer_id}):")
                print(f"  Allocations: {data['alloc']}, Deallocations: {data['dealloc']}")
                print(f"  Net leak: {data['alloc'] - data['dealloc']} buffer(s)")

                # Show timeline
                print(f"  Timeline:")
                for event_type, event in data["events"][:15]:  # Show first 15 events
                    if event_type == "alloc":
                        print(f"    Line {event['line']}: ALLOC {event['size']} bytes ({event['type']})")
                    else:
                        status = "UNKNOWN" if event.get("unknown") else ("FINAL" if event.get("is_final") else "ref--")
                        print(f"    Line {event['line']}: FREE {event['size']} bytes ({status})")

                if len(data["events"]) > 15:
                    print(f"    ... ({len(data['events']) - 15} more events)")

        # Show double-freed buffers
        if double_freed:
            print(f"\nDOUBLE-FREED BUFFERS (Top 10):")
            print(f"-" * 80)

            for buffer_id, data in sorted(
                double_freed.items(), key=lambda x: x[1]["dealloc"] - x[1]["alloc"], reverse=True
            )[:10]:
                print(f"\nBuffer 0x{buffer_id:x} ({buffer_id}):")
                print(f"  Allocations: {data['alloc']}, Deallocations: {data['dealloc']}")
                print(f"  Excess frees: {data['dealloc'] - data['alloc']}")

                # Show timeline
                print(f"  Timeline:")
                for event_type, event in data["events"][:10]:
                    if event_type == "alloc":
                        print(f"    Line {event['line']}: ALLOC {event['size']} bytes ({event['type']})")
                    else:
                        status = "UNKNOWN" if event.get("unknown") else ("FINAL" if event.get("is_final") else "ref--")
                        print(f"    Line {event['line']}: FREE {event['size']} bytes ({status})")


def find_size_changes(allocations):
    """Find buffers where the same address was used for different sizes"""

    print("\n" + "=" * 80)
    print("BUFFER ADDRESS REUSE (Same address, different sizes)")
    print("=" * 80)

    size_changes = []

    for key, events in allocations.items():
        if len(events) < 2:
            continue

        device, buffer_id = key
        sizes = [e["size"] for e in events]

        # Check if sizes changed
        if len(set(sizes)) > 1:
            size_changes.append((device, buffer_id, events))

    if not size_changes:
        print("\nNo size changes detected.")
        return

    print(f"\nFound {len(size_changes)} buffers with size changes:")

    for device, buffer_id, events in size_changes[:15]:  # Show first 15
        print(f"\nDevice {device}, Buffer 0x{buffer_id:x} ({buffer_id}):")
        print(f"  Allocated {len(events)} times with different sizes:")

        for i, event in enumerate(events[:10], 1):
            print(f"    {i}. Line {event['line']}: {event['size']} bytes ({event['type']})")

        if len(events) > 10:
            print(f"    ... ({len(events) - 10} more allocations)")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <debug.log>")
        print()
        print("Analyzes buffer allocation server debug.log to find:")
        print("  - Leaked buffers (allocated but not deallocated)")
        print("  - Double-freed buffers (deallocated more than allocated)")
        print("  - Buffer address reuse with size changes")
        sys.exit(1)

    log_file = sys.argv[1]

    allocations, deallocations = parse_server_log(log_file)

    print(f"\nTotal unique buffers: {len(set(list(allocations.keys()) + list(deallocations.keys())))}")
    print(f"Buffers with allocations: {len(allocations)}")
    print(f"Buffers with deallocations: {len(deallocations)}")

    analyze_buffers(allocations, deallocations)
    find_size_changes(allocations)

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("To trace where these buffers were created:")
    print("1. Apply the debug logging patch: ADD_BUFFER_DEBUG_LOGGING.patch")
    print("2. Rebuild TT-Metal: cmake --build build")
    print("3. Run with: export TT_BUFFER_DEBUG_LOG=1")
    print("4. Check /tmp/tt_buffer_debug.log for call stacks")
    print()
    print("The leaked buffer addresses from this analysis can be correlated")
    print("with the call stacks in tt_buffer_debug.log to find their origin.")


if __name__ == "__main__":
    main()
