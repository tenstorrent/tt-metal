#!/usr/bin/env python3
"""
Analyze buffer allocation/deallocation logs to find:
1. Buffers that were never deallocated (leaks)
2. Buffers that were deallocated more times than allocated (double-frees)
3. Buffer ID reuse patterns
4. Size mismatches during reuse
"""

import re
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from enum import Enum


class EventType(Enum):
    ALLOC = "ALLOC"
    FREE = "FREE"
    UNKNOWN_FREE = "UNKNOWN_FREE"


@dataclass
class BufferEvent:
    line_num: int
    event_type: EventType
    pid: int
    device_id: int
    buffer_id: int
    size: int
    ref_count: int = 0  # For FREE events that show ref_count
    is_final: bool = False  # For FREE events marked as FINAL


@dataclass
class BufferHistory:
    buffer_id: int
    device_id: int
    events: List[BufferEvent] = field(default_factory=list)

    def total_allocs(self) -> int:
        return sum(1 for e in self.events if e.event_type == EventType.ALLOC)

    def total_frees(self) -> int:
        return sum(1 for e in self.events if e.event_type in [EventType.FREE, EventType.UNKNOWN_FREE])

    def is_leaked(self) -> bool:
        return self.total_allocs() > self.total_frees()

    def is_double_freed(self) -> bool:
        return self.total_frees() > self.total_allocs()

    def size_changes(self) -> List[Tuple[int, int, int]]:
        """Returns list of (line_num, old_size, new_size) where size changed"""
        changes = []
        last_size = None
        for event in self.events:
            if event.event_type == EventType.ALLOC:
                if last_size is not None and last_size != event.size:
                    changes.append((event.line_num, last_size, event.size))
                last_size = event.size
        return changes


def parse_log_file(filename: str) -> Dict[Tuple[int, int], BufferHistory]:
    """Parse debug.log and extract buffer allocation history"""

    # Regex patterns (no line numbers in actual file, we'll add them)
    alloc_pattern = re.compile(
        r"✓ \[PID (\d+)\] Allocated (\d+) bytes of \w+ on device (\d+) \(buffer_id=(\w+)(?:, ref_count=(\d+))?\)"
    )
    free_pattern = re.compile(
        r"✗ \[PID (\d+)\] Freed buffer (\w+) on device (\d+) \((\d+) bytes(?:, (?:ref_count=(\d+) remaining|FINAL))?\)"
    )
    unknown_free_pattern = re.compile(r"⚠ \[PID (\d+)\] Deallocation for unknown buffer (\w+) on device (\d+)")

    histories: Dict[Tuple[int, int], BufferHistory] = defaultdict(lambda: BufferHistory(0, 0))

    print(f"Parsing {filename}...")
    with open(filename, "r") as f:
        for line_num, line in enumerate(f, 1):
            # Try to match allocation
            match = alloc_pattern.search(line)
            if match:
                pid = int(match.group(1))
                size = int(match.group(2))
                device_id = int(match.group(3))
                buffer_id = int(match.group(4), 0)  # Handle hex or decimal
                ref_count = int(match.group(5)) if match.group(5) else 1

                key = (device_id, buffer_id)
                if histories[key].buffer_id == 0:
                    histories[key].buffer_id = buffer_id
                    histories[key].device_id = device_id

                event = BufferEvent(
                    line_num=line_num,
                    event_type=EventType.ALLOC,
                    pid=pid,
                    device_id=device_id,
                    buffer_id=buffer_id,
                    size=size,
                    ref_count=ref_count,
                )
                histories[key].events.append(event)
                continue

            # Try to match free
            match = free_pattern.search(line)
            if match:
                pid = int(match.group(1))
                buffer_id = int(match.group(2), 0)
                device_id = int(match.group(3))
                size = int(match.group(4))
                ref_count_str = match.group(5)

                key = (device_id, buffer_id)
                if histories[key].buffer_id == 0:
                    histories[key].buffer_id = buffer_id
                    histories[key].device_id = device_id

                is_final = "FINAL" in line
                ref_count = int(ref_count_str) if ref_count_str else 0

                event = BufferEvent(
                    line_num=line_num,
                    event_type=EventType.FREE,
                    pid=pid,
                    device_id=device_id,
                    buffer_id=buffer_id,
                    size=size,
                    ref_count=ref_count,
                    is_final=is_final,
                )
                histories[key].events.append(event)
                continue

            # Try to match unknown free
            match = unknown_free_pattern.search(line)
            if match:
                pid = int(match.group(1))
                buffer_id = int(match.group(2), 0)
                device_id = int(match.group(3))

                key = (device_id, buffer_id)
                if histories[key].buffer_id == 0:
                    histories[key].buffer_id = buffer_id
                    histories[key].device_id = device_id

                event = BufferEvent(
                    line_num=line_num,
                    event_type=EventType.UNKNOWN_FREE,
                    pid=pid,
                    device_id=device_id,
                    buffer_id=buffer_id,
                    size=0,  # Unknown
                )
                histories[key].events.append(event)

    print(f"Found {len(histories)} unique buffers\n")
    return dict(histories)


def analyze_buffers(histories: Dict[Tuple[int, int], BufferHistory]):
    """Analyze buffer histories and print reports"""

    # Find leaked buffers
    leaked = [(key, hist) for key, hist in histories.items() if hist.is_leaked()]
    double_freed = [(key, hist) for key, hist in histories.items() if hist.is_double_freed()]
    perfectly_managed = [
        (key, hist)
        for key, hist in histories.items()
        if hist.total_allocs() == hist.total_frees() and hist.total_allocs() > 0
    ]

    # Find size changes
    size_changers = [(key, hist) for key, hist in histories.items() if hist.size_changes()]

    print("=" * 80)
    print("BUFFER ALLOCATION ANALYSIS")
    print("=" * 80)
    print(f"Total unique buffers: {len(histories)}")
    print(f"Leaked buffers (alloc > free): {len(leaked)}")
    print(f"Double-freed buffers (free > alloc): {len(double_freed)}")
    print(f"Perfectly managed buffers: {len(perfectly_managed)}")
    print(f"Buffers with size changes: {len(size_changers)}")
    print()

    # === LEAKED BUFFERS ===
    if leaked:
        print("=" * 80)
        print("LEAKED BUFFERS (Not Deallocated)")
        print("=" * 80)

        # Group by device
        by_device = defaultdict(list)
        for key, hist in leaked:
            by_device[hist.device_id].append((key, hist))

        total_leaked_bytes = 0
        for device_id in sorted(by_device.keys()):
            buffers = by_device[device_id]
            device_bytes = sum(hist.events[-1].size for _, hist in buffers if hist.events)
            total_leaked_bytes += device_bytes

            print(f"\nDevice {device_id}: {len(buffers)} leaked buffers ({device_bytes / 1024:.1f} KB)")

            # Show top 10 largest
            buffers_sorted = sorted(buffers, key=lambda x: x[1].events[-1].size if x[1].events else 0, reverse=True)
            for (dev_id, buf_id), hist in buffers_sorted[:10]:
                if not hist.events:
                    continue
                last_alloc = next((e for e in reversed(hist.events) if e.event_type == EventType.ALLOC), None)
                if last_alloc:
                    print(f"  Buffer 0x{buf_id:x}: {last_alloc.size / 1024:.1f} KB")
                    print(f"    Allocated at line {last_alloc.line_num} by PID {last_alloc.pid}")
                    print(f"    Total: {hist.total_allocs()} allocs, {hist.total_frees()} frees")

                    # Show first and last events
                    if len(hist.events) > 1:
                        print(f"    First event: line {hist.events[0].line_num} ({hist.events[0].event_type.value})")
                        print(f"    Last event: line {hist.events[-1].line_num} ({hist.events[-1].event_type.value})")

        print(f"\nTotal leaked memory: {total_leaked_bytes / (1024 * 1024):.2f} MB")
        print()

    # === DOUBLE-FREED BUFFERS ===
    if double_freed:
        print("=" * 80)
        print("DOUBLE-FREED BUFFERS (More Frees than Allocs)")
        print("=" * 80)

        for (dev_id, buf_id), hist in double_freed[:20]:  # Top 20
            print(f"\nBuffer 0x{buf_id:x} on device {dev_id}:")
            print(f"  Total: {hist.total_allocs()} allocs, {hist.total_frees()} frees")
            print(f"  Event timeline:")
            for event in hist.events[-10:]:  # Last 10 events
                event_str = f"    Line {event.line_num}: {event.event_type.value}"
                if event.event_type == EventType.ALLOC:
                    event_str += f" size={event.size}"
                elif event.event_type == EventType.FREE:
                    event_str += f" size={event.size}"
                    if event.is_final:
                        event_str += " (FINAL)"
                    elif event.ref_count > 0:
                        event_str += f" (ref={event.ref_count})"
                print(event_str)
        print()

    # === SIZE CHANGES ===
    if size_changers:
        print("=" * 80)
        print("BUFFERS WITH SIZE CHANGES (Buffer ID Reused for Different Sizes)")
        print("=" * 80)

        for (dev_id, buf_id), hist in size_changers[:20]:  # Top 20
            changes = hist.size_changes()
            print(f"\nBuffer 0x{buf_id:x} on device {dev_id}:")
            print(f"  Total size changes: {len(changes)}")
            for line_num, old_size, new_size in changes[:5]:  # Show first 5
                print(f"    Line {line_num}: {old_size} bytes -> {new_size} bytes")

            # Show allocation pattern
            alloc_sizes = [e.size for e in hist.events if e.event_type == EventType.ALLOC]
            size_counts = Counter(alloc_sizes)
            print(f"  Size frequency:", end="")
            for size, count in sorted(size_counts.items(), key=lambda x: -x[1])[:3]:
                print(f" {size}B ({count}x)", end="")
            print()
        print()

    # === SUMMARY OF ISSUES ===
    print("=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print()

    print("1. LEAKED BUFFERS (WHY NOT DEALLOCATED):")
    if leaked:
        print("   - Application did NOT send FREE messages for these buffers")
        print("   - Likely causes:")
        print("     * Process crashed/exited before cleanup")
        print("     * Exception thrown before deallocate() call")
        print("     * Missing deallocate() in error paths")
        print("     * Circular references preventing cleanup")
    else:
        print("   ✓ No leaked buffers!")
    print()

    print("2. DOUBLE-FREED BUFFERS (WHY UNKNOWN DEALLOCATION):")
    if double_freed:
        print("   - Buffer ID was reused without proper tracking")
        print("   - Likely causes:")
        print("     * Buffer freed -> reallocated -> freed multiple times")
        print("     * ref_count mechanism doesn't track SIZE changes")
        print("     * Old pointer used after buffer was reallocated")
    else:
        print("   ✓ No double-freed buffers!")
    print()

    print("3. SIZE CHANGES (BUFFER ID REUSE):")
    if size_changers:
        print("   - Same memory address reused for different buffer sizes")
        print("   - This is NORMAL for memory allocators")
        print("   - BUT current ref_count tracking doesn't handle this properly")
        print("   - Solution: Track (device_id, buffer_id, size) as key instead")
    else:
        print("   ✓ No problematic buffer ID reuse!")
    print()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <debug.log>")
        sys.exit(1)

    log_file = sys.argv[1]
    histories = parse_log_file(log_file)
    analyze_buffers(histories)

    print("\nAnalysis complete!")
    print("\nTo fix these issues:")
    print("1. Ensure all buffers are properly deallocated before process exit")
    print("2. Use RAII pattern (destructors) to guarantee cleanup")
    print("3. Add exception handlers that ensure cleanup")
    print("4. Consider changing buffer key to include size")


if __name__ == "__main__":
    main()
