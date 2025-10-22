#!/usr/bin/env python3
"""
Real-time memory monitor - polls allocation server statistics
Shows memory usage changes during inference
"""

import socket
import struct
import time
import sys
import os
from dataclasses import dataclass
from typing import Dict

TT_ALLOC_SERVER_SOCKET = "/tmp/tt_allocation_server.sock"


@dataclass
class DeviceStats:
    num_buffers: int
    dram_bytes: int
    l1_bytes: int
    l1_small_bytes: int
    trace_bytes: int

    @property
    def total_bytes(self):
        return self.dram_bytes + self.l1_bytes + self.l1_small_bytes + self.trace_bytes

    def __sub__(self, other):
        """Calculate delta between two stats snapshots"""
        return DeviceStats(
            num_buffers=self.num_buffers - other.num_buffers,
            dram_bytes=self.dram_bytes - other.dram_bytes,
            l1_bytes=self.l1_bytes - other.l1_bytes,
            l1_small_bytes=self.l1_small_bytes - other.l1_small_bytes,
            trace_bytes=self.trace_bytes - other.trace_bytes,
        )


def bytes_to_mb(bytes_val):
    return bytes_val / (1024.0 * 1024.0)


def format_delta(value, unit="MB"):
    """Format delta with + or - sign"""
    if value > 0:
        return f"+{value:.2f} {unit}"
    elif value < 0:
        return f"{value:.2f} {unit}"
    else:
        return f" {value:.2f} {unit}"


def request_stats() -> Dict[int, DeviceStats]:
    """Request statistics from allocation server"""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(TT_ALLOC_SERVER_SOCKET)

        # Send STATS request (type=2)
        msg = struct.pack("BxxxIQQ", 2, 0, 0, 0)  # type=2 (STATS), rest zeros
        sock.send(msg)

        # Receive response
        # Format: per device: device_id, num_buffers, dram, l1, l1_small, trace (all uint64_t)
        stats = {}
        while True:
            data = sock.recv(8 * 6)  # 6 uint64_t values
            if len(data) < 8 * 6:
                break

            values = struct.unpack("QQQQQQ", data)
            device_id, num_buffers, dram, l1, l1_small, trace = values

            stats[device_id] = DeviceStats(
                num_buffers=num_buffers, dram_bytes=dram, l1_bytes=l1, l1_small_bytes=l1_small, trace_bytes=trace
            )

        sock.close()
        return stats
    except Exception as e:
        print(f"Error requesting stats: {e}", file=sys.stderr)
        return {}


def clear_screen():
    os.system("clear" if os.name == "posix" else "cls")


def print_stats_table(
    current_stats: Dict[int, DeviceStats], prev_stats: Dict[int, DeviceStats] = None, show_delta: bool = True
):
    """Print stats in a nice table format"""

    print("=" * 100)
    print(f"{'Device':<8} {'Buffers':<10} {'DRAM':<15} {'L1':<15} {'L1_SMALL':<15} {'TRACE':<15} {'Total':<15}")
    print("=" * 100)

    for device_id in sorted(current_stats.keys()):
        stats = current_stats[device_id]

        # Current values
        line = f"{device_id:<8} {stats.num_buffers:<10} "
        line += f"{bytes_to_mb(stats.dram_bytes):>6.2f} MB   "
        line += f"{bytes_to_mb(stats.l1_bytes):>6.2f} MB   "
        line += f"{bytes_to_mb(stats.l1_small_bytes):>6.2f} MB   "
        line += f"{bytes_to_mb(stats.trace_bytes):>6.2f} MB   "
        line += f"{bytes_to_mb(stats.total_bytes):>6.2f} MB"
        print(line)

        # Delta (if requested and available)
        if show_delta and prev_stats and device_id in prev_stats:
            delta = stats - prev_stats[device_id]
            if any([delta.dram_bytes, delta.l1_bytes, delta.l1_small_bytes, delta.trace_bytes]):
                delta_line = f"{'(Δ)':<8} {delta.num_buffers:+<10} "
                delta_line += f"{format_delta(bytes_to_mb(delta.dram_bytes)):>12}   "
                delta_line += f"{format_delta(bytes_to_mb(delta.l1_bytes)):>12}   "
                delta_line += f"{format_delta(bytes_to_mb(delta.l1_small_bytes)):>12}   "
                delta_line += f"{format_delta(bytes_to_mb(delta.trace_bytes)):>12}   "
                delta_line += f"{format_delta(bytes_to_mb(delta.total_bytes)):>12}"
                print(f"\033[90m{delta_line}\033[0m")  # Gray color for delta

    print("=" * 100)


def monitor_loop(interval: float = 0.5, show_delta: bool = True):
    """Continuously monitor and display memory stats"""
    prev_stats = None
    iteration = 0

    print("\n🔍 Real-Time Memory Monitor")
    print(f"   Update interval: {interval}s")
    print(f"   Press Ctrl+C to stop\n")
    time.sleep(1)

    try:
        while True:
            current_stats = request_stats()

            if not current_stats:
                print("⚠️  Could not connect to allocation server")
                print(
                    f"   Ensure server is running: TT_ALLOC_TRACKING_ENABLED=1 ./build/install/bin/allocation_server_poc"
                )
                time.sleep(2)
                continue

            clear_screen()
            print(f"\n📊 Memory Usage Monitor (Update #{iteration})")
            print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            print_stats_table(current_stats, prev_stats, show_delta)

            # Show hints
            if iteration == 0:
                print("\n💡 Tips:")
                print("   - Δ (delta) shows changes since last update")
                print("   - Run inference in another terminal to see allocations")
                print("   - Large TRACE allocations = trace capture")
                print("   - Small L1 allocations = kernel/firmware loading")
                print("   - Steady state = trace execution (no new allocations)\n")

            prev_stats = current_stats
            iteration += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Real-time memory usage monitor")
    parser.add_argument("-i", "--interval", type=float, default=0.5, help="Update interval in seconds (default: 0.5)")
    parser.add_argument("--no-delta", action="store_true", help="Hide delta values")
    parser.add_argument("--once", action="store_true", help="Show stats once and exit")

    args = parser.parse_args()

    if args.once:
        stats = request_stats()
        if stats:
            print_stats_table(stats, show_delta=False)
        else:
            print("⚠️  Could not connect to allocation server")
            sys.exit(1)
    else:
        monitor_loop(args.interval, not args.no_delta)


if __name__ == "__main__":
    main()
