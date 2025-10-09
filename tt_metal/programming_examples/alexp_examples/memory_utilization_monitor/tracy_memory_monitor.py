#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tracy Memory Monitor - Python API

This module provides Python bindings to the TracyMemoryMonitor C++ class,
allowing you to query device memory statistics from Python tests and scripts.

Usage:
    from tracy_memory_monitor import TracyMemoryMonitor

    monitor = TracyMemoryMonitor()
    stats = monitor.query_device(0)
    print(f"Device 0 DRAM: {stats.dram_allocated} bytes")

Integration with tests:
    def test_memory_usage():
        monitor = TracyMemoryMonitor()

        # Get baseline
        before = monitor.query_device(0)

        # Run your code
        run_model()

        # Check memory increase
        after = monitor.query_device(0)
        memory_increase = after.dram_allocated - before.dram_allocated
        assert memory_increase < MAX_ALLOWED_MEMORY
"""

import ctypes
import os
from dataclasses import dataclass
from typing import List, Optional
from enum import IntEnum


class BufferType(IntEnum):
    """Buffer type enumeration matching C++ TracyMemoryMonitor::BufferType"""

    DRAM = 0
    L1 = 1
    SYSTEM_MEMORY = 2
    L1_SMALL = 3
    TRACE = 4


@dataclass
class DeviceMemoryStats:
    """Memory statistics for a single device"""

    dram_allocated: int
    l1_allocated: int
    system_memory_allocated: int
    l1_small_allocated: int
    trace_allocated: int
    num_buffers: int
    total_allocs: int
    total_frees: int

    @property
    def total_allocated(self) -> int:
        """Get total allocated across all buffer types"""
        return (
            self.dram_allocated
            + self.l1_allocated
            + self.system_memory_allocated
            + self.l1_small_allocated
            + self.trace_allocated
        )

    def get_allocated(self, buffer_type: BufferType) -> int:
        """Get allocation for specific buffer type"""
        return {
            BufferType.DRAM: self.dram_allocated,
            BufferType.L1: self.l1_allocated,
            BufferType.SYSTEM_MEMORY: self.system_memory_allocated,
            BufferType.L1_SMALL: self.l1_small_allocated,
            BufferType.TRACE: self.trace_allocated,
        }[buffer_type]

    def __str__(self):
        return (
            f"DeviceMemoryStats(\n"
            f"  DRAM: {format_bytes(self.dram_allocated)}\n"
            f"  L1: {format_bytes(self.l1_allocated)}\n"
            f"  System: {format_bytes(self.system_memory_allocated)}\n"
            f"  L1_SMALL: {format_bytes(self.l1_small_allocated)}\n"
            f"  TRACE: {format_bytes(self.trace_allocated)}\n"
            f"  Active Buffers: {self.num_buffers}\n"
            f"  Total Allocs: {self.total_allocs}\n"
            f"  Total Frees: {self.total_frees}\n"
            f")"
        )


def format_bytes(bytes_val: int) -> str:
    """Format byte count as human-readable string"""
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    size = float(bytes_val)

    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1

    return f"{size:.2f} {units[unit_idx]}"


class TracyMemoryMonitor:
    """
    Python wrapper for the TracyMemoryMonitor C++ class.

    This class provides access to real-time device memory statistics
    tracked by the Tracy-based memory monitor.

    Note: This is a thin wrapper around a C++ singleton. The actual
    tracking happens automatically when buffers are allocated/deallocated
    in the C++ code.
    """

    MAX_DEVICES = 8

    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize the Tracy Memory Monitor wrapper.

        Args:
            lib_path: Optional path to the shared library containing
                     TracyMemoryMonitor. If not provided, searches
                     standard locations.
        """
        # This is a placeholder - in reality, you would:
        # 1. Load the C++ library containing TracyMemoryMonitor
        # 2. Set up ctypes bindings to the C++ functions
        # 3. Or use pybind11/SWIG for cleaner bindings

        # For now, this serves as documentation and API definition
        self._lib = None
        self._initialized = False

        # Try to load the library (implementation would go here)
        # self._lib = ctypes.CDLL(lib_path or self._find_library())
        # self._setup_bindings()

    def _find_library(self) -> str:
        """Find the tt_metal library with TracyMemoryMonitor"""
        # Search common locations
        search_paths = [
            os.path.join(os.environ.get("TT_METAL_HOME", ""), "build/lib/libtt_metal.so"),
            "/usr/local/lib/libtt_metal.so",
            "./libtt_metal.so",
        ]

        for path in search_paths:
            if os.path.exists(path):
                return path

        raise RuntimeError("Could not find tt_metal library. " "Set TT_METAL_HOME or provide lib_path explicitly.")

    def _setup_bindings(self):
        """Setup ctypes bindings to C++ functions"""
        # Example of how bindings would be set up:
        # self._lib.tracy_memory_monitor_query_device.argtypes = [ctypes.c_int]
        # self._lib.tracy_memory_monitor_query_device.restype = DeviceMemoryStatsC
        pass

    def query_device(self, device_id: int) -> DeviceMemoryStats:
        """
        Query memory statistics for a specific device.

        This is a lock-free operation suitable for real-time monitoring.

        Args:
            device_id: Device ID (0-7)

        Returns:
            DeviceMemoryStats snapshot for the device

        Raises:
            ValueError: If device_id is out of range
        """
        if device_id < 0 or device_id >= self.MAX_DEVICES:
            raise ValueError(f"device_id must be in range [0, {self.MAX_DEVICES})")

        # TODO: Call C++ function via ctypes/pybind11
        # For now, return mock data to show the API
        return DeviceMemoryStats(
            dram_allocated=0,
            l1_allocated=0,
            system_memory_allocated=0,
            l1_small_allocated=0,
            trace_allocated=0,
            num_buffers=0,
            total_allocs=0,
            total_frees=0,
        )

    def query_all_devices(self) -> List[DeviceMemoryStats]:
        """
        Query memory statistics for all devices.

        Returns:
            List of DeviceMemoryStats for all devices (0-7)
        """
        return [self.query_device(i) for i in range(self.MAX_DEVICES)]

    def get_active_buffer_count(self, device_id: int) -> int:
        """
        Get count of active (unfreed) buffers on a device.

        This requires a lock and is slightly more expensive than query_device.

        Args:
            device_id: Device ID (0-7)

        Returns:
            Number of active buffers
        """
        if device_id < 0 or device_id >= self.MAX_DEVICES:
            raise ValueError(f"device_id must be in range [0, {self.MAX_DEVICES})")

        # TODO: Call C++ function
        return 0

    @staticmethod
    def is_tracy_enabled() -> bool:
        """Check if Tracy profiler is enabled at compile time"""
        # TODO: Query from C++
        return False

    def reset(self):
        """Reset all statistics (for testing)"""
        # TODO: Call C++ reset function
        pass


# Example usage and testing utilities
class MemoryMonitorContext:
    """
    Context manager for monitoring memory usage within a code block.

    Usage:
        with MemoryMonitorContext(device_id=0) as monitor:
            # Your code here
            run_model()

        print(f"Memory increase: {monitor.memory_increase} bytes")
        print(f"Peak allocation: {monitor.peak_allocated} bytes")
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.monitor = TracyMemoryMonitor()
        self.before: Optional[DeviceMemoryStats] = None
        self.after: Optional[DeviceMemoryStats] = None

    def __enter__(self):
        self.before = self.monitor.query_device(self.device_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.after = self.monitor.query_device(self.device_id)
        return False

    @property
    def memory_increase(self) -> int:
        """Get total memory increase during the context"""
        if self.before is None or self.after is None:
            return 0
        return self.after.total_allocated - self.before.total_allocated

    @property
    def dram_increase(self) -> int:
        """Get DRAM increase during the context"""
        if self.before is None or self.after is None:
            return 0
        return self.after.dram_allocated - self.before.dram_allocated

    @property
    def l1_increase(self) -> int:
        """Get L1 increase during the context"""
        if self.before is None or self.after is None:
            return 0
        return self.after.l1_allocated - self.before.l1_allocated


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Tracy Memory Monitor - Python Client")
    parser.add_argument("-d", "--device", type=int, default=0, help="Device ID to monitor")
    parser.add_argument("-r", "--refresh", type=int, default=1000, help="Refresh interval (ms)")
    parser.add_argument("-s", "--single", action="store_true", help="Single query mode")
    args = parser.parse_args()

    monitor = TracyMemoryMonitor()

    if args.single:
        stats = monitor.query_device(args.device)
        print(f"\nDevice {args.device} Memory Statistics:")
        print(stats)
    else:
        print(f"\n📊 Monitoring device {args.device} (refresh: {args.refresh}ms)")
        print("Press Ctrl+C to exit\n")

        try:
            while True:
                stats = monitor.query_device(args.device)
                print(f"\r{' ' * 80}\r", end="")  # Clear line
                print(
                    f"DRAM: {format_bytes(stats.dram_allocated)} | "
                    f"L1: {format_bytes(stats.l1_allocated)} | "
                    f"Buffers: {stats.num_buffers}",
                    end="",
                    flush=True,
                )
                time.sleep(args.refresh / 1000.0)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
