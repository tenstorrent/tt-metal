# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Core API for TT-SMI - high-level Python interface."""

from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

from .bindings import native


class Device:
    """Device interface wrapping native C++ device."""

    def __init__(self, native_dev):
        self.chip_id = native_dev.chip_id
        self.asic_id = native_dev.asic_id
        self.arch_name = native_dev.arch_name
        self.is_remote = native_dev.is_remote
        self.display_id = native_dev.display_id
        self.tray_id = native_dev.tray_id
        self.chip_in_tray = native_dev.chip_in_tray

        self.temperature = native_dev.telemetry.temperature
        self.power = native_dev.telemetry.power
        self.voltage_mv = native_dev.telemetry.voltage_mv
        self.current_ma = native_dev.telemetry.current_ma
        self.aiclk_mhz = native_dev.telemetry.aiclk_mhz
        self.telemetry_status = native_dev.telemetry.status

        self.total_dram = native_dev.total_dram
        self.used_dram = native_dev.used_dram
        self.total_l1 = native_dev.total_l1
        self.used_l1 = native_dev.used_l1
        self.used_l1_small = native_dev.used_l1_small
        self.used_trace = native_dev.used_trace
        self.used_cb = native_dev.used_cb
        self.used_kernel = native_dev.used_kernel
        self.has_shm = native_dev.has_shm

        self.processes = [
            {
                "pid": p.pid,
                "name": p.name,
                "dram": p.dram_allocated,
                "l1": p.l1_allocated,
                "l1_small": p.l1_small_allocated,
                "trace": p.trace_allocated,
                "cb": p.cb_allocated,
                "kernel": p.kernel_allocated,
            }
            for p in native_dev.processes
        ]

        self._native = native_dev

    @property
    def dram_utilization(self) -> float:
        """DRAM utilization percentage."""
        if self.total_dram == 0:
            return 0.0
        return (self.used_dram / self.total_dram) * 100.0

    @property
    def l1_utilization(self) -> float:
        """L1 utilization percentage."""
        if self.total_l1 == 0:
            return 0.0
        total_l1_used = self.used_l1 + self.used_l1_small + self.used_cb
        return (total_l1_used / self.total_l1) * 100.0


def get_devices() -> List[Device]:
    """
    Enumerate all Tenstorrent devices using native C++ backend.

    Returns:
        List of Device objects
    """
    native_devices = native.enumerate_devices()
    return [Device(native_dev=d) for d in native_devices]


def update_telemetry(device: Device) -> bool:
    """Update telemetry for a device."""
    success = native.update_device_telemetry(device._native)
    if success:
        # Re-sync telemetry values from C++ to Python
        device.temperature = device._native.telemetry.temperature
        device.power = device._native.telemetry.power
        device.voltage_mv = device._native.telemetry.voltage_mv
        device.current_ma = device._native.telemetry.current_ma
        device.aiclk_mhz = device._native.telemetry.aiclk_mhz
        device.telemetry_status = device._native.telemetry.status
    else:
        # Update failed - sync the status to see why
        device.telemetry_status = device._native.telemetry.status
    return success


def update_telemetry_parallel(devices: List[Device], timeout: float = 1.0) -> None:
    """
    Update telemetry for all devices in parallel (like C++ tt_smi).

    Each device is polled in a separate thread with a timeout to prevent
    one stuck device from blocking others.

    Args:
        devices: List of Device objects to update
        timeout: Timeout per device in seconds (default: 1.0)
    """

    def update_one(dev: Device) -> None:
        try:
            success = update_telemetry(dev)
            if not success and dev.telemetry_status == "Unknown":
                # Update returned False but didn't set a status
                dev.telemetry_status = "Error"
        except Exception as e:
            # Mark as failed if exception occurs
            dev.telemetry_status = "Error"

    # Launch parallel telemetry updates (like C++ std::async)
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {executor.submit(update_one, dev): dev for dev in devices}

        # Wait for all with timeout
        try:
            for future in as_completed(futures, timeout=timeout):
                try:
                    future.result(timeout=0.1)
                except Exception as e:
                    # Timeout or error - device will show stale/error status
                    pass
        except TimeoutError:
            # Some futures didn't complete in time (e.g., after chip reset)
            # Mark devices with unfinished futures as having timeout status
            for future, dev in futures.items():
                if not future.done():
                    dev.telemetry_status = "Timeout"


def update_memory(device: Device) -> bool:
    """Update memory stats for a device."""
    success = native.update_device_memory(device._native)
    if success:
        # Re-sync memory values from C++ to Python
        device.used_dram = device._native.used_dram
        device.used_l1 = device._native.used_l1
        device.used_l1_small = device._native.used_l1_small
        device.used_trace = device._native.used_trace
        device.used_cb = device._native.used_cb
        # Note: used_kernel NOT synced - kernels live in RESERVED L1, not allocatable
        device.processes = [
            {
                "pid": p.pid,
                "name": p.name,
                "dram": p.dram_allocated,
                "l1": p.l1_allocated,
                "l1_small": p.l1_small_allocated,
                "trace": p.trace_allocated,
                "cb": p.cb_allocated,
                # kernel field omitted - not part of allocatable memory
            }
            for p in device._native.processes
        ]
    return success


def cleanup_dead_processes() -> int:
    """Clean up dead processes from SHM. Returns number cleaned."""
    return native.cleanup_dead_processes()


def format_bytes(bytes_val: int) -> str:
    """Format bytes with units (KiB, MiB, GiB)."""
    return native.format_bytes(bytes_val)
