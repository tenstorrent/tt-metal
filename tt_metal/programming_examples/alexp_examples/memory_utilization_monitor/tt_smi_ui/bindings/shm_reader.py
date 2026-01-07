# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Pure Python SHM reader (fallback when C++ bindings unavailable)."""

import os
import mmap
import struct
import signal
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class TelemetryData:
    temperature: float = -1.0
    power: float = -1.0
    aiclk_mhz: int = 0
    status: str = "Unknown"
    available: bool = False


@dataclass
class ProcessMemory:
    pid: int = 0
    name: str = ""
    dram_allocated: int = 0
    l1_allocated: int = 0
    l1_small_allocated: int = 0
    trace_allocated: int = 0
    cb_allocated: int = 0
    kernel_allocated: int = 0


@dataclass
class Device:
    asic_id: int = 0
    arch_name: str = "Unknown"
    has_shm: bool = False

    total_dram: int = 0
    used_dram: int = 0
    total_l1: int = 0
    used_l1: int = 0
    used_l1_small: int = 0
    used_trace: int = 0
    used_cb: int = 0
    used_kernel: int = 0

    processes: List[dict] = field(default_factory=list)


class SHMReader:
    """Pure Python shared memory reader."""

    SHM_DIR = Path("/dev/shm")

    @staticmethod
    def format_bytes(bytes_val: int) -> str:
        """Format bytes with units."""
        if bytes_val == 0:
            return "0B"
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        idx = 0
        val = float(bytes_val)
        while val >= 1024.0 and idx < 4:
            val /= 1024.0
            idx += 1
        return f"{val:.1f}{units[idx]}"

    @staticmethod
    def is_process_alive(pid: int) -> bool:
        """Check if process is alive."""
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def enumerate_devices(self) -> List[Device]:
        """Scan /dev/shm for TT device SHM files."""
        devices = []

        if not self.SHM_DIR.exists():
            return devices

        for shm_file in self.SHM_DIR.glob("tt_device_*_memory"):
            try:
                # Parse asic_id from filename: tt_device_<asic_id>_memory
                parts = shm_file.name.split("_")
                if len(parts) >= 3:
                    asic_id = int(parts[2])
                    dev = Device(asic_id=asic_id)

                    if self._read_shm(shm_file, dev):
                        devices.append(dev)
            except (ValueError, IndexError):
                continue

        return devices

    def _read_shm(self, shm_path: Path, dev: Device) -> bool:
        """Read SHM file and populate device."""
        try:
            with open(shm_path, "rb") as f:
                # Memory map the file
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

                # Parse header (simplified - just get atomics)
                # Offsets based on SHMDeviceMemoryRegion structure
                offset = 24  # Skip version, num_active, timestamp, refcount

                # board_serial, asic_id, device_id
                mm.seek(offset)
                offset += 24

                # Read atomic counters (8 bytes each, relaxed ordering)
                mm.seek(offset)
                dev.used_dram = struct.unpack("Q", mm.read(8))[0]
                dev.used_l1 = struct.unpack("Q", mm.read(8))[0]
                dev.used_l1_small = struct.unpack("Q", mm.read(8))[0]
                dev.used_trace = struct.unpack("Q", mm.read(8))[0]
                dev.used_cb = struct.unpack("Q", mm.read(8))[0]
                dev.used_kernel = struct.unpack("Q", mm.read(8))[0]

                # Hardcoded memory sizes (fallback)
                dev.total_dram = 12 * 1024**3  # 12 GiB
                dev.total_l1 = 93 * 1024**2  # 93 MiB

                dev.has_shm = True
                mm.close()
                return True

        except Exception:
            return False

    def update_device_memory(self, dev: Device) -> bool:
        """Update memory stats for device."""
        shm_name = f"tt_device_{dev.asic_id}_memory"
        shm_path = self.SHM_DIR / shm_name
        return self._read_shm(shm_path, dev)

    def cleanup_dead_processes(self) -> int:
        """Clean up dead processes (requires write access)."""
        # Not implemented in pure Python fallback (requires mmap PROT_WRITE)
        return 0
