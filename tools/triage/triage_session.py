#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import threading

from ttexalens.coordinate import OnChipCoordinate
from ttexalens.device import Device
from ttexalens.hardware.risc_debug import RiscLocation


class TriageSession:
    """Singleton that holds all mutable state accumulated during a triage run."""

    def __init__(self):
        self._lock = threading.Lock()
        self._broken_devices: set[Device] = set()
        self._broken_cores: set[RiscLocation] = set()
        self._halted_cores: set[RiscLocation] = set()

    def add_broken_device(self, device: Device) -> None:
        with self._lock:
            self._broken_devices.add(device)

    def is_device_broken(self, device: Device) -> bool:
        with self._lock:
            return device in self._broken_devices

    def add_broken_core(self, location: OnChipCoordinate, risc_name: str) -> None:
        with self._lock:
            self._broken_cores.add(RiscLocation(location, None, risc_name))

    def get_device_broken_cores(self, device: Device) -> set[RiscLocation]:
        with self._lock:
            return {rl for rl in self._broken_cores if rl.location.device == device}

    def get_location_broken_cores(self, location: OnChipCoordinate) -> set[RiscLocation]:
        with self._lock:
            return {rl for rl in self._broken_cores if rl.location == location}

    def is_halted_core(self, location: OnChipCoordinate, risc_name: str) -> bool:
        with self._lock:
            return RiscLocation(location, None, risc_name) in self._halted_cores

    def add_halted_core(self, location: OnChipCoordinate, risc_name: str) -> None:
        with self._lock:
            self._halted_cores.add(RiscLocation(location, None, risc_name))

    @property
    def halted_cores(self) -> set[RiscLocation]:
        with self._lock:
            return self._halted_cores.copy()


_triage_session: TriageSession = TriageSession()


def get_triage_session() -> TriageSession:
    global _triage_session
    return _triage_session
