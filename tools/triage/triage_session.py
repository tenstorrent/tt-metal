#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import threading
from dataclasses import dataclass

from ttexalens.coordinate import OnChipCoordinate
from ttexalens.device import Device


@dataclass(frozen=True)
class BrokenCore:
    location: OnChipCoordinate
    risc_name: str

    def __hash__(self):
        return hash((self.location, self.risc_name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BrokenCore):
            return False
        return self.location == other.location and self.risc_name == other.risc_name

    def __str__(self) -> str:
        return f"{self.risc_name} at {self.location.to_user_str()}"


class TriageSession:
    """Singleton that holds all mutable state accumulated during a triage run."""

    def __init__(self):
        self._lock = threading.Lock()
        self._broken_devices: set[Device] = set()
        self._broken_cores: dict[Device, set[BrokenCore]] = {}
        self._halted_cores: set[tuple[OnChipCoordinate, str]] = set()

    def add_broken_device(self, device: Device) -> None:
        with self._lock:
            self._broken_devices.add(device)

    def is_device_broken(self, device: Device) -> bool:
        with self._lock:
            return device in self._broken_devices

    def add_broken_core(self, device: Device, broken_core: BrokenCore) -> None:
        with self._lock:
            if device in self._broken_cores:
                self._broken_cores[device].add(broken_core)
            else:
                self._broken_cores[device] = {broken_core}

    def is_core_broken(self, device: Device, location: OnChipCoordinate, risc_name: str) -> bool:
        with self._lock:
            if device not in self._broken_cores:
                return False
            return BrokenCore(location, risc_name) in self._broken_cores[device]

    def is_device_in_broken_cores(self, device: Device) -> bool:
        with self._lock:
            return device in self._broken_cores

    def get_device_broken_cores(self, device: Device) -> set[BrokenCore] | None:
        with self._lock:
            cores = self._broken_cores.get(device)
            return cores.copy() if cores is not None else None

    def add_halted_core(self, location: OnChipCoordinate, risc_name: str) -> None:
        with self._lock:
            self._halted_cores.add((location, risc_name))

    @property
    def halted_cores(self) -> set[tuple[OnChipCoordinate, str]]:
        with self._lock:
            return self._halted_cores.copy()


_triage_session: TriageSession | None = None


def get_triage_session() -> TriageSession:
    global _triage_session
    if _triage_session is None:
        _triage_session = TriageSession()
    return _triage_session
