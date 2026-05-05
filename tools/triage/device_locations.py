#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    device_locations

Description:
    Data provider that resolves each ttexalens Device to its physical location:
    PCI BDF (e.g. "0000:01:00.0") and Galaxy tray number / tray bus id.

    Wraps `galaxy_topology.get_pci_bdf` + `compute_tray` so dump scripts that
    want tray-aware output don't have to repeat the per-device walk or reach
    into the name-mangled ttexalens UMD attributes themselves.

    Cached with @triage_singleton — computed once per triage run.

Owner:
    miacim
"""

from __future__ import annotations

from dataclasses import dataclass

from galaxy_topology import compute_tray, get_pci_bdf, ubb_table_for_devices
from run_checks import run as get_run_checks, RunChecks
from triage import ScriptConfig, triage_singleton, run_script
from ttexalens.context import Context


script_config = ScriptConfig(
    data_provider=True,
    depends=["run_checks"],
)


@dataclass
class DeviceLocation:
    """Per-device physical-location summary."""

    pci_bdf: str | None
    tray_num: int | None
    tray_bus_id: int | None


class DeviceLocations:
    """Lookup table from logical device id -> DeviceLocation, plus cached arch / tray map."""

    def __init__(
        self,
        by_device_id: dict[int, DeviceLocation],
        ubb_table: dict[int, int] | None,
    ):
        self._by_id = by_device_id
        self._ubb_table = ubb_table

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, device_id: int) -> DeviceLocation | None:
        return self._by_id.get(device_id)

    def pci_bdf(self, device_id: int) -> str | None:
        loc = self._by_id.get(device_id)
        return loc.pci_bdf if loc is not None else None

    def tray_num(self, device_id: int) -> int | None:
        loc = self._by_id.get(device_id)
        return loc.tray_num if loc is not None else None

    def tray_bus_id(self, device_id: int) -> int | None:
        loc = self._by_id.get(device_id)
        return loc.tray_bus_id if loc is not None else None

    def tray_to_devices(self) -> dict[int, list[int]]:
        """Return {tray_num: sorted list of device ids on that tray}."""
        result: dict[int, list[int]] = {}
        for device_id, loc in self._by_id.items():
            if loc.tray_num is None:
                continue
            result.setdefault(loc.tray_num, []).append(device_id)
        for tray in result:
            result[tray].sort()
        return result

    @property
    def ubb_table(self) -> dict[int, int] | None:
        """Return the UBB tray->bus_id table for the host's arch (WH or BH), or None."""
        return self._ubb_table


@triage_singleton
def run(args, context: Context) -> DeviceLocations:
    """Build per-device PCI/tray info. Cached per (args, context)."""
    run_checks: RunChecks = get_run_checks(args, context)

    by_id: dict[int, DeviceLocation] = {}
    for device in run_checks.devices:
        bdf = get_pci_bdf(device)
        arch = getattr(getattr(device, "_umd_device", None), "arch", None)
        tray_num, tray_bus_id = compute_tray(bdf, arch)
        by_id[int(device.id)] = DeviceLocation(
            pci_bdf=bdf,
            tray_num=tray_num,
            tray_bus_id=tray_bus_id,
        )

    return DeviceLocations(by_id, ubb_table_for_devices(run_checks.devices))


if __name__ == "__main__":
    run_script()
