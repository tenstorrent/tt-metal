#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for mapping ttexalens devices to Galaxy tray topology.

Mirrors the logic tt-smi already uses for `-glx_list_tray_to_device`:
  - PCI BDF is read from the underlying tt_umd device.
  - Tray number is derived from `bus_id & 0xF0` via the WH/BH UBB tables.
"""

from __future__ import annotations

# UBB bus-id tables. Sourced from tt_smi.constants when available; fall back to
# inline literals so triage still works on hosts without tt-smi installed.
try:
    from tt_smi import constants as _tt_smi_constants

    WH_UBB_BUS_IDS: dict[int, int] = dict(_tt_smi_constants.WH_UBB_BUS_IDS)
    BH_UBB_BUS_IDS: dict[int, int] = dict(_tt_smi_constants.BH_UBB_BUS_IDS)
except ImportError:
    WH_UBB_BUS_IDS = {1: 0xC0, 2: 0x80, 3: 0x00, 4: 0x40}
    BH_UBB_BUS_IDS = {1: 0x00, 2: 0x40, 3: 0xC0, 4: 0x80}


SUPPORTED_GALAXY_SHAPES: set[tuple[int, int]] = {(8, 4), (4, 8)}


# One-shot warning state so a UMD-ABI break (the name-mangled access path
# below stops working) shows up loudly on the first device but doesn't spam
# every subsequent one.
_pci_bdf_warned = False


def get_pci_bdf(device) -> str | None:
    """Return the PCI BDF (e.g. "0000:01:00.0") for a ttexalens Device, or None.

    Mirrors tt_smi's UMD path: device.get_pci_device().get_device_info().pci_bdf.
    Returns None silently for remote/JTAG-only devices (legitimate skip).
    On a name-mangled access break (UMD ABI change), emits a one-shot
    triage warning so the regression is visible instead of degrading to
    "no devices map to a UBB tray; skipping".

    TODO(ttexalens): UmdDevice should grow a public `pci_bdf` property so we
    don't have to reach through `_UmdDevice__device`.
    """
    global _pci_bdf_warned
    umd_device = getattr(device, "_umd_device", None)
    if umd_device is None:
        return None
    if not getattr(umd_device, "is_mmio_capable", True):
        return None
    try:
        tt_dev = umd_device._UmdDevice__device  # name-mangled — fragile, see TODO
        return tt_dev.get_pci_device().get_device_info().pci_bdf
    except AttributeError as e:
        if not _pci_bdf_warned:
            try:
                from triage import log_check

                log_check(
                    False,
                    f"galaxy_topology.get_pci_bdf: UMD attribute access failed ({e}); "
                    f"the name-mangled `_UmdDevice__device.get_pci_device().get_device_info().pci_bdf` "
                    f"path is broken — likely a tt_umd ABI change. Tray mapping will be unavailable.",
                )
            except Exception:
                pass
            _pci_bdf_warned = True
        return None


def ubb_table_for_arch(arch) -> dict[int, int] | None:
    """Pick WH or BH UBB table from a tt_umd ARCH enum, or None for unknown arch."""
    name = getattr(arch, "name", str(arch) if arch is not None else "").upper()
    if "WORMHOLE" in name:
        return WH_UBB_BUS_IDS
    if "BLACKHOLE" in name:
        return BH_UBB_BUS_IDS
    return None


def ubb_table_for_devices(devices) -> dict[int, int] | None:
    """Pick the WH/BH UBB table from the arch of the first available device."""
    for device in devices:
        arch = getattr(getattr(device, "_umd_device", None), "arch", None)
        if arch is not None:
            table = ubb_table_for_arch(arch)
            if table is not None:
                return table
    return None


def compute_tray(bdf: str | None, arch) -> tuple[int | None, int | None]:
    """Return (tray_num, tray_bus_id) for a BDF + arch, or (None, None) if not on a UBB tray."""
    if not bdf:
        return None, None
    table = ubb_table_for_arch(arch)
    if table is None:
        return None, None
    try:
        bus_id = int(bdf.split(":")[1], 16)
    except (IndexError, ValueError):
        return None, None
    tray_bus_id = bus_id & 0xF0
    bus_id_to_tray = {bus: tray for tray, bus in table.items()}
    tray_num = bus_id_to_tray.get(tray_bus_id)
    if tray_num is None:
        return None, None
    return tray_num, tray_bus_id


def build_tray_to_devices(devices) -> dict[int, list[int]]:
    """Map tray number -> sorted list of logical device ids running on that tray.

    Devices that don't resolve to a UBB tray (single-card hosts, remote chips,
    unknown arch) are simply omitted.
    """
    tray_to_devices: dict[int, list[int]] = {}
    for device in devices:
        bdf = get_pci_bdf(device)
        arch = getattr(getattr(device, "_umd_device", None), "arch", None)
        tray_num, _ = compute_tray(bdf, arch)
        if tray_num is None:
            continue
        tray_to_devices.setdefault(tray_num, []).append(int(device.id))
    for tray in tray_to_devices:
        tray_to_devices[tray].sort()
    return tray_to_devices


def device_to_cell(
    device_id: int,
    tray_num: int,
    sorted_tray_devices: list[int],
    shape: tuple[int, int],
) -> tuple[int, int]:
    """Map a logical device id to its (row, col) in the Galaxy mesh.

    Layout matches the physical Galaxy topology that `tt-smi
    -glx_list_tray_to_device` reflects:

      8x4 — 2x2 tray quadrants {T1:TL, T3:TR, T2:BL, T4:BR}, each tray a 4x2
            sub-grid arranged column-major (idx %4 -> sub-row, idx //4 -> sub-col).
      4x8 — 2x2 tray quadrants {T1:TL, T2:TR, T3:BL, T4:BR}, each tray a 2x4
            sub-grid arranged row-major (idx //4 -> sub-row, idx %4 -> sub-col).

    `sorted_tray_devices` must be the sorted list of device ids on `tray_num`;
    `idx` (0..7) is the device's position within that list.
    """
    if shape not in SUPPORTED_GALAXY_SHAPES:
        raise ValueError(f"Unsupported Galaxy shape {shape!r}; expected one of {SUPPORTED_GALAXY_SHAPES}")
    idx = sorted_tray_devices.index(device_id)
    if shape == (8, 4):
        quadrant = {1: (0, 0), 2: (1, 0), 3: (0, 1), 4: (1, 1)}[tray_num]
        sub_r, sub_c = idx % 4, idx // 4
        return quadrant[0] * 4 + sub_r, quadrant[1] * 2 + sub_c
    # shape == (4, 8)
    quadrant = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}[tray_num]
    sub_r, sub_c = idx // 4, idx % 4
    return quadrant[0] * 2 + sub_r, quadrant[1] * 4 + sub_c
