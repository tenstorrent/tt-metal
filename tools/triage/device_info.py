#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    device_info

Description:
    Per-device hardware report: tray/board grouping, ASIC position, architecture,
    board type, PCI bus id and BDF, unique id, ARC firmware version, and ARC
    postcode state. One row per device.

    Groups multiple chips that share a physical sub-unit through the `Tray/Board`
    column - UBB tray (1..4) on Wormhole/Blackhole Galaxy boards, board id on
    N300/P300, and board id on single-chip boards. Sort by that column to
    recover the "chips per tray/board" view.

Owner:
    macimovic
"""

from dataclasses import dataclass

from tt_umd import ClusterDescriptor

from triage import ScriptConfig, triage_field, run_script
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.umd_device import TimeoutDeviceRegisterError

from run_checks import run as get_run_checks

script_config = ScriptConfig(depends=["run_checks"])


@dataclass
class DeviceInfoRow:
    tray_or_board: str = triage_field("Tray/Board")
    chip_id: int = triage_field("Chip")
    asic_loc: str = triage_field("ASIC")
    arch: str = triage_field("Arch")
    board_type: str = triage_field("Board")
    bus_id: str = triage_field("Bus ID")
    pci_bdf: str = triage_field("PCI BDF")
    unique_id: str = triage_field("Unique ID")
    arc_fw: str = triage_field("ARC FW")
    postcode: str = triage_field("Postcode")


def _tray_or_board_label(cd: ClusterDescriptor, chip_id: int) -> str:
    tray = cd.get_tray_id(chip_id)
    if tray is not None:
        return f"Tray {tray}"
    return f"Board {cd.get_board_id_for_chip(chip_id):#x}"


def _bus_id_label(cd: ClusterDescriptor, chip_id: int) -> str:
    if not cd.is_chip_mmio_capable(chip_id):
        return ""
    return f"{cd.get_bus_id(chip_id):#04x}"


def get_device_info(device: Device, cd: ClusterDescriptor, pci_bdfs: dict) -> DeviceInfoRow:
    fw = device.firmware_version
    chip_id = device.id

    if not device.is_blackhole():
        try:
            raw = device.arc_block.get_register_store().read_register("ARC_RESET_SCRATCH0")
            postcode = hex(raw)
        except TimeoutDeviceRegisterError:
            raise
        except Exception as e:
            postcode = f"error: {e}"
    else:
        postcode = "N/A"

    return DeviceInfoRow(
        tray_or_board=_tray_or_board_label(cd, chip_id),
        chip_id=chip_id,
        asic_loc=str(cd.get_asic_location(chip_id)),
        arch=str(device._arch),
        board_type=str(device.board_type),
        bus_id=_bus_id_label(cd, chip_id),
        pci_bdf=pci_bdfs.get(chip_id, ""),
        unique_id=hex(device.unique_id),
        arc_fw=f"{fw.major}.{fw.minor}.{fw.patch}",
        postcode=postcode,
    )


def _row_sort_key(per_device_result) -> tuple:
    info = per_device_result.result
    label = info.tray_or_board if isinstance(info, DeviceInfoRow) else ""
    return (label, per_device_result.device_description.device.id)


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    cd = context.cluster_descriptor
    pci_bdfs = cd.get_chip_pci_bdfs()
    results = run_checks.run_per_device_check(lambda device: get_device_info(device, cd, pci_bdfs))
    if results:
        results.sort(key=_row_sort_key)
    return results


if __name__ == "__main__":
    run_script()
