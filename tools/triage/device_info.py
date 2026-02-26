#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    device_info

Description:
    Reports per-device hardware information: architecture, board type, unique ID,
    ARC firmware version and ARC postcode state.

Owner:
    macimovic
"""

from dataclasses import dataclass

from triage import ScriptConfig, triage_field, run_script
from ttexalens.context import Context

script_config = ScriptConfig()


@dataclass
class DeviceInfoRow:
    dev: str = triage_field("Dev")
    arch: str = triage_field("Arch")
    board_type: str = triage_field("Board")
    unique_id: str = triage_field("Unique ID")
    arc_fw: str = triage_field("ARC FW")
    postcode: str = triage_field("Postcode")


def run(args, context: Context):
    rows = []
    for device_id, device in context.devices.items():
        fw = device.firmware_version

        if not device.is_blackhole():
            try:
                raw = device.arc_block.get_register_store().read_register("ARC_RESET_SCRATCH0")
                postcode = hex(raw)
            except Exception as e:
                postcode = f"error: {e}"
        else:
            postcode = "N/A"

        rows.append(
            DeviceInfoRow(
                dev=str(device.id),
                arch=str(device._arch),
                board_type=str(device.board_type),
                unique_id=hex(device.unique_id),
                arc_fw=f"{fw.major}.{fw.minor}.{fw.patch}",
                postcode=postcode,
            )
        )
    return rows


if __name__ == "__main__":
    run_script()
