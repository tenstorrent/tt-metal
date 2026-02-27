#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    firmware_versions

Description:
    Reports per-device firmware component versions read from ARC telemetry:
    ETH, Board Manager (app and bootloader), Flash Bundle and CM.
    Requires ARC firmware >= 18.4.

Owner:
    macimovic
"""

from dataclasses import dataclass

from triage import ScriptConfig, triage_field, run_script
from ttexalens.context import Context
from triage_hw_utils import read_tag

script_config = ScriptConfig()


@dataclass
class FirmwareVersionsRow:
    dev: str = triage_field("Dev")
    eth_fw: str = triage_field("ETH FW")
    bm_app_fw: str = triage_field("BM App FW")
    bm_bl_fw: str = triage_field("BM BL FW")
    flash_bundle: str = triage_field("Flash Bundle")
    cm_fw: str = triage_field("CM FW")


def run(args, context: Context):
    rows = []
    for device_id, device in context.devices.items():
        rows.append(
            FirmwareVersionsRow(
                dev=str(device.id),
                eth_fw=read_tag(device_id, "ETH_FW_VERSION"),
                bm_app_fw=read_tag(device_id, "BM_APP_FW_VERSION"),
                bm_bl_fw=read_tag(device_id, "BM_BL_FW_VERSION"),
                flash_bundle=read_tag(device_id, "FLASH_BUNDLE_VERSION"),
                cm_fw=read_tag(device_id, "CM_FW_VERSION"),
            )
        )
    return rows


if __name__ == "__main__":
    run_script()
