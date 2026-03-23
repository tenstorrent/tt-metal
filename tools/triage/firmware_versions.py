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
from ttexalens.device import Device
from triage_hw_utils import read_tag

from run_checks import run as get_run_checks

script_config = ScriptConfig(depends=["run_checks"])


@dataclass
class FirmwareVersionsRow:
    eth_fw: str = triage_field("ETH FW")
    bm_app_fw: str = triage_field("BM App FW")
    bm_bl_fw: str = triage_field("BM BL FW")
    flash_bundle: str = triage_field("Flash Bundle")
    cm_fw: str = triage_field("CM FW")


def get_firmware_versions(device: Device) -> FirmwareVersionsRow:
    device_id = device.id
    return FirmwareVersionsRow(
        eth_fw=read_tag(device_id, "ETH_FW_VERSION"),
        bm_app_fw=read_tag(device_id, "BM_APP_FW_VERSION"),
        bm_bl_fw=read_tag(device_id, "BM_BL_FW_VERSION"),
        flash_bundle=read_tag(device_id, "FLASH_BUNDLE_VERSION"),
        cm_fw=read_tag(device_id, "CM_FW_VERSION"),
    )


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    return run_checks.run_per_device_check(lambda device: get_firmware_versions(device))


if __name__ == "__main__":
    run_script()
