#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    device_telemetry

Description:
    Reports per-device live telemetry from ARC: AI and ARC clock speeds,
    ASIC and board temperatures, ETH port status, DDR status and speed,
    and estimated ARC uptime from the heartbeat counter.
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
class DeviceTelemetryRow:
    aiclk: str = triage_field("AI Clk")
    arcclk: str = triage_field("ARC Clk")
    asic_temp: str = triage_field("ASIC Temp")
    board_temp: str = triage_field("Board Temp")
    eth_live: str = triage_field("ETH Live")
    ddr_status: str = triage_field("DDR Status")
    ddr_speed: str = triage_field("DDR Speed")
    uptime: str = triage_field("ARC Uptime")


def get_device_telemetry(device: Device) -> DeviceTelemetryRow:
    device_id = device.id
    return DeviceTelemetryRow(
        aiclk=read_tag(device_id, "AICLK"),
        arcclk=read_tag(device_id, "ARCCLK"),
        asic_temp=read_tag(device_id, "ASIC_TEMPERATURE"),
        board_temp=read_tag(device_id, "BOARD_TEMPERATURE"),
        eth_live=read_tag(device_id, "ETH_LIVE_STATUS"),
        ddr_status=read_tag(device_id, "DDR_STATUS"),
        ddr_speed=read_tag(device_id, "DDR_SPEED"),
        uptime=read_tag(device_id, "TIMER_HEARTBEAT"),
    )


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    return run_checks.run_per_device_check(lambda device: get_device_telemetry(device))


if __name__ == "__main__":
    run_script()
