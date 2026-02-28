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
from triage_hw_utils import read_tag

script_config = ScriptConfig()


@dataclass
class DeviceTelemetryRow:
    dev: str = triage_field("Dev")
    aiclk: str = triage_field("AI Clk")
    arcclk: str = triage_field("ARC Clk")
    asic_temp: str = triage_field("ASIC Temp")
    board_temp: str = triage_field("Board Temp")
    eth_live: str = triage_field("ETH Live")
    ddr_status: str = triage_field("DDR Status")
    ddr_speed: str = triage_field("DDR Speed")
    uptime: str = triage_field("ARC Uptime")


def run(args, context: Context):
    rows = []
    for device_id, device in context.devices.items():
        rows.append(
            DeviceTelemetryRow(
                dev=str(device.id),
                aiclk=read_tag(device_id, "AICLK"),
                arcclk=read_tag(device_id, "ARCCLK"),
                asic_temp=read_tag(device_id, "ASIC_TEMPERATURE"),
                board_temp=read_tag(device_id, "BOARD_TEMPERATURE"),
                eth_live=read_tag(device_id, "ETH_LIVE_STATUS"),
                ddr_status=read_tag(device_id, "DDR_STATUS"),
                ddr_speed=read_tag(device_id, "DDR_SPEED"),
                uptime=read_tag(device_id, "TIMER_HEARTBEAT"),
            )
        )
    return rows


if __name__ == "__main__":
    run_script()
