#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_arc

Description:
    Checking that ARC heartbeat is running. Estimating ARC uptime.
"""

from dataclasses import dataclass
from triage import ScriptConfig, triage_field, hex_serializer, log_check_device, run_script
from run_checks import run as get_run_checks
from datetime import timedelta
import time
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.hardware.noc_block import NocBlock
from ttexalens.tt_exalens_lib import read_arc_telemetry_entry
import utils
from utils import RED, BLUE, RST

script_config = ScriptConfig(
    depends=["run_checks"],
)


@dataclass
class ArcCheckData:
    location: OnChipCoordinate = triage_field("Loc")
    postcode: int = triage_field("Postcode", hex_serializer)
    uptime: timedelta = triage_field("Up time")
    clock_mhz: int = triage_field("Clock MHz")
    heartbeats_per_second: float = triage_field("Heartbeats/s")


def check_arc_block(arc: NocBlock, postcode: int) -> ArcCheckData:
    device = arc.location.device
    device_id = arc.location.device_id
    # Heartbeat must be increasing
    heartbeat_0 = read_arc_telemetry_entry(device_id, "TIMER_HEARTBEAT")
    delay_seconds = 0.2
    time.sleep(delay_seconds)
    heartbeat_1 = read_arc_telemetry_entry(device_id, "TIMER_HEARTBEAT")
    log_check_device(device, heartbeat_1 > heartbeat_0, f"ARC heartbeat not increasing: {RED}{heartbeat_1}{RST}.")

    arcclk_mhz = read_arc_telemetry_entry(device_id, "ARCCLK")
    heartbeats_per_second = (heartbeat_1 - heartbeat_0) / delay_seconds

    # We do this in order to support all firmware versions
    # This way we do not support uptime longer than around 8 years, but that is unrealistic
    heartbeat_offset = 0xA5A5A5A5 if heartbeat_1 >= 0xA5A5A5A5 else 0
    assert (
        heartbeat_1 > heartbeat_offset
    ), f"ARC heartbeat lower than default value: {RED}{heartbeat_1}{RST}. Expected at least {BLUE}{heartbeat_offset}{RST}"
    uptime_seconds = (heartbeat_1 - heartbeat_offset) / heartbeats_per_second

    # Heartbeat must be between 5 and 50
    log_check_device(
        device,
        heartbeats_per_second >= 5,
        f"ARC heartbeat is too low: {RED}{heartbeats_per_second}{RST}hb/s. Expected at least {BLUE}5{RST}hb/s",
    )
    log_check_device(
        device,
        heartbeats_per_second <= 50,
        f"ARC heartbeat is too high: {RED}{heartbeats_per_second}{RST}hb/s. Expected at most {BLUE}50{RST}hb/s",
    )

    return ArcCheckData(
        location=arc.location,
        postcode=postcode,
        uptime=timedelta(seconds=uptime_seconds),
        clock_mhz=arcclk_mhz,
        heartbeats_per_second=heartbeats_per_second,
    )


def check_arc(device: Device):
    arc = device.arc_block
    postcode = arc.get_register_store().read_register("ARC_RESET_SCRATCH0")
    log_check_device(
        device,
        (postcode & 0xFFFF0000) == 0xC0DE0000,
        f"ARC postcode: {RED}0x{postcode:08x}{RST}. Expected {BLUE}0xc0de____{RST}",
    )
    if device.is_wormhole() or device.is_blackhole():
        return check_arc_block(arc, postcode)
    else:
        utils.DEBUG(f"Unsupported architecture for check_arc: {device._arch}")


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    return run_checks.run_per_device_check(lambda device: check_arc(device))


if __name__ == "__main__":
    run_script()
