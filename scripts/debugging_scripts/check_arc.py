#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_arc

Description:
    Checking that ARC heartbeat is running. Estimating ARC uptime.
"""

from dataclasses import dataclass
from triage import ScriptConfig, triage_field, hex_serializer, log_check, run_script
from run_checks import run as get_run_checks
from datetime import timedelta
import time
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.hardware.noc_block import NocBlock
from ttexalens.hw.tensix.blackhole.blackhole import BlackholeDevice
from ttexalens.hw.tensix.wormhole.wormhole import WormholeDevice
from ttexalens.tt_exalens_lib import read_arc_telemetry_entry
import utils
from utils import RED, BLUE, GREEN, ORANGE, RST

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


def check_wormhole_arc(arc: NocBlock, postcode: int) -> ArcCheckData:
    device_id = arc.location._device._id
    # Heartbeat must be increasing
    heartbeat_0 = read_arc_telemetry_entry(device_id, "TAG_ARC0_HEALTH")
    delay_seconds = 0.1
    time.sleep(delay_seconds)
    heartbeat_1 = read_arc_telemetry_entry(device_id, "TAG_ARC0_HEALTH")
    log_check(heartbeat_1 > heartbeat_0, f"ARC heartbeat not increasing: {RED}{heartbeat_1}{RST}.")

    # Compute uptime
    arcclk_mhz = read_arc_telemetry_entry(device_id, "TAG_ARCCLK")
    heartbeats_per_second = (heartbeat_1 - heartbeat_0) / delay_seconds
    uptime_seconds = heartbeat_1 / heartbeats_per_second

    # Heartbeat must be between 500 and 20000 hb/s
    log_check(
        heartbeats_per_second >= 500,
        f"ARC heartbeat is too low: {RED}{heartbeats_per_second}{RST}hb/s. Expected at least {BLUE}500{RST}hb/s",
    )
    log_check(
        heartbeats_per_second <= 20000,
        f"ARC heartbeat is too high: {RED}{heartbeats_per_second}{RST}hb/s. Expected at most {BLUE}20000{RST}hb/s",
    )

    return ArcCheckData(
        location=arc.location,
        postcode=postcode,
        uptime=timedelta(seconds=uptime_seconds),
        clock_mhz=arcclk_mhz,
        heartbeats_per_second=heartbeats_per_second,
    )


def check_blackhole_arc(arc: NocBlock, postcode: int) -> ArcCheckData:
    device_id = arc.location._device._id
    # Heartbeat must be increasing
    heartbeat_0 = read_arc_telemetry_entry(device_id, "TAG_TIMER_HEARTBEAT")
    delay_seconds = 0.2
    time.sleep(delay_seconds)
    heartbeat_1 = read_arc_telemetry_entry(device_id, "TAG_TIMER_HEARTBEAT")
    log_check(heartbeat_1 > heartbeat_0, f"ARC heartbeat not increasing: {RED}{heartbeat_1}{RST}.")

    # Compute uptime
    arcclk_mhz = read_arc_telemetry_entry(device_id, "TAG_ARCCLK")
    heartbeats_per_second = (heartbeat_1 - heartbeat_0) / delay_seconds
    uptime_seconds = heartbeat_1 / heartbeats_per_second

    # Heartbeat must be between 10 and 50
    log_check(
        heartbeats_per_second >= 10,
        f"ARC heartbeat is too low: {RED}{heartbeats_per_second}{RST}hb/s. Expected at least {BLUE}10{RST}hb/s",
    )
    log_check(
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
    log_check(
        (postcode & 0xFFFF0000) == 0xC0DE0000,
        f"ARC postcode: {RED}0x{postcode:08x}{RST}. Expected {BLUE}0xc0de____{RST}",
    )
    if type(device) == WormholeDevice:
        return check_wormhole_arc(arc, postcode)
    elif type(device) == BlackholeDevice:
        return check_blackhole_arc(arc, postcode)
    else:
        utils.DEBUG(f"Unsupported architecture for check_arc: {device._arch}")


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    return run_checks.run_per_device_check(lambda device: check_arc(device))


if __name__ == "__main__":
    run_script()
