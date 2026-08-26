#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_arc

Description:
    Checking that ARC heartbeat is running. Estimating ARC uptime.

Owner:
    adjordjevic-TT
"""

from dataclasses import dataclass
from triage import ScriptConfig, ScriptPriority, triage_field, hex_serializer, log_check_device, run_script
from run_checks import run as get_run_checks
from arc_heartbeat_sampling import run as get_arc_heartbeat_sampling, ArcHeartbeatSampling
from datetime import timedelta
import time
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.hardware.noc_block import NocBlock
from ttexalens.tt_exalens_lib import read_arc_telemetry_entry
import utils

script_config = ScriptConfig(
    depends=["run_checks", "arc_heartbeat_sampling"],
    # We want this script to run as late as possible so more time passes and we have more precise hb/s estimation.
    priority=ScriptPriority.LOW,
)

MINIMAL_WAIT_SECONDS = 2
HEARTBEAT_INTERVAL_SECONDS = 0.1


@dataclass
class ArcCheckData:
    location: OnChipCoordinate = triage_field("Loc")
    postcode: int | None = triage_field("Postcode", hex_serializer)
    uptime: timedelta = triage_field("Up time")
    clock_mhz: int = triage_field("Clock MHz")
    heartbeats_per_second: float = triage_field("Heartbeats/s")


def check_arc_block(arc: NocBlock, postcode: int | None, arc_heartbeat_sampling: ArcHeartbeatSampling) -> ArcCheckData:
    device = arc.location.device
    device_id = arc.location.device_id
    # Heartbeat must be increasing
    current_heartbeat_sample = arc_heartbeat_sampling.get_heartbeat_sample(device)
    initial_heartbeat_sample = arc_heartbeat_sampling.get_initial_heartbeat_sample(device)
    arcclk_mhz = read_arc_telemetry_entry(device_id, "ARCCLK")
    heartbeats_per_second = (current_heartbeat_sample.heartbeat - initial_heartbeat_sample.heartbeat) / (
        current_heartbeat_sample.timestamp - initial_heartbeat_sample.timestamp
    )

    # We do this in order to support all firmware versions
    # This way we do not support uptime longer than around 8 years, but that is unrealistic
    heartbeat_offset = 0xA5A5A5A5 if current_heartbeat_sample.heartbeat >= 0xA5A5A5A5 else 0
    log_check_device(
        device,
        current_heartbeat_sample.heartbeat > heartbeat_offset,
        f"ARC heartbeat lower than default value: [error]{current_heartbeat_sample.heartbeat}[/]. Expected at least [info]{heartbeat_offset}[/]",
    )

    # Heartbeat must be between 9 and 11
    heartbeats_per_second_lower_bound = 9
    heartbeats_per_second_upper_bound = 11
    log_check_device(
        device,
        heartbeats_per_second >= heartbeats_per_second_lower_bound,
        f"ARC heartbeat is too low: [error]{heartbeats_per_second}[/]hb/s. Expected at least [info]{heartbeats_per_second_lower_bound}[/]hb/s",
    )
    log_check_device(
        device,
        heartbeats_per_second <= heartbeats_per_second_upper_bound,
        f"ARC heartbeat is too high: [error]{heartbeats_per_second}[/]hb/s. Expected at most [info]{heartbeats_per_second_upper_bound}[/]hb/s",
    )

    uptime_seconds = (current_heartbeat_sample.heartbeat - heartbeat_offset) * HEARTBEAT_INTERVAL_SECONDS

    return ArcCheckData(
        location=arc.location,
        postcode=postcode,
        uptime=timedelta(seconds=uptime_seconds),
        clock_mhz=arcclk_mhz,
        heartbeats_per_second=heartbeats_per_second,
    )


def check_arc(device: Device, arc_heartbeat_sampling: ArcHeartbeatSampling):
    arc = device.arc_block
    # We skip postcode check for blackhole devices due to https://github.com/tenstorrent/tt-exalens/issues/535
    if device.is_blackhole():
        postcode = None
    else:
        postcode = arc.get_register_store().read_register("ARC_RESET_SCRATCH0")
        log_check_device(
            device,
            (postcode & 0xFFFF0000) == 0xC0DE0000,
            f"ARC postcode: [error]0x{postcode:08x}[/]. Expected [info]0xc0de____[/]",
        )
    if device.is_wormhole() or device.is_blackhole():
        return check_arc_block(arc, postcode, arc_heartbeat_sampling)
    else:
        utils.DEBUG(f"Unsupported architecture for check_arc: {device._arch}")


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    arc_heartbeat_sampling = get_arc_heartbeat_sampling(args, context)

    # Ensuring that we wait at least MINIMAL_WAIT_SECONDS to ensure hb/s prediction is precise.
    latest_timestamp = max(sample.timestamp for sample in arc_heartbeat_sampling.initial_samples.values())
    smallest_wait = time.monotonic() - latest_timestamp
    if smallest_wait < MINIMAL_WAIT_SECONDS:
        time.sleep(MINIMAL_WAIT_SECONDS - smallest_wait)

    return run_checks.run_per_device_check(lambda device: check_arc(device, arc_heartbeat_sampling))


if __name__ == "__main__":
    run_script()
