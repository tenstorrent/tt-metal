#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    heartbeat_samples

Description:
    Data provider script that collects a baseline ARC heartbeat sample for each device.

    A heartbeat sample pairs the current TIMER_HEARTBEAT telemetry value with a monotonic
    timestamp. Taking two samples separated in time allows consumers (e.g. check_arc) to
    estimate the ARC heartbeat rate and uptime.

Owner:
    ihamer-tt
"""

from dataclasses import dataclass
import time

from triage import ScriptPriority, triage_singleton, ScriptConfig, run_script
from run_checks import run as get_run_checks
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.tt_exalens_lib import read_arc_telemetry_entry

script_config = ScriptConfig(
    data_provider=True,
    depends=["run_checks"],
    priority=ScriptPriority.HIGH,
)


@dataclass
class HeartbeatSample:
    heartbeat: int
    timestamp: float


def get_heartbeat_sample(device: Device) -> HeartbeatSample:
    return HeartbeatSample(
        heartbeat=read_arc_telemetry_entry(device.arc_block.location.device_id, "TIMER_HEARTBEAT"),
        timestamp=time.monotonic(),
    )


@triage_singleton
def run(args, context: Context) -> dict[Device, HeartbeatSample]:
    run_checks = get_run_checks(args, context)
    return {
        sample.device_description.device: sample.result
        for sample in run_checks.run_per_device_check(lambda device: get_heartbeat_sample(device))
    }


if __name__ == "__main__":
    run_script()
