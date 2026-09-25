#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    arc_heartbeat_sampling

Description:
    Data provider script that takes inital snapshot of ARC heartbeat sample for each device and allows taking new snapshots.

Owner:
    adjordjevic-TT
"""

from dataclasses import dataclass
from typing import cast
import time

from triage import ScriptPriority, triage_singleton, ScriptConfig, run_script
from run_checks import run as get_run_checks, RunChecks
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.tt_exalens_lib import read_firmware_telemetry_entry
from triage_hw_utils import device_has_firmware

script_config = ScriptConfig(
    data_provider=True,
    depends=["run_checks"],
    # We want this script to run as early as possible so more time passes between this script and its clients.
    # This way we get more precise hb/s estimation in check_arc.
    priority=ScriptPriority.HIGH,
)


@dataclass
class HeartbeatSample:
    heartbeat: int
    timestamp: float


class ArcHeartbeatSampling:
    def __init__(self, run_checks: RunChecks):
        self.initial_samples: dict[Device, HeartbeatSample] = {
            sample.device_description.device: cast(HeartbeatSample, sample.result)
            for sample in (run_checks.run_per_device_check(self.get_heartbeat_sample) or [])
        }

    def get_heartbeat_sample(self, device: Device) -> HeartbeatSample | None:
        # No readable device firmware means no heartbeat to sample.
        if not device_has_firmware(device):
            return None
        return HeartbeatSample(
            heartbeat=read_firmware_telemetry_entry(device.id, "TIMER_HEARTBEAT"),
            timestamp=time.monotonic(),
        )

    def get_initial_heartbeat_sample(self, device: Device) -> HeartbeatSample | None:
        return self.initial_samples.get(device)


@triage_singleton
def run(args, context: Context) -> ArcHeartbeatSampling:
    run_checks = get_run_checks(args, context)
    return ArcHeartbeatSampling(run_checks)


if __name__ == "__main__":
    run_script()
