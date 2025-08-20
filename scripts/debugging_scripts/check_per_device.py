#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_per_device

Description:
    Provides a way to run checks on each device in the system.
    This script will iterate over all devices and apply the specified checks.
    If check method returns value, data will be collected and returned. You can use this as a way to dump data for device (see `dump_callstacks.py` for example).
    If check method returns None, no data will be collected for that device. You can use this just to execute some checks on devices without collecting data (see `check_noc_locations.py` for example).
"""

from collections.abc import Callable
from dataclasses import dataclass
from devices_to_check import run as get_devices_to_check
from triage import triage_singleton, ScriptConfig, triage_field, recurse_field, run_script
from ttexalens.context import Context
from ttexalens.device import Device

script_config = ScriptConfig(
    data_provider=True,
    depends=["devices_to_check"],
)


@dataclass
class PerDeviceCheckResult:
    device: Device = triage_field("Dev")
    result: object = recurse_field()


class PerDeviceCheck:
    def __init__(self, devices: list[Device]):
        self.devices = devices

    def run_check(self, check: Callable[[Device], object]):
        result: list[PerDeviceCheckResult] = []
        for device in self.devices:
            check_result = check(device)
            if check_result is None:
                continue
            if isinstance(check_result, list):
                for item in check_result:
                    result.append(PerDeviceCheckResult(device=device, result=item))
            else:
                result.append(PerDeviceCheckResult(device=device, result=check_result))
        if len(result) == 0:
            return None
        return result


@triage_singleton
def run(args, context: Context):
    devices = get_devices_to_check(args, context)
    return PerDeviceCheck(devices)


if __name__ == "__main__":
    run_script()
