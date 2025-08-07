#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    devices_to_check [--dev=<device_id>]...

Options:
    --dev=<device_id>   Specify the device id. 'all' is also an option  [default: in_use]

Description:
    Provides list of devices that should be checked for other scripts.
"""

from inspector_data import run as get_inspector_data, InspectorData
from triage import triage_singleton, ScriptConfig, run_script
from ttexalens.context import Context
from ttexalens.device import Device
from utils import ORANGE, RST

script_config = ScriptConfig(
    data_provider=True,
    depends=["inspector_data"],
)


def get_devices(devices: list[str], inspector_data: InspectorData | None, context: Context) -> list[Device]:
    if len(devices) == 1 and devices[0].lower() == "in_use":
        if inspector_data is not None:
            device_ids = list(inspector_data.devices_in_use)
            if len(device_ids) == 0:
                print(
                    f"  {ORANGE}No devices in use found in inspector data. Switching to use all available devices. If you are using ttnn check if you have enabled program cache.{RST}"
                )
                device_ids = [int(id) for id in context.devices.keys()]
        else:
            print(f"  {ORANGE}Using all available devices.{RST}")
            device_ids = [int(id) for id in context.devices.keys()]
    elif len(devices) == 1 and devices[0].lower() == "all":
        device_ids = [int(id) for id in context.devices.keys()]
    else:
        device_ids = [int(id) for id in devices]
    return [context.devices[id] for id in device_ids]


@triage_singleton
def run(args, context: Context):
    devices = args["--dev"]
    inspector_data = get_inspector_data(args, context)
    return get_devices(devices, inspector_data, context)


if __name__ == "__main__":
    run_script()
