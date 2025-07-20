#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Script Name: check_noc_locations.py

Usage:
    check_noc_locations

Description:
    Checking that we can reach all NOC endpoints through NOC0 and NOC1.
"""

from check_per_device import run as get_check_per_device
from ttexalens.context import Context
from ttexalens.device import Device
from triage import ScriptConfig, log_check

script_config = ScriptConfig(
    depends=["check_per_device"],
)


def check_noc_locations(device: Device, noc_id: int):
    noc_str = f"noc{noc_id}"
    block_types = ["functional_workers", "eth"]
    for block_type in block_types:
        for location in device.get_block_locations(block_type):
            noc_block = device.get_block(location)
            register_store = noc_block.get_register_store(noc_id)
            data = register_store.read_register("NOC_NODE_ID")
            n_x = data & 0x3F
            n_y = (data >> 6) & 0x3F
            loc_to_noc = location.to(noc_str)
            log_check(
                loc_to_noc != (n_x, n_y),
                f"Device {device._id} {block_type} block at {location.to_user_str()} has wrong NOC location ({n_x}-{n_y})",
            )


def run(args, context: Context):
    check_per_device = get_check_per_device(args, context)
    check_per_device.run_check(lambda device: check_noc_locations(device, noc_id=0))
    check_per_device.run_check(lambda device: check_noc_locations(device, noc_id=1))


if __name__ == "__main__":
    from triage import run_script

    run_script()
