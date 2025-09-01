#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    block_to_check

Description:
    Provides list of block locations of given type that should be checked for other scripts.
"""

from ttexalens.hw.tensix.wormhole.wormhole import WormholeDevice
from devices_to_check import run as get_devices_to_check
from triage import triage_singleton, ScriptConfig, run_script, TTTriageError
from ttexalens.context import Context
from ttexalens.device import Device, OnChipCoordinate

script_config = ScriptConfig(
    data_provider=True,
    depends=["devices_to_check"],
)

VALID_BLOCK_TYPES = {
    "idle_eth",
    "active_eth",
    "tensix",
    "eth",
}


def is_galaxy(device: Device) -> str:
    return device.cluster_desc["chip_to_boardtype"] == "GALAXY"


def get_idle_eth_block_locations(device: Device) -> list[OnChipCoordinate]:
    block_locations = device.idle_eth_block_locations
    # We remove idle eth blocks that are reserved for syseng use
    # These are blocks on wormhole mmio capable devices with connections to remote devices
    # If board type is galaxy, we remove idle eth blocks at locations e0,0 e0,1 e0,2 e0,3 and e0,15,
    # if not we just remove e0,15
    if isinstance(device, WormholeDevice) and device._has_mmio:
        locations_to_remove = {"e0,0", "e0,1", "e0,2", "e0,3", "e0,15"} if is_galaxy(device) else {"e0,15"}
        for location in block_locations:
            if location.to_str("logical") in locations_to_remove:
                block_locations.remove(location)

    return block_locations


def get_block_locations_to_check(block_type: str, device: Device) -> list[OnChipCoordinate]:
    if block_type not in VALID_BLOCK_TYPES:
        raise TTTriageError(f"Invalid block type {block_type}")

    block_locations: dict[Device, list[OnChipCoordinate]] = {}
    match block_type:
        case "idle_eth":
            return get_idle_eth_block_locations(device)
        case "active_eth":
            return device.active_eth_block_locations
        case _:
            # In exalens we call tensix blocks functional_workers
            block_type = "functional_workers" if block_type == "tensix" else block_type
            return device.get_block_locations(block_type)


@triage_singleton
def run(args, context: Context):
    devices = get_devices_to_check(args, context)
    return {
        device: {block_type: get_block_locations_to_check(block_type, device) for block_type in VALID_BLOCK_TYPES}
        for device in devices
    }


if __name__ == "__main__":
    run_script()
