#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    block_locations_to_check

Description:
    Provides list of block locations of supported types that should be checked for other scripts.
"""

from typing import Literal, Tuple, TypeAlias
from ttexalens.hw.tensix.wormhole.wormhole import WormholeDevice
from devices_to_check import run as get_devices_to_check
from triage import triage_singleton, ScriptConfig, run_script
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.coordinate import OnChipCoordinate

script_config = ScriptConfig(
    data_provider=True,
    depends=["devices_to_check"],
)

# List of block types that script returns, can be extended if other block types are needed
BLOCK_TYPES = [
    "idle_eth",
    "active_eth",
    "tensix",
    "eth",
]

BlockType: TypeAlias = Literal[BLOCK_TYPES]


def is_galaxy(device: Device) -> bool:
    return device.cluster_desc["chip_to_boardtype"][device._id] == "GALAXY"


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


def get_block_locations_to_check(block_type: BlockType, device: Device) -> list[OnChipCoordinate]:
    match block_type:
        case "idle_eth":
            return get_idle_eth_block_locations(device)
        case "active_eth":
            return device.active_eth_block_locations
        case _:
            # In exalens we call tensix blocks functional_workers
            block_type = "functional_workers" if block_type == "tensix" else block_type
            return device.get_block_locations(block_type)


class BlockLocationsToCheck:
    def __init__(self, devices: list[Device]):
        self._data: dict[Device, dict[BlockType, list[OnChipCoordinate]]] = {
            device: {block_type: get_block_locations_to_check(block_type, device) for block_type in BLOCK_TYPES}
            for device in devices
        }

    def __getitem__(self, key: Tuple[Device, BlockType]) -> list[OnChipCoordinate]:
        device, block_type = key
        return self._data[device][block_type]

    def get_devices(self) -> list[Device]:
        return list(self._data.keys())


@triage_singleton
def run(args, context: Context):
    devices = get_devices_to_check(args, context)
    return BlockLocationsToCheck(devices)


if __name__ == "__main__":
    run_script()
