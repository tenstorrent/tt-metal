#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    blocks_to_check --type=<block_type>

Arguments:
    --type=<block_type> Specify the block type. options: tensix, idle_eth, active_eth, eth, arc, pcie, dram, router_only, security,
                                                         l2cpu, functional_workers, harvested_workers, harvested_eth, harvested_dram

Description:
    Provides list of block locations that should be checked for other scripts.
"""

from devices_to_check import run as get_devices_to_check
from triage import triage_singleton, ScriptConfig, run_script, TTTriageError
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.noc_block import NocBlock

script_config = ScriptConfig(
    data_provider=True,
    depends=["devices_to_check"],
)

VALID_BLOCK_TYPES = {
    "idle_eth",
    "active_eth",
    "tensix",
    "eth",
    "arc",
    "pcie",
    "dram",
    "router_only",
    "security",
    "l2cpu",
    "functional_workers",
    "harvested_workers",
    "harvested_eth",
    "harvested_dram",
}


def is_galaxy(device: Device) -> str:
    board_type = device.cluster_desc["chip_to_boardtype"]
    if board_type == "GALAXY":
        return True
    return False


def get_idle_eth_blocks(device: Device) -> list[NocBlock]:
    blocks = device.idle_eth_blocks
    # Remove idle eth block at location e0,15 (if galaxy also remove e0,0 e0,1 e0,2 e0,3)
    if device._has_mmio and device.cluster_desc["ethernet_connections_to_remote_devices"]:
        locations_to_remove = {"e0,0", "e0,1", "e0,2", "e0,3", "e0,15"} if is_galaxy(device) else {"e0,15"}
        for block in blocks:
            if block.location.to_str("logical") in locations_to_remove:
                blocks.remove(block)

    return blocks


def get_blocks(block_type: str, devices: list[Device]) -> dict[Device, list[NocBlock]]:
    if block_type not in VALID_BLOCK_TYPES:
        raise TTTriageError(f"Invalid block type {block_type}")

    blocks: dict[Device, list[NocBlock]] = {}
    match block_type:
        case "idle_eth":
            for device in devices:
                blocks[device] = get_idle_eth_blocks(device)
        case "active_eth":
            for device in devices:
                blocks[device] = device.active_eth_blocks
        case _:
            # In exalens we call tensix blocks functional_workers
            block_type = "functional_workers" if block_type == "tensix" else block_type
            for device in devices:
                blocks[device] = device.get_blocks(block_type)

    return blocks


@triage_singleton
def run(args, context: Context):
    block_type = args["--type"]
    devices = get_devices_to_check(args, context)
    return get_blocks(block_type, devices)


if __name__ == "__main__":
    run_script()
